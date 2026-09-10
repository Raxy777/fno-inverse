"""
The four-stage inversion (§8.4, §11.2 steps 10-12).

    Stage 0  CNN regressor, one forward pass            -> one candidate
    Stage 1  256-candidate envelope screen, m = 1..6    -> 16 survivors
    Stage 2  Adam, 200 steps, m = 1..10, envelope       -> best survivor
    Stage 3  L-BFGS, full band, complex residual        -> the answer

No stage hard-codes its own objective or band: both come from
`cfg.STAGE_OBJECTIVE` and `misfit.STAGE_BANDS` through `Objective.for_stage`.  v2.0
had the pipeline written down in three places -- the document, `screen`'s
`amplitude=True`, and `refine_adam`'s default -- and they disagreed with each other
(the document said envelope refinement in stage 2; the code ran a complex misfit).
One table now decides, and `config.self_check` prints it.

Why four stages and not one Adam run from a random start: the misfit is non-convex
with a complex-residual basin about lambda_s/4 across (§11.3's landscape figure), so a
start further away than that converges to a cycle-skipped minimum -- one where the
predicted arrival is a whole period out and the residual is locally minimal but
globally wrong.  Every stage exists to hand the next one a starting point inside its
basin, and the widest basin belongs to the envelope, which is why the screen and
stage 2 use it.

Three implementation choices worth defending:

**The 16 survivors are optimised as one batch, not in a loop.**  They share the
incident field, the observation and the network, so a batch of 16 candidate
geometries is one forward pass of batch 16 x F instead of 16 passes of F.  The
objective is summed across candidates before `backward()`; since the candidates are
independent, row i's gradient depends only on row i, so the sum is exactly the same
as 16 separate backward passes and about an order of magnitude cheaper.

**The interface width is held at the trained value.**  v2.0 annealed eps from 2.0 to
1.0 cells while the surrogate had only ever seen 1.5; bracketing a training value is
not the same as validating either endpoint, and the surrogate -- not the geometry --
is what is being extrapolated.  `cfg.EPS_INVERT_ANNEAL` is False, so the schedule is
one entry long; setting it True restores the anneal, and
`inverse.sensitivity.eps_transfer_report` is what has to pass first.

**eps is annealed by restarting L-BFGS, not by changing it mid-run** (when it is
annealed at all).  L-BFGS approximates curvature from differences of gradients taken
at different iterates; if the objective changes underneath it, those differences
describe two different functions and the approximation is not merely stale but wrong
-- it will confidently step in a direction that was never a descent direction.  So
each eps gets a fresh history.
"""

from __future__ import annotations

import itertools
import math
import time
import warnings
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .. import config as cfg
from ..geometry.sdf import Circle, ShapeFamily, equivalent_circle
from .misfit import (InverseCase, Objective, SurrogateForward,
                     envelope_basin_width, tikhonov)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class InversionResult:
    theta: Tensor                       # [P] final estimate, physical
    misfit: float                       # final full-band complex misfit
    stages: dict = field(default_factory=dict)
    theta_true: Tensor | None = None
    lambda_s: float = 1.0
    seconds: float = 0.0
    n_forward: int = 0
    family: ShapeFamily | None = None   # needed for shape-appropriate scoring
    truth_family: ShapeFamily | None = None   # when the truth is a *different* family

    # -- scoring ----------------------------------------------------------
    #
    # Every metric below is permutation-invariant and family-aware.  v2.0's versions
    # indexed theta[:2] and theta[2] literally, which silently scored an ellipse's
    # semi-major axis as a radius, compared a two-void recovery's *first* blob against
    # whichever blob the generator happened to list first, and called a run that found
    # one void twice a success.  §9 of the review asks for shape-appropriate metrics;
    # these are they.  `family` is optional so that an unpickled v2.0 result still
    # loads, and when it is absent the circle interpretation is used and said so.
    #
    # `truth_family` is separate from `family` for one case only, and it is the case
    # the direct-regressor baseline lives in: a *circle* fit scored against an
    # *ellipse* or two-void truth.  Leaving it None used to mean the truth's columns
    # were read with the prediction's family, so a 5-parameter ellipse truth was
    # silently truncated to (xc, yc, a) and its semi-major axis scored as a radius --
    # the same defect the paragraph above says was fixed, one level up.  When the two
    # families differ, `iou` compares each shape through its own SDF (which is the
    # honest cross-family number, and the one §8.5's transfer claim rests on) while the
    # centre and size errors compare the truth's *equal-area circle*, because blob
    # slots are not in correspondence across families.  That reduction is lossy and it
    # is named: it is why IoU is the primary score and position is the secondary one.
    def _fam(self) -> ShapeFamily:
        return self.family if self.family is not None else Circle()

    def _fam_true(self) -> ShapeFamily:
        return self.truth_family if self.truth_family is not None else self._fam()

    def _cross_family(self) -> bool:
        return self._fam_true().name != self._fam().name

    def _reduced(self) -> tuple[Tensor, Tensor, ShapeFamily] | None:
        """
        (theta_pred, theta_true, family) in one common parameterisation, or None.

        The identity when the two families agree, which is the ordinary case.  Across
        families *both* sides are reduced to their area-equivalent circle -- not just
        the truth -- so the comparison is symmetric and neither side is read with the
        other's column layout.  That last part is the bug this replaces: applying the
        prediction's family to `theta_true` truncated a 5-parameter ellipse truth to
        (xc, yc, a) and scored its semi-major axis as a radius, and in the other
        direction (an ellipse fit against a circle truth) indexed off the end of a
        3-vector.  Neither raised anything visible for the first case.

        The reduction is lossy, and it is why `iou` -- which compares each theta through
        its own SDF and reduces nothing -- is the primary cross-family score.
        """
        if self.theta_true is None:
            return None
        if not self._cross_family():
            return self.theta, self.theta_true, self._fam()
        p = equivalent_circle(self.theta.reshape(1, -1), self._fam()).reshape(-1)
        t = equivalent_circle(self.theta_true.reshape(1, -1),
                              self._fam_true()).reshape(-1)
        return p, t, Circle()

    def _matching(self) -> list[tuple[int, int]] | None:
        """
        Which predicted blob corresponds to which true blob.

        Returns [(pred_slot, true_slot), ...] over `_blob_slots` indices, choosing the
        permutation that minimises the total centre distance.  For one blob this is
        [(0, 0)]; for two it is a two-way comparison.  Blob *labels* carry no meaning --
        a two-void ground truth is a set of two voids, not an ordered pair -- so any
        metric that does not minimise over permutations is reporting a labelling
        accident as an error.
        """
        red = self._reduced()
        if red is None:
            return None
        p, t, fam = red
        slots = _blob_slots(fam)
        n = len(slots)
        p = p.detach().cpu().to(torch.float64)
        t = t.detach().cpu().to(torch.float64)

        def d(i: int, j: int) -> float:
            xi, yi = slots[i]
            xj, yj = slots[j]
            return float(((p[xi] - t[xj]) ** 2 + (p[yi] - t[yj]) ** 2).sqrt())

        best, best_cost = None, math.inf
        for perm in itertools.permutations(range(n)):
            cost = sum(d(i, perm[i]) for i in range(n))
            if cost < best_cost:
                best, best_cost = perm, cost
        return [(i, best[i]) for i in range(n)]

    @staticmethod
    def _radius_of(theta: Tensor, fam: ShapeFamily, blob: int) -> float | None:
        """The equal-area radius of one blob, or None if the family has no size."""
        names = fam.param_names
        if "a" in names and "b" in names:
            ia, ib = names.index("a"), names.index("b")
            return float((theta[ia] * theta[ib]).abs().sqrt())
        rs = [i for i, s in enumerate(names) if s.startswith("R")]
        return None if blob >= len(rs) else float(theta[rs[blob]].abs())

    def _radius_slot(self, blob: int) -> float | None:
        """The equal-area radius of one predicted blob, in physical units."""
        return self._radius_of(self.theta, self._fam(), blob)

    @property
    def position_error_ls(self) -> float | None:
        """
        The *worst* matched blob-centre error, in lambda_s.

        Worst rather than mean: a two-void run that nails one void and misses the other
        has not recovered the geometry, and averaging says it half did.
        """
        m = self._matching()
        if m is None:
            return None
        p, t, fam = self._reduced()
        slots = _blob_slots(fam)
        p = p.detach().cpu().to(torch.float64)
        t = t.detach().cpu().to(torch.float64)
        worst = 0.0
        for i, j in m:
            xi, yi = slots[i]
            xj, yj = slots[j]
            worst = max(worst, float(((p[xi] - t[xj]) ** 2
                                      + (p[yi] - t[yj]) ** 2).sqrt()))
        return worst / self.lambda_s

    @property
    def radius_error_ls(self) -> float | None:
        """
        Worst matched equal-area-radius error, in lambda_s.

        `sqrt(a b)` for an ellipse, so an axis swap costs nothing here (it is not a
        size error) and shows up in `axis_ratio_error` and `iou` instead.
        """
        m = self._matching()
        if m is None:
            return None
        p, t, fam = self._reduced()
        p = p.detach().cpu().to(torch.float64)
        t = t.detach().cpu().to(torch.float64)
        errs = []
        for i, j in m:
            rp = self._radius_of(p, fam, i)
            rt = self._radius_of(t, fam, j)
            if rp is not None and rt is not None:
                errs.append(abs(rp - rt))
        return max(errs) / self.lambda_s if errs else None

    @staticmethod
    def _canonical_ellipse(theta: Tensor, fam: ShapeFamily
                           ) -> tuple[float, float, float] | None:
        """
        (major, minor, orientation-of-the-major-axis) for an axis-swap-free comparison.

        (a, b, alpha) is a two-to-one parameterisation: (a, b, alpha) and
        (b, a, alpha + pi/2) are the same ellipse.  Differencing the raw parameters
        therefore reports a pure 90-degree rotation as both a large axis-ratio error
        *and* zero orientation error, which is exactly backwards.  Canonicalising to
        the major axis first makes the two errors independent and each one true.

        `fam` is the family *of that theta*, so an ellipse fit against a circle truth
        reads the truth with three columns instead of indexing off the end of it.
        """
        if "a" not in fam.param_names or "alpha" not in fam.param_names:
            return None
        ia, ib = fam.param_names.index("a"), fam.param_names.index("b")
        k = fam.param_names.index("alpha")
        a, b, al = (abs(float(theta[ia])), abs(float(theta[ib])), float(theta[k]))
        if b > a:
            a, b, al = b, a, al + math.pi / 2.0
        return a, b, al % math.pi

    @property
    def axis_ratio_error(self) -> float | None:
        """
        |(major/minor)_pred - (major/minor)_true|, else None.  Rotation-invariant.

        None whenever *either* side is not an ellipse, including the circle-fit-against-
        ellipse-truth case: a circle has no axis ratio, so there is no error to report,
        and reporting the truth's eccentricity as one would credit the fit with a
        measurement it never made.  `iou` is what registers that failure.
        """
        if self.theta_true is None:
            return None
        p = self._canonical_ellipse(self.theta, self._fam())
        t = self._canonical_ellipse(self.theta_true, self._fam_true())
        if p is None or t is None:
            return None
        return abs(p[0] / p[1] - t[0] / t[1])

    @property
    def orientation_error_deg(self) -> float | None:
        """
        Major-axis orientation error in degrees, modulo the ellipse's pi symmetry.

        NaN when *either* shape is within 2% of circular: a circle has no orientation,
        so the error is undefined whether it is the truth that is round (nothing to
        recover) or the fit (nothing recovered).  Averaging those in as numbers would
        let a family of circular fits to eccentric truths report a small mean
        orientation error while having found no orientation at all; `axis_ratio_error`
        and `iou` are what catch that case.
        """
        if self.theta_true is None:
            return None
        p = self._canonical_ellipse(self.theta, self._fam())
        t = self._canonical_ellipse(self.theta_true, self._fam_true())
        if p is None or t is None:
            return None
        if abs(t[0] / t[1] - 1.0) < 0.02 or abs(p[0] / p[1] - 1.0) < 0.02:
            return float("nan")
        d = (p[2] - t[2]) % math.pi
        return math.degrees(min(d, math.pi - d))

    def iou(self, *, eps_cells: float | None = None, n: int | None = None) -> float | None:
        """
        Soft-indicator intersection-over-union against the truth.  The primary score.

        sum min(chi_pred, chi_true) / sum max(chi_pred, chi_true), evaluated on the
        network grid at the interface width the surrogate was trained at, because that
        is the geometry the forward model actually saw.  Size-normalised, orientation-
        aware, permutation-free (it never has to decide which blob is which), and
        defined across families -- so an ellipse truth scored against a circle fit
        gives a meaningful number, which is exactly the transfer case of §8.5.

        Each side goes through *its own* family's SDF, which is what makes the last
        claim true.  It was not true before `truth_family` existed: the predicted
        family was applied to both thetas, so a circle fit against an ellipse truth
        compared itself to a circle of radius `a` and returned a number that looked
        like an overlap and was not one.
        """
        if self.theta_true is None:
            return None
        from ..geometry.sdf import grid_coords, soft_indicator
        fam, fam_true = self._fam(), self._fam_true()
        eps_cells = cfg.EPS_INTERFACE_CELLS if eps_cells is None else eps_cells
        n = cfg.N_NET if n is None else n
        dx = cfg.L_DOMAIN / n
        dev = self.theta.device
        yy, xx = grid_coords(n, dx, device=dev, dtype=torch.float64)
        eps_len = eps_cells * dx
        with torch.no_grad():
            tp = self.theta.reshape(1, -1).to(device=dev, dtype=torch.float64)
            tt = self.theta_true.reshape(1, -1).to(device=dev, dtype=torch.float64)
            a = soft_indicator(fam.sdf(tp, yy, xx), eps_len)
            b = soft_indicator(fam_true.sdf(tt, yy, xx), eps_len)
            inter = torch.minimum(a, b).sum()
            union = torch.maximum(a, b).sum()
        return float(inter / union) if float(union) > 0.0 else 0.0

    @property
    def success(self) -> bool:
        """
        Both gates: the centre is in the right place *and* the shape overlaps.

        The position gate alone passes a run that put a void of the wrong size or
        eccentricity in the right place; the IoU gate alone passes a large void that
        swallows a small misplaced one.  Requiring both is the honest conjunction, and
        for the circle family they nearly coincide by construction (see GATE_IOU).
        """
        e = self.position_error_ls
        if e is None or not e < cfg.GATE_POSITION_LS:
            return False
        v = self.iou()
        return v is None or v >= cfg.GATE_IOU

    @property
    def wall_saturation(self) -> float | None:
        """
        max |z| over the final estimate, in the unconstrained coordinates.

        The sigmoid reparameterisation keeps theta feasible without clipping, so the
        gradient is nonzero for every finite z *in exact arithmetic*.  In float32
        sigmoid'(z) underflows to exactly zero at |z| >= 16.75, at which point theta is
        1.2e-6 network cells from the wall and L-BFGS reads the dead gradient as
        convergence -- a run pinned to the edge of the feasible box reporting success.

        Note what this number can and cannot see.  It is computed *from theta*, and
        `to_unconstrained` clamps the normalised coordinate to [1e-4, 1 - 1e-4], so its
        value saturates at `cfg.WALL_Z_CLAMP` = 9.2103 no matter how far into the tail
        the optimiser's own z travelled.  It therefore detects "the answer is on the
        wall" (which is the actionable signal) and not "the gradient was dead" (which
        needs the live z and is pinned in test_geometry instead).  `summarise` counts
        runs above `cfg.WALL_Z_WARN` = 9.0; an earlier threshold of 12.0 was
        unreachable by construction and counted nothing.
        """
        if self.family is None:
            return None
        z = self.family.to_unconstrained(self.theta.reshape(1, -1), self.lambda_s)
        return float(z.abs().max())

    def summary(self) -> str:
        rows = [f"theta = {[round(float(v), 4) for v in self.theta]}",
                f"final misfit {self.misfit:.4e}   "
                f"{self.n_forward} forward evaluations   {self.seconds:.1f} s"]
        if self.theta_true is not None:
            v = self.iou()
            rad = self.radius_error_ls
            rows.append(
                f"position error {self.position_error_ls:.4f} lambda_s   "
                f"radius error {'n/a' if rad is None else f'{rad:.4f}'} lambda_s   "
                f"IoU {v:.3f}   "
                f"{'PASS' if self.success else 'FAIL'} "
                f"(gates: position < {cfg.GATE_POSITION_LS}, IoU >= {cfg.GATE_IOU})")
            extra = []
            if self.axis_ratio_error is not None:
                extra.append(f"axis-ratio error {self.axis_ratio_error:.3f}")
            if self.orientation_error_deg is not None:
                extra.append(f"orientation error {self.orientation_error_deg:.1f} deg")
            if len(_blob_slots(self._fam())) > 1:
                extra.append("blobs matched permutation-invariantly")
            if self._cross_family():
                extra.append(f"{self._fam().name} fit vs {self._fam_true().name} truth: "
                             "IoU is exact, position/radius via equal-area circles")
            if extra:
                rows.append("  " + "   ".join(extra))
        return "\n".join(rows)



# ---------------------------------------------------------------------------
# Stage 1: screening
# ---------------------------------------------------------------------------
def _blob_slots(family: ShapeFamily) -> list[tuple[int, int]]:
    """
    The (x-index, y-index) pairs in `family.param_names`.

    Read off the names rather than hard-coded per family, because the thing that made
    `screen` circle-only was three literal `3`s.  Every family in geometry/sdf.py
    names its centres xc/yc (or xc1/yc1, xc2/yc2), so this returns [(0,1)] for a
    circle or an ellipse and [(0,1), (3,4)] for a two-circle.
    """
    n = family.param_names
    xs = [i for i, s in enumerate(n) if s.startswith("xc")]
    ys = [i for i, s in enumerate(n) if s.startswith("yc")]
    if not xs or len(xs) != len(ys):
        raise ValueError(
            f"family {family.name!r} has parameters {n}, from which the screen cannot "
            "identify centre coordinates; name them xc/yc or supply candidates "
            "explicitly via screen(theta=...)")
    return list(zip(xs, ys))


def screen_candidates(family: ShapeFamily, lambda_s: float, *,
                      n_grid: int = cfg.SCREEN_GRID, radius: float | None = None,
                      n_orient: int = 4, separation: float | None = None) -> Tensor:
    """
    The screen's candidate set for *any* family.  [N, P] physical.

    Every parameter starts at the midpoint of its own bound, which for a circle is
    mid-radius, for an ellipse is a = b = mid-radius with alpha = 0 (i.e. a circle),
    and for a two-circle is two mid-radius circles.  The scan is then over *position*
    only: the centroid of the blobs moves on an n_grid x n_grid lattice spanning the
    feasible box.

    Position-only is a deliberate restriction, not an oversight.  The screen exists to
    put a starting point inside the basin, positional error is what closes the basin,
    and the shape parameters enter the scattered amplitude much more weakly (radius
    mostly as an overall factor, ~R^2 in the Rayleigh regime, which
    `scale_invariant=True` divides out).  Eccentricity and orientation are left to
    stages 2 and 3, which have gradients for them.

    The one exception is a family with more than one blob, where relative placement is
    not a weak parameter at all -- two voids side by side and the same two voids
    stacked scatter differently.  So for K > 1 blobs the lattice is crossed with
    `n_orient` orientations of a fixed centre-to-centre `separation`, giving
    n_grid^2 x n_orient candidates.  That is a coarse cover of a 6-D space and is
    honestly labelled as such: the two-void screen is a starting-point generator whose
    capture rate has to be measured (`screen_capture_rate`), not assumed.
    """
    lo, hi = family.bounds(lambda_s)
    mid = 0.5 * (lo + hi)
    if radius is None:
        radius = float(0.5 * (cfg.R_MIN_LS + cfg.R_MAX_LS) * lambda_s)
    slots = _blob_slots(family)
    k = len(slots)
    if separation is None:
        separation = 4.0 * radius           # a clear gap of two radii between surfaces

    xs = torch.linspace(float(lo[0]), float(hi[0]), n_grid)
    ys = torch.linspace(float(lo[1]), float(hi[1]), n_grid)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy = gx.reshape(-1), gy.reshape(-1)

    angles = [0.0] if k == 1 else [math.pi * i / n_orient for i in range(n_orient)]
    rows = []
    for phi in angles:
        th = mid.view(1, -1).repeat(cx.numel(), 1).clone()
        for b, (ix, iy) in enumerate(slots):
            off = separation * (b - 0.5 * (k - 1))
            th[:, ix] = cx + off * math.cos(phi)
            th[:, iy] = cy + off * math.sin(phi)
            # the radius-like slot of each blob, if the family has one
            for j, nm in enumerate(family.param_names):
                if nm in (f"R{b + 1}", "R") or (k == 1 and nm in ("a", "b")):
                    th[:, j] = radius
        rows.append(th)
    theta = torch.cat(rows, dim=0)
    return theta.clamp(min=lo, max=hi)      # a rotated pair must stay feasible


@torch.no_grad()
def screen(forward: SurrogateForward, case: InverseCase, family: ShapeFamily, *,
           n_grid: int = cfg.SCREEN_GRID, n_keep: int = cfg.N_SURVIVORS,
           chunk: int = cfg.SCREEN_CHUNK, radius: float | None = None,
           objective: str | None = None, theta: Tensor | None = None
           ) -> tuple[Tensor, Tensor]:
    """
    Coarse screen over a position grid, scored by `cfg.SCREEN_OBJECTIVE`.  (theta, J)

    The grid spans the feasible box, so its spacing is about
    (L - 2 x 1.5 lambda_s) / 16 ~ 0.3 lambda_s.  That is coarser than the
    lambda_s/4 complex-residual basin, which is why 16 survivors are kept rather than
    one: the true optimum may sit between grid nodes, and the nearest few nodes are
    then all mediocre and nearly tied.

    Whether it is coarser than the basin of the objective actually used here is a
    measured question, not an asserted one -- see `misfit.envelope_basin_width` and
    `screen_capture_rate`.  v2.0 asserted a factor of nearly three of margin, on the
    strength of an "envelope basin = N_c/2 x waveform basin" rule applied to a
    magnitude-spectrum misfit that is provably blind to the shift the basin is a basin
    for.  Both halves of that have been withdrawn.

    Runs at the *trained* interface width (`cfg.EPS_INTERFACE_CELLS`), not at
    `EPS_INVERT_START`: the screen is where the surrogate is asked the most questions,
    so it is the last place to be feeding it out-of-distribution inputs.
    """
    lam_s = case.lambda_s
    dev = forward.device
    if theta is None:
        theta = screen_candidates(family, lam_s, n_grid=n_grid, radius=radius)
    theta = theta.to(dev)

    obj = Objective.for_stage(1, forward, case, family,
                              objective=objective or cfg.SCREEN_OBJECTIVE,
                              eps_cells=cfg.EPS_INTERFACE_CELLS,
                              scale_invariant=True)
    j = torch.cat([obj.residual(theta[i:i + chunk]).cpu()
                   for i in range(0, theta.shape[0], chunk)])
    order = torch.argsort(j)[:n_keep]
    return theta[order.to(dev)], j[order]


# ---------------------------------------------------------------------------
# Does the screen actually capture the true basin?  (§11.2 step 10a)
# ---------------------------------------------------------------------------
def centroid(theta: Tensor, family: ShapeFamily) -> Tensor:
    """Mean of the blob centres, [..., 2] -- the one position every family has."""
    slots = _blob_slots(family)
    xs = torch.stack([theta[..., ix] for ix, _ in slots], dim=-1).mean(-1)
    ys = torch.stack([theta[..., iy] for _, iy in slots], dim=-1).mean(-1)
    return torch.stack([xs, ys], dim=-1)


@torch.no_grad()
def screen_capture_rate(forward: SurrogateForward, cases: list[InverseCase],
                        family: ShapeFamily | None = None, *,
                        objectives: tuple[str, ...] | None = None,
                        capture_ls: float | None = None,
                        n_keep: int = cfg.N_SURVIVORS,
                        n_grid: int = cfg.SCREEN_GRID,
                        chunk: int = cfg.SCREEN_CHUNK,
                        progress=None) -> dict:
    """
    Fraction of cases whose survivor set contains a candidate inside the true basin.

    This is the measurement that replaces v2.0's assertion.  The claim being tested is
    the only thing the screen is for: that at least one of the `n_keep` survivors
    starts stage 2 close enough to converge.  It is reported for
    `cfg.SCREEN_OBJECTIVE` and for `cfg.SCREEN_FALLBACK_OBJECTIVE` side by side, so
    the cost of the cheap magnitude screen is a number rather than an argument -- if
    the fallback captures as often, it is the better choice and should be made the
    default; if it does not, the review's first finding is confirmed empirically as
    well as algebraically.

    `capture_ls` is the capture radius in shear wavelengths.  Default: half the
    narrower measured envelope-basin width from `misfit.envelope_basin_width`, run once
    on the first case that has a ground truth.  That is the honest definition -- inside
    the basin means inside the *measured* basin -- but it costs an n x n misfit map, so
    pass a number to skip it.

    Position is compared as the centroid of the blob centres, which is the only
    quantity all three families share.  A two-void case whose survivors have the
    centroid right and the separation wrong therefore counts as captured; that is
    deliberate, since separation is what stages 2 and 3 have gradients for, but it
    means this rate is an upper bound on end-to-end success for multi-blob families.
    """
    family = family or Circle()
    objectives = objectives or (cfg.SCREEN_OBJECTIVE, cfg.SCREEN_FALLBACK_OBJECTIVE)
    truth = [c for c in cases if c.theta_true is not None]
    if not truth:
        raise ValueError("screen_capture_rate needs cases with theta_true")

    basin = None
    if capture_ls is None:
        obj = Objective.for_stage(1, forward, truth[0], family,
                                  eps_cells=cfg.EPS_INTERFACE_CELLS,
                                  scale_invariant=True)
        basin = envelope_basin_width(obj)
        capture_ls = 0.5 * min(basin["per_objective"]["envelope"]["along_ls"],
                               basin["per_objective"]["envelope"]["across_ls"])


    grid = screen_candidates(family, truth[0].lambda_s, n_grid=n_grid)
    out: dict = {}
    for name in objectives:
        errs: list[float] = []
        it = progress(truth) if progress is not None else truth
        for c in it:
            th, _ = screen(forward, c, family, n_keep=n_keep, chunk=chunk,
                           objective=name,
                           theta=(grid if c.lambda_s == truth[0].lambda_s
                                  else screen_candidates(family, c.lambda_s,
                                                         n_grid=n_grid)))
            p = centroid(th.cpu(), family)
            t = centroid(c.theta_true.reshape(1, -1).cpu(),
                         c.truth_family or family)
            errs.append(float((p - t).pow(2).sum(-1).sqrt().min()) / c.lambda_s)
        e = torch.tensor(errs)
        rate = float((e < capture_ls).double().mean())
        out[name] = dict(n=len(errs), capture_rate=rate,
                         gate_pass=bool(rate >= cfg.GATE_SCREEN_CAPTURE),
                         min_error_ls_median=float(e.median()),
                         min_error_ls_p90=float(e.quantile(0.9)),
                         min_error_ls_max=float(e.max()))
    return dict(per_objective=out, capture_ls=float(capture_ls),
                capture_ls_source=("measured envelope basin" if basin is not None
                                   else "caller-supplied"),
                gate=cfg.GATE_SCREEN_CAPTURE, n_candidates=int(grid.shape[0]),
                n_keep=n_keep, family=family.name, basin=basin)


# ---------------------------------------------------------------------------
# Stage 2: Adam on the survivors
# ---------------------------------------------------------------------------
def refine_adam(forward: SurrogateForward, case: InverseCase, family: ShapeFamily,
                theta0: Tensor, *, steps: int = cfg.ADAM_STEPS_STAGE2,
                lr: float = cfg.ADAM_LR_STAGE2, band: slice | None = None,
                eps_cells: float = cfg.EPS_INTERFACE_CELLS,
                objective: str | None = None,
                log_every: int = 0) -> tuple[Tensor, Tensor, list[float]]:
    """
    Adam on all candidates at once.  Returns (theta, J, trace of mean J).

    Adam rather than L-BFGS here because the iterate is still far from the optimum
    and the objective is not yet locally quadratic; a curvature model fitted to a
    non-quadratic region is worse than no curvature model.  lr = 5e-2 is in the
    unconstrained coordinates, where the feasible range of every parameter is O(1),
    so one step moves a position by at most a few percent of the domain.

    The objective is `cfg.STAGE_OBJECTIVE[2]`, which is the envelope.  It used to be
    the complex residual, silently, because this function simply omitted the flag that
    would have selected anything else -- and §8.4 has always said envelope refinement
    here.  Handing the complex misfit an iterate that is still most of a wavelength out
    is exactly the situation cycle skipping describes.
    """
    lam_s = case.lambda_s
    z = family.to_unconstrained(theta0, lam_s).clone().requires_grad_(True)
    obj = Objective.for_stage(2, forward, case, family, eps_cells=eps_cells,
                              **({} if band is None else {"band": band}),
                              **({} if objective is None else {"objective": objective}))
    opt = torch.optim.Adam([z], lr=lr)
    trace: list[float] = []
    for it in range(steps):
        j = obj.of_z(z)
        loss = j.sum()          # candidates are independent; see module docstring
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append(float(j.mean()))
        if log_every and it % log_every == 0:
            print(f"    adam {it:4d}  mean J {trace[-1]:.4e}  "
                  f"best {float(j.min()):.4e}")
    with torch.no_grad():
        theta = family.to_physical(z.detach(), lam_s)
        j = obj.residual(theta)
    return theta, j, trace


# ---------------------------------------------------------------------------
# Stage 3: L-BFGS, at the trained interface width
# ---------------------------------------------------------------------------
def eps_schedule_default() -> tuple[float, ...]:
    """
    The Stage-3 interface-width schedule: one entry unless the anneal is enabled.

    `cfg.EPS_INVERT_ANNEAL` is False, so this returns `(EPS_INTERFACE_CELLS,)` -- the
    width the surrogate was trained at and the only width it has evidence for.  The
    geometric argument for annealing (a wider interface spreads d chi / d theta over
    more grid points, a narrower one localises the estimate) is sound about the
    geometry and silent about the network, which is the part being extrapolated.
    Turning the flag on restores (2.0, 1.5, 1.0); it should not be turned on before
    `inverse.sensitivity.eps_transfer_report` shows the surrogate's receiver
    sensitivities still track the solver's at the endpoints.
    """
    if not cfg.EPS_INVERT_ANNEAL:
        return (cfg.EPS_INTERFACE_CELLS,)
    a, b = cfg.EPS_INVERT_START, cfg.EPS_INVERT_END
    return (a, 0.5 * (a + b), b)


def refine_lbfgs(forward: SurrogateForward, case: InverseCase,
                 family: ShapeFamily, theta0: Tensor, *,
                 steps: int = cfg.LBFGS_STEPS_STAGE3,
                 band: slice | None = None,
                 eps_schedule: tuple[float, ...] | None = None,
                 objective: str | None = None,
                 mu: float = cfg.TIKHONOV_MU,
                 log: bool = False) -> tuple[Tensor, float, list[float]]:
    """
    Short L-BFGS runs on the complex residual.  (theta, J, trace)

    theta0 is a single candidate, [1, P].  Strong-Wolfe line search is on: without a
    line search L-BFGS can take a step that overshoots into a region where the
    surrogate has never been evaluated (a void overlapping the domain edge, say) and
    the returned "descent" direction is then based on a garbage gradient.

    The returned J is the *data* misfit at the final width, with no Tikhonov term in
    it, because it is the statistic the lack-of-fit indicator consumes (§9.3) and a
    regulariser's contribution to a goodness-of-fit number is noise.  The `trace`, by
    contrast, is what L-BFGS actually minimised, penalty included.
    """
    if eps_schedule is None:
        eps_schedule = eps_schedule_default()
    lam_s = case.lambda_s
    z = family.to_unconstrained(theta0, lam_s).clone()
    trace: list[float] = []
    per = max(1, steps // len(eps_schedule))
    kw = {**({} if band is None else {"band": band}),
          **({} if objective is None else {"objective": objective})}

    for eps in eps_schedule:
        z = z.detach().clone().requires_grad_(True)
        obj = Objective.for_stage(3, forward, case, family, eps_cells=eps, **kw)
        opt = torch.optim.LBFGS([z], max_iter=per, history_size=10,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = (obj.residual(family.to_physical(z, lam_s)) + tikhonov(z, mu)).sum()
            loss.backward()
            trace.append(float(loss))
            return loss

        opt.step(closure)
        if log:
            print(f"    lbfgs eps={eps:.2f} cells  J {trace[-1]:.4e}")

    with torch.no_grad():
        theta = family.to_physical(z.detach(), lam_s)
        obj = Objective.for_stage(3, forward, case, family,
                                  eps_cells=eps_schedule[-1], **kw)
        j = float(obj.residual(theta)[0])
    return theta, j, trace


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def invert(forward: SurrogateForward, case: InverseCase, *,
           family: ShapeFamily | None = None, theta_init: Tensor | None = None,
           n_keep: int = cfg.N_SURVIVORS, skip_screen: bool = False,
           log: bool = False) -> InversionResult:
    """
    Run stages 0-3 and return the estimate with its per-stage trace.

    `theta_init` is the Stage-0 CNN guess, or None.  It is *added to* the screen's
    survivors rather than replacing them: if the CNN is right the extra candidate
    costs one row in a batch of 17, and if the defect is out of distribution -- which
    is the case the whole thesis is about -- the CNN's guess can be badly wrong and
    must not be the only starting point.  `skip_screen=True` trusts it alone, which
    is only sensible for timing comparisons.
    """
    family = family or Circle()
    t0 = time.perf_counter()
    stages: dict = {}
    lam_s = case.lambda_s
    dev = forward.device
    n_fwd = 0

    # -- stages 0 and 1 ---------------------------------------------------
    cands: list[Tensor] = []
    if theta_init is not None:
        cands.append(theta_init.reshape(1, -1).to(dev))
        stages["stage0_theta"] = theta_init.reshape(-1).tolist()
    if not (skip_screen and cands):
        grid = screen_candidates(family, lam_s)
        th, j = screen(forward, case, family, n_keep=n_keep, theta=grid)
        n_fwd += grid.shape[0]
        cands.append(th)
        stages["stage1_best_J"] = float(j[0])
        stages["stage1_J"] = j.tolist()
        stages["stage1_n_candidates"] = int(grid.shape[0])
    theta = torch.cat(cands, dim=0)
    # provenance: a misfit or a position error is not interpretable without the
    # functional and the interface width that produced it (§9, and the review's
    # repeated point that reported numbers must carry their settings)
    stages["objectives"] = {1: cfg.SCREEN_OBJECTIVE, 2: cfg.STAGE_OBJECTIVE[2],
                            3: cfg.STAGE_OBJECTIVE[3]}
    stages["eps_schedule"] = list(eps_schedule_default())
    stages["family"] = family.name
    if log:
        print(f"  stage 1: {theta.shape[0]} candidates, "
              f"best J {stages.get('stage1_best_J', float('nan')):.4e}")

    # -- stage 2 ----------------------------------------------------------
    theta, j, trace2 = refine_adam(forward, case, family, theta,
                                   log_every=50 if log else 0)
    n_fwd += cfg.ADAM_STEPS_STAGE2 * theta.shape[0]
    k = int(j.argmin())
    stages["stage2_trace"] = trace2
    stages["stage2_J"] = float(j[k])
    stages["stage2_theta"] = theta[k].tolist()
    if log:
        print(f"  stage 2: J {float(j[k]):.4e} at "
              f"{[round(float(v), 3) for v in theta[k]]}")

    # -- stage 3 ----------------------------------------------------------
    theta_f, j_f, trace3 = refine_lbfgs(forward, case, family,
                                        theta[k:k + 1], log=log)
    n_fwd += len(trace3)
    stages["stage3_trace"] = trace3
    stages["stage3_J"] = j_f

    res = InversionResult(theta=theta_f[0].detach().cpu(), misfit=j_f,
                          stages=stages,
                          theta_true=(None if case.theta_true is None
                                      else case.theta_true.detach().cpu()),
                          lambda_s=lam_s,
                          seconds=time.perf_counter() - t0, n_forward=n_fwd,
                          family=family, truth_family=case.truth_family)
    stages["wall_saturation"] = res.wall_saturation
    if log:
        print(res.summary())
    return res


# ---------------------------------------------------------------------------
# Batch evaluation, for the §11.2 step 11 and step 12 statistics
# ---------------------------------------------------------------------------
def run_many(forward: SurrogateForward, cases: list[InverseCase], *,
             family: ShapeFamily | None = None,
             theta_inits: Tensor | None = None,
             progress=None) -> list[InversionResult]:
    """Invert a list of cases sequentially, returning every result."""
    it = range(len(cases))
    if progress is not None:
        it = progress(it)
    out = []
    for i in it:
        ti = None if theta_inits is None else theta_inits[i]
        out.append(invert(forward, cases[i], family=family, theta_init=ti))
    return out


def summarise(results: list[InversionResult]) -> dict:
    """
    Success rate and error statistics, plus the §11.2 step 11 gate.

    The *median* position error is reported alongside the mean because the failure
    mode here is bimodal, not heavy-tailed-continuous: an inversion either lands in
    the right basin (error ~ lambda_s/20) or cycle-skips into a neighbouring one
    (error ~ lambda_s/2 or worse).  A mean over a bimodal distribution describes
    neither mode.

    `success` is now the conjunction of the position and IoU gates (§9), so the rate
    reported here is not comparable with a v2.0 number; `success_rate_position_only`
    is carried alongside it so the two can be told apart.  `n_wall_saturated` counts
    runs whose final estimate sits far enough into the sigmoid's tail that the float32
    gradient is at risk of being identically zero -- see `wall_saturation`.
    """
    pos = torch.tensor([r.position_error_ls for r in results
                        if r.position_error_ls is not None])
    rad = torch.tensor([r.radius_error_ls for r in results
                        if r.radius_error_ls is not None])
    ious = [r.iou() for r in results]
    iou_t = torch.tensor([v for v in ious if v is not None])
    ok = torch.tensor([float(r.success) for r in results])
    ok_pos = torch.tensor([float(r.position_error_ls is not None
                                and r.position_error_ls < cfg.GATE_POSITION_LS)
                           for r in results])
    mis = torch.tensor([r.misfit for r in results])
    sat = [r.wall_saturation for r in results]
    rate = float(ok.mean()) if len(ok) else float("nan")
    ar = [r.axis_ratio_error for r in results if r.axis_ratio_error is not None]
    orient = [r.orientation_error_deg for r in results
              if r.orientation_error_deg is not None
              and not math.isnan(r.orientation_error_deg)]
    out = dict(
        n=len(results),
        success_rate=rate,
        success_rate_position_only=float(ok_pos.mean()) if len(ok_pos) else float("nan"),
        gate_pass=bool(rate >= cfg.GATE_SUCCESS_RATE),
        position_ls_mean=float(pos.mean()) if len(pos) else float("nan"),
        position_ls_median=float(pos.median()) if len(pos) else float("nan"),
        position_ls_p90=float(pos.quantile(0.9)) if len(pos) else float("nan"),
        radius_ls_median=float(rad.median()) if len(rad) else float("nan"),
        iou_median=float(iou_t.median()) if len(iou_t) else float("nan"),
        iou_p10=float(iou_t.quantile(0.1)) if len(iou_t) else float("nan"),
        misfit_median=float(mis.median()) if len(mis) else float("nan"),
        seconds_mean=float(sum(r.seconds for r in results) / max(len(results), 1)),
        n_wall_saturated=sum(1 for s in sat if s is not None
                             and s > cfg.WALL_Z_WARN),
        gates=dict(position_ls=cfg.GATE_POSITION_LS, iou=cfg.GATE_IOU,
                   success_rate=cfg.GATE_SUCCESS_RATE),
    )
    if ar:
        out["axis_ratio_error_median"] = float(torch.tensor(ar).median())
    if orient:
        out["orientation_error_deg_median"] = float(torch.tensor(orient).median())
    if out["n_wall_saturated"]:
        warnings.warn(
            f"{out['n_wall_saturated']} of {len(results)} runs ended with "
            f"max|z| > {cfg.WALL_Z_WARN} in the unconstrained coordinates, i.e. within "
            "0.013 network cells of the edge of the feasible box, where float32 "
            "sigmoid' underflows by |z| = 16.75.  Those runs' 'convergence' may be a "
            "dead gradient rather than a minimum.",
            RuntimeWarning, stacklevel=2)
    return out


# ---------------------------------------------------------------------------
# Lack-of-fit indicator (§9.3, §11.2 step 12)
# ---------------------------------------------------------------------------
# Renamed from "model-mismatch detector".  The old name claimed more than the
# statistic can deliver, in three separate ways the review separated out:
#
#   * A large residual is not evidence about *shape* specifically.  It is consistent
#     with wrong geometry, surrogate error, a wrong source or material, unmodelled
#     noise, or an optimiser that simply did not converge.  The statistic detects lack
#     of fit; attributing it needs a second experiment.
#   * A small residual does not certify correctness.  An optimiser is free to exploit
#     surrogate error to reach a low misfit at a wrong geometry, which is precisely why
#     `sensitivity.verify_with_solver` re-scores the answer with the reference solver.
#   * The residual is not made available by the physics penalty, and never was.  A
#     supervised forward surrogate provides one just as well; the penalty changes the
#     surrogate's accuracy, not the existence of an observational residual.
#
# What remains, and is worth having: a monotone scalar whose distribution differs
# between in-family and out-of-family data, with a threshold *frozen on calibration
# data* before it is applied to test data.
def lack_of_fit_statistic(misfit: float, *, snr_db: float | None = None,
                          floor: float = cfg.LOF_MODEL_FLOOR,
                          amplification: float | None = None) -> float:
    """
    The converged data residual, normalised by what noise and model error explain.

        T = J_data / (eta * 10^(-snr_db/10) + floor)

    `J_data` must exclude the Tikhonov term -- `InversionResult.misfit` does, and
    `stages["stage3_trace"]` does not.  A regulariser's contribution to a
    goodness-of-fit statistic is a systematic offset that varies with how far the
    solution sat from the bounds, i.e. noise in exactly the quantity being thresholded.

    `floor` is the nonzero model-error floor.  Without it the noiseless case
    (`snr_db=None`) has a zero denominator, and the high-SNR cases are all reported as
    catastrophic lack of fit because they are being compared against a noise level far
    below the surrogate's own accuracy.

    `eta` is the deconvolution's noise amplification, mean of |1/(i omega s_hat)|^2 over
    the band relative to its minimum -- the misfit is quadratic, so the squared profile
    is the right average.  Flat time-domain noise does *not* stay flat after
    deconvolution (`solver.harmonic.conditioning_report`), and using a single scalar for
    it is still an approximation: the transform also correlates the noise across
    frequency through the finite window, so `eta` understates the effective variance at
    the band edges.  Deriving the exact covariance is the honest fix and is not done
    here; what is done is to stop pretending the noise is white in the phasor domain.
    """
    if snr_db is None:
        return float(misfit) / max(floor, 1e-30)
    if amplification is None:
        from ..solver.harmonic import conditioning_report
        amp = torch.tensor(conditioning_report()["amplification"])
        amplification = float(amp.pow(2).mean())
    return float(misfit) / max(amplification * 10.0 ** (-snr_db / 10.0) + floor, 1e-30)


@dataclass(frozen=True)
class LackOfFit:
    """
    A threshold frozen on calibration data, plus the provenance to defend it.

    Frozen (in the dataclass sense as well as the methodological one) because the
    entire value of the number depends on it not having been chosen after seeing the
    test set.
    """
    threshold: float
    target_fpr: float
    calibration_fpr: float
    calibration_tpr: float
    n_in: int
    n_out: int
    floor: float
    snr_db: tuple[float, ...] = ()
    note: str = ""

    def flag(self, stat: float) -> bool:
        """True = "the assumed family cannot explain this data"."""
        return stat > self.threshold

    def evaluate(self, stats_in: list[float], stats_out: list[float]) -> dict:
        """Apply the frozen threshold to held-out data.  Nothing is refitted."""
        a = torch.tensor(stats_in, dtype=torch.float64)
        b = torch.tensor(stats_out, dtype=torch.float64)
        fpr = float((a > self.threshold).double().mean()) if len(a) else float("nan")
        tpr = float((b > self.threshold).double().mean()) if len(b) else float("nan")
        return dict(threshold=self.threshold, fpr=fpr, tpr=tpr,
                    gate_pass=bool(fpr <= cfg.GATE_LOF_FPR),
                    gate=cfg.GATE_LOF_FPR, n_in=len(a), n_out=len(b))


def calibrate_lack_of_fit(stats_in: list[float], stats_out: list[float] | None = None,
                          *, target_fpr: float = cfg.GATE_LOF_FPR,
                          floor: float = cfg.LOF_MODEL_FLOOR,
                          snr_db: tuple[float, ...] = (),
                          note: str = "") -> LackOfFit:
    """
    Choose the threshold from *in-family* statistics alone, at a target FPR.

    Deliberately one-sided: the threshold is the (1 - target_fpr) quantile of the
    in-family calibration statistics, so it is set by what correctly-specified cases
    look like and is not tuned against the out-of-family set.  `stats_out` is optional
    and is used only to report the TPR that this threshold happens to achieve.

    Two things the caller must get right, and which no code can check for them:

      * `stats_in` has to include the *difficult but correctly specified* cases -- low
        SNR, small radius, deep standoff, shadowed geometry -- because those are the
        false-positive controls.  Calibrating on easy in-family cases sets the threshold
        too low and every hard-but-correct case is then flagged as mismatched.
      * `stats_in` and `stats_out` must be independent of the test set the threshold is
        later applied to, and of anything used to select the model.

    Warns when the in-family residuals sit well above `floor`, which means the model
    floor is understated and the statistic is being dominated by surrogate error rather
    than by noise -- the threshold still works, but it is measuring the surrogate.
    """
    a = torch.tensor(stats_in, dtype=torch.float64)
    if a.numel() == 0:
        raise ValueError("calibration needs in-family statistics")
    q = float(a.quantile(1.0 - target_fpr))
    fpr = float((a > q).double().mean())
    tpr = float("nan")
    if stats_out:
        b = torch.tensor(stats_out, dtype=torch.float64)
        tpr = float((b > q).double().mean())
    if float(a.median()) > 3.0:
        warnings.warn(
            f"in-family lack-of-fit statistic has median {float(a.median()):.2f} >> 1: "
            f"the residual is {float(a.median()):.1f}x what noise and the "
            f"LOF_MODEL_FLOOR = {floor:.2e} explain, so the floor is understated and "
            "the statistic is dominated by surrogate error.  Substitute the measured "
            "receiver-restricted validation misfit before quoting a calibrated rate.",
            RuntimeWarning, stacklevel=2)
    return LackOfFit(threshold=q, target_fpr=target_fpr, calibration_fpr=fpr,
                     calibration_tpr=tpr, n_in=int(a.numel()),
                     n_out=len(stats_out or ()), floor=floor, snr_db=tuple(snr_db),
                     note=note)


def lack_of_fit_roc(stats_in: list[float], stats_out: list[float],
                    n_thresh: int = 200) -> dict:
    """
    ROC for "is this data explained by the family I inverted with?".

    A diagnostic curve, not the operating point: the operating point is whatever
    `calibrate_lack_of_fit` froze.  Reporting the AUC alongside is still worth it,
    because an AUC near 0.5 says the statistic carries no information at all and no
    choice of threshold will rescue it.

    `stats_in` come from in-family cases, `stats_out` from out-of-family ones -- both
    the *statistic*, not the raw misfit, so that cases at different SNRs are
    commensurable.  Returns the ROC points and the AUC, computed by the rank identity
    rather than by trapezoidal integration so it is exact at the sample size involved.
    """
    a = torch.tensor(stats_in, dtype=torch.float64)
    b = torch.tensor(stats_out, dtype=torch.float64)
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    th = torch.linspace(lo, hi, n_thresh, dtype=torch.float64)
    tpr = [(b > t).double().mean().item() for t in th]      # out-of-family flagged
    fpr = [(a > t).double().mean().item() for t in th]
    # AUC = P(stat_out > stat_in), ties counted as half
    diff = b.view(-1, 1) - a.view(1, -1)
    auc = float((diff > 0).double().mean() + 0.5 * (diff == 0).double().mean())
    return dict(threshold=th.tolist(), tpr=tpr, fpr=fpr, auc=auc,
                median_in=float(a.median()), median_out=float(b.median()))


def detector_roc(misfits_in: list[float], misfits_out: list[float],
                 n_thresh: int = 200) -> dict:
    """Deprecated alias for `lack_of_fit_roc`; the old name overclaimed (§9.3)."""
    warnings.warn(
        "detector_roc is deprecated: the statistic is a lack-of-fit indicator, not a "
        "shape-specific model-mismatch detector, and it should be fed the normalised "
        "statistic from lack_of_fit_statistic rather than raw misfits.  Use "
        "lack_of_fit_roc.", DeprecationWarning, stacklevel=2)
    return lack_of_fit_roc(misfits_in, misfits_out, n_thresh)


__all__ = [
    "InversionResult",
    "LackOfFit",
    "calibrate_lack_of_fit",
    "centroid",
    "detector_roc",
    "eps_schedule_default",
    "invert",
    "lack_of_fit_roc",
    "lack_of_fit_statistic",
    "refine_adam",
    "refine_lbfgs",
    "run_many",
    "screen",
    "screen_candidates",
    "screen_capture_rate",
    "summarise",
]
