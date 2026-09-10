"""
The inversion's objective (§8.1-§8.4).

The forward map is the trained surrogate, and the quantity being differentiated is

    J(theta) = sum_m w_m || G(theta)|_ring - d_obs ||^2 / sum_m || d_obs ||^2

with G the network's predicted *scattered* displacement phasor field, restricted to
the 32 receivers.  Everything delicate about this file is bookkeeping, and the
bookkeeping is load-bearing, so it is written down rather than inferred.

**Units.**  The network was trained on inputs and targets divided by the per-(source,
nu, frequency) incident field amplitude (`harmonic.incident_scale`).  Its output is
therefore in those units, and the observed receiver data is not.  `SurrogateForward`
multiplies the prediction by the same scale before it is compared, rather than
dividing the observation: a relative misfit is invariant to a common factor, but the
two would silently disagree the moment an absolute residual is reported -- and §11.2
step 12 reports exactly that, as the lack-of-fit statistic.

**The interface width is fixed by default, not annealed.**  Training used
EPS_INTERFACE_CELLS = 1.5 cells and nothing else.  v2.0 ran the inversion with eps
annealed from 2.0 down to 1.0 cells on the argument that d chi / d theta is a bump of
width eps around the boundary (§8.2), so a wider interface has gradient support over
more grid points and a broader basin, while a narrower one sharpens the final
estimate.  Both halves of that are true *about the geometry*; neither says anything
about the surrogate, which was never shown either endpoint.  Bracketing the training
value is not validation of the endpoints -- it is two extrapolations that happen to
straddle the interpolation.  So `cfg.EPS_INVERT_ANNEAL` is False, every stage runs at
the trained width, and the schedule is opt-in behind
`inverse.sensitivity.eps_transfer_report`, which measures whether the surrogate's
receiver sensitivities still track the solver's at 2.0 and 1.0 cells.

**Which objective each stage optimises, and why the old answer was wrong.**  v2.0
screened on |G| vs |d| -- the magnitude spectrum -- and defended it as "phase-free by
construction", the property that supposedly makes an envelope misfit immune to cycle
skipping.  That defence inverts the actual mathematics.  For a trace delayed by tau,
|g_hat_tau(omega)| = |g_hat(omega)| *exactly*, so a magnitude-spectrum misfit is not
merely phase-free, it is travel-time **blind**: it cannot distinguish a defect from
the same defect moved anywhere along a locus of equal scattering amplitude, which is
precisely what a position screen has to decide.  `inverse.timedomain` proves the point
executably (`shift_invariance_demo`) and supplies the objective §8.4 always meant: the
modulus of a band-limited analytic signal reconstructed from the phasors, which is
shift-*equivariant* and therefore does carry arrival time.

The pipeline is now declared in one place, `cfg.STAGE_OBJECTIVE`: stage 1 screen and
stage 2 refinement on the envelope, stage 3 on the complex residual.  The magnitude
screen survives as `spectral_magnitude_misfit`, named as the heuristic it is and kept
because it costs one `abs()` instead of a [M, 128] reconstruction; what it is *not*
allowed to do any more is claim an envelope's basin.  Its capture rate is measured
against the envelope screen's by `inverse.invert.screen_capture_rate`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from . import timedomain as td
from .. import config as cfg
from .. import features as feat
from ..geometry.sdf import (ShapeFamily, equivalent_circle, geometry_channels,
                           net_coords)
from ..models.fno2d import FNO2d


# ---------------------------------------------------------------------------
# Bundling one inverse problem
# ---------------------------------------------------------------------------
@dataclass
class InverseCase:
    """
    One inversion: the data, the acquisition it came from, and (for scoring) truth.

    d_obs   [1, R, 2, M]  complex, *physical* units -- scattered displacement phasors
    src_idx, nu_idx       which cached incident solve applies
    theta_true            [P] or None when inverting real / mismatched data
    truth_family          the family theta_true is written in, when it is *not* the
                          family being inverted for -- the transfer experiment of §8.5,
                          where a circle is fitted to an ellipse or a pair of voids.
                          Carrying the real truth and its family, rather than a
                          pre-reduced equal-area circle, is what lets `InversionResult`
                          report an honest IoU against the actual shape; leave it None
                          when the families agree.
    """
    d_obs: Tensor
    src_idx: int
    nu_idx: int
    theta_true: Tensor | None = None
    snr_db: float | None = None
    truth_family: ShapeFamily | None = None

    @property
    def nu(self) -> float:
        return float(cfg.NU_LIST[self.nu_idx])

    @property
    def lambda_s(self) -> float:
        return cfg.cs_over_cp(self.nu) / cfg.FC

    def to(self, device) -> "InverseCase":
        t = None if self.theta_true is None else self.theta_true.to(device)
        return InverseCase(self.d_obs.to(device), self.src_idx, self.nu_idx, t,
                           self.snr_db, self.truth_family)

    @staticmethod
    def from_dict(d: dict) -> "InverseCase":
        """Adapt `data.dataset.load_inversion_case` output."""
        return InverseCase(d_obs=d["d_obs"], src_idx=int(d["src_idx"]),
                           nu_idx=int(d["nu_idx"]), theta_true=d.get("theta_true"),
                           snr_db=d.get("snr_db"),
                           truth_family=d.get("truth_family"))


# ---------------------------------------------------------------------------
# The differentiable forward model
# ---------------------------------------------------------------------------
class SurrogateForward:
    """
    theta -> predicted scattered phasors, differentiably, in physical units.

    The network's weights are frozen (`requires_grad_(False)`) and it is kept in
    eval mode.  Neither is cosmetic: a live `requires_grad` on 9.9 M parameters
    would have every inversion step building a graph through all of them, and the
    L-BFGS closure would allocate the whole backward tape for gradients nobody
    reads.  Freezing makes the graph what it should be -- three numbers in, one
    scalar out.
    """

    def __init__(self, model: FNO2d, incident: dict, *, device=None,
                 receivers: Tensor | None = None,
                 dtype: torch.dtype = torch.float32):
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.inc_phasors = incident["phasors"].to(self.device, self.cdtype)
        self.inc_scale = incident["scale"].to(self.device, dtype)     # [S,NU,M]
        assert self.inc_phasors.is_complex(), "incident phasors must be complex"
        self.recv = (receivers if receivers is not None
                     else torch.tensor(cfg.RECEIVERS_NET, dtype=torch.long)
                     ).to(self.device)
        ny = self.inc_phasors.shape[-2]
        self._coords = feat.coordinate_channels(ny, device=self.device, dtype=dtype)
        self._grid = net_coords(device=self.device, dtype=dtype)

    # -- pieces ------------------------------------------------------------
    def freqs(self, band: slice) -> Tensor:
        return torch.tensor(cfg.FREQS[band], dtype=self.dtype, device=self.device)

    def incident(self, src_idx: int, nu_idx: int, band: slice) -> tuple[Tensor, Tensor]:
        """(normalised u_inc [F,2,ny,nx], scale [F]) for one acquisition."""
        u = self.inc_phasors[src_idx, nu_idx][:, band]               # [2,F,ny,nx]
        s = self.inc_scale[src_idx, nu_idx][band]                    # [F]
        u = u.permute(1, 0, 2, 3) / s.view(-1, 1, 1, 1)
        return u, s

    def field(self, theta: Tensor, family: ShapeFamily, *, src_idx: int,
              nu_idx: int, band: slice = cfg.BAND_STAGE3,
              eps_cells: float = cfg.EPS_INTERFACE_CELLS) -> Tensor:
        """
        Predicted scattered displacement phasors, [B, F, 2, ny, nx] complex,
        in physical units.
        """
        B = theta.shape[0]
        u_inc, scale = self.incident(src_idx, nu_idx, band)
        F_ = u_inc.shape[0]
        ny, nx = u_inc.shape[-2], u_inc.shape[-1]

        yy, xx = self._grid
        phi_t, chi = geometry_channels(theta, family, dx=cfg.DX_NET,
                                       eps_cells=eps_cells, yy=yy, xx=xx)
        x = feat.pack_inputs(
            phi_t, chi,
            u_inc.unsqueeze(0).expand(B, F_, 2, ny, nx),
            self.freqs(band).unsqueeze(0).expand(B, F_),
            torch.full((B,), float(cfg.NU_LIST[nu_idx]), device=self.device,
                       dtype=self.dtype),
            dx=cfg.DX_NET, coords=self._coords, dtype=self.dtype)
        pred = self.model(x)                                    # [B*F, 4, ny, nx]
        # rows are (sample, frequency) with frequency major, so the per-row scale
        # is the band tiled B times -- not repeat_interleave, which would apply
        # one frequency's scale to a whole sample
        pred = pred * scale.repeat(B).view(-1, 1, 1, 1)
        return feat.unflatten_freq(feat.channels_to_complex(pred), F_)

    def at_receivers(self, u: Tensor) -> Tensor:
        """[B, F, 2, ny, nx] -> [B, R, 2, F], the layout the observations use."""
        ry, rx = self.recv[:, 0], self.recv[:, 1]
        return u[..., ry, rx].permute(0, 3, 2, 1)

    def predict(self, theta: Tensor, family: ShapeFamily, *, src_idx: int,
                nu_idx: int, band: slice = cfg.BAND_STAGE3,
                eps_cells: float = cfg.EPS_INTERFACE_CELLS) -> Tensor:
        """theta [B, P] -> [B, R, 2, F] complex, physical units."""
        return self.at_receivers(self.field(theta, family, src_idx=src_idx,
                                            nu_idx=nu_idx, band=band,
                                            eps_cells=eps_cells))


# ---------------------------------------------------------------------------
# Misfits
# ---------------------------------------------------------------------------
def _reduce(num: Tensor, den: Tensor, eps: float) -> Tensor:
    return num / den.clamp_min(eps)


def complex_misfit(pred: Tensor, obs: Tensor, *, weights: Tensor | None = None,
                   eps: float = 1e-30) -> Tensor:
    """
    Relative squared error on the complex residual, per candidate.  [B]

    pred [B, R, 2, F], obs [1 or B, R, 2, F], both complex and in the same units.
    Normalised by ||obs||^2 rather than left absolute so that the value is
    comparable across SNRs, source positions and defect sizes -- which is what lets
    one threshold serve `invert.lack_of_fit_statistic` (§11.2 step 12) across an
    acquisition rather than needing one threshold per source position.  Note what the
    normalisation does *not* do: it does not make the value comparable across noise
    levels, which is why the statistic divides by an explicit noise-plus-model-floor
    term on top of it.
    """
    r = (pred - obs).abs().pow(2)
    o = obs.abs().pow(2).expand_as(r)
    if weights is not None:
        w = weights.view(1, 1, 1, -1)
        r, o = r * w, o * w
    dims = (1, 2, 3)
    return _reduce(r.sum(dims), o.sum(dims), eps)


def spectral_magnitude_misfit(pred: Tensor, obs: Tensor, *,
                              weights: Tensor | None = None,
                              scale_invariant: bool = False,
                              eps: float = 1e-30) -> Tensor:
    """
    The same, on |pred| vs |obs| -- the cheap screen, and a *heuristic*.  [B]

    Named for what it compares, because the old name (`amplitude_misfit`) invited the
    reading that it is an envelope misfit, and it is not.  A time shift multiplies
    every phasor by exp(-i omega tau) and leaves every modulus untouched, so this
    functional is not "phase-free" in a benign sense -- it is **travel-time blind**:
    two candidates whose predicted fields differ only by a delay score identically.
    `timedomain.shift_invariance_demo` exhibits that as a number rather than an
    argument, and `timedomain.envelope_misfit` is the objective that does see arrival
    time.

    What survives of the original justification is narrower but real.  The scattered
    *amplitude pattern* across a 32-receiver ring and across frequency does constrain
    position -- through shadowing and through the interference of the direct and
    creeping contributions, not through travel time -- and evaluating it costs one
    `abs()` where the envelope costs a [M, n_t] matrix product per candidate.  So it
    is kept as `cfg.SCREEN_FALLBACK_OBJECTIVE`, its capture rate is measured against
    the envelope screen's by `inverse.invert.screen_capture_rate`, and nothing in the
    pipeline is allowed to claim an envelope's basin on its behalf.

    `scale_invariant` divides out the least-squares optimal real scalar
    a = <|p|,|d|> / <|p|,|p|> before comparing.  The screen evaluates candidates at a
    single fixed radius while the true radius varies over a factor of three, and
    radius enters the scattered amplitude mostly as an overall factor (~R^2 in the
    Rayleigh regime).  Without this the screen would rank candidates partly on how
    close the *assumed* radius was, which is information it does not have; with it,
    the ranking depends only on the shape of the amplitude pattern across the ring
    and across frequency, which is what actually carries position.
    """
    p, o = pred.abs(), obs.abs().expand_as(pred.abs())
    if scale_invariant:
        dims = (1, 2, 3)
        a = ((p * o).sum(dims, keepdim=True)
             / p.pow(2).sum(dims, keepdim=True).clamp_min(eps))
        p = a * p
    r = (p - o).pow(2)
    o2 = o.pow(2)
    if weights is not None:
        w = weights.view(1, 1, 1, -1)
        r, o2 = r * w, o2 * w
    dims = (1, 2, 3)
    return _reduce(r.sum(dims), o2.sum(dims), eps)


#: Deprecated name.  Kept because `.nbgen` sources and saved notebooks import it, but
#: it is the name that caused the confusion the review flagged: "amplitude misfit"
#: reads as "envelope misfit", and the two differ by exactly the information the
#: screen needs.  New code should say what it means.
amplitude_misfit = spectral_magnitude_misfit


# ---------------------------------------------------------------------------
# The registry `cfg.STAGE_OBJECTIVE` names
# ---------------------------------------------------------------------------
# Every objective here has the signature
#
#     f(pred, obs, *, band, weights, scale_invariant, recon) -> [B]
#
# so that a stage can be reconfigured by editing one string in config.py.  The
# uniform signature costs five ignored keyword arguments per call and buys the
# property that no caller has to know which family an objective belongs to.
#
# `weights` is *spectral* weighting -- one real number per frequency.  It is
# meaningful for the two objectives that compare phasors frequency by frequency and
# meaningless for the three that first reconstruct a trace, because there the
# frequency weighting is the reconstruction's own taper (`cfg.RECON_TAPER`) and a
# second one would silently make the objective depend on two tapers.  Rather than
# drop it, those three refuse it.
def _needs_no_weights(name: str, weights: Tensor | None) -> None:
    if weights is not None:
        raise ValueError(
            f"objective '{name}' reconstructs a trace, so per-frequency weights are "
            f"already fixed by cfg.RECON_TAPER = {cfg.RECON_TAPER!r}; pass weights "
            "only to 'spectral_magnitude' or 'complex', or change the taper")


def _o_spectral_magnitude(pred: Tensor, obs: Tensor, *, band: slice,
                          weights: Tensor | None = None,
                          scale_invariant: bool = False,
                          recon: td.Reconstruction | None = None) -> Tensor:
    return spectral_magnitude_misfit(pred, obs, weights=weights,
                                     scale_invariant=scale_invariant)


def _o_complex(pred: Tensor, obs: Tensor, *, band: slice,
               weights: Tensor | None = None, scale_invariant: bool = False,
               recon: td.Reconstruction | None = None) -> Tensor:
    return complex_misfit(pred, obs, weights=weights)


def _o_envelope(pred: Tensor, obs: Tensor, *, band: slice,
                weights: Tensor | None = None, scale_invariant: bool = False,
                recon: td.Reconstruction | None = None) -> Tensor:
    _needs_no_weights("envelope", weights)
    return td.envelope_misfit(pred, obs, recon=recon, band=band,
                               scale_invariant=scale_invariant)


def _o_traveltime(pred: Tensor, obs: Tensor, *, band: slice,
                  weights: Tensor | None = None, scale_invariant: bool = False,
                  recon: td.Reconstruction | None = None) -> Tensor:
    _needs_no_weights("traveltime", weights)
    return td.traveltime_misfit(pred, obs, recon=recon, band=band)


def _o_correlation(pred: Tensor, obs: Tensor, *, band: slice,
                   weights: Tensor | None = None, scale_invariant: bool = False,
                   recon: td.Reconstruction | None = None) -> Tensor:
    _needs_no_weights("correlation", weights)
    return td.correlation_misfit(pred, obs, recon=recon, band=band)


OBJECTIVES: dict[str, Callable[..., Tensor]] = {
    "spectral_magnitude": _o_spectral_magnitude,
    "envelope": _o_envelope,
    "traveltime": _o_traveltime,
    "correlation": _o_correlation,
    "complex": _o_complex,
}

#: Frequency band per continuation stage.  Duplicated from `solver.harmonic.band_slice`
#: on purpose: the inverse package should not import the FDTD solver to look up three
#: slices, and `tests/test_config.py` asserts the two agree.
STAGE_BANDS: dict[int, slice] = {1: cfg.BAND_STAGE1, 2: cfg.BAND_STAGE2,
                                 3: cfg.BAND_STAGE3}

#: The objectives that go through `timedomain.Reconstruction`, i.e. the ones that need
#: a [M, n_t] basis and therefore a `recon`.
RECONSTRUCTED: frozenset[str] = frozenset({"envelope", "traveltime", "correlation"})


def check_objective(name: str) -> str:
    """Validate an objective name against the registry, with a useful message."""
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; "
                       f"config.STAGE_OBJECTIVE and inverse.misfit.OBJECTIVES must "
                       f"agree, and the registry has {sorted(OBJECTIVES)}")
    return name


for _s, _n in cfg.STAGE_OBJECTIVE.items():                  # fail at import, not in
    check_objective(_n)                                     # the middle of an inversion
check_objective(cfg.SCREEN_OBJECTIVE)
check_objective(cfg.SCREEN_FALLBACK_OBJECTIVE)
del _s, _n


def tikhonov(z: Tensor, mu: float = cfg.TIKHONOV_MU) -> Tensor:
    """
    mu * mean(z^2) in the *unconstrained* coordinates.  [B]

    Regularising z rather than theta is deliberate.  The bounds are imposed by
    theta = lo + (hi-lo) sigmoid(z) (§8.1), so |z| large means the parameter has
    pinned itself against a bound, where dtheta/dz -> 0: the gradient vanishes, and
    L-BFGS's curvature estimate -- built from differences of vanishing gradients --
    degenerates.  A weak penalty on |z| keeps the iterate off the saturated tails.
    It also does the usual Tikhonov job of lifting the small eigenvalue of the
    Gauss-Newton Hessian, which here belongs to the radius/standoff trade-off: a
    slightly larger void slightly further away scatters almost the same field.

    mu = 1e-3 against a relative data misfit that starts at O(1), so it is
    negligible until the data term is nearly flat, which is where it is wanted.
    """
    return mu * z.pow(2).mean(dim=-1)


# ---------------------------------------------------------------------------
# The objective the optimiser sees
# ---------------------------------------------------------------------------
@dataclass
class Objective:
    """
    A callable J(theta) for one case, one band, one interface width, one objective.

    Kept as an object rather than a closure so that `stage`, `objective`, `eps_cells`
    and `band` are inspectable after the fact: a reported position error is
    meaningless without knowing which band and which functional produced it, and a
    debug session that has to reverse-engineer a closure's captured variables is a
    debug session wasted.

    `objective` is a key of `OBJECTIVES`, and the intended value for each stage lives
    in `cfg.STAGE_OBJECTIVE` -- use `Objective.for_stage` rather than repeating the
    table at the call site.  The old boolean `amplitude=` is still accepted, warns,
    and maps to `cfg.SCREEN_FALLBACK_OBJECTIVE`, which is what it used to select.
    """
    forward: SurrogateForward
    case: InverseCase
    family: ShapeFamily
    band: slice = cfg.BAND_STAGE3
    eps_cells: float = cfg.EPS_INTERFACE_CELLS
    objective: str = "complex"
    scale_invariant: bool = False
    weights: Tensor | None = None
    stage: int | None = None
    amplitude: bool | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.amplitude is not None:
            warnings.warn(
                "Objective(amplitude=...) is deprecated: the flag chose between the "
                "magnitude-spectrum screen and the complex misfit, and the pipeline "
                "now names its objective explicitly.  Pass "
                "objective='spectral_magnitude' (or, for the screen §8.4 actually "
                "specifies, objective='envelope') instead.",
                DeprecationWarning, stacklevel=3)
            self.objective = (cfg.SCREEN_FALLBACK_OBJECTIVE if self.amplitude
                              else "complex")
        check_objective(self.objective)

    # -- construction ------------------------------------------------------
    @classmethod
    def for_stage(cls, stage: int, forward: SurrogateForward, case: InverseCase,
                  family: ShapeFamily, **kw) -> "Objective":
        """
        The objective `cfg.STAGE_OBJECTIVE` declares for `stage`, on that stage's band.

        This is the single place the stage -> (band, functional) mapping is read, so
        the pipeline the paper describes and the pipeline the code runs cannot drift
        apart the way they did in v2.0.
        """
        kw.setdefault("band", STAGE_BANDS[stage])
        kw.setdefault("objective", cfg.STAGE_OBJECTIVE[stage])
        return cls(forward, case, family, stage=stage, **kw)

    # -- pieces ------------------------------------------------------------
    @property
    def reconstruction(self) -> td.Reconstruction | None:
        """The [M, n_t] analytic basis, or None for the two phasor-domain objectives."""
        if self.objective not in RECONSTRUCTED:
            return None
        return td.reconstruction(self.band, n_t=cfg.RECON_N_T, taper=cfg.RECON_TAPER,
                                 device=self.forward.device)

    def data(self) -> Tensor:
        return self.case.d_obs[..., self.band].to(self.forward.device)

    def predict(self, theta: Tensor) -> Tensor:
        return self.forward.predict(theta, self.family,
                                    src_idx=self.case.src_idx,
                                    nu_idx=self.case.nu_idx,
                                    band=self.band, eps_cells=self.eps_cells)

    def residual(self, theta: Tensor) -> Tensor:
        """Per-candidate misfit, [B].  No regularisation."""
        return OBJECTIVES[self.objective](
            self.predict(theta), self.data(), band=self.band, weights=self.weights,
            scale_invariant=self.scale_invariant, recon=self.reconstruction)

    def __call__(self, theta: Tensor) -> Tensor:
        return self.residual(theta)

    def of_z(self, z: Tensor, *, mu: float = 0.0) -> Tensor:
        """Misfit as a function of the unconstrained parameters, plus Tikhonov."""
        theta = self.family.to_physical(z, self.case.lambda_s)
        j = self.residual(theta)
        return j if mu == 0.0 else j + tikhonov(z, mu)

    def with_objective(self, name: str, **kw) -> "Objective":
        """A copy scoring the same data with a different functional."""
        return Objective(self.forward, self.case, self.family,
                         band=kw.pop("band", self.band),
                         eps_cells=kw.pop("eps_cells", self.eps_cells),
                         objective=check_objective(name),
                         scale_invariant=kw.pop("scale_invariant",
                                                self.scale_invariant),
                         weights=kw.pop("weights", None), stage=self.stage, **kw)


# ---------------------------------------------------------------------------
# Diagnostics: the misfit landscape figure of §11.3
# ---------------------------------------------------------------------------
@torch.no_grad()
def misfit_map(obj: Objective, *, n: int = 41, radius: float | None = None,
               chunk: int = cfg.SCREEN_CHUNK) -> dict:
    """
    J on a grid of (xc, yc) at fixed R -- the figure §11.3 calls non-negotiable.

    Two features are being looked for, and both are physics rather than decoration.

    The basin should be about lambda_s/4 across for the complex objective, which is
    the resolution a half-wavelength criterion predicts, and several times wider for
    the envelope -- `envelope_basin_width` measures the ratio instead of asserting it.

    It should also be *anisotropic*, and the direction matters: the long axis is
    **transverse** to the source-defect line, not along it.  v2.0 had this backwards
    (§8.3, and the review's smaller-corrections list).  The reason is first-order path
    length.  Displace the defect by delta along the ray and the two-way path changes
    by ~2 delta, so the arrival moves by 2 delta / c and the misfit climbs
    immediately; displace it by delta transverse to the ray and the path length
    changes only at second order, ~delta^2 / L, so the misfit is nearly flat.  Fast
    variation means a *narrow* basin, hence narrow along the ray and long across it.

    That argument is for one source and one receiver.  This acquisition has 32
    receivers on a ring around the defect, so a transverse displacement that leaves
    one path length unchanged shortens the paths to receivers on one side and
    lengthens them on the other: the ring constrains the transverse direction too, and
    the elongation is milder than the single-path argument suggests.  The falsifiable
    statement is therefore the *ordering*, across >= along, and not a specific ratio.
    """
    lam_s = obj.case.lambda_s
    lo, hi = obj.family.bounds(lam_s)
    if radius is None:
        if obj.case.theta_true is None:
            radius = 0.5 * (cfg.R_MIN_LS + cfg.R_MAX_LS) * lam_s
        else:
            # The map's third column is a circle radius, so an out-of-family truth is
            # reduced to its equal-area radius rather than having column 2 read as one
            # (which for an ellipse is the semi-major axis, i.e. the wrong slice).
            fam_t = obj.case.truth_family or obj.family
            radius = float(equivalent_circle(
                obj.case.theta_true.reshape(1, -1).cpu(), fam_t)[0, 2])
    xs = torch.linspace(float(lo[0]), float(hi[0]), n)
    ys = torch.linspace(float(lo[1]), float(hi[1]), n)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    theta = torch.stack([gx.reshape(-1), gy.reshape(-1),
                         torch.full((n * n,), radius)], dim=-1).to(obj.forward.device)

    out = []
    for i in range(0, theta.shape[0], chunk):
        out.append(obj.residual(theta[i:i + chunk]).cpu())
    j = torch.cat(out).view(n, n)
    k = int(j.argmin())
    return dict(J=j, x=xs, y=ys, radius=radius, lambda_s=lam_s,
                argmin=(float(xs[k % n]), float(ys[k // n])),
                theta_true=obj.case.theta_true,
                source_xy=cfg.SOURCE_XY[obj.case.src_idx])


def basin_width(m: dict, *, factor: float = 2.0) -> dict:
    """
    Size of the region where J < factor * J_min, in shear wavelengths.

    Reported as (along, across) relative to the source-to-minimum direction, so the
    anisotropy the landscape figure is supposed to show comes out as a number and not
    only as a picture.  `elongation` = across / along, and the expectation is that it
    is >= 1 -- see `misfit_map` for why that direction and not the other one.

    Two honest limitations.  The extent is measured as the bounding box of the
    sub-level set in the rotated frame, so a level set with a disconnected secondary
    lobe (which elastic multipath does produce) is reported as one wide basin; and it
    is quantised by the grid, so `n_cells` should be at least a few tens before the
    widths mean much.
    """
    j = m["J"]
    mask = j <= factor * float(j.min())
    iy, ix = torch.nonzero(mask, as_tuple=True)
    x = m["x"][ix]
    y = m["y"][iy]
    cx, cy = m["argmin"]
    sx, sy = m["source_xy"]
    ang = math.atan2(cy - sy, cx - sx)
    ca, sa = math.cos(ang), math.sin(ang)
    dx_, dy_ = x - cx, y - cy
    along = (ca * dx_ + sa * dy_)
    across = (-sa * dx_ + ca * dy_)
    ls = m["lambda_s"]
    a_ls = float(along.max() - along.min()) / ls
    c_ls = float(across.max() - across.min()) / ls
    return dict(n_cells=int(mask.sum()), along_ls=a_ls, across_ls=c_ls,
                elongation=c_ls / max(a_ls, 1e-12),
                angle_deg=math.degrees(ang), factor=factor,
                grid_step_ls=float(m["x"][1] - m["x"][0]) / ls)


@torch.no_grad()
def envelope_basin_width(obj: Objective, *,
                         objectives: tuple[str, ...] = ("envelope", "complex"),
                         n: int = 41, factor: float = 2.0,
                         radius: float | None = None,
                         chunk: int = cfg.SCREEN_CHUNK) -> dict:
    """
    Measure each objective's basin on the same grid, and the screen's coverage of it.

    This is the function `config.self_check` defers to.  v2.0 *asserted* that the
    envelope basin is N_CYCLES / 2 = 2.5 times the waveform basin and then asserted
    that the screen grid therefore covers it, so a claim about elastic multipath was
    doing load-bearing work with no measurement behind it.  Two things were wrong with
    that: the factor was asserted, and the objective it was asserted about was the
    magnitude spectrum, which has no basin in the relevant sense at all because it
    cannot see the shift the basin is supposed to be a basin for.

    What replaces it: run `misfit_map` once per objective over the same (xc, yc) grid
    at fixed R, take `basin_width` of each, and report the screen's worst-case node
    spacing next to the measured half-widths.  `covered` is then a measured fact.  Note
    which way the inequality goes with band width -- the envelope's duration is ~1/B,
    so a *wider* band gives a *narrower* basin, and the screen's job gets harder as
    frequency continuation proceeds.  That is why the screen runs on BAND_STAGE1.

    A caveat the number cannot express: this is one case, at one true radius, with the
    radius held fixed at truth.  A basin measured with R free is wider, because the
    radius/standoff trade-off gives the level set another direction to extend in.
    """
    out: dict = {}
    for name in objectives:
        o = obj.with_objective(name)
        m = misfit_map(o, n=n, radius=radius, chunk=chunk)
        out[name] = dict(band=(o.band.start, o.band.stop), **basin_width(m, factor=factor))
        out[name]["J_min"] = float(m["J"].min())
        radius = m["radius"]                       # keep every objective on one R

    lam_s = obj.case.lambda_s
    lo, hi = obj.family.bounds(lam_s)
    step = float(hi[0] - lo[0]) / max(cfg.SCREEN_GRID - 1, 1)
    worst = math.sqrt(2.0) * 0.5 * step           # node to farthest point of its cell
    ref = out.get("envelope", next(iter(out.values())))
    half = 0.5 * min(ref["along_ls"], ref["across_ls"]) * lam_s
    return dict(per_objective=out, radius=radius, lambda_s=lam_s,
                screen_step=step, screen_step_ls=step / lam_s,
                worst_node_distance=worst, worst_node_distance_ls=worst / lam_s,
                narrow_half_width_ls=half / lam_s,
                margin=half / max(worst, 1e-12), covered=bool(half >= worst),
                nominal_v2_ratio=cfg.N_CYCLES / 2.0,
                measured_ratio=(out["envelope"]["along_ls"] / out["complex"]["along_ls"]
                                if {"envelope", "complex"} <= set(out) else None))


__all__ = [
    "InverseCase",
    "OBJECTIVES",
    "Objective",
    "RECONSTRUCTED",
    "STAGE_BANDS",
    "SurrogateForward",
    "amplitude_misfit",
    "basin_width",
    "check_objective",
    "complex_misfit",
    "envelope_basin_width",
    "misfit_map",
    "spectral_magnitude_misfit",
    "tikhonov",
]
