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
step 12 reports exactly that, as the model-mismatch detector statistic.

**The interface width is annealed, not fixed.**  Training used
EPS_INTERFACE_CELLS = 1.5 cells; the inversion runs eps from 2.0 down to 1.0 cells
(`cfg.EPS_INVERT_START/END`).  Wider at the start because the sensitivity
d chi / d theta is a bump of width eps around the boundary (§8.2), so a wider
interface has gradient support over more grid points and a correspondingly broader
basin; narrower at the end because the position estimate cannot be sharper than the
interface it is estimating.  The schedule brackets the training value, so the network
is never asked about a sharpness far outside what it saw -- an eps of 0.3 cells would
put the whole transition inside one cell and the input would alias.

**Two misfits, and why the screening one is amplitude-only.**  §8.4 asks for an
envelope misfit in the low-frequency screening stage, to avoid cycle skipping.  An
envelope needs a time trace, and a trace synthesised from 20 phasors spanning
0.66-1.34 f_c has a time resolution of about 1/(0.68 f_c) = 1.5 periods: its envelope
is a smoothed version of the amplitude spectrum with an FFT in between.  So the screen
compares |G| to |d| directly.  That is phase-free by construction, which is the
property that matters -- cycle skipping is the misfit oscillating with the carrier,
and an amplitude comparison has no carrier in it.  Stages 2 and 3 use the complex
misfit, where the phase carries the resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .. import config as cfg
from .. import features as feat
from ..geometry.sdf import ShapeFamily, geometry_channels, net_coords
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
    """
    d_obs: Tensor
    src_idx: int
    nu_idx: int
    theta_true: Tensor | None = None
    snr_db: float | None = None

    @property
    def nu(self) -> float:
        return float(cfg.NU_LIST[self.nu_idx])

    @property
    def lambda_s(self) -> float:
        return cfg.cs_over_cp(self.nu) / cfg.FC

    def to(self, device) -> "InverseCase":
        t = None if self.theta_true is None else self.theta_true.to(device)
        return InverseCase(self.d_obs.to(device), self.src_idx, self.nu_idx, t,
                           self.snr_db)

    @staticmethod
    def from_dict(d: dict) -> "InverseCase":
        """Adapt `data.dataset.load_inversion_case` output."""
        return InverseCase(d_obs=d["d_obs"], src_idx=int(d["src_idx"]),
                           nu_idx=int(d["nu_idx"]), theta_true=d.get("theta_true"),
                           snr_db=d.get("snr_db"))


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
    a single threshold serve as the model-mismatch detector of §11.2 step 12.
    """
    r = (pred - obs).abs().pow(2)
    o = obs.abs().pow(2).expand_as(r)
    if weights is not None:
        w = weights.view(1, 1, 1, -1)
        r, o = r * w, o * w
    dims = (1, 2, 3)
    return _reduce(r.sum(dims), o.sum(dims), eps)


def amplitude_misfit(pred: Tensor, obs: Tensor, *, weights: Tensor | None = None,
                     scale_invariant: bool = False, eps: float = 1e-30) -> Tensor:
    """
    The same, on |pred| vs |obs| -- the phase-free screening misfit.  [B]

    Loses the resolution that phase carries (an amplitude spectrum constrains the
    defect to roughly a wavelength, not a tenth of one), which is the point: it is
    used to choose starting points, and it cannot cycle-skip because there is no
    carrier left in it to skip a cycle of.

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
    A callable J(theta) for one case, one band and one interface width.

    Kept as an object rather than a closure so that `stage`, `eps_cells` and `band`
    are inspectable after the fact: a reported position error is meaningless without
    knowing which band produced it, and a debug session that has to reverse-engineer
    a closure's captured variables is a debug session wasted.
    """
    forward: SurrogateForward
    case: InverseCase
    family: ShapeFamily
    band: slice = cfg.BAND_STAGE3
    eps_cells: float = cfg.EPS_INTERFACE_CELLS
    amplitude: bool = False
    scale_invariant: bool = False
    weights: Tensor | None = None

    def data(self) -> Tensor:
        return self.case.d_obs[..., self.band].to(self.forward.device)

    def residual(self, theta: Tensor) -> Tensor:
        """Per-candidate misfit, [B].  No regularisation."""
        pred = self.forward.predict(theta, self.family,
                                    src_idx=self.case.src_idx,
                                    nu_idx=self.case.nu_idx,
                                    band=self.band, eps_cells=self.eps_cells)
        if self.amplitude:
            return amplitude_misfit(pred, self.data(), weights=self.weights,
                                    scale_invariant=self.scale_invariant)
        return complex_misfit(pred, self.data(), weights=self.weights)

    def __call__(self, theta: Tensor) -> Tensor:
        return self.residual(theta)

    def of_z(self, z: Tensor, *, mu: float = 0.0) -> Tensor:
        """Misfit as a function of the unconstrained parameters, plus Tikhonov."""
        theta = self.family.to_physical(z, self.case.lambda_s)
        j = self.residual(theta)
        return j if mu == 0.0 else j + tikhonov(z, mu)


# ---------------------------------------------------------------------------
# Diagnostics: the misfit landscape figure of §11.3
# ---------------------------------------------------------------------------
@torch.no_grad()
def misfit_map(obj: Objective, *, n: int = 41, radius: float | None = None,
               chunk: int = cfg.SCREEN_CHUNK) -> dict:
    """
    J on a grid of (xc, yc) at fixed R -- the figure §11.3 calls non-negotiable.

    Two features are being looked for, and both are physics rather than decoration.
    The basin should be about lambda_s/4 across, which is the resolution a
    half-wavelength criterion predicts; and it should be *elongated* along the
    source-to-defect line, because a single source constrains travel time (hence
    range) far better than it constrains the angle.  A circular basin would mean
    something is wrong -- most likely that the misfit is being dominated by
    amplitude rather than phase.
    """
    lam_s = obj.case.lambda_s
    lo, hi = obj.family.bounds(lam_s)
    if radius is None:
        radius = (float(obj.case.theta_true[2]) if obj.case.theta_true is not None
                  else 0.5 * (cfg.R_MIN_LS + cfg.R_MAX_LS) * lam_s)
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
    elongation the landscape figure is supposed to show comes out as a number and
    not only as a picture.
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
    return dict(n_cells=int(mask.sum()),
                along_ls=float(along.max() - along.min()) / ls,
                across_ls=float(across.max() - across.min()) / ls,
                angle_deg=math.degrees(ang))


__all__ = [
    "InverseCase",
    "Objective",
    "SurrogateForward",
    "amplitude_misfit",
    "basin_width",
    "complex_misfit",
    "misfit_map",
    "tikhonov",
]
