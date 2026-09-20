"""
Training objective (§7).

Four terms:

    L = L_field + gamma * L_H1 + beta * L_meas + alpha * L_phys

`L_field` is the relative L2 error on the scattered displacement phasor -- relative,
per (sample, frequency), because the scattered amplitude varies by more than an order
of magnitude across the dataset (roughly R^2 in the Rayleigh regime, and a further
factor from source-to-defect distance) and an absolute L2 would let the handful of
large, close voids supply almost the whole gradient.

`L_H1` penalises the error in the *gradient* of the field.  A pure L2 loss is
notoriously happy to blur: it will trade a sharp scattered wavefront for a smooth
approximation of the same energy, because the L2 penalty for smearing a front over
two cells is small.  The inversion reads phase at the receivers, and a smeared front
is a phase error, so the derivative has to be in the objective explicitly.
gamma = 0.1 rather than 1.0 because at 16 points per P wavelength the discrete
gradient amplifies the high-wavenumber part of the error by roughly k dx ~ 0.4-0.7,
and matching the two terms' natural scales rather than their nominal weights is what
keeps L_field in charge.

`L_meas` is the error at the 32 receivers, which is what the inversion actually
reads.  32 pixels out of 16384 would otherwise contribute 0.2% of L_field and be
effectively unconstrained, and a surrogate that is excellent in the field interior
and mediocre on the measurement ring is useless for the inverse problem.  A random
subset of 8 of the 32 is used each step rather than all 32: the estimator is
unbiased either way, and subsetting stops the ring from becoming a set of 32
memorised values.

`L_phys` is the residual of the governing equation itself, evaluated on the predicted
*total* field.  Two things are excluded from it, both for real reasons rather than
convenience: three cells at each edge, because a 4th-order stencil reaches two cells
and the derivative of the field there is not defined by data the network was given;
and a disc of three cells around the source, where the true residual is a delta
function and finite differences of it are meaningless.

What sets the ceiling on `L_phys`, and why alpha is balanced rather than fixed: the
residual is evaluated on the 128^2 network grid, which is half the solver's
resolution, so the 4th-order stencil's own dispersion error (3 (k dx)^4 / 640, about
1e-3 relative at 9 points per shear wavelength) is a floor the network cannot go
below no matter how right it is.  The coefficient transition at the void boundary is
0.82 network cells wide (10-90%), which is *narrower* than the five-point stencil is
wide, so the stencil straddles it everywhere along the interface and adds a second
floor there.  A fixed alpha would therefore either be swamped or would spend the
network's capacity fitting discretisation error.  `balance_alpha` sets alpha from the
ratio of gradient norms so the physics term contributes a chosen *fraction* of the
data term's gradient, which is a statement about influence rather than about units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F_nn

from . import config as cfg
from . import features as feat

# 4th-order collocated central first derivative: (-f2 + 8 f1 - 8 f-1 + f-2) / 12h
_D1 = (1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0)   # at -2, -1, +1, +2


def d1(f: Tensor, dim: int, dx: float) -> Tensor:
    """
    4th-order central derivative along `dim`, returned on the *interior*.

    Shrinks the axis by 4 (two cells at each end) rather than padding.  Padding
    would be tidier and wrong: a zero or replicate pad invents field values, and the
    residual computed from invented values is largest exactly at the edge, where the
    absorbing layer has already been cropped away and the field is not small.
    """
    s = [slice(None)] * f.dim()

    def sl(off: int) -> Tensor:
        s[dim] = slice(2 + off, f.shape[dim] - 2 + off)
        return f[tuple(s)]

    return (_D1[0] * sl(-2) + _D1[1] * sl(-1)
            + _D1[2] * sl(+1) + _D1[3] * sl(+2)) / dx


def _crop(f: Tensor, n: int = 2) -> Tensor:
    """Trim n cells from both ends of the last two axes, to match d1's output."""
    return f[..., n:-n, n:-n] if n else f


# ---------------------------------------------------------------------------
# Data terms
# ---------------------------------------------------------------------------
def rel_l2(pred: Tensor, target: Tensor, *, eps: float = 1e-12) -> Tensor:
    """
    Mean over rows of ||pred - target||_2 / ||target||_2.

    Reduces over every axis but the first, so with the frequency axis folded into
    the batch (features.pack_inputs) each (sample, frequency) pair gets its own
    denominator.
    """
    dims = tuple(range(1, pred.dim()))
    num = (pred - target).pow(2).sum(dims).sqrt()
    den = target.pow(2).sum(dims).sqrt().clamp_min(eps)
    return (num / den).mean()


def h1_seminorm(pred: Tensor, target: Tensor, *, dx: float = cfg.DX_NET,
                eps: float = 1e-12) -> Tensor:
    """Relative L2 of the error in the spatial gradient.  Common interior only."""
    dims = tuple(range(1, pred.dim()))
    num, den = 0.0, 0.0
    for dim, other in ((-2, -1), (-1, -2)):
        gp, gt = d1(pred, dim, dx), d1(target, dim, dx)
        sl = [slice(None)] * pred.dim()
        sl[other] = slice(2, pred.shape[other] - 2)
        gp, gt = gp[tuple(sl)], gt[tuple(sl)]
        num = num + (gp - gt).pow(2).sum(dims)
        den = den + gt.pow(2).sum(dims)
    return (num.sqrt() / den.sqrt().clamp_min(eps)).mean()


def measurement_loss(pred: Tensor, target: Tensor, recv_yx: Tensor, *,
                     n_subset: int = 8, generator: torch.Generator | None = None,
                     eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """
    Relative error at a random subset of receivers.  Returns (loss, chosen indices).

    pred/target [B, C, ny, nx] real channels; recv_yx [R, 2] long, network-grid
    indices -- the same indices the solver sampled the A-scans at, which is why an
    A-scan and this field value are the same number rather than merely similar.

    The indices are bounds-checked rather than left to advanced indexing, because a
    receiver ring built for one grid and applied to another silently *works* for
    negative-looking indices (Python wraps them) and raises a bare IndexError with no
    context for the rest.  Both failure modes read as "the loss is fine" for a while.
    """
    if recv_yx.dim() != 2 or recv_yx.shape[1] != 2:
        raise ValueError(f"recv_yx must be [R, 2]; got {tuple(recv_yx.shape)}")
    ny, nx = pred.shape[-2], pred.shape[-1]
    lo = int(recv_yx.min()) if recv_yx.numel() else 0
    hi_y, hi_x = int(recv_yx[:, 0].max()), int(recv_yx[:, 1].max())
    if lo < 0 or hi_y >= ny or hi_x >= nx:
        raise IndexError(
            f"receiver indices span y<={hi_y}, x<={hi_x} (min {lo}) but the field is "
            f"{ny}x{nx}; the ring and the field are on different grids")
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    R = recv_yx.shape[0]
    n_subset = min(n_subset, R)
    # drawn on the CPU: a CUDA randperm cannot take a CPU generator, and tying the
    # subset choice to the device would make a run irreproducible across machines
    sel = torch.randperm(R, generator=generator)[:n_subset].to(recv_yx.device)
    ry, rx = recv_yx[sel, 0], recv_yx[sel, 1]
    p = pred[..., ry, rx]                      # [B, C, n_subset]
    t = target[..., ry, rx]
    dims = tuple(range(1, p.dim()))
    num = (p - t).pow(2).sum(dims).sqrt()
    den = t.pow(2).sum(dims).sqrt().clamp_min(eps)
    return (num / den).mean(), sel


# ---------------------------------------------------------------------------
# Physics term
# ---------------------------------------------------------------------------
@dataclass
class PhysicsContext:
    """
    Everything the residual needs, assembled once per batch.

    lam, mu, rho  [N, 1, ny, nx]   material fields on the *network* grid
    omega         [N, 1, 1, 1]     angular frequency per row
    weight        [N, 1, ny, nx]   full-size, >= 0; cropped by the consumer
    """
    lam: Tensor
    mu: Tensor
    rho: Tensor
    omega: Tensor
    weight: Tensor
    dx: float = cfg.DX_NET


def source_mask(src_idx: Tensor, ny: int = cfg.N_NET, nx: int = cfg.N_NET, *,
                radius_cells: float = cfg.PHYS_SOURCE_EXCLUDE_CELLS,
                device=None) -> Tensor:
    """
    0 inside `radius_cells` of the source, 1 elsewhere.  [N, 1, ny, nx].

    The point force is a delta: the exact residual is unbounded there, and a finite
    difference of it is a large finite number that would dominate the loss and pull
    the network towards cancelling an artefact.
    """
    iy = torch.arange(ny, device=device).view(1, -1, 1).float()
    ix = torch.arange(nx, device=device).view(1, 1, -1).float()
    sy = torch.tensor([cfg.SOURCES_NET[int(i)][0] for i in src_idx],
                      device=device).view(-1, 1, 1).float()
    sx = torch.tensor([cfg.SOURCES_NET[int(i)][1] for i in src_idx],
                      device=device).view(-1, 1, 1).float()
    far = torch.hypot(iy - sy, ix - sx) > radius_cells
    return far.float().unsqueeze(1)


def make_context(chi: Tensor, nu: Tensor, freqs: Tensor, src_idx: Tensor, *,
                 dx: float = cfg.DX_NET,
                 erode: int = cfg.ERODE_CELLS,
                 interface_weight: float = 0.0,
                 interface_cells: int = cfg.STENCIL_HALF_WIDTH) -> PhysicsContext:
    """
    Build a PhysicsContext for a batch, with the frequency axis already flattened.

    chi     [B, ny, nx]   soft indicator on the network grid
    nu      [B]           Poisson ratio
    freqs   [B, F]        frequency per column, in units of f_c
    src_idx [B]           source index

    `erode` cells are removed from each edge on top of the two the stencil already
    costs: 2 for the stencil, 1 more because the *coefficient* fields are also
    differentiated in the divergence, so the outermost usable cell needs its
    neighbours' coefficients too.  ERODE_CELLS is defined in config as
    STENCIL_HALF_WIDTH + 1 for exactly this reason.

    The frequency axis is flattened B-major (`repeat_interleave`), so row `b`'s F
    frequencies are consecutive and a label tensor shaped [B, F, ...] flattens to match
    with a plain `reshape`.  This is the opposite convention to `features.pack_inputs`,
    which is F-major because the *network* consumes one frequency per forward pass; the
    two are not interchangeable and mixing them silently pairs the wrong omega with the
    wrong field.

    `interface_weight` is review §3.7's objection.  The default weight is the solid
    fraction 1 - chi, which is zero exactly on the void boundary -- and the void boundary
    is where the traction-free condition lives, i.e. the only place the scattering
    problem is actually posed.  A physics loss that is blind there is not weakly
    enforcing the boundary condition, it is not enforcing it at all; what it constrains
    is the Navier equation in a medium that happens to have a hole in it.

    Setting `interface_weight = w` restores a band of weight `w` around the chi = 0.5
    contour, `interface_cells` network cells wide on each side.  The band is built by
    dilating `4 chi (1 - chi)` -- which peaks at 1 on the contour -- with a max filter,
    rather than used directly, and the reason is a measurement: the interface is 1.6
    *fine* cells wide, so on the network grid chi goes from 0.03 to 0.97 inside a single
    cell and `4 chi (1 - chi)` alone is nonzero on at most one cell per boundary normal,
    where 1 - chi is already larger than it.  Used directly the knob would change the
    weight sum by nothing at all.  Dilation also selects the right set: the 4th-order
    stencil reaches 2 cells, so `STENCIL_HALF_WIDTH` is the radius over which a cell's
    residual is contaminated by the discontinuity, and that band is precisely what the
    default weight discards.

    It defaults to 0 because the residual in the band is genuinely large -- the field is
    not smooth on the stencil's scale there, so the finite difference of the *exact*
    solution has an O(1) error and the term would be minimised by smoothing the true
    answer.  Turn it on only with a weight small enough that the term informs rather than
    dominates, and read `validate.check_physics_residual_on_labels`, which reports the
    residual with and without the band on labels the solver produced, first.
    """
    from .geometry.sdf import material_fields

    B, F = freqs.shape
    ny, nx = chi.shape[-2], chi.shape[-1]
    dev = chi.device

    mu0 = ((1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))).view(-1, 1, 1)
    lam0 = 1.0 - 2.0 * mu0
    lam, mu, rho = material_fields(chi, lam0, mu0)              # [B, ny, nx]

    rep = lambda t: t.repeat_interleave(F, dim=0).unsqueeze(1)  # [B*F, 1, ny, nx]
    omega = (2.0 * math.pi * freqs.reshape(-1)).view(-1, 1, 1, 1)

    # The residual is only meaningful in the solid: inside the void the stiffness is
    # at the floor (1e-4) and the equation is numerically degenerate.  Weighting by
    # the solid fraction is a smooth version of "solve where there is material".
    solid = (1.0 - chi).clamp_min(0.0).unsqueeze(1)             # [B, 1, ny, nx]
    if interface_weight:
        r = int(interface_cells)
        shell = (4.0 * chi * (1.0 - chi)).clamp_min(0.0).unsqueeze(1)
        if r > 0:
            shell = F_nn.max_pool2d(shell, 2 * r + 1, stride=1, padding=r)
        solid = torch.maximum(solid, interface_weight * shell)
    w = solid * source_mask(src_idx, ny, nx, device=dev)
    if erode:
        keep = torch.zeros_like(w)
        keep[..., erode:-erode, erode:-erode] = 1.0
        w = w * keep
    w = w.repeat_interleave(F, dim=0)                           # [B*F, 1, ny, nx]

    return PhysicsContext(lam=rep(lam), mu=rep(mu), rho=rep(rho),
                          omega=omega, weight=w, dx=dx)


def navier_residual(u: Tensor, ctx: PhysicsContext) -> Tensor:
    """
    Residual of the time-harmonic Navier equation, [N, 2, ny-8, nx-8] complex.

        R_i = d_j sigma_ij + rho omega^2 u_i,
        sigma_ij = lam delta_ij d_k u_k + mu (d_i u_j + d_j u_i)

    u is complex [N, 2, ny, nx] with component 0 = x, 1 = y.  Two nested 4th-order
    stencils cost four cells at each end of each axis, hence the -8.

    The divergence is taken of the *stress*, not expanded into the constant-coefficient
    Navier-Cauchy form.  Expanding would give (lam + mu) grad div u + mu lap u, which
    is only equal to this for spatially constant lam and mu -- and the entire point of
    the problem is that they are not constant.  Writing it in stress-divergence form
    also means the traction-free condition at the void boundary is enforced implicitly
    by mu -> 0 there, exactly as in the solver, instead of needing a separate boundary
    term.
    """
    assert u.is_complex(), "the residual is time-harmonic; u must be complex"
    dx = ctx.dx
    ux, uy = u[:, 0:1], u[:, 1:2]

    # Each d1 shrinks only the differentiated axis, so bring all four first
    # derivatives onto the common [ny-4, nx-4] interior before combining.
    dux_dx = d1(ux, -1, dx)[..., 2:-2, :]
    dux_dy = d1(ux, -2, dx)[..., :, 2:-2]
    duy_dx = d1(uy, -1, dx)[..., 2:-2, :]
    duy_dy = d1(uy, -2, dx)[..., :, 2:-2]

    lam, mu = _crop(ctx.lam), _crop(ctx.mu)
    div = dux_dx + duy_dy
    sxx = lam * div + 2.0 * mu * dux_dx
    syy = lam * div + 2.0 * mu * duy_dy
    sxy = mu * (dux_dy + duy_dx)

    dsxx_dx = d1(sxx, -1, dx)[..., 2:-2, :]
    dsxy_dy = d1(sxy, -2, dx)[..., :, 2:-2]
    dsxy_dx = d1(sxy, -1, dx)[..., 2:-2, :]
    dsyy_dy = d1(syy, -2, dx)[..., :, 2:-2]

    rho = _crop(ctx.rho, 4)
    w2 = ctx.omega ** 2
    ui = _crop(u, 4)
    rx = dsxx_dx + dsxy_dy + rho * w2 * ui[:, 0:1]
    ry = dsxy_dx + dsyy_dy + rho * w2 * ui[:, 1:2]
    return torch.cat([rx, ry], dim=1)


def physics_loss(u_total: Tensor, ctx: PhysicsContext, *,
                 eps: float = 1e-12) -> Tensor:
    """
    Weighted RMS residual, normalised by rho omega^2 |u| so it is dimensionless.

    The normaliser is the inertial term rather than the field itself: dividing by
    ||R|| of anything derived from the prediction would let the network reduce the
    loss by shrinking its own output, which is a degenerate minimum the physics term
    is specifically supposed to rule out.
    """
    r = navier_residual(u_total, ctx)
    w = _crop(ctx.weight, 4)
    assert w.shape[-2:] == r.shape[-2:], (
        f"weight {tuple(w.shape[-2:])} does not match residual "
        f"{tuple(r.shape[-2:])}; the crop bookkeeping has drifted")
    num = (r.abs().pow(2) * w).sum(dim=(1, 2, 3))
    scale = (_crop(ctx.rho, 4) * ctx.omega ** 2
             * _crop(u_total, 4).abs()).pow(2)
    den = (scale.sum(dim=1, keepdim=True) * w).sum(dim=(1, 2, 3)).clamp_min(eps)
    return (num / den).sqrt().mean()


# ---------------------------------------------------------------------------
# Assembly and balancing
# ---------------------------------------------------------------------------
@dataclass
class LossTerms:
    total: Tensor
    field: Tensor
    h1: Tensor
    meas: Tensor
    phys: Tensor
    alpha: float

    def as_dict(self) -> dict[str, float]:
        return dict(total=float(self.total), field=float(self.field),
                    h1=float(self.h1), meas=float(self.meas),
                    phys=float(self.phys), alpha=self.alpha)


def compute(pred: Tensor, target: Tensor, *, recv_yx: Tensor,
            ctx: PhysicsContext | None = None, u_inc: Tensor | None = None,
            alpha: float = cfg.ALPHA_PHYS, beta: float = cfg.BETA_MEAS,
            gamma: float = cfg.GAMMA_H1, n_meas: int = 8,
            generator: torch.Generator | None = None) -> LossTerms:
    """
    All four terms.  `pred`/`target` are [N, 4, ny, nx] real channel form.

    `u_inc` is the flattened complex incident field [N, 2, ny, nx]; the physics
    residual needs the *total* field, and the network only predicts the scattered
    part, so they have to be added back together here.  Skipping that -- taking the
    residual of the scattered field alone -- would be penalising the network for
    failing to satisfy an equation it does not satisfy: the scattered field obeys the
    Navier equation with a source term supported on the void, not a homogeneous one.
    """
    l_field = rel_l2(pred, target)
    l_h1 = h1_seminorm(pred, target)
    l_meas, _ = measurement_loss(pred, target, recv_yx, n_subset=n_meas,
                                 generator=generator)
    if ctx is None or u_inc is None:
        l_phys = torch.zeros((), device=pred.device, dtype=pred.dtype)
        alpha = 0.0
    else:
        u_s = feat.channels_to_complex(pred)
        l_phys = physics_loss(u_inc + u_s, ctx)
    total = l_field + gamma * l_h1 + beta * l_meas + alpha * l_phys
    return LossTerms(total=total, field=l_field, h1=l_h1, meas=l_meas,
                     phys=l_phys, alpha=float(alpha))


def balance_alpha(l_data: Tensor, l_phys: Tensor, params: list[Tensor], *,
                  target_ratio: float = 0.1, alpha_prev: float | None = None,
                  ema: float = 0.9, clamp: tuple[float, float] = (1e-5, 1.0)
                  ) -> float:
    """
    alpha such that alpha * ||grad L_phys|| = target_ratio * ||grad L_data||.

    Measured on `params` -- pass the projection layer's weights, i.e. the last place
    the two terms' gradients are still comparable quantities.  Measuring on the whole
    network would average over the lifting layers, where the physics term's gradient
    is diluted by the depth it has passed through and the ratio stops meaning
    anything.

    EMA-smoothed because the ratio is noisy from step to step (the physics residual
    is dominated by whichever samples in the batch have a void nearest the receiver
    ring), and an alpha that jumps by an order of magnitude between steps makes the
    total loss non-stationary for Adam's second-moment estimate.

    Call inside a `torch.enable_grad()` region but *outside* the optimiser step, and
    note that it costs two extra backward passes: use it every N steps, not every
    step.
    """
    def gnorm(loss: Tensor) -> float:
        g = torch.autograd.grad(loss, params, retain_graph=True,
                                allow_unused=True, create_graph=False)
        # |x|**2, not x**2: gradients w.r.t. the spectral weights are complex, and
        # float(complex_tensor) raises "cannot be converted to double without
        # overflow".  abs().pow(2) is the right L2 summand and a no-op for real x.
        return math.sqrt(sum(float(x.abs().pow(2).sum()) for x in g if x is not None))

    gd, gp = gnorm(l_data), gnorm(l_phys)
    if gp <= 0.0 or not math.isfinite(gp):
        return alpha_prev if alpha_prev is not None else cfg.ALPHA_PHYS
    a = target_ratio * gd / gp
    if alpha_prev is not None:
        a = ema * alpha_prev + (1.0 - ema) * a
    return float(min(max(a, clamp[0]), clamp[1]))


__all__ = [
    "LossTerms",
    "PhysicsContext",
    "balance_alpha",
    "compute",
    "d1",
    "h1_seminorm",
    "make_context",
    "measurement_loss",
    "navier_residual",
    "physics_loss",
    "rel_l2",
    "source_mask",
]
