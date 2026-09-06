"""
Batched 4th-order staggered-grid velocity-stress elastic FDTD (§3.1 - §3.6).

Written in PyTorch so that many samples run in parallel on one GPU, and so the
same finite-difference kernels can be reused by the differential scattered-field
physics loss (§7.2) -- consistency between the label-generation path and the
physics-loss path is worth more than elegance in either one.  ("Lippmann-Schwinger"
is what earlier versions of this file and of §7.2 called that residual; it is not.
Lippmann-Schwinger is the *integral* equation, u = u_inc + G * V u, and nothing here
builds a Green's operator.  What §7.2 writes is the differential equation the
scattered field satisfies, with the contrast terms as a source, and what
`losses.physics_loss` evaluates is its residual under this stencil.)

Grid convention
---------------
Arrays are [B, iy, ix]; axis -2 is y, axis -1 is x.  Variables sit at offset
positions inside each cell (§3.2):

    sxx, syy   (i,       j    )   cell centre
    vx         (i + 1/2, j    )   x-face
    vy         (i,       j + 1/2)  y-face
    sxy        (i + 1/2, j + 1/2)  corner

Time is staggered too: velocities live at t^{n+1/2}, stresses at t^n.  The
resulting leapfrog is second-order in time and *non-dissipative* -- it conserves
a discrete energy, which is why `validate.py` can assert energy conservation as
a sharp number rather than a trend.  A dissipative scheme would damp the wave
over the ~1400 steps needed to cross the domain and the network would learn
attenuation that is not physical.

Which derivative goes where follows from the stagger and is the easiest thing in
the whole project to get subtly wrong, so each one is annotated at the call site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import config as cfg

# 4th-order staggered coefficients (§3.2).  The s^3 term of the symbol expansion
# cancels between these two -- that cancellation *is* the fourth-order accuracy.
C1: float = 9.0 / 8.0
C2: float = 1.0 / 24.0


# ---------------------------------------------------------------------------
# Staggered derivative operators
# ---------------------------------------------------------------------------
def _pad_axis(g: Tensor, axis: int, before: int, after: int,
              mode: str = "constant") -> Tensor:
    """Pad a single axis of a [..., ny, nx] tensor."""
    if axis in (-1, g.dim() - 1):
        pad = (before, after, 0, 0)
    elif axis in (-2, g.dim() - 2):
        pad = (0, 0, before, after)
    else:
        raise ValueError("only the last two axes are spatial")
    if mode == "constant":
        return F.pad(g, pad)
    # replicate needs a 4D input
    squeeze = g.dim() == 3
    gg = g.unsqueeze(1) if squeeze else g
    out = F.pad(gg, pad, mode="replicate")
    return out.squeeze(1) if squeeze else out


def _take(g: Tensor, axis: int, start: int, n: int) -> Tensor:
    if axis in (-1, g.dim() - 1):
        return g[..., start:start + n]
    return g[..., start:start + n, :]


def d_plus(g: Tensor, axis: int, dx: float) -> Tensor:
    """
    Derivative of a field living on integer nodes, evaluated at i + 1/2:

        (D+ g)_{i+1/2} = [ 9/8 (g_{i+1} - g_i) - 1/24 (g_{i+2} - g_{i-1}) ] / dx
    """
    n = g.shape[axis]
    gp = _pad_axis(g, axis, before=1, after=2)
    return (C1 * (_take(gp, axis, 2, n) - _take(gp, axis, 1, n))
            - C2 * (_take(gp, axis, 3, n) - _take(gp, axis, 0, n))) / dx


def d_minus(g: Tensor, axis: int, dx: float) -> Tensor:
    """
    Derivative of a field living on half-integer nodes, evaluated at i:

        (D- g)_i = [ 9/8 (g_i - g_{i-1}) - 1/24 (g_{i+1} - g_{i-2}) ] / dx
    """
    n = g.shape[axis]
    gp = _pad_axis(g, axis, before=2, after=1)
    return (C1 * (_take(gp, axis, 2, n) - _take(gp, axis, 1, n))
            - C2 * (_take(gp, axis, 3, n) - _take(gp, axis, 0, n))) / dx


def avg_plus(g: Tensor, axis: int) -> Tensor:
    """Arithmetic average of a nodal field onto i + 1/2."""
    n = g.shape[axis]
    gp = _pad_axis(g, axis, before=0, after=1, mode="replicate")
    return 0.5 * (_take(gp, axis, 0, n) + _take(gp, axis, 1, n))


def avg_minus(g: Tensor, axis: int) -> Tensor:
    """Arithmetic average of a half-integer field onto i."""
    n = g.shape[axis]
    gp = _pad_axis(g, axis, before=1, after=0, mode="replicate")
    return 0.5 * (_take(gp, axis, 0, n) + _take(gp, axis, 1, n))


def harmonic_plus(g: Tensor, axis: int, floor: float) -> Tensor:
    """Harmonic average onto i + 1/2.  Correct for moduli across an interface."""
    n = g.shape[axis]
    gp = _pad_axis(g, axis, before=0, after=1, mode="replicate").clamp_min(floor)
    a, b = _take(gp, axis, 0, n), _take(gp, axis, 1, n)
    return 2.0 * a * b / (a + b)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
def tone_burst(t: Tensor, fc: float = cfg.FC, n_cycles: int = cfg.N_CYCLES
               ) -> Tensor:
    """
    N_c-cycle Hann-windowed sinusoid (§2.3), zero outside [0, N_c/f_c].

    The window spectrum is *exactly* zero at f_c(1 +- 2/N_c); §5.4 places all 20
    operating frequencies strictly inside those nulls, which is why the
    deconvolution of harmonic.py stays conditioned.
    """
    dur = n_cycles / fc
    w = 0.5 * (1.0 - torch.cos(2.0 * math.pi * fc * t / n_cycles))
    s = w * torch.sin(2.0 * math.pi * fc * t)
    return torch.where((t >= 0) & (t <= dur), s, torch.zeros_like(s))


def source_spectrum(omegas: Tensor, dt: float, nt: int, *,
                    fc: float = cfg.FC, n_cycles: int = cfg.N_CYCLES,
                    dtype=torch.float64) -> Tensor:
    """
    s_hat(omega) on the grid the force is *injected* on: t_n = n dt.

    Not the velocity grid.  The force enters the leapfrog velocity update, which
    steps v^{n-1/2} -> v^{n+1/2} about the stress level t^n, so the body force in
    that update is the force at t^n and the DFT that inverts it must use integer
    sample times.  Deconvolving a velocity phasor (half-integer times) with a
    source spectrum computed on the same half-integer grid leaves a factor
    exp(+i omega dt/2) on every phasor: 0.072 rad at f_max, a 6% error that looks
    exactly like every wave arriving half a step early -- indistinguishable from a
    uniform wave-speed error, which is the quantity the inversion measures.

    `tests/test_dft_consistency.py` pins this against the injection site, and
    `solver.validate.check_green_incident` measures the consequence against the
    analytic Green's tensor.
    """
    t = torch.arange(nt, dtype=dtype, device=omegas.device) * dt
    s = tone_burst(t, fc, n_cycles)
    return dft_at_freqs(s.unsqueeze(0), omegas.to(dtype), dt, nt,
                        t_offset=0.0).squeeze(0)


def dft_at_freqs(signal: Tensor, omegas: Tensor, dt: float, nt: int, *,
                 t_offset: float = 0.5) -> Tensor:
    """
    DFT of a signal sampled at t_n = (n + `t_offset`) dt:

        X(omega) = sum_n x_n exp(-i omega (n + t_offset) dt) dt

    `signal` is [..., nt]; returns [..., n_omega] complex.  The default 1/2 is the
    velocity grid, because the leapfrog puts velocities at half-integer time
    levels and that is the grid the recorded data actually lives on.  Pass 0.0 for
    a quantity defined at integer levels -- the stresses, or the injected source.
    """
    t = ((torch.arange(nt, dtype=torch.float64, device=signal.device) + t_offset)
         * dt)
    phase = torch.exp(-1j * omegas.to(torch.float64).view(-1, 1) * t.view(1, -1))
    x = signal.to(torch.complex128)
    return torch.einsum("...t,mt->...m", x, phase) * dt


# ---------------------------------------------------------------------------
# Absorbing layer
# ---------------------------------------------------------------------------
def absorber_profile(n_total: int, n_pml: int, dx: float, *,
                     shift_y: float = 0.0, shift_x: float = 0.0,
                     order: float = cfg.ABSORBER_ORDER,
                     r_target: float = cfg.ABSORBER_R_TARGET,
                     c: float = cfg.CP,
                     device=None, dtype=torch.float32) -> Tensor:
    """
    Graded polynomial damping d(x) = d0 (depth / L_abs)^p, summed over the two
    directions (§3.6).

    Graded rather than constant because an abrupt jump in damping is itself an
    impedance discontinuity and reflects; ramping from zero at the interface
    means the wave never sees a sharp change.

    `shift_y/shift_x` evaluate the profile at a staggered location (pass 0.5 for
    a half-cell offset) so each field is damped at its own position.

    This is a sponge, not a PML: the damping is added to the velocity update
    without any complex coordinate stretch, so `r_target` is a design input to the
    d0 formula and not the reflection the layer achieves.  `solver.validate.
    check_absorber_reflection` measures the achieved reflection.
    """
    if n_pml <= 0:
        return torch.zeros(n_total, n_total, device=device, dtype=dtype)
    l_pml = n_pml * dx
    d0 = -(order + 1.0) * c * math.log(r_target) / (2.0 * l_pml)

    def axis_profile(shift: float) -> Tensor:
        i = torch.arange(n_total, device=device, dtype=dtype) + shift
        depth_lo = (n_pml - i).clamp_min(0.0)
        depth_hi = (i - (n_total - 1 - n_pml)).clamp_min(0.0)
        depth = torch.maximum(depth_lo, depth_hi) * dx
        return d0 * (depth / l_pml) ** order

    dy = axis_profile(shift_y).view(-1, 1)
    dx_ = axis_profile(shift_x).view(1, -1)
    return dy + dx_


damping_profile = absorber_profile      # earlier name, kept for callers/notebooks


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    """
    ascans      [B, n_recv, 2, nt]   velocity at the ring, full time resolution
    phasors     [B, 2, M, ny, nx]    complex velocity phasors on the network grid
    frames      [B, n_frames, 2, ny, nx] or None
    energy      [n_diag] discrete total energy, or None
    energy_t    [n_diag] sample times
    """
    ascans: Tensor
    phasors: Tensor | None = None
    frames: Tensor | None = None
    energy: Tensor | None = None
    energy_t: Tensor | None = None
    peak_energy: float = 0.0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class ElasticFDTD2D:
    """
    One instance holds the material and damping arrays for a batch of samples;
    `run()` steps it in time.

    Parameters
    ----------
    lam, mu, rho : [B, NY, NX] on the *padded* fine grid, cell-centred.
    n_pml        : absorbing-layer thickness in fine cells (0 disables it).
    """

    def __init__(self, lam: Tensor, mu: Tensor, rho: Tensor, *,
                 dx: float = cfg.DX_FINE, dt: float = cfg.DT,
                 n_pml: int = cfg.N_PML_FINE,
                 downsample: int = cfg.DOWNSAMPLE,
                 absorber_order: float = cfg.ABSORBER_ORDER,
                 absorber_r_target: float = cfg.ABSORBER_R_TARGET):
        assert lam.shape == mu.shape == rho.shape and lam.dim() == 3
        self.B, self.ny, self.nx = lam.shape
        assert self.ny == self.nx, "square grids only"
        self.dx, self.dt, self.n_pml = dx, dt, n_pml
        self.downsample = downsample
        self.device, self.dtype = lam.device, lam.dtype

        # The absorbing layer is cropped away before anything is handed to the
        # network (§3.6): it is not physical, and the FFT inside the FNO assumes
        # periodic boundaries, so an absorbing layer at the edge would create a
        # discontinuity across the periodic wrap and show up as Gibbs ringing at
        # high wavenumber.
        self.core = slice(n_pml, self.ny - n_pml)
        self.n_core = self.ny - 2 * n_pml
        assert self.n_core % downsample == 0, (
            f"cropped grid {self.n_core} is not divisible by the downsample "
            f"factor {downsample}")
        self.n_net = self.n_core // downsample

        # -- CFL, checked rather than assumed -----------------------------
        c_max = torch.sqrt(((lam + 2 * mu) / rho.clamp_min(1e-30)).max()).item()
        courant = c_max * dt / dx
        if courant > cfg.CFL_LIMIT_4TH:
            raise ValueError(
                f"CFL violated: c_max dt/dx = {courant:.4f} > "
                f"{cfg.CFL_LIMIT_4TH:.4f}.  c_max={c_max:.4f}.  "
                "Reduce dt or check the material fields.")
        self.courant = courant

        # -- moduli at their own staggered positions (§3.1) ----------------
        floor = cfg.VOID_STIFFNESS_FLOOR * float(mu.max())
        self.lam = lam
        self.mu = mu
        self.c11 = lam + 2.0 * mu                       # centres
        # mu at corners: harmonic in both directions, which is the standard
        # interface-consistent averaging and correctly gives ~0 at a void edge.
        self.mu_xy = harmonic_plus(harmonic_plus(mu, -1, floor), -2, floor)
        self.rho_vx = avg_plus(rho, -1)                 # x-faces
        self.rho_vy = avg_plus(rho, -2)                 # y-faces

        # -- damping, one profile per field position -----------------------
        # `absorber_order` and `absorber_r_target` are arguments rather than reads of
        # the config so that `solver.validate.check_absorber_reflection` can measure
        # what the layer actually reflects as a function of its own design inputs.  The
        # production values are the config defaults.
        self.absorber_order = absorber_order
        self.absorber_r_target = absorber_r_target
        kw = dict(device=self.device, dtype=self.dtype, order=absorber_order,
                  r_target=absorber_r_target)
        self.d_c = absorber_profile(self.ny, n_pml, dx, **kw)
        self.d_vx = absorber_profile(self.ny, n_pml, dx, shift_x=0.5, **kw)
        self.d_vy = absorber_profile(self.ny, n_pml, dx, shift_y=0.5, **kw)
        self.d_xy = absorber_profile(self.ny, n_pml, dx, shift_x=0.5, shift_y=0.5, **kw)

        # Crank-Nicolson factors for  dF/dt + d F = RHS  (§3.6)
        def cn(d: Tensor) -> tuple[Tensor, Tensor]:
            half = 0.5 * dt * d
            return (1.0 - half) / (1.0 + half), dt / (1.0 + half)

        self.a_c, self.b_c = cn(self.d_c)
        self.a_vx, self.b_vx = cn(self.d_vx)
        self.a_vy, self.b_vy = cn(self.d_vy)
        self.a_xy, self.b_xy = cn(self.d_xy)

    # -- energy diagnostic (§3.7 checks 2 and 3) --------------------------
    def _energy(self, vx, vy, sxx, syy, sxy, core_only: bool = True) -> Tensor:
        """
        Discrete total energy 0.5 * integral( rho|v|^2 + sigma:epsilon ) dA.

        Plane strain inverts to
            eps_xx = [(lam+2mu) s_xx - lam s_yy] / det,   det = 4 mu (lam + mu)
        so the strain energy is expressible in the primary variables directly.

        `core_only` restricts the integral to the physical region, which is what
        check 3 needs: "residual energy *in the domain* after the wave has
        exited" must not include the energy still being dissipated inside the
        absorbing layer.
        """
        c = self.core if core_only else slice(None)
        lam, mu, c11 = self.lam[:, c, c], self.mu[:, c, c], self.c11[:, c, c]
        det = (4.0 * mu * (lam + mu)).clamp_min(1e-30)
        sxx_, syy_, sxy_ = sxx[:, c, c], syy[:, c, c], sxy[:, c, c]
        strain = 0.5 * (
            (c11 * (sxx_ ** 2 + syy_ ** 2) - 2.0 * lam * sxx_ * syy_) / det
            + sxy_ ** 2 / mu.clamp_min(1e-30)
        )
        kinetic = 0.5 * (self.rho_vx[:, c, c] * vx[:, c, c] ** 2
                         + self.rho_vy[:, c, c] * vy[:, c, c] ** 2)
        return (kinetic + strain).sum(dim=(-2, -1)) * self.dx ** 2

    # -- main loop ---------------------------------------------------------
    @torch.no_grad()
    def run(self, src_yx: list[tuple[int, int]], *,
            nt: int = cfg.NT,
            recv_yx: list[tuple[int, int]] | None = None,
            omegas: Tensor | None = None,
            save_frames: int = 0,
            energy_every: int = 0,
            amplitude: float = 1.0,
            n_cycles: int = cfg.N_CYCLES,
            dft_every: int = cfg.DFT_EVERY,
            progress=None) -> SimResult:
        """
        Step the scheme `nt` times.

        src_yx   one (iy, ix) per batch item, in *padded fine grid* indices --
                 the force has to be injected on the grid the solver runs on.
        recv_yx  receiver positions in *network grid* indices (cropped and
                 downsampled).  Deliberately a different convention from
                 src_yx, and worth the asymmetry: the A-scans are then sampled
                 from exactly the same cropped, 2x2-averaged field that the
                 network predicts, so an A-scan and the corresponding field
                 phasor at that receiver are consistent to the last bit.
                 Sampling the fine grid instead would introduce an O(dx^2)
                 offset between the training labels and the inversion data --
                 small, but it would sit precisely where the inversion reads
                 phase.
        omegas   if given, accumulate the running DFT at these angular
                 frequencies.  This is done on the fly rather than by FFT-ing
                 saved frames: with only 64 frames the save interval puts
                 Nyquist at 1.33 f_c, right on top of the operating band, so an
                 FFT of the frames would alias exactly where it hurts.  The
                 running sum is exact at every frequency for a few tens of MB.
        """
        B, ny, nx = self.B, self.ny, self.nx
        dev, dt, dx = self.device, self.dt, self.dx
        z = lambda: torch.zeros(B, ny, nx, device=dev, dtype=self.dtype)
        vx, vy, sxx, syy, sxy = z(), z(), z(), z(), z()

        assert len(src_yx) == B, f"{len(src_yx)} sources for batch {B}"
        bidx = torch.arange(B, device=dev)
        sy = torch.tensor([p[0] for p in src_yx], device=dev)
        sx = torch.tensor([p[1] for p in src_yx], device=dev)

        # receivers
        if recv_yx is not None:
            bad = [p for p in recv_yx
                   if not (0 <= p[0] < self.n_net and 0 <= p[1] < self.n_net)]
            assert not bad, (
                f"receivers {bad[:3]} are outside the {self.n_net}^2 network grid; "
                "recv_yx uses network-grid indices, not padded fine-grid indices")
            ry = torch.tensor([p[0] for p in recv_yx], device=dev)
            rx = torch.tensor([p[1] for p in recv_yx], device=dev)
            ascans = torch.zeros(B, len(recv_yx), 2, nt, device=dev, dtype=self.dtype)
        else:
            ry = rx = None
            ascans = torch.zeros(B, 0, 2, nt, device=dev, dtype=self.dtype)

        # running DFT accumulators, on the *cropped, downsampled* network grid.
        # Cropping first is required (§3.6); downsampling first is free, because
        # average-pooling is linear and therefore commutes with the accumulation,
        # and it cuts the accumulator memory by 4x.
        nnet = self.n_net
        core = self.core
        if omegas is not None:
            M = omegas.numel()
            phasors = torch.zeros(B, 2, M, nnet, nnet, device=dev, dtype=torch.complex64)
            om = omegas.to(dev).to(torch.float64).view(-1, 1, 1)
        else:
            phasors = None

        frames = None
        if save_frames:
            every = max(1, nt // save_frames)
            frames = torch.zeros(B, save_frames, 2, nnet, nnet,
                                 device=dev, dtype=self.dtype)

        e_list, e_t = [], []
        # Source samples at the *stress* time levels t = n dt, not the velocity
        # levels: the force is added to the velocity update that steps
        # v^{n-1/2} -> v^{n+1/2}, and that update is centred on t^n.  Sampling the
        # burst at (n+1/2) dt instead is a half-step time shift of the effective
        # source -- first order, and worth 6% of the phasor at f_max.
        # `source_spectrum` uses this same grid so the deconvolution inverts what
        # was actually injected.
        ts = torch.arange(nt, device=dev, dtype=torch.float64) * dt
        tv = (torch.arange(nt, device=dev, dtype=torch.float64) + 0.5) * dt
        src_wave = tone_burst(ts, cfg.FC, n_cycles).to(self.dtype) * amplitude
        # point force -> body-force density
        src_scale = 1.0 / dx ** 2

        it = range(nt)
        if progress is not None:
            it = progress(it)

        for n in it:
            # ---- velocity update, t^{n-1/2} -> t^{n+1/2} -----------------
            # vx sits at (i+1/2, j): d/dx of a centred field  -> d_plus  on x
            #                        d/dy of a corner field   -> d_minus on y
            fx = d_plus(sxx, -1, dx) + d_minus(sxy, -2, dx)
            # vy sits at (i, j+1/2): d/dx of a corner field   -> d_minus on x
            #                        d/dy of a centred field  -> d_plus  on y
            fy = d_minus(sxy, -1, dx) + d_plus(syy, -2, dx)

            vx = self.a_vx * vx + self.b_vx * fx / self.rho_vx
            vy = self.a_vy * vy + self.b_vy * fy / self.rho_vy

            # vertical point force on the ring (§3.7)
            vy[bidx, sy, sx] = vy[bidx, sy, sx] + (
                self.b_vy[sy, sx] * src_wave[n] * src_scale / self.rho_vy[bidx, sy, sx])

            # ---- record at the velocity level ---------------------------
            need_dft = phasors is not None and n % dft_every == 0
            need_frame = (frames is not None and n % every == 0
                          and n // every < save_frames)
            if ry is not None or need_dft or need_frame:
                vx_c = avg_minus(vx, -1)      # x-faces -> centres
                vy_c = avg_minus(vy, -2)      # y-faces -> centres
                vx_d = F.avg_pool2d(vx_c[:, core, core].unsqueeze(1),
                                    self.downsample).squeeze(1)
                vy_d = F.avg_pool2d(vy_c[:, core, core].unsqueeze(1),
                                    self.downsample).squeeze(1)

            if ry is not None:
                # Sampled from vx_d, not vx_c: the A-scan is then the same
                # number the network's output carries at that cell.
                ascans[:, :, 0, n] = vx_d[:, ry, rx]
                ascans[:, :, 1, n] = vy_d[:, ry, rx]

            if phasors is not None and need_dft:
                w = torch.exp(-1j * om * tv[n]).to(torch.complex64) * (dt * dft_every)
                phasors[:, 0] += vx_d.unsqueeze(1).to(torch.complex64) * w
                phasors[:, 1] += vy_d.unsqueeze(1).to(torch.complex64) * w

            if need_frame:
                k = n // every
                frames[:, k, 0] = vx_d
                frames[:, k, 1] = vy_d

            # ---- stress update, t^n -> t^{n+1} ---------------------------
            # centres: d/dx of an x-face field -> d_minus; d/dy of a y-face
            # field -> d_minus
            exx = d_minus(vx, -1, dx)
            eyy = d_minus(vy, -2, dx)
            sxx = self.a_c * sxx + self.b_c * (self.c11 * exx + self.lam * eyy)
            syy = self.a_c * syy + self.b_c * (self.lam * exx + self.c11 * eyy)

            # corners: d/dy of an x-face field -> d_plus; d/dx of a y-face
            # field -> d_plus
            exy = d_plus(vx, -2, dx) + d_plus(vy, -1, dx)
            sxy = self.a_xy * sxy + self.b_xy * (self.mu_xy * exy)

            if energy_every and n % energy_every == 0:
                e_list.append(self._energy(vx, vy, sxx, syy, sxy))
                e_t.append(float(tv[n]))

            # Divergence check every 100 steps rather than every step: each test
            # forces a GPU synchronisation, and 1400 of those cost more than the
            # solve.
            if n % 100 == 0 and not torch.isfinite(sxx[..., ::37, ::37]).all():
                raise FloatingPointError(
                    f"solver diverged at step {n}/{nt} (Courant={self.courant:.4f})")

        if not torch.isfinite(sxx).all():
            raise FloatingPointError(
                f"solver diverged before step {nt} (Courant={self.courant:.4f})")

        energy = torch.stack(e_list, dim=-1) if e_list else None
        return SimResult(
            ascans=ascans,
            phasors=phasors,
            frames=frames,
            energy=energy,
            energy_t=torch.tensor(e_t) if e_t else None,
            peak_energy=float(energy.max()) if energy is not None else 0.0,
            meta=dict(nt=nt, dt=dt, dx=dx, courant=self.courant,
                      n_pml=self.n_pml, grid=nx, n_cycles=n_cycles,
                      n_net=self.n_net, downsample=self.downsample,
                      dft_every=dft_every, absorber_kind=cfg.ABSORBER_KIND,
                      absorber_order=self.absorber_order,
                      absorber_r_target=self.absorber_r_target),
        )


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------
def homogeneous_material(nu: float, batch: int = 1, n_total: int = cfg.N_FINE_TOTAL,
                         device=None, dtype=torch.float32
                         ) -> tuple[Tensor, Tensor, Tensor]:
    """Constant (lam, mu, rho) fields for one Poisson ratio."""
    lam0, mu0 = cfg.lame_from_nu(nu)
    ones = torch.ones(batch, n_total, n_total, device=device, dtype=dtype)
    return lam0 * ones, mu0 * ones, cfg.RHO0 * ones


def material_with_voids(chi: Tensor, nu: float) -> tuple[Tensor, Tensor, Tensor]:
    """
    (lam, mu, rho) from a soft indicator on the padded fine grid.

    Delegates to geometry.sdf.material_fields so that there is exactly one
    definition of what a void is, shared with the physics loss.
    """
    from ..geometry.sdf import material_fields
    lam0, mu0 = cfg.lame_from_nu(nu)
    return material_fields(chi, lam0, mu0, cfg.RHO0)
