"""
The solver sanity checks of §3.7, as code (§11.2 steps 1-5), plus the five the
architecture review added.

These are the checks the document says never to cut, and the reason is not
diligence for its own sake: every one of them fails *silently* in the sense that a
broken solver still produces plausible-looking wavefields.  A sign error in one
staggered derivative gives a solver that propagates at the wrong speed; a badly
graded absorber gives one that reflects 5% off the boundary, which the network then
faithfully learns as part of the operator, and the inversion then interprets as a
second defect.  Nothing downstream can distinguish a wrong solver from a hard
problem, so the checks have to happen here.

Checks 1-5 are internal consistency: energy, arrival times, absorbed energy, a
Rayleigh amplitude slope, grid convergence.  Every one of them can pass while the
solver solves the wrong problem, which is what the review objected to, and the
absorber is the worked example -- check 3 says 1e-4 of the energy is left in the
domain at t_end while check 9 says the layer was corrupting the ring field by 22%,
because energy that leaks *out* and energy that reflects *back* are different
failures.  So checks 6-9 compare against things outside this file: the analytic
Green's function for a point force (6), the Pao-Mow series for a traction-free
cavity (7), a separated interface-width and grid study (8), and the same
acquisition in a padded open domain (9).  Check 10 points the other way -- it takes
the solver as truth and measures what the *training loss* reads on it, which is the
only way to learn the floor of a regulariser whose discretisation is not the
solver's.

Each function returns a `CheckResult` carrying the number, the gate it was compared
against and a one-line explanation.  `run_all()` prints the table that goes in
notebook 01.

Cost note: checks 4, 5, 8 and 9 need grids the production configuration does not use
-- check 4 because Rayleigh scattering requires kR << 1 *and* R >> dx at the same
time, checks 5 and 8 because a convergence test needs a finer grid to compare
against, check 9 because an open-domain reference needs a much larger one.  All four
are marked slow and want a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .. import config as cfg
from .. import losses as LOSS
from ..geometry.sdf import Circle, grid_coords, material_fields, soft_indicator
from . import cavity as CAV
from . import harmonic as H
from .fdtd_elastic import ElasticFDTD2D, homogeneous_material, tone_burst


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    gate: float
    units: str = ""
    detail: str = ""
    extras: dict = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (f"[{mark}] {self.name:34s} {self.value:12.4e} {self.units:<10s} "
                f"(gate {self.gate:.1e})  {self.detail}")


# The physical width of the void interface, aliased here because every check in this
# file that builds a void needs it and because it is a *length*: the network sees chi
# on the network grid and the solver builds its material on the fine grid, and both
# must smooth the void over the same physical distance or the label describes a
# slightly different void from the one the input describes.  Holding this fixed is
# also what makes a grid-refinement study mean anything (`check_interface_width`).
EPS_LEN_PHYS: float = cfg.EPS_INTERFACE_PHYS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _circle_chi(radius: float, *, n_total: int, n_pml: int, dx: float,
                eps_len: float = EPS_LEN_PHYS,
                centre: tuple[float, float] | None = None,
                l_domain: float = cfg.L_DOMAIN, device=None,
                dtype=torch.float32) -> Tensor:
    """Soft indicator [1, n_total, n_total] for one circular void."""
    yy, xx = grid_coords(n_total, dx, offset=n_pml, device=device, dtype=dtype)
    if centre is None:
        centre = (0.5 * l_domain, 0.5 * l_domain)
    theta = torch.tensor([[centre[0], centre[1], radius]], device=device, dtype=dtype)
    phi = Circle().sdf(theta, yy, xx)
    return soft_indicator(phi, eps_len)


def _refine_index(i_coarse: int, refine: int, n_pml_coarse: int = cfg.N_PML_FINE
                  ) -> int:
    """
    The index on an `refine`-times finer grid whose cell centre coincides exactly
    with cell centre `i_coarse` of the production grid.

    Cell centres sit at (i + 1/2 - P) dx.  Requiring the two to coincide gives

        i_f + 1/2 - P_f = refine (i_c + 1/2 - P_c),   P_f = refine P_c
        =>  i_f = refine i_c + (refine - 1)/2

    which is an integer only for *odd* refine.  This is why the convergence check
    refines by 3 and not by 2: with an even factor the source and receivers land
    half a fine cell away from where they sit on the coarse grid, and that offset
    is a phase error of k_s dx/4 ~ 0.09 rad at the top of the band -- around 1.4%,
    comparable to the discretisation error the test is trying to measure.  The
    test would then be measuring its own setup.
    """
    assert refine % 2 == 1, f"refine must be odd for exact centre alignment, got {refine}"
    return refine * i_coarse + (refine - 1) // 2


def _refine_face_index(i_coarse: int, refine: int) -> int:
    """
    The same thing for a quantity living on the *upper face* of cell `i_coarse`.

    The point force is added to `vy`, which sits at (i + 1 - P) dx rather than
    (i + 1/2 - P) dx, so requiring the two to coincide gives

        i_f + 1 - P_f = refine (i_c + 1 - P_c),  P_f = refine P_c
        =>  i_f = refine i_c + refine - 1

    with no parity condition: a face refines exactly for any factor.  Worth a
    separate function because using `_refine_index` here would move the source half a
    fine cell, and 0.26 rad of shear phase at f_max is larger than the effect any of
    these checks is measuring.  `check_interface_width` asserts the resulting
    physical position against the coarse one to 1e-12.
    """
    return refine * i_coarse + refine - 1


def _receiver_ring(n_net: int, inset: int = cfg.RING_INSET_NET
                   ) -> list[tuple[int, int]]:
    return cfg.ring_positions(cfg.N_RECV_PER_SIDE, n_grid=n_net, inset=inset)


def _ring_xy(recv_net: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """
    Physical `(x, y)` of a list of network cells, in the frame the analytic
    reference uses.

    Identical to `cfg.receiver_position` for the production ring, but written in
    terms of the cells actually passed to `run` so that a check using a different
    ring cannot silently compare against the production positions.
    """
    return [((rx + 0.5) * cfg.DX_NET, (ry + 0.5) * cfg.DX_NET) for ry, rx in recv_net]


def _force_xy(fy: int, fx: int) -> tuple[float, float]:
    """
    Physical `(x, y)` of the point force injected at padded-fine index `(fy, fx)`.

    The `+1.0` rather than `+0.5` in y is the y-face offset of `vy`; see
    `cfg.source_force_position`, which this reproduces for the production sources
    (asserted in `check_green_incident`) and generalises to sources that are not
    on the acquisition ring.
    """
    return ((fx + 0.5 - cfg.N_PML_FINE) * cfg.DX_FINE,
            (fy + 1.0 - cfg.N_PML_FINE) * cfg.DX_FINE)


def _rel_l2(a: Tensor, b: Tensor) -> float:
    """||a - b|| / ||b||, over everything."""
    return float((a - b).abs().pow(2).sum().sqrt()
                 / b.abs().pow(2).sum().sqrt().clamp_min(1e-30))


# ---------------------------------------------------------------------------
# Check 1 -- energy conservation with the absorber switched off
# ---------------------------------------------------------------------------
def check_energy_conservation(nu: float = 1.0 / 3.0, *, device=None,
                              gate: float = cfg.GATE_ENERGY_DRIFT) -> CheckResult:
    """
    Homogeneous medium, no absorber, source at the centre, stopped before the
    wave reaches the boundary.  Gate: energy *drift* below GATE_ENERGY_DRIFT.

    Two subtleties, both of which have bitten people writing this exact test.

    First, the domain is doubled to L = 16 lambda_p.  The production domain is
    8 lambda_p, so a centred source reaches the wall in 4 T_p -- before the
    5-cycle burst has even finished radiating.  There is then no window in which
    energy *should* be constant, and the test measures nothing.

    Second, drift is not the same as oscillation.  The leapfrog is symplectic and
    conserves a discrete energy exactly, but the discrete energy it conserves pairs
    v^{n+1/2} with the *average* of sigma^n and sigma^{n+1}; the naive expression
    used here pairs v^{n+1/2} with sigma^{n+1}, which differs by O(dt) and
    therefore oscillates at the time-step frequency with a bounded amplitude that
    never grows.  A symplectic integrator can show a large bounded oscillation and
    still be perfectly stable, whereas any secular drift means energy is being
    created or destroyed.  So the gate is applied to the difference of window means
    (drift) and the oscillation is reported alongside, not gated.
    """
    n_fine = 2 * cfg.N_FINE                     # L = 16 lambda_p
    l_dom = 2.0 * cfg.L_DOMAIN
    lam, mu, rho = homogeneous_material(nu, batch=1, n_total=n_fine, device=device)
    sim = ElasticFDTD2D(lam, mu, rho, dx=cfg.DX_FINE, dt=cfg.DT, n_pml=0,
                        downsample=cfg.DOWNSAMPLE)

    # stop 1 T_p before the P wave reaches the nearest wall
    t_stop = 0.5 * l_dom / cfg.CP - 1.0
    nt = int(t_stop / cfg.DT)
    src = [(n_fine // 2, n_fine // 2)]
    res = sim.run(src, nt=nt, energy_every=8)

    t = res.energy_t
    e = res.energy[0]
    # measurement window: after the burst has finished radiating
    mask = t > cfg.BURST_DURATION + 0.5
    assert bool(mask.any()), "no post-burst window; increase t_stop"
    tw, ew = t[mask], e[mask]
    q = max(2, ew.numel() // 4)
    drift = float((ew[-q:].mean() - ew[:q].mean()).abs() / ew.mean())
    osc = float((ew.max() - ew.min()) / ew.mean())

    return CheckResult(
        name="1. energy drift (no absorber)",
        passed=drift < gate, value=drift, gate=gate, units="rel",
        detail=f"nt={nt}, window t in [{float(tw[0]):.1f}, {float(tw[-1]):.1f}] T_p, "
               f"bounded oscillation {osc:.2%}",
        extras=dict(t=t, energy=e, oscillation=osc, nt=nt, l_domain=l_dom),
    )


# ---------------------------------------------------------------------------
# The scheme's own group velocity (check 2's reference)
# ---------------------------------------------------------------------------
def _stencil_wavenumber(theta: Tensor) -> Tensor:
    """K dx for the 4th-order staggered first difference, theta = k dx.

    Applying [c1 (f(x+dx/2) - f(x-dx/2)) + c2 (f(x+3dx/2) - f(x-3dx/2))]/dx with
    (c1, c2) = (9/8, -1/24) to e^{ikx} gives i K with
    K dx = 2 [c1 sin(theta/2) + c2 sin(3 theta/2)], whose expansion is
    K dx = theta - 0.0046875 theta^5 + O(theta^7): fourth-order accurate, and always
    an *under*-estimate of theta, so the discrete wave is spatially slow.
    """
    return 2.0 * (1.125 * torch.sin(0.5 * theta)
                  - (1.0 / 24.0) * torch.sin(1.5 * theta))


def _numerical_wavenumber(omega: Tensor, c: float, n_hat: tuple[float, float], *,
                          dx: float = cfg.DX_FINE, dt: float = cfg.DT,
                          iters: int = 60) -> Tensor:
    """k(omega) for this scheme along the direction n_hat, by bisection.

    The staggered leapfrog dispersion relation in 2-D is

        (2/dt) sin(omega dt / 2) = (c/dx) |K(k n_y dx), K(k n_x dx)|

    which reduces to omega = c k as dx, dt -> 0.  Expanding to leading order gives
    v_g/c = 1 + nu^2 theta^2 / 8 - 0.046875 theta^4 (nu = c dt/dx): the second-order
    time stepping makes the wave *fast*, the fourth-order space stencil makes it
    slow, and at this grid's ppw the time term wins below about 1.7 f_c.  The full
    relation is solved rather than the expansion because the burst carries content
    out to 2 f_c, where the two terms are the same size for S.
    """
    lhs = (2.0 / dt) * torch.sin(0.5 * omega * dt)
    ny, nx = abs(n_hat[0]), abs(n_hat[1])
    k_lo = torch.zeros_like(lhs)
    k_hi = torch.full_like(lhs, 1.7 / dx)        # below the stencil's turning point
    for _ in range(iters):
        k_mid = 0.5 * (k_lo + k_hi)
        rhs = (c / dx) * torch.hypot(_stencil_wavenumber(k_mid * ny * dx),
                                     _stencil_wavenumber(k_mid * nx * dx))
        below = rhs < lhs
        k_lo = torch.where(below, k_mid, k_lo)
        k_hi = torch.where(below, k_hi, k_mid)
    return 0.5 * (k_lo + k_hi)


def _dispersive_arrival(d: float, c: float, n_hat: tuple[float, float], *,
                        n_cycles: int, nt: int = cfg.NT, dt: float = cfg.DT,
                        fc: float = cfg.FC) -> float:
    """Envelope-peak time of the burst after travelling d through *this scheme*.

    Propagates the actual `tone_burst` spectrum by exp(-i k(omega) d) with k from
    `_numerical_wavenumber` and returns the peak of the resulting analytic envelope.
    Nothing is fitted and no recorded trace is read: the stencil coefficients, dt,
    dx and the burst are all fixed before the solver runs, so this is a prediction,
    not a calibration.

    Only the dispersion relation enters -- no Green's function.  That is deliberate:
    it keeps this check about wave speeds and leaves the amplitude and phase of the
    2-D response to check 6, which compares against the Hankel function absolutely.
    The omission is bounded, not assumed: propagating the same burst with the exact
    H_0^(2) instead of a plane wave moves this prediction by about 0.1 step over the
    receivers in use, and the 2-D coda alone (exact k, exact Hankel) accounts for only
    0.05 steps of arrival error, which is why the coda cannot be the explanation for a
    2-step discrepancy.  0.1 step is also roughly the sensitivity of the parabolic
    refinement below to where the band is truncated, so quoting it more precisely
    would be false precision.
    """
    t_src = torch.arange(nt, dtype=torch.float64) * dt
    s_hat = torch.fft.rfft(tone_burst(t_src, fc, n_cycles))
    omega = 2.0 * math.pi * torch.fft.rfftfreq(nt, d=dt).to(torch.float64)
    keep = (omega > 0.0) & (omega < 2.0 * math.pi * 3.0 * fc)
    k = _numerical_wavenumber(omega[keep], c, n_hat, dt=dt)
    v = torch.zeros_like(s_hat)
    v[keep] = s_hat[keep] * torch.exp(-1j * k * d)
    # analytic signal from the one-sided spectrum: double the positive bins
    z = torch.zeros(nt, dtype=v.dtype)
    z[: v.shape[0]] = 2.0 * v
    z[0] = v[0]
    envelope = torch.fft.ifft(z).abs()
    k0 = int(envelope.argmax())
    pos = float(k0)
    if 0 < k0 < nt - 1:
        y0, y1, y2 = (float(envelope[k0 - 1]), float(envelope[k0]),
                      float(envelope[k0 + 1]))
        den = y0 - 2.0 * y1 + y2
        if den < 0.0:
            pos += min(0.5, max(-0.5, 0.5 * (y0 - y2) / den))
    return pos * dt


# ---------------------------------------------------------------------------
# Check 2 -- P and S arrival times
# ---------------------------------------------------------------------------
def check_arrival_times(nu: float = 1.0 / 3.0, *, device=None,
                        n_cycles: int = 2,
                        gate: float = cfg.GATE_ARRIVAL_STEPS,
                        pattern_min: float = 0.3,
                        far_field_bursts: float = 2.5) -> CheckResult:
    """
    Homogeneous medium with the absorber on.  Gate: measured minus predicted arrival
    within GATE_ARRIVAL_STEPS time steps, for both P and S, where "predicted" is the
    arrival *this discretisation* produces (see `_dispersive_arrival`); the error
    against the continuum d/c + N_c/(2 f_c) is reported alongside, ungated.

    Measured on the *envelope peak*, not the leading edge.  The envelope peak of a
    Hann-windowed burst arrives at d/c + N_c/(2 f_c) -- the group delay is the
    centre of the window -- which is an exact analytic prediction.  A 5%-of-peak
    threshold crossing has no closed form, so comparing it to d/c would be
    comparing two different quantities and the "error" would be dominated by the
    burst shape rather than by the solver.  The peak is located to sub-sample
    precision by fitting a parabola through the three envelope samples around the
    argmax: the sample spacing is dt, so an integer argmax carries a +-0.5-step
    quantisation error, which is half the gate and was the largest single term in
    what this check used to report for the receivers it *could* measure.

    The envelope is `H.vector_envelope`: the analytic signal of each signed velocity
    component, then the norm.  This check used to take the analytic signal of
    `sqrt(vx^2 + vy^2)`, which is not an envelope -- rectifying first moves the
    spectrum to DC and 2 f_c -- and that convention alone misplaces the peak by 16
    to 18 steps against a synthetic burst with exactly known arrivals, against a
    gate of 1.0.

    Three conditions decide whether a receiver can time a phase at all.  They are
    stated from the acquisition geometry and the known radiation pattern, before any
    trace is looked at, and receivers that fail them are excluded rather than fudged:

    1. **The phase has to be radiated towards the receiver.**  The source is a
       *vertical* point force (`fdtd_elastic.run`), whose far field is P polarised
       radially with amplitude proportional to |cos t| and S polarised transversely
       with |sin t|, where t is the angle between the force axis and the
       source-receiver line.  The eight receivers that share the source's grid row
       are at t = 90 degrees *exactly*, so |cos t| = 0 identically and no P wave is
       radiated to them.  Timing a P arrival there measures whatever else is in the
       window -- the S skirt, the near field, absorber leakage -- and it is the
       entire content of the 57.5-step "P arrival error" this check reported for
       several revisions.  It was never the wave speeds.  `pattern_min = 0.3` keeps
       receivers where the phase carries at least 9% of the energy it has at the
       pattern maximum, while the *other* phase is at most 0.95 of its own maximum;
       below that the leakage between the two is comparable to the phase being timed.

    2. **The two packets have to be separable.**  The P and S envelope peaks are
       separated by d(1/c_s - 1/c_p), which at nu = 1/3 is exactly d, since
       c_s/c_p = 1/2 there.  Resolving them as two peaks needs that gap to exceed
       the burst duration N_c/f_c, i.e. d > N_c lambda_p.  At N_c = 5 that is
       d > 5 lambda_p, which barely fits in an 8 lambda_p domain; at N_c = 2 it is
       d > 2 lambda_p, which every cross-domain receiver satisfies.  The travel
       times being checked belong to the solver, not to the burst, so shortening the
       burst is legitimate.

    3. **The prediction has to be in its far field.**  d/c + group delay treats the
       2-D Green's function as a pure delay, when in fact a point source in two
       dimensions has a trailing coda of duration comparable to d/c, which drags the
       envelope peak later when d/c is not long compared with the burst.
       `far_field_bursts = 2.5` requires the travel time of the phase to exceed 2.5
       burst durations.  The receivers this excludes are reported in `detail` and
       carried in `extras` rather than dropped silently.  Beyond that radius the coda
       is worth 0.05 steps and no longer matters (see `_dispersive_arrival`), so it is
       not what the surviving error is made of.

    What the surviving error *is* made of is two things, both of them properties of
    the discretisation rather than of the physics, and both removed by fixing the
    prediction rather than the gate.

    First, the geometry has to be the geometry the solver used.  The offsets come from
    `_force_xy` and `_ring_xy`, not from differences of grid indices: the point force
    is added to `vy`, which lives on a y-face, and `net_to_fine` lands on the
    lower-left of the four fine cells straddling a network centre, so the force sits
    dx_fine/2 to the -x side of the nominal network cell centre and exactly on it in
    y (`cfg.source_force_position`).  dx_fine/2 is 0.917 steps of P travel time and
    1.833 of S, so timing against index arithmetic put a direction-dependent bias of
    up to 1.8 steps on a 1.0-step gate -- visible as a P error that swung from 0.00 to
    -1.21 steps with |cos t| at fixed distance, and gone once the physical positions
    are used.  The receiver side needs no such correction: the A-scan is a 2x2 average
    of fine cell centres, whose centroid is exactly the network cell centre.

    Second, the gate is read against the arrival *this scheme* predicts rather than
    the continuum one.  `_dispersive_arrival` propagates the burst by
    exp(-i k(omega) d) with k from the staggered-leapfrog dispersion relation, a
    prediction fixed by the stencil coefficients, dt, dx and N_c before any solve
    happens -- no free parameter and no trace read, so this is not the gate being
    widened to fit the measurement.  It matters because the correction is larger than
    the gate: over the scored receivers it reaches 0.83 steps for P and 1.36 for S.
    Comparing an envelope peak with the *continuum* d/c on this grid would read 1.57
    steps and fail, and a check that cannot pass measures nothing.  That the model is
    the right one is visible in the residual: before the correction the P error tracks
    d and the direction, after it the P residual is flat at -0.26 to -0.40 steps
    across every distance and every angle, which is a leftover convention constant and
    not an accumulating propagation error.

    Both continuum-reference errors are reported in `extras` as
    `errs_p_ray`/`errs_s_ray` and in `detail`, because that is the number a reader who
    wants "does it travel at c_p" is asking for and it should not be buried.  The
    correction has one approximation left: the dispersion relation is the scalar one
    applied separately to each wave type, exact along the grid axes and only
    approximate off them, and the 0.4-step residual is the honest measure of that.
    """
    lam, mu, rho = homogeneous_material(nu, batch=1, n_total=cfg.N_FINE_TOTAL,
                                        device=device)
    sim = ElasticFDTD2D(lam, mu, rho)
    recv = _receiver_ring(sim.n_net)
    src_net = cfg.SOURCES_NET[0]
    src_fine = [cfg.net_to_fine(*src_net)]
    res = sim.run(src_fine, nt=cfg.NT, recv_yx=recv, n_cycles=n_cycles)

    cs = cfg.cs_over_cp(nu)
    group_delay = 0.5 * n_cycles / cfg.FC
    burst = n_cycles / cfg.FC
    env = H.vector_envelope(res.ascans)[0]                       # [R, nt]
    nt = env.shape[-1]

    # Physical offsets, not index differences.  The force is not at the nominal
    # network cell centre: `cfg.source_force_position` documents a dx_fine/2
    # displacement in x (the y one cancels against the staggered vy face), and
    # dx_fine/2 is 0.917 steps of P travel time and 1.833 of S -- at and above the
    # gate, so index arithmetic here is not a rounding detail.  The receiver side has
    # no such offset: the A-scan is a 2x2 average of fine cell centres, whose
    # centroid is exactly the network cell centre.  `_force_xy`/`_ring_xy` are the
    # same helpers checks 6 and 7 compare their analytic references against.
    src_x, src_y = _force_xy(*src_fine[0])
    ring = _ring_xy(recv)
    off_y = torch.tensor([ry - src_y for _, ry in ring], dtype=torch.float64)
    off_x = torch.tensor([rx - src_x for rx, _ in ring], dtype=torch.float64)
    d = torch.hypot(off_y, off_x).clamp_min(1e-30)

    # far-field radiation pattern of a vertical point force: P as |cos|, S as |sin|
    pattern_p = off_y.abs() / d
    pattern_s = off_x.abs() / d

    t_p_pred = d / cfg.CP + group_delay
    t_s_pred = d / cs + group_delay
    separation = d * (1.0 / cs - 1.0 / cfg.CP)
    resolved = separation > burst
    far_p = (d / cfg.CP) > far_field_bursts * burst
    far_s = (d / cs) > far_field_bursts * burst
    use_p = resolved & far_p & (pattern_p > pattern_min)
    use_s = resolved & far_s & (pattern_s > pattern_min)
    assert bool(use_p.any()) and bool(use_s.any()), (
        "no receiver can time both phases: check the ring, the source index and "
        "pattern_min / far_field_bursts")

    half = 0.5 * n_cycles / cfg.FC

    def _peak_error(row: Tensor, pred: float) -> float | None:
        """|measured - predicted| in steps, with a sub-sample parabolic peak."""
        lo = int(max(0, (pred - half) / cfg.DT))
        hi = int(min(nt, (pred + half) / cfg.DT))
        if hi - lo < 4:
            return None
        k = int(row[lo:hi].argmax()) + lo
        pos = k + 0.5                                # samples sit at (k + 1/2) dt
        if 0 < k < nt - 1:
            y0, y1, y2 = float(row[k - 1]), float(row[k]), float(row[k + 1])
            den = y0 - 2.0 * y1 + y2
            if den < 0.0:                            # a maximum, not a plateau
                pos += min(0.5, max(-0.5, 0.5 * (y0 - y2) / den))
        return abs(pos * cfg.DT - pred) / cfg.DT

    nan = float("nan")
    errs_p = [nan] * len(recv)               # against this scheme's own arrival
    errs_s = [nan] * len(recv)
    errs_p_ray = [nan] * len(recv)           # against the continuum d/c + group
    errs_s_ray = [nan] * len(recv)
    disp_p = [nan] * len(recv)               # the correction itself, in steps
    disp_s = [nan] * len(recv)
    near_p, near_s = [], []                  # excluded by the far-field test only
    for r in range(len(recv)):
        n_hat = (float(off_y[r] / d[r]), float(off_x[r] / d[r]))
        for c_, pred, ok, bag, bag_ray, bag_d, near in (
                (cfg.CP, t_p_pred[r], use_p[r], errs_p, errs_p_ray, disp_p, near_p),
                (cs, t_s_pred[r], use_s[r], errs_s, errs_s_ray, disp_s, near_s)):
            pred = float(pred)
            if not (bool(ok) or bool(resolved[r])):
                continue
            e_ray = _peak_error(env[r], pred)
            if e_ray is None:
                continue
            t_disp = _dispersive_arrival(float(d[r]), c_, n_hat,
                                         n_cycles=n_cycles)
            e = _peak_error(env[r], t_disp)
            if bool(ok):
                bag[r] = e if e is not None else nan
                bag_ray[r] = e_ray
                bag_d[r] = (t_disp - pred) / cfg.DT
            else:
                near.append(e_ray)

    got_p = [e for e in errs_p if e == e]
    got_s = [e for e in errs_s if e == e]
    ray_p = [e for e in errs_p_ray if e == e]
    ray_s = [e for e in errs_s_ray if e == e]
    assert got_p and got_s, (
        "no usable P/S window; the record is too short or the ring too tight")
    worst = max(max(got_p), max(got_s))
    med_p = float(torch.tensor(got_p).median())
    med_s = float(torch.tensor(got_s).median())
    excl = f"{int(len(recv) - use_p.sum())}/{len(recv)} P"
    return CheckResult(
        name="2. P/S arrival error",
        passed=worst < gate, value=worst, gate=gate, units="steps",
        detail=f"P worst {max(got_p):.2f} median {med_p:.2f} over {len(got_p)} recv, "
               f"S worst {max(got_s):.2f} median {med_s:.2f} over {len(got_s)}; "
               f"vs continuum d/c: P {max(ray_p):.2f} S {max(ray_s):.2f} "
               f"(scheme dispersion accounts for up to "
               f"{max(abs(x) for x in disp_p + disp_s if x == x):.2f}); "
               f"excluded {excl} and {int(len(recv) - use_s.sum())}/{len(recv)} S "
               f"(pattern < {pattern_min}, overlap, or inside "
               f"{far_field_bursts:g} burst lengths); near-field worst "
               f"{max(near_p + near_s, default=float('nan')):.2f}; c_s/c_p={cs:.3f}",
        extras=dict(errs_p=errs_p, errs_s=errs_s, errs_p_ray=errs_p_ray,
                    errs_s_ray=errs_s_ray, dispersion_p=disp_p,
                    dispersion_s=disp_s, distances=d.tolist(),
                    usable=resolved.tolist(), usable_p=use_p.tolist(),
                    usable_s=use_s.tolist(), pattern_p=pattern_p.tolist(),
                    pattern_s=pattern_s.tolist(), near_field_p=near_p,
                    near_field_s=near_s, median_p=med_p, median_s=med_s,
                    worst_ray=max(max(ray_p), max(ray_s)), ascans=res.ascans),
    )


# ---------------------------------------------------------------------------
# Check 3 -- absorber residual
# ---------------------------------------------------------------------------
def check_absorber(nu: float = 1.0 / 3.0, *, device=None,
                   gate: float = cfg.GATE_PML_RESIDUAL) -> CheckResult:
    """
    Homogeneous medium, absorber on, run the full T_end.  Gate: energy remaining
    inside the *physical* region, as a fraction of the peak, below
    GATE_PML_RESIDUAL.

    `_energy(core_only=True)` is what makes this meaningful: energy still being
    dissipated inside the absorbing layer is not a reflection, it is the absorber
    doing its job, and counting it would make a perfect absorber look like a
    failure.  What must vanish is energy that came *back*.

    Also reports the A-scan tail-energy fraction, because the frequency-domain
    labels are a finite-window DFT and anything still ringing at T_end wraps
    around onto t = 0.  A clean absorber and a clean DFT are the same requirement
    seen from two sides.
    """
    lam, mu, rho = homogeneous_material(nu, batch=1, n_total=cfg.N_FINE_TOTAL,
                                        device=device)
    sim = ElasticFDTD2D(lam, mu, rho)
    recv = _receiver_ring(sim.n_net)
    res = sim.run([cfg.net_to_fine(*cfg.SOURCES_NET[0])], nt=cfg.NT,
                  recv_yx=recv, energy_every=8)
    e = res.energy[0]
    residual = float(e[-1] / e.max())
    tail = float(H.tail_energy_fraction(res.ascans)[0])
    return CheckResult(
        name="3. absorber residual energy",
        passed=residual < gate, value=residual, gate=gate, units="rel peak",
        detail=f"{cfg.N_PML_FINE} cells = {cfg.N_PML_FINE*cfg.DX_FINE:.2f} lambda_p, "
               f"A-scan tail energy {tail:.2e}",
        extras=dict(t=res.energy_t, energy=e, tail_fraction=tail),
    )


# ---------------------------------------------------------------------------
# Check 4 -- Rayleigh scaling and mode conversion
# ---------------------------------------------------------------------------
def check_rayleigh_and_mode_conversion(
        nu: float = 1.0 / 3.0, *, device=None,
        radii_ls: tuple[float, ...] = (0.05, 0.06, 0.07, 0.08),
        refine: int = 8, l_domain: float = 3.0,
        slope_window: tuple[float, float] = (3.3, 4.7)) -> CheckResult:
    """
    Scattered energy versus void radius, in the long-wavelength limit.

    Theory (2D, long wavelength): a small void scatters through a monopole and a
    dipole term, both O((kR)^2) in *amplitude*, so the scattered energy collected
    on a fixed ring scales as R^4 at fixed frequency.  Gate: the fitted log-log
    slope of energy against R lies in `slope_window`.

    Why the gate is a window and not a tight number.  The check needs kR << 1 and
    R >> dx simultaneously, and a fixed grid cannot give both: at the production
    resolution the smallest resolvable void already has kR ~ 1.  The radii and the
    refinement therefore move together -- on this 8x-refined grid the radii used
    span kR ~ 0.31 to 0.50, small enough that the fitted slope recovers to ~3.7
    while the smallest void is still ~6.4 fine cells across.  (An earlier
    4x/kR~0.6-1.3 choice sat too far out of the long-wavelength limit: the R^4 law
    had already begun to saturate, and even the best-conditioned pair fitted below
    3.3.  The failure was in the sampling, not the solver -- checks 6 and 7 certify
    the scattered amplitude against analytic references with no fitted parameters.)
    Even so this is a *scaling-consistency* test rather than a precision test -- it
    catches a slope of 2 or 6, which is what broken interface averaging or a
    mis-normalised soft indicator produces, and it is not sensitive enough to
    certify the coefficient.  Said plainly rather than dressed up.

    The interface width is 1 cell of *this* grid rather than the production
    EPS_LEN_PHYS.  Legitimate here because the quantity under test is the solver's
    scattering law, not the dataset's geometry convention, and necessary because
    these radii are 0.05-0.08 lambda_s -- an eighth of the smallest production void
    -- so any fixed physical width would be a sizeable fraction of the void itself.

    All radii plus the incident field run as one batch, so this is one solve.

    Also reports mode conversion: scattered energy arriving in the S-wave time
    window relative to the P window.  Mode conversion at a traction-free void is
    the physical content that makes this problem elastic rather than acoustic; if
    it is absent, the two equations are not coupled and the whole premise of the
    project is broken.
    """
    dx = cfg.DX_FINE / refine
    n_fine = int(round(l_domain / dx))
    n_pml = int(round(cfg.N_PML_FINE * cfg.DX_FINE / dx))
    n_total = n_fine + 2 * n_pml
    down = refine
    while n_fine % down:                          # keep the crop divisible
        down -= 1
    dt = cfg.CFL_NUMBER * dx / cfg.CP
    lam_s = cfg.cs_over_cp(nu)
    # long enough for the slowest useful path: across the domain at c_s, plus the
    # burst, plus a margin
    t_end = math.sqrt(2.0) * l_domain / lam_s + cfg.BURST_DURATION + 1.0
    nt = int(math.ceil(t_end / dt))

    radii = [r * lam_s for r in radii_ls]
    lam0, mu0 = cfg.lame_from_nu(nu)

    chis = [torch.zeros(1, n_total, n_total, device=device)]      # incident first
    for r in radii:
        chis.append(_circle_chi(r, n_total=n_total, n_pml=n_pml, dx=dx,
                                eps_len=dx, l_domain=l_domain, device=device))
    chi = torch.cat(chis, dim=0)
    lam, mu, rho = material_fields(chi, lam0, mu0, cfg.RHO0)

    sim = ElasticFDTD2D(lam, mu, rho, dx=dx, dt=dt, n_pml=n_pml, downsample=down)
    recv = _receiver_ring(sim.n_net)
    src_net = cfg.ring_positions(cfg.N_SRC_PER_SIDE, n_grid=sim.n_net,
                                 inset=cfg.RING_INSET_NET)[0]
    src_fine = [(src_net[0] * down + n_pml, src_net[1] * down + n_pml)] * chi.shape[0]
    res = sim.run(src_fine, nt=nt, recv_yx=recv)

    inc = res.ascans[0:1]
    scat = res.ascans[1:] - inc
    energy = scat.pow(2).sum(dim=(1, 2, 3))                       # [n_radii]

    lr = torch.log(torch.tensor(radii, dtype=torch.float64))
    le = torch.log(energy.detach().cpu().to(torch.float64).clamp_min(1e-300))
    A = torch.stack([lr, torch.ones_like(lr)], dim=1)
    slope = float(torch.linalg.lstsq(A, le.unsqueeze(1)).solution[0, 0])
    local = float((le[1] - le[0]) / (lr[1] - lr[0]))

    # mode conversion: scattered energy in the S window versus the P window
    #
    # Index arithmetic is deliberate here, unlike in check 2.  These distances only
    # centre energy windows of half-width 0.5 * BURST_DURATION = 2.5 time units, and
    # the sub-cell force offset check 2 has to respect (dx_fine/2, worth 0.016 units
    # of P travel and 0.031 of S) is under 1.3% of that half-width.  This check also
    # runs on a refined grid of its own, so `dx_net = dx * down` rather than
    # cfg.DX_NET, and it scores a *slope*, which a common offset cannot move.
    dx_net = dx * down
    tgrid = (torch.arange(nt, device=res.ascans.device, dtype=torch.float64) + 0.5) * dt
    d = torch.tensor(
        [math.hypot((ry - src_net[0]) * dx_net, (rx - src_net[1]) * dx_net)
         for ry, rx in recv], dtype=torch.float64, device=tgrid.device)
    half = 0.5 * cfg.BURST_DURATION
    biggest = scat[-1].pow(2).sum(dim=1).to(torch.float64)        # [R, nt]
    e_p = torch.zeros((), dtype=torch.float64, device=tgrid.device)
    e_s = torch.zeros((), dtype=torch.float64, device=tgrid.device)
    for r in range(biggest.shape[0]):
        wp = (tgrid > d[r] / cfg.CP - half) & (tgrid < d[r] / cfg.CP + half)
        ws = (tgrid > d[r] / lam_s - half) & (tgrid < d[r] / lam_s + half)
        e_p = e_p + biggest[r][wp].sum()
        e_s = e_s + biggest[r][ws].sum()
    conv = float(e_s / e_p.clamp_min(1e-30))

    lo, hi = slope_window
    kr = [2.0 * math.pi * r / lam_s for r in radii]
    return CheckResult(
        name="4. Rayleigh slope d(logE)/d(logR)",
        passed=(lo < slope < hi) and conv > 0.01, value=slope,
        gate=4.0, units="",
        detail=f"expect 4; smallest-pair slope {local:.2f}; kR in "
               f"[{min(kr):.2f}, {max(kr):.2f}]; S/P scattered energy {conv:.1%} "
               f"(needs >1%); grid {n_total}^2 x {nt} steps",
        extras=dict(radii=radii, energy=energy.tolist(), slope=slope,
                    local_slope=local, kR=kr, mode_conversion=conv,
                    grid=n_total, dx=dx, nt=nt),
    )


# ---------------------------------------------------------------------------
# Check 5 -- grid convergence, i.e. the label noise floor
# ---------------------------------------------------------------------------
def check_grid_convergence(nu: float = 1.0 / 3.0, *, device=None,
                           radius_ls: float = 0.8, refine: int = 3,
                           gate: float = cfg.GATE_GRID_CONVERGENCE) -> CheckResult:
    """
    The same defect solved at dx and dx/refine, compared on receiver displacement
    phasors.  Gate: relative L2 below GATE_GRID_CONVERGENCE.

    This number is not a solver check so much as a *floor*: it is the accuracy of
    the training labels themselves, and no surrogate can be asked to beat it.  If
    the surrogate reaches 4% relative error and this check says 2%, the surrogate
    is within a factor of two of the data it was given, and the honest conclusion
    is that the labels have become the limiting factor -- a different research task
    from making the network bigger.  Reporting a surrogate error without reporting
    this floor is the most common way to overstate an operator-learning result.

    `refine` must be odd; see `_refine_index` for why (an even factor moves every
    source and receiver half a fine cell, and the resulting phase error is the same
    size as the effect being measured).  The refined run divides dx and dt by the
    same factor, holding the Courant number fixed, so the comparison isolates
    spatial discretisation; T_end is unchanged, so both runs' phasors are the same
    functionals of the same time window.
    """
    lam_s = cfg.cs_over_cp(nu)
    radius = radius_ls * lam_s
    lam0, mu0 = cfg.lame_from_nu(nu)
    om = H.omegas_tensor(device)
    src_c = cfg.SOURCES_NET[0]

    def solve(r: int) -> Tensor:
        dx = cfg.DX_FINE / r
        n_pml = cfg.N_PML_FINE * r
        n_total = cfg.N_FINE * r + 2 * n_pml
        dt = cfg.DT / r
        nt = cfg.NT * r
        chi = _circle_chi(radius, n_total=n_total, n_pml=n_pml, dx=dx, device=device)
        lam, mu, rho = material_fields(chi, lam0, mu0, cfg.RHO0)
        sim = ElasticFDTD2D(lam, mu, rho, dx=dx, dt=dt, n_pml=n_pml,
                            downsample=cfg.DOWNSAMPLE * r)
        assert sim.n_net == cfg.N_NET, f"refine={r} gives n_net={sim.n_net}"
        iy, ix = cfg.net_to_fine(*src_c)                     # production indices
        if r > 1:
            # The +y force is injected on the vy face, not at the y-cell centre.
            # Preserve its physical location exactly under refinement.
            iy, ix = _refine_face_index(iy, r), _refine_index(ix, r)
        res = sim.run([(iy, ix)], nt=nt, recv_yx=_receiver_ring(sim.n_net))
        return H.displacement_from_ascans(res.ascans, omegas=om, dt=dt, nt=nt)

    coarse = solve(1)
    fine = solve(refine)
    err = _rel_l2(coarse, fine)

    # per-frequency, because the error is not flat in frequency: it grows like
    # (k dx)^4, so the top of the band is always the worst case
    per_f = [_rel_l2(coarse[..., m], fine[..., m]) for m in range(coarse.shape[-1])]
    m_worst = max(range(len(per_f)), key=per_f.__getitem__)
    return CheckResult(
        name="5. grid convergence (label floor)",
        passed=err < gate, value=err, gate=gate, units="rel-L2",
        detail=f"{cfg.N_FINE}^2 vs {cfg.N_FINE*refine}^2; worst frequency "
               f"{per_f[m_worst]:.2%} at f={cfg.FREQS[m_worst]:.3f} f_c",
        extras=dict(per_frequency=per_f, radius=radius, refine=refine),
    )


# ---------------------------------------------------------------------------
# Check 6 -- the incident field against the analytic Green's tensor
# ---------------------------------------------------------------------------
def check_green_incident(nu: float = 1.0 / 3.0, *, device=None, src_idx: int = 0,
                         gate: float = cfg.GATE_GREEN_REL_L2,
                         gate_phase: float = cfg.GATE_CAVITY_PHASE_RAD
                         ) -> CheckResult:
    """
    Homogeneous medium, receiver ring, compared against `cavity.green_displacement`
    with *no free amplitude and no free phase*.

    Checks 1-5 are all internal: they compare the solver against itself on a finer
    grid, or against a scaling law fitted to its own output.  Every one of them
    passes for a solver that propagates at the wrong speed or carries the wrong
    source amplitude, as long as it does so consistently.  This is the first check
    with an external answer, and it establishes the two things the cavity reference
    then depends on: that the deconvolved A-scan really is the elastodynamic
    Green's tensor for a unit `+y` force, and that the Fourier and outgoing-wave
    conventions of `harmonic.py` and `cavity.py` agree.  That is why the comparison
    is absolute -- a fitted scale would forgive exactly the errors it exists to
    find -- and why `rel_l2_conj` is reported: if the conventions were backwards,
    that number would be the small one.

    TWO SOLVES, BECAUSE THE PRODUCTION GEOMETRY MEASURES TWO THINGS AT ONCE.
    The acquisition ring puts sources `RING_INSET_NET` = 3 network cells from the
    absorber, so a production source radiates its *near* field straight into a
    sponge that is 0.25 lambda_s away, at every angle including grazing, where a
    damping layer is at its worst.  Measured here: 1.9% relative L2 against free
    space with the production source, against 2.6% with the same solver and the
    same absorber and the source moved to the domain centre.  The difference is
    geometry, not solver, so the gate is applied to the centre-source solve and the
    production number is reported beside it.

    Those two numbers used to be 23% and 3.6%, and the collapse is the redesigned
    absorber (`config.N_ABSORBER_FINE`, 60 fine cells at p = 4 rather than 30 at p = 2).
    Note that the ordering flipped: with a thin layer the production source was six times
    worse than a centred one, because near-field grazing incidence is where a sponge
    fails first; with a thick one the production source reads *better*, since the
    centre-source comparison is then dominated by the longer propagation path and its
    numerical dispersion rather than by the boundary.  Both are now the same handful of
    percent, so the geometric distinction this check was built to expose has stopped
    mattering -- which is the outcome, not a reason to stop measuring it.

    That production number is not the label error either, and the reason is worth
    writing down.  The absorber artefact is linear in the field, and `d_obs` builds the
    scattered field as a difference of two solves that share the source, the domain and
    the absorber (`(u_tot + R u_tot) - (u_inc + R u_inc) = u_scat + R u_scat`), so the
    source-side artefact cancels *exactly* and what survives is the sponge acting
    on the scattered field alone.  Voids sit at least BOUNDARY_KEEPOUT_LS from the
    edge, and check 9 measures that surviving term directly against an open domain:
    1.8%, the same order as the grid-convergence floor of check 5, and it belongs in the
    same sentence whenever the label accuracy is quoted.  What does *not* cancel
    from the network's inputs is `incident_scale`, which normalises by an incident
    field carrying the full artefact -- but that is one deterministic scalar per
    (source, frequency, nu) and is applied to inputs and targets alike.

    Three known discrepancies are modelled rather than tolerated, because each is
    the size of the gate:

    * the source sits on a y-face, half a fine cell above the cell centre
      (`cfg.source_force_position`, 0.26 rad at f_max);
    * an A-scan is the field through a face-average and a 2x2 block average, a 5%
      amplitude reduction at f_max (`cavity.sample_like_solver`);
    * the discrete delta is one cell wide, a 0.3% form factor, which is *not*
      modelled and is part of what the residual contains.

    What is left is numerical dispersion: at 14.5 points per shear wavelength the
    staggered symbol is 0.99969 and the leapfrog's temporal factor 0.99914, so the
    numerical shear wavenumber is 0.054% low and the phase error reaches about
    0.02 rad at r = 2.  With no boundary at all this check reads 1.0%, rising like
    f^2 across the band, which is the floor it can reach.  A residual that is flat
    in frequency, or that appears as a single global phase, is not dispersion and
    should be read as a bug -- that is how the half-time-step source offset
    documented in `fdtd_elastic.source_spectrum` was found.
    """
    lam, mu, rho = homogeneous_material(nu, batch=1, n_total=cfg.N_FINE_TOTAL,
                                        device=device)
    sim = ElasticFDTD2D(lam, mu, rho)
    recv_net = _receiver_ring(sim.n_net)
    recv_xy = _ring_xy(recv_net)
    om = H.omegas_tensor(device)

    centre_net = (cfg.N_NET // 2, cfg.N_NET // 2)
    reports = {}
    for tag, src_net in (("centre", centre_net), ("production", cfg.SOURCES_NET[src_idx])):
        fy, fx = cfg.net_to_fine(*src_net)
        src_xy = _force_xy(fy, fx)
        if tag == "production":                 # one definition of the y-face offset
            assert src_xy == cfg.source_force_position(src_idx), (
                src_xy, cfg.source_force_position(src_idx))
        res = sim.run([(fy, fx)], nt=cfg.NT, recv_yx=recv_net)
        u = H.displacement_from_ascans(res.ascans, omegas=om, dt=cfg.DT, nt=cfg.NT)
        meas = u[0].detach().cpu().numpy().astype(complex)          # [R, 2, M]

        def field(points, _s=src_xy) -> np.ndarray:
            return CAV.green_displacement(_s, points, cfg.FREQS, nu=nu)

        ref = CAV.sample_like_solver(field, recv_xy)
        rep = CAV.cavity_accuracy_report(meas, ref, freqs=cfg.FREQS)
        # the same comparison without the sampling operator, to show how much of
        # the agreement is the modelling and how much is the solver
        rep["rel_l2_point_sampled"] = CAV.cavity_accuracy_report(
            meas, field(np.asarray(recv_xy, float)))["rel_l2"]
        rep["src_xy"] = src_xy
        reports[tag] = rep

    rep = reports["centre"]
    err, phase = rep["rel_l2"], rep["phase_rad"]
    m_worst = int(np.argmax(rep["rel_l2_per_freq"]))
    return CheckResult(
        name="6. Green's function (absolute)",
        passed=err < gate and phase < gate_phase, value=err, gate=gate,
        units="rel-L2",
        detail=f"centre source: phase {phase:.3f} rad (gate {gate_phase:.2f}), amp "
               f"{rep['amp_ratio']:.4f}, worst f "
               f"{rep['rel_l2_per_freq'][m_worst]:.1%} at "
               f"{cfg.FREQS[m_worst]:.3f} f_c, conjugated "
               f"{rep['rel_l2_conj']:.1%}; production source (ring inset "
               f"{cfg.RING_INSET_NET}) {reports['production']['rel_l2']:.1%}, which "
               f"cancels from the scattered field",
        extras=dict(report=rep, report_production=reports["production"]),
    )


# ---------------------------------------------------------------------------
# Check 7 -- the void against the analytic traction-free cavity
# ---------------------------------------------------------------------------
def _cavity_scattered(radii: list[float], *, nu: float, src_idx: int, device,
                      eps_len: float | list[float], void_density_scale: float,
                      dx: float = cfg.DX_FINE, refine: int = 1
                      ) -> tuple[np.ndarray, tuple[float, float], int, int]:
    """
    Solver scattered phasors at the receiver ring, one row per radius, [nR, R, 2, M].

    The incident field is row 0 of the same batch, so it shares the source, the
    domain and the absorber with every total field and the subtraction removes the
    source-side sponge artefact exactly (see `check_green_incident`).

    `eps_len` may be a single width or one per radius; the latter is how
    `check_interface_width` sweeps the interface at fixed geometry in one solve.
    """
    n_pml = cfg.N_PML_FINE * refine
    n_total = cfg.N_FINE_TOTAL * refine
    dt, nt, down = cfg.DT / refine, cfg.NT * refine, cfg.DOWNSAMPLE * refine
    lam0, mu0 = cfg.lame_from_nu(nu)
    om = H.omegas_tensor(device)
    eps = [eps_len] * len(radii) if isinstance(eps_len, float) else list(eps_len)
    assert len(eps) == len(radii), (len(eps), len(radii))

    fy, fx = cfg.net_to_fine(*cfg.SOURCES_NET[src_idx])
    src_xy = _force_xy(fy, fx)
    if refine > 1:
        fy, fx = _refine_face_index(fy, refine), _refine_index(fx, refine)
        moved = ((fx + 0.5 - n_pml) * dx, (fy + 1.0 - n_pml) * dx)
        assert max(abs(a - b) for a, b in zip(moved, src_xy)) < 1e-12, (moved, src_xy)

    chis = [torch.zeros(1, n_total, n_total, device=device)]        # incident first
    for r, e in zip(radii, eps):
        chis.append(_circle_chi(r, n_total=n_total, n_pml=n_pml, dx=dx,
                                eps_len=e, device=device))
    chi = torch.cat(chis, dim=0)
    solid = (1.0 - chi).clamp_min(cfg.VOID_STIFFNESS_FLOOR)
    rho = cfg.RHO0 * (1.0 - (1.0 - void_density_scale) * chi)

    sim = ElasticFDTD2D(lam0 * solid, mu0 * solid, rho, dx=dx, dt=dt, n_pml=n_pml,
                        downsample=down)
    assert sim.n_net == cfg.N_NET, f"refine={refine} gives n_net={sim.n_net}"
    recv_net = _receiver_ring(sim.n_net)
    res = sim.run([(fy, fx)] * chi.shape[0], nt=nt, recv_yx=recv_net)
    u = H.displacement_from_ascans(res.ascans, omegas=om, dt=dt, nt=nt)
    u = u.detach().cpu().numpy().astype(complex)
    return u[1:] - u[0:1], src_xy, down, nt


def _cavity_reference(radius: float, src_xy, *, nu: float, dx: float = cfg.DX_FINE,
                      downsample: int = cfg.DOWNSAMPLE) -> np.ndarray:
    """Analytic traction-free cavity, through the solver's own sampling operator."""
    recv_xy = _ring_xy(_receiver_ring(cfg.N_NET))
    centre = (0.5 * cfg.L_DOMAIN, 0.5 * cfg.L_DOMAIN)
    return CAV.sample_like_solver(
        lambda p: CAV.cavity_field(src_xy, p, cfg.FREQS, radius=radius,
                                   centre=centre, nu=nu).scattered,
        recv_xy, dx_fine=dx, downsample=downsample)


def check_cavity_scattering(nu: float = 1.0 / 3.0, *, device=None,
                            radii_ls: tuple[float, ...] = (cfg.R_MIN_LS, 0.8,
                                                           cfg.R_MAX_LS),
                            src_idx: int = 0, eps_len: float | None = None,
                            void_density_scale: float | None = None,
                            gate: float = cfg.GATE_CAVITY_REL_L2,
                            gate_phase: float = cfg.GATE_CAVITY_PHASE_RAD
                            ) -> CheckResult:
    """
    The claim this check exists to test: *the labels are cavity scattering*.

    Everything downstream is stated in those terms -- the physics residual of §7.2,
    the Rayleigh scaling of check 4, the whole shape-recovery premise -- and until
    this was measured it was an assumption about a soft inclusion with a smoothed
    boundary.  Here the solver's scattered field at the receiver ring is compared
    against the analytic traction-free cavity (`cavity.cavity_field`) with no fitted
    amplitude and no fitted phase, at three radii spanning the specified range.

    Two things had to change before the comparison passed, and neither works without
    the other.  Measured at R = 1.2 lambda_s (relative L2 of the scattered receiver
    field, fitted amplitude ratio in brackets, nu = 1/3):

                                 eps = 1.5 net cells   eps = 0.375 fine cells
        rho_void = RHO0             87.4% (0.52)            76.7% (0.71)
        rho_void = 1e-2 RHO0        86.3% (0.36)             9.3% (0.99)

    A wide interface is not a cavity boundary, and a void that keeps its mass loads
    the boundary it is supposed to free -- see `config.EPS_INTERFACE_FINE_CELLS` and
    `config.VOID_DENSITY_SCALE`, which quote the mechanism and the stability limit.
    Note that the two changes are not additive: either one alone leaves the error near
    85%, and the amplitude ratio says why -- the three failing corners return between a
    third and three quarters of the right scattered amplitude, so they are not slightly
    wrong cavities, they are different scatterers.  Both rows are reproducible from here
    by passing `eps_len` and `void_density_scale`; they are not re-measured on every run
    because the point of a gate is to hold the *current* model, not to re-derive rejected
    ones.

    All radii and the incident field are one batched solve, so the source-side
    absorber artefact cancels exactly and what is left is the sponge acting on the
    scattered field, the illumination error of check 6, and the interface.
    `config.GATE_CAVITY_REL_L2` carries that budget, and carries the finding that the
    budget does *not* close: those three terms are ~3.4% in quadrature against 9.3%
    measured, and the remainder is the void model itself rather than any discretisation
    (check 8 is the evidence).  Roughly 9% is what it costs to represent a traction-free
    boundary as a soft light inclusion on a fixed grid.

    Note what the gate is *not*: a claim that the labels are accurate to 12%.  It is
    a claim that the forward model solves the problem the write-up says it solves, at
    the fidelity this grid allows.  The label noise floor is check 5, and the phase --
    the travel-time information the inversion actually uses -- comes out an order of
    magnitude inside its own gate.
    """
    eps = EPS_LEN_PHYS if eps_len is None else eps_len
    rho_s = (cfg.VOID_DENSITY_SCALE if void_density_scale is None
             else void_density_scale)
    lam_s = cfg.cs_over_cp(nu) / cfg.FC
    radii = [r * lam_s for r in radii_ls]

    scat, src_xy, _, _ = _cavity_scattered(
        radii, nu=nu, src_idx=src_idx, device=device, eps_len=eps,
        void_density_scale=rho_s)
    reports = [CAV.cavity_accuracy_report(scat[i],
                                          _cavity_reference(r, src_xy, nu=nu),
                                          freqs=cfg.FREQS)
               for i, r in enumerate(radii)]

    errs = [r["rel_l2"] for r in reports]
    phases = [abs(r["phase_rad"]) for r in reports]
    rep = reports[max(range(len(errs)), key=errs.__getitem__)]
    m_worst = int(np.argmax(rep["rel_l2_per_freq"]))
    per_r = ", ".join(f"{rl:.2f} l_s {e:.1%}" for rl, e in zip(radii_ls, errs))
    return CheckResult(
        name="7. void vs traction-free cavity",
        passed=max(errs) < gate and max(phases) < gate_phase, value=max(errs),
        gate=gate, units="rel-L2",
        detail=f"R = {per_r}; worst radius: amp {rep['amp_ratio']:.4f}, phase "
               f"{rep['phase_rad']:+.3f} rad (gate {gate_phase:.2f}), calibrated "
               f"{rep['rel_l2_cal']:.1%}, worst f "
               f"{rep['rel_l2_per_freq'][m_worst]:.1%} at "
               f"{cfg.FREQS[m_worst]:.3f} f_c; eps {eps/cfg.DX_FINE:.3f} fine cells, "
               f"rho_void {rho_s:.0e}, impedance "
               f"{math.sqrt(cfg.VOID_STIFFNESS_FLOOR * rho_s):.0e}",
        extras=dict(reports=reports, radii=radii, radii_ls=list(radii_ls),
                    rel_l2=errs, eps_len=eps, void_density_scale=rho_s),
    )


# ---------------------------------------------------------------------------
# Check 8 -- interface width and grid spacing, separated
# ---------------------------------------------------------------------------
def check_interface_width(nu: float = 1.0 / 3.0, *, device=None,
                          radius_ls: float = cfg.R_MAX_LS, src_idx: int = 0,
                          factors: tuple[float, ...] = (2.0, 1.0, 0.5),
                          refine: int = 3,
                          gate: float = cfg.GATE_GRID_CONVERGENCE) -> CheckResult:
    """
    Two refinements that a single "make the grid finer" study would conflate.

    The cavity error of check 7 has two candidate sources with opposite remedies.  A
    smoothed interface is a *physical* deviation from a traction-free cavity and gets
    better as eps shrinks; a sigmoid narrower than the cell that samples it is a
    *discretisation* error and gets worse.  Sweeping eps and dx together cannot tell
    them apart, and the review's objection was exactly that.

    Part A holds dx fixed and varies eps.  Measured (10-90% width in fine cells,
    rel-L2 of the scattered ring field at R = 1.2 lambda_s, fitted amplitude ratio
    beneath):

        2.64      1.98      1.65      1.32      0.99
        15.7%     11.0%      9.3%      8.9%     11.4%
        0.948     0.975     0.989     1.000     1.013

    a U with a broad flat bottom, and the production width sits on it.  The amplitude
    ratio crosses 1.0 between the last two, which is the signature of the right-hand
    branch: a sub-cell sigmoid is a staircase, and a staircase over-scatters.  The
    minimum is one step narrower than production and is deliberately not taken; see
    `config.EPS_INTERFACE_FINE_CELLS` for why 0.4 points is not worth moving towards the
    staircase branch.  The default `factors` sweep is coarser than this table (3.30,
    1.65, 0.82) precisely so that the gate asks whether production is in an interior
    minimum rather than whether it is at the exact optimum, which is a question about
    the shape of the U and not about a fourth digit.

    Part B holds eps *as a length* and refines dx by `refine`, with `n_pml`,
    `n_total`, `nt` and `downsample` all scaled so that `n_net` and the physical
    positions of the source and the ring are unchanged -- otherwise the two runs
    differ in their acquisition geometry as well as their grid.  Measured: 20.7% ->
    20.9% at a narrow eps and 84.4% -> 84.6% at the old wide one.  Both are flat to
    0.2 points across a 3x refinement, which is the actual finding: at this width the
    discretisation is not the limitation, and the remaining cavity error is the
    absorber and the illumination (checks 6 and 9), not the mesh.

    Gate: the eps sweep must have its minimum at the production width (an interior
    minimum, so neither branch is being ridden), and the dx refinement must move the
    error by less than GATE_GRID_CONVERGENCE.
    """
    lam_s = cfg.cs_over_cp(nu) / cfg.FC
    radius = radius_ls * lam_s
    eps_list = [f * EPS_LEN_PHYS for f in factors]
    i_prod = factors.index(1.0)

    scat, src_xy, _, _ = _cavity_scattered(
        [radius] * len(eps_list), nu=nu, src_idx=src_idx, device=device,
        eps_len=eps_list, void_density_scale=cfg.VOID_DENSITY_SCALE)
    ref = _cavity_reference(radius, src_xy, nu=nu)
    reps = [CAV.cavity_accuracy_report(scat[i], ref, freqs=cfg.FREQS)
            for i in range(len(eps_list))]
    errs = [r["rel_l2"] for r in reps]
    at_min = errs[i_prod] == min(errs) and 0 < i_prod < len(errs) - 1

    delta, err_fine = float("nan"), float("nan")
    if refine > 1:
        scat_f, src_f, _, _ = _cavity_scattered(
            [radius], nu=nu, src_idx=src_idx, device=device, eps_len=EPS_LEN_PHYS,
            void_density_scale=cfg.VOID_DENSITY_SCALE, dx=cfg.DX_FINE / refine,
            refine=refine)
        assert max(abs(a - b) for a, b in zip(src_f, src_xy)) < 1e-12
        err_fine = CAV.cavity_accuracy_report(scat_f[0], ref,
                                             freqs=cfg.FREQS)["rel_l2"]
        delta = abs(err_fine - errs[i_prod])

    w = [f * EPS_LEN_PHYS * cfg.EPS_TRANSITION_FACTOR / cfg.DX_FINE for f in factors]
    table = ", ".join(f"{wi:.2f} cells {e:.1%}" for wi, e in zip(w, errs))
    return CheckResult(
        name="8. interface width vs grid",
        passed=at_min and not (delta >= gate), value=delta, gate=gate,
        units="rel-L2 shift",
        detail=f"R = {radius_ls:.2f} l_s; 10-90% width {table} "
               f"({'interior min at the production width' if at_min else 'NOT at the minimum'}); "
               f"dx/{refine} at fixed eps: {errs[i_prod]:.1%} -> {err_fine:.1%}",
        extras=dict(reports=reps, factors=list(factors), rel_l2=errs,
                    width_fine_cells=w, rel_l2_refined=err_fine, refine=refine,
                    radius=radius),
    )


# ---------------------------------------------------------------------------
# Check 9 -- what the sponge actually reflects
# ---------------------------------------------------------------------------
def _open_domain_ring(*, nu: float, src_idx: int, device, n_pml: int = cfg.N_PML_FINE,
                      pad: int = 0, order: float = cfg.ABSORBER_ORDER,
                      r_target: float = cfg.ABSORBER_R_TARGET,
                      radius: float | None = None) -> np.ndarray:
    """
    Ring phasors for one absorber design, [nrows, R, 2, M]; row 0 incident.

    Two independent ways to enlarge the grid, and the distinction is the whole point of
    the check.  `n_pml` thickens the *absorber* outward, leaving the physical region at
    N_FINE cells so the production ring keeps its network indices -- that is a different
    absorber on the same problem.  `pad` enlarges the *physical* region instead, so the
    production ring sits `pad // downsample` cells in and any boundary reflection has
    `2 * pad * dx` further to travel; that is the same absorber on a bigger problem, and
    with `pad` large enough that the round trip exceeds T_end it is an open domain as far
    as the recording window can tell.
    """
    grow = n_pml - cfg.N_PML_FINE
    n_total = cfg.N_FINE_TOTAL + 2 * (grow + pad)
    # `grow` is already carried by the coordinate offset (= n_pml), so only the extra
    # physical cells move the domain centre away from 0.5 * L_DOMAIN.
    ctr = pad * cfg.DX_FINE + 0.5 * cfg.L_DOMAIN
    lam0, mu0 = cfg.lame_from_nu(nu)
    rows = [torch.zeros(1, n_total, n_total, device=device)]
    if radius is not None:
        rows.append(_circle_chi(radius, n_total=n_total, n_pml=n_pml, dx=cfg.DX_FINE,
                                centre=(ctr, ctr), device=device))
    chi = torch.cat(rows, dim=0)
    solid = (1.0 - chi).clamp_min(cfg.VOID_STIFFNESS_FLOOR)
    rho = cfg.RHO0 * (1.0 - (1.0 - cfg.VOID_DENSITY_SCALE) * chi)

    sim = ElasticFDTD2D(lam0 * solid, mu0 * solid, rho, n_pml=n_pml,
                        absorber_order=order, absorber_r_target=r_target)
    assert sim.n_net == cfg.N_NET + 2 * (pad // cfg.DOWNSAMPLE), sim.n_net
    shift = pad // cfg.DOWNSAMPLE
    recv = [(y + shift, x + shift) for y, x in _receiver_ring(cfg.N_NET)]
    fy, fx = cfg.net_to_fine(*cfg.SOURCES_NET[src_idx])
    res = sim.run([(fy + grow + pad, fx + grow + pad)] * chi.shape[0], nt=cfg.NT,
                  recv_yx=recv)
    u = H.displacement_from_ascans(res.ascans, omegas=H.omegas_tensor(device),
                                   dt=cfg.DT, nt=cfg.NT)
    return u.detach().cpu().numpy().astype(complex)


def check_absorber_reflection(nu: float = 1.0 / 3.0, *, device=None,
                              src_idx: int = 0, radius_ls: float = cfg.R_MAX_LS,
                              pad: int = 208, n_pml: int | None = None,
                              order: float | None = None,
                              r_target: float | None = None,
                              gate: float = cfg.GATE_ABSORBER_REFLECTION
                              ) -> CheckResult:
    """
    What the sponge reflects, measured as a difference against an open domain.

    ABSORBER_R_TARGET is a design input to a WKB formula, not an achieved reflection,
    and finding (c) of the review is that nothing here had checked the difference.  A
    textbook normal-incidence reflection coefficient would need a plane-wave source and
    would answer a question the production geometry never asks: sources sit
    RING_INSET_NET cells from the layer, so the layer is struck at every angle out to
    grazing, and grazing incidence is where a sponge is worst.  Instead this runs the
    production acquisition twice -- once as built, once with the physical region padded
    so far that nothing the walls send back can reach the ring inside T_END -- and calls
    the difference the artefact.  That measures the layer in situ, at the angles it
    actually sees, with no plane-wave synthesis and no analytic reference.

    `pad = 208` fine cells is 6.5 length units, so a wall echo takes 13 units of extra
    travel; the residual is converged in `pad` (22.34% at 104, 22.33% at 208 for the
    v2.0 layer), which is how one knows the reference is not being measured against
    itself.

    Both fields are reported because they answer different questions.  The incident
    artefact is what the layer does, full stop.  The scattered artefact is what survives
    into a label: the incident field is subtracted, and since check 7's batch shares the
    source and the domain, the source-side reflection cancels in that subtraction.  For
    the v2.0 layer the two were 22.33% and 6.26% -- the cancellation is real and it was
    not nearly enough, since 6.26% was then the largest single term in check 7's error
    budget.  For the layer that replaced it they are 1.08% and 1.81%, and the *ordering
    flips*: once the common-mode reflection is small there is no large term left for the
    subtraction to remove, and the scattered field is the smaller signal, so the same
    absolute artefact reads as a larger relative one.  Gate on both.

    The design was chosen by measuring, not by lowering R_target; see `config.py`'s
    absorber block for the 42-design sweep and the reason R_target wants to go *up*.
    """
    n_pml = cfg.N_ABSORBER_FINE if n_pml is None else n_pml
    order = cfg.ABSORBER_ORDER if order is None else order
    r_target = cfg.ABSORBER_R_TARGET if r_target is None else r_target
    radius = radius_ls * cfg.cs_over_cp(nu) / cfg.FC
    kw = dict(nu=nu, src_idx=src_idx, device=device, radius=radius)

    ref = _open_domain_ring(pad=pad, **kw)
    got = _open_domain_ring(n_pml=n_pml, order=order, r_target=r_target, **kw)

    def rel(a: np.ndarray, b: np.ndarray, axis=None) -> np.ndarray:
        return np.linalg.norm(a - b, axis=axis) / np.linalg.norm(b, axis=axis)

    inc = float(rel(got[0], ref[0]))
    scat = float(rel(got[1] - got[0], ref[1] - ref[0]))
    per_f = rel(got[0], ref[0], axis=(0, 1))
    d0 = cfg.absorber_d0(n_pml, p=order, r_target=r_target)
    thick = n_pml * cfg.DX_FINE
    return CheckResult(
        name="9. absorber vs open domain",
        passed=inc < gate and scat < gate, value=inc, gate=gate, units="rel-L2",
        detail=f"incident {inc:.2%}, scattered {scat:.2%}; {n_pml} fine cells = "
               f"{thick / (cfg.CP / cfg.FREQS[0]):.2f} lam_p(f_lo), p={order:g}, "
               f"design R={r_target:g}, d0={d0:.2f} ({d0 * cfg.DT:.3f} per step); "
               f"per-f {per_f[0]:.1%} -> {per_f[-1]:.1%}; open-domain pad {pad} cells",
        extras=dict(incident=inc, scattered=scat, per_freq=per_f.tolist(), d0=d0,
                    n_absorber_fine=n_pml, order=order, r_target=r_target, pad=pad,
                    thickness_lambda_p_lo=thick / (cfg.CP / cfg.FREQS[0])),
    )


# ---------------------------------------------------------------------------
# Check 10 -- the physics loss, evaluated on labels the solver produced
# ---------------------------------------------------------------------------
def check_physics_residual_on_labels(nu: float = 1.0 / 3.0, *, device=None,
                                     radii_ls: tuple[float, ...] = (cfg.R_MIN_LS, 0.8,
                                                                    cfg.R_MAX_LS),
                                     src_idx: int = 0,
                                     interface_weight: float = 1.0,
                                     gate: float = cfg.GATE_PHYS_RESIDUAL_LABEL
                                     ) -> CheckResult:
    """
    What `losses.physics_loss` reads on a field that is, by construction, correct.

    Review finding (d): the physics residual and the solver do not share a
    discretisation.  The solver is a staggered velocity-stress scheme on the 376^2 fine
    grid with dt-coupled updates; the residual is a pair of nested 4th-order *centred*
    differences of the time-harmonic Navier operator on the 128^2 network grid.  Nothing
    forces the second to vanish on a solution of the first, and a regulariser with an
    unmeasured floor is a regulariser that can be trading physics for numerics.  The
    floor is what this measures.

    Four rows in one batch: the homogeneous medium, then three radii spanning the
    specified range.  The homogeneous row is the control that separates the two
    contributions -- it has no interface anywhere, so whatever it reads is pure
    discretisation mismatch (plus the source, which the weight excludes), and any excess
    in the void rows is the interface.  Reported both with the default solid-fraction
    weight and with `interface_weight` restoring the band the default discards, which is
    the other half of review §3.7: the traction-free condition lives exactly where the
    default weight is zero.

    Measured (nu = 1/3, production grid, default weight / interface band):

        homogeneous   0.033 / 0.033
        R = 0.4 l_s   0.044 / 0.048
        R = 0.8 l_s   0.047 / 0.056
        R = 1.2 l_s   0.060 / 0.075

    Three things follow.  The floor is a third of the total, so most of what training
    would charge a perfect prediction is the *scheme*, not the geometry -- and it is
    genuinely irreducible, since the network grid is the network grid.  The interface adds
    +0.027 at the largest radius and scales with perimeter, which is why the gate carries
    headroom for the transfer families rather than sitting just above this table.  And
    restoring the interface band adds +0.016 more, monotonically in radius: the band is
    the worst-conditioned place in the domain for a centred stencil, which is exactly why
    the default weight drops it and exactly why dropping it is a real omission.  Neither
    choice is free, and `losses.make_context`'s `interface_weight` exists so the trade is
    a number rather than an assumption.

    The gate is on the *default* weight, since that is what training uses.  It is a
    floor, not an accuracy claim: a residual of 0.06 means a perfect prediction would
    still be charged 0.06, so `cfg.ALPHA_PHYS` has to be small enough that 0.06 of
    something is not worth more than the data term -- see the arithmetic there.
    """
    lam_s = cfg.cs_over_cp(nu) / cfg.FC
    radii = [r * lam_s for r in radii_ls]
    om = H.omegas_tensor(device)
    freqs = torch.tensor(cfg.FREQS, device=device, dtype=torch.float32)

    rows = [torch.zeros(1, cfg.N_FINE_TOTAL, cfg.N_FINE_TOTAL, device=device)]
    rows += [_circle_chi(r, n_total=cfg.N_FINE_TOTAL, n_pml=cfg.N_ABSORBER_FINE,
                         dx=cfg.DX_FINE, device=device) for r in radii]
    chi_f = torch.cat(rows, dim=0)
    B = chi_f.shape[0]
    lam0, mu0 = cfg.lame_from_nu(nu)
    solid = (1.0 - chi_f).clamp_min(cfg.VOID_STIFFNESS_FLOOR)
    rho = cfg.RHO0 * (1.0 - (1.0 - cfg.VOID_DENSITY_SCALE) * chi_f)

    sim = ElasticFDTD2D(lam0 * solid, mu0 * solid, rho)
    fy, fx = cfg.net_to_fine(*cfg.SOURCES_NET[src_idx])
    res = sim.run([(fy, fx)] * B, nt=cfg.NT, recv_yx=cfg.RECEIVERS_NET, omegas=om)
    u = H.displacement_from_field(res.phasors, omegas=om)   # [B, 2, M, ny, nx]

    # chi on the *network* grid, built the way the dataloader builds it -- not pooled
    # from the fine grid, since the network never sees the fine one.
    yy, xx = grid_coords(cfg.N_NET, cfg.DX_NET, device=device)
    ctr = 0.5 * cfg.L_DOMAIN
    chi_n = [torch.zeros(1, cfg.N_NET, cfg.N_NET, device=device)]
    chi_n += [soft_indicator(Circle().sdf(
        torch.tensor([[ctr, ctr, r]], device=device), yy, xx), EPS_LEN_PHYS)
        for r in radii]
    chi_n = torch.cat(chi_n, dim=0)

    nu_t = torch.full((1,), nu, device=device)
    src_t = torch.tensor([src_idx], device=device)
    out = []
    for b in range(B):
        # B-major flatten of one row's M frequencies, which is the order make_context
        # produces with repeat_interleave.
        ub = u[b].permute(1, 0, 2, 3).contiguous()           # [M, 2, ny, nx]
        vals = []
        for w_if in (0.0, interface_weight):
            ctx = LOSS.make_context(chi_n[b:b + 1], nu_t, freqs.unsqueeze(0), src_t,
                                    interface_weight=w_if)
            vals.append(float(LOSS.physics_loss(ub, ctx)))
        out.append(vals)

    floor, void = out[0], out[1:]
    worst = max(v[0] for v in void)
    per_r = ", ".join(f"{rl:.2f} l_s {v[0]:.3f}/{v[1]:.3f}"
                      for rl, v in zip(radii_ls, void))
    return CheckResult(
        name="10. physics residual on labels",
        passed=worst < gate, value=worst, gate=gate, units="rel residual",
        detail=f"homogeneous floor {floor[0]:.3f} (band {floor[1]:.3f}); "
               f"R = {per_r} (default/band); interface costs "
               f"{worst - floor[0]:+.3f} over the floor, the band "
               f"{max(v[1] for v in void) - worst:+.3f} more",
        extras=dict(floor=floor, per_radius=out[1:], radii_ls=list(radii_ls),
                    interface_weight=interface_weight),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_all(device=None, *, include_slow: bool = True,
            nu: float = 1.0 / 3.0) -> list[CheckResult]:
    """
    Run the checks in the order of §11.2 and print the table.

    Checks 1-5 are the original set.  6-10 were added because the review found that the
    forward model had never been compared against anything external and that the physics
    regulariser had never been measured at all: 6 and 7 test the incident and the
    scattered field against closed-form elastodynamics, 8 separates the two refinements
    that could explain 7's residual, 9 measures the absorber against an open domain, and
    10 reads `losses.physics_loss` on labels the solver produced, which is the only way
    to know its floor.  6, 7 and 10 are cheap enough to always run (thirteen solves,
    about two minutes on CPU at the production grid).  8 and 9 are slow for the same
    reason they are informative -- 8 refines the grid threefold, 9 pads the domain to 792
    cells a side -- so they sit behind `include_slow` with 4 and 5.
    """
    results = [
        check_energy_conservation(nu, device=device),
        check_arrival_times(nu, device=device),
        check_absorber(nu, device=device),
    ]
    if include_slow:
        results.append(check_rayleigh_and_mode_conversion(nu, device=device))
        results.append(check_grid_convergence(nu, device=device))
    results.append(check_green_incident(nu, device=device))
    results.append(check_cavity_scattering(nu, device=device))
    results.append(check_physics_residual_on_labels(nu, device=device))
    if include_slow:
        results.append(check_interface_width(nu, device=device))
        results.append(check_absorber_reflection(nu, device=device))

    print(f"solver validation, nu = {nu:.3f}, Courant = {cfg.CFL_NUMBER:.4f}, "
          f"nt = {cfg.NT}")
    print("-" * 100)
    for r in results:
        print(r)
    print("-" * 100)
    n_ok = sum(r.passed for r in results)
    print(f"{n_ok}/{len(results)} checks passed")
    if n_ok < len(results):
        print("Do not proceed to dataset generation until every check passes: "
              "the network will learn whatever the solver does, including its bugs.")
    return results


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_all(device=dev, include_slow=(dev == "cuda"))
