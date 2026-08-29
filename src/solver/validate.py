"""
The five solver sanity checks of §3.7, as code (§11.2 steps 1-5).

These are the checks the document says never to cut, and the reason is not
diligence for its own sake: every one of them fails *silently* in the sense that a
broken solver still produces plausible-looking wavefields.  A sign error in one
staggered derivative gives a solver that propagates at the wrong speed; a badly
graded absorber gives one that reflects 5% off the boundary, which the network then
faithfully learns as part of the operator, and the inversion then interprets as a
second defect.  Nothing downstream can distinguish a wrong solver from a hard
problem, so the checks have to happen here.

Each function returns a `CheckResult` carrying the number, the gate it was compared
against and a one-line explanation.  `run_all()` prints the table that goes in
notebook 01.

Cost note: checks 4 and 5 need finer grids than the production configuration --
check 4 because Rayleigh scattering requires kR << 1 *and* R >> dx at the same
time, check 5 because a convergence test needs a grid the production grid can be
compared against.  Both are marked slow and want a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .. import config as cfg
from ..geometry.sdf import Circle, grid_coords, material_fields, soft_indicator
from . import harmonic as H
from .fdtd_elastic import ElasticFDTD2D, homogeneous_material


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


# The physical width of the void interface.  The network sees chi on the *network*
# grid with a sigmoid width of EPS_INTERFACE_CELLS network cells, so the solver --
# whatever its own dx -- must smooth the void over that same physical distance, or
# the label describes a slightly different void from the one the input describes.
EPS_LEN_PHYS: float = cfg.EPS_INTERFACE_CELLS * cfg.DX_NET


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


def _receiver_ring(n_net: int, inset: int = cfg.RING_INSET_NET
                   ) -> list[tuple[int, int]]:
    return cfg.ring_positions(cfg.N_RECV_PER_SIDE, n_grid=n_net, inset=inset)


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
# Check 2 -- P and S arrival times
# ---------------------------------------------------------------------------
def check_arrival_times(nu: float = 1.0 / 3.0, *, device=None,
                        n_cycles: int = 2,
                        gate: float = cfg.GATE_ARRIVAL_STEPS) -> CheckResult:
    """
    Homogeneous medium with the absorber on.  Gate: measured minus predicted
    arrival within GATE_ARRIVAL_STEPS time steps, for both P and S.

    Measured on the *envelope peak*, not the leading edge.  The envelope peak of a
    Hann-windowed burst arrives at d/c + N_c/(2 f_c) -- the group delay is the
    centre of the window -- which is an exact analytic prediction.  A 5%-of-peak
    threshold crossing has no closed form, so comparing it to d/c would be
    comparing two different quantities and the "error" would be dominated by the
    burst shape rather than by the solver.

    A 2-cycle burst is used rather than the production 5-cycle one, for a reason
    worth stating: the P and S envelope peaks are separated by d(1/c_s - 1/c_p),
    which at nu = 1/3 is 0.73 d.  Resolving them as two peaks needs that gap to
    exceed the burst duration N_c/f_c, i.e. d > 1.37 N_c.  At N_c = 5 that is
    d > 6.9 lambda_p, which barely fits in an 8 lambda_p domain; at N_c = 2 it is
    d > 2.7 lambda_p, which every cross-domain receiver satisfies.  The travel
    times being checked belong to the solver, not to the burst, so shortening the
    burst is legitimate.
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
    env = H.envelope(res.ascans.pow(2).sum(dim=2).sqrt())[0]     # [R, nt]
    tgrid = (torch.arange(env.shape[-1], device=env.device,
                          dtype=torch.float64) + 0.5) * cfg.DT

    # distance from source to each receiver, in network-grid coordinates
    d = torch.tensor(
        [math.hypot((ry - src_net[0]) * cfg.DX_NET, (rx - src_net[1]) * cfg.DX_NET)
         for ry, rx in recv], dtype=torch.float64)

    t_p_pred = d / cfg.CP + group_delay
    t_s_pred = d / cs + group_delay
    separation = d * (1.0 / cs - 1.0 / cfg.CP)
    usable = separation > n_cycles / cfg.FC
    assert bool(usable.any()), "no receiver resolves P and S separately"

    errs_p, errs_s = [], []
    half = 0.5 * n_cycles / cfg.FC
    for r in range(env.shape[0]):
        if not bool(usable[r]):
            continue
        for pred, bag in ((t_p_pred[r], errs_p), (t_s_pred[r], errs_s)):
            lo = int(max(0, (pred - half) / cfg.DT))
            hi = int(min(env.shape[-1], (pred + half) / cfg.DT))
            if hi - lo < 4:
                continue
            k = int(env[r, lo:hi].argmax()) + lo
            bag.append(abs(float(tgrid[k]) - float(pred)) / cfg.DT)

    assert errs_p and errs_s, (
        "no usable P/S window; the record is too short or the ring too tight")
    worst = max(max(errs_p), max(errs_s))
    return CheckResult(
        name="2. P/S arrival error",
        passed=worst < gate, value=worst, gate=gate, units="steps",
        detail=f"{int(usable.sum())}/{len(recv)} receivers resolve both; "
               f"P worst {max(errs_p):.2f}, S worst {max(errs_s):.2f} steps "
               f"(c_s/c_p={cs:.3f})",
        extras=dict(errs_p=errs_p, errs_s=errs_s, distances=d.tolist(),
                    usable=usable.tolist(), ascans=res.ascans),
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
        radii_ls: tuple[float, ...] = (0.10, 0.125, 0.15, 0.20),
        refine: int = 4, l_domain: float = 3.0,
        slope_window: tuple[float, float] = (3.3, 4.7)) -> CheckResult:
    """
    Scattered energy versus void radius, in the long-wavelength limit.

    Theory (2D, long wavelength): a small void scatters through a monopole and a
    dipole term, both O((kR)^2) in *amplitude*, so the scattered energy collected
    on a fixed ring scales as R^4 at fixed frequency.  Gate: the fitted log-log
    slope of energy against R lies in `slope_window`.

    Why the gate is a window and not a tight number.  The check needs kR << 1 and
    R >> dx simultaneously, and a fixed grid cannot give both: at the production
    resolution the smallest resolvable void already has kR ~ 1.  Even on this
    4x-refined grid the radii used span kR ~ 0.6 to 1.3, where the next term in the
    long-wavelength expansion contributes tens of percent.  So this is a
    *scaling-consistency* test rather than a precision test -- it catches a slope of
    2 or 6, which is what broken interface averaging or a mis-normalised soft
    indicator produces, and it is not sensitive enough to certify the coefficient.
    Said plainly rather than dressed up.

    The interface width is 1 cell of *this* grid rather than the production
    EPS_LEN_PHYS: at these radii the production smoothing (1.5 network cells =
    0.094 lambda_p) is wider than the void itself, so there would be no void left
    to scatter.  Legitimate here because the quantity under test is the solver's
    scattering, not the dataset's geometry convention.

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
            iy, ix = _refine_index(iy, r), _refine_index(ix, r)
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
# Driver
# ---------------------------------------------------------------------------
def run_all(device=None, *, include_slow: bool = True,
            nu: float = 1.0 / 3.0) -> list[CheckResult]:
    """Run the checks in the order of §11.2 and print the table."""
    results = [
        check_energy_conservation(nu, device=device),
        check_arrival_times(nu, device=device),
        check_absorber(nu, device=device),
    ]
    if include_slow:
        results.append(check_rayleigh_and_mode_conversion(nu, device=device))
        results.append(check_grid_convergence(nu, device=device))

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
