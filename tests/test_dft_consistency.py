"""
The two time grids of the harmonic reduction, pinned (harmonic.py docstring item 1).

Deconvolution divides a transform of the *recorded velocities* by a transform of the
*injected force*, and those two sequences do not live on the same time grid: the
leapfrog stores velocities at t = (n+1/2)dt, while the force is added to the update
centred on t = n dt.  Every function here exists because getting that wrong is
invisible -- it multiplies every phasor by exp(+i omega dt/2), 0.072 rad at f_max,
which reads as a 0.85% wave-speed error, and the inversion measures wave speeds.

The bug these tests were written for was real and is fixed; `test_source_spectrum_*`
pins the source grid, `test_first_step_*` pins the sequence the solver actually
injects, and `test_deconvolution_recovers_a_pure_delay` pins the whole chain to
twelve digits without running a solve.
"""

from __future__ import annotations

import math

import pytest
import torch

from src import config as cfg
from src.solver import harmonic as H
from src.solver.fdtd_elastic import (ElasticFDTD2D, dft_at_freqs,
                                     homogeneous_material, source_spectrum,
                                     tone_burst)

NT_SHORT = 256           # enough to hold the burst; the tests are all linear


def _burst(nt: int, offset: float) -> torch.Tensor:
    t = (torch.arange(nt, dtype=torch.float64) + offset) * cfg.DT
    return tone_burst(t, cfg.FC, cfg.N_CYCLES)


# ---------------------------------------------------------------------------
# The source spectrum is the transform of the injected sequence
# ---------------------------------------------------------------------------
def test_source_spectrum_uses_the_integer_time_grid():
    om = H.omegas_tensor()
    got = source_spectrum(om, cfg.DT, NT_SHORT)
    want = dft_at_freqs(_burst(NT_SHORT, 0.0).unsqueeze(0), om, cfg.DT, NT_SHORT,
                        t_offset=0.0).squeeze(0)
    assert torch.allclose(got, want, rtol=1e-13, atol=0.0)


def test_source_spectrum_is_not_the_half_grid_quadrature():
    """
    The regression this file exists for.  The mistake was not sampling the burst at
    the wrong times -- it was transforming samples taken at `n dt` with the
    quadrature times `(n + 1/2) dt` that the velocity DFT needs.  That is an exact
    constant phase, so no amplitude test could ever have caught it.
    """
    om = H.omegas_tensor()
    got = source_spectrum(om, cfg.DT, NT_SHORT)
    bad = dft_at_freqs(_burst(NT_SHORT, 0.0).unsqueeze(0), om, cfg.DT, NT_SHORT,
                       t_offset=0.5).squeeze(0)
    assert torch.allclose(got.abs(), bad.abs(), rtol=1e-13)
    lag = -torch.angle(bad / got) / om            # the implied time shift, [M]
    assert torch.allclose(lag, torch.full_like(lag, 0.5 * cfg.DT), rtol=1e-9)


def test_dft_offset_is_the_sample_time():
    """
    `t_offset` must be the time of sample 0, not a fudge factor.  Tested on a
    complex exponential, where the transform is exact and free of the finite-window
    leakage a real cosine would carry: sampling `exp(+i w t_n)` and transforming
    with matched times sums `nt` ones, and any mismatch delta shows up as
    `exp(+i w delta dt)` and nothing else.
    """
    w = cfg.OMEGAS[-1]
    om = torch.tensor([w], dtype=torch.float64)
    for offset in (0.0, 0.5, 1.0):
        t = (torch.arange(NT_SHORT, dtype=torch.float64) + offset) * cfg.DT
        x = torch.exp(1j * w * t).unsqueeze(0)
        got = dft_at_freqs(x, om, cfg.DT, NT_SHORT, t_offset=offset)
        assert got.real.item() == pytest.approx(NT_SHORT * cfg.DT, rel=1e-12)
        assert abs(got.imag.item()) < 1e-12 * NT_SHORT * cfg.DT, offset
        wrong = dft_at_freqs(x, om, cfg.DT, NT_SHORT, t_offset=offset + 0.5)
        assert torch.angle(wrong / got).item() == pytest.approx(
            -0.5 * w * cfg.DT, rel=1e-9)


# ---------------------------------------------------------------------------
# The solver injects s(n dt), starting from s(0) = 0
# ---------------------------------------------------------------------------
def test_the_two_grids_differ_by_a_cube_at_the_start_of_the_burst():
    """
    How distinguishable the two grids are in the first few samples, which is what
    the two solver tests below rely on.  A Hann-windowed sinusoid starts as t^3
    (the window is quadratic, the sinusoid linear), so `s(0)` is *exactly* zero
    while `s(dt/2)` is not -- but it is only 1.5e-6, so "the field is zero after one
    step" is an exact statement rather than a large one.  By the second sample the
    two grids differ by a factor of (3/2)^3, which is a comfortable 3.4x.
    """
    assert float(_burst(1, 0.0)[0]) == 0.0
    assert 1e-7 < abs(float(_burst(1, 0.5)[0])) < 1e-5
    assert float(_burst(2, 0.5)[1]) / float(_burst(2, 0.0)[1]) == pytest.approx(
        1.5 ** 3, rel=0.02)


def test_first_step_is_exactly_zero():
    """
    One step of the real solver.  The force added at step 0 is s(0) = 0 on the
    integer grid, so the whole field after one step is *identically* zero -- not
    small, zero -- and any other choice of grid makes it nonzero.
    """
    n = 64
    lam, mu, rho = homogeneous_material(1.0 / 3.0, batch=1, n_total=n)
    sim = ElasticFDTD2D(lam, mu, rho, n_pml=4)
    res = sim.run([(n // 2, n // 2)], nt=1,
                  recv_yx=[(sim.n_net // 2, sim.n_net // 2)])
    assert float(res.ascans.abs().max()) == 0.0


def test_second_step_is_the_second_sample():
    """
    Two steps.  Nothing has propagated yet -- the first step injected zero, so the
    stresses are still zero -- and the y-face at the source therefore carries
    exactly `dt * s[1] / (rho dx^2)`.  Pins the *index*: `s[1] = s(dt)`, which the
    test above shows differs from `s(3dt/2)` by 3.4x, far outside the tolerance.
    """
    n, nu = 64, 1.0 / 3.0
    n_pml, fy = 4, 32
    lam, mu, rho = homogeneous_material(nu, batch=1, n_total=n)
    sim = ElasticFDTD2D(lam, mu, rho, n_pml=n_pml)
    res = sim.run([(fy, fy)], nt=2, recv_yx=[(sim.n_net // 2, sim.n_net // 2)])
    # recording: avg_minus puts half the face value on each of the centres fy and
    # fy+1, then a downsample^2 block mean.  Both centres are in the same block
    # unless the face sits on the last row of one, hence the parity.
    d = sim.downsample
    same_block = (fy - n_pml) % d < d - 1
    weight = (1.0 if same_block else 0.5) / d ** 2
    want = (cfg.DT * float(_burst(2, 0.0)[1]) / (cfg.RHO0 * cfg.DX_FINE ** 2)
            * weight)
    assert float(res.ascans[0, 0, 1, 1]) == pytest.approx(want, rel=1e-6)
    assert float(res.ascans[0, 0, :, 0].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# End to end: a delay must come out as a phase, and nothing else
# ---------------------------------------------------------------------------
def test_deconvolution_recovers_a_pure_delay():
    """
    Feed `displacement_from_ascans` a velocity trace that is *exactly* the time
    derivative of a delayed copy of the injected burst, differenced onto the half
    levels the way the leapfrog produces it.  Then the deconvolved displacement is
    known in closed form:

        v[n] = (s[n+1] - s[n]) / dt   at t = (n+1/2) dt
        =>  v_hat = i * omega_tilde * s_hat,   omega_tilde = (2/dt) sin(omega dt/2)

    so `u_hat` must equal exp(-i omega t0) * omega_tilde / omega -- a pure phase
    from the delay, times the 0.086% leapfrog symbol correction that
    `transfer_factor` deliberately does not apply.  A source spectrum on the wrong
    grid would show up here as an extra 0.072 rad at f_max, 800x the tolerance.
    """
    nt, n0 = cfg.NT, 100
    om = H.omegas_tensor()
    s = _burst(nt + 1, 0.0)
    s_shift = torch.zeros(nt + 1, dtype=torch.float64)
    s_shift[n0:] = s[:nt + 1 - n0]
    v = (s_shift[1:] - s_shift[:-1]) / cfg.DT          # at (n + 1/2) dt

    u = H.displacement_from_ascans(v.view(1, 1, 1, nt), omegas=om, dt=cfg.DT, nt=nt)
    om_tilde = (2.0 / cfg.DT) * torch.sin(om * cfg.DT / 2.0)
    want = torch.exp(-1j * om.to(torch.complex128) * (n0 * cfg.DT)) * (om_tilde / om)
    got = u[0, 0, 0].to(torch.complex128)
    assert torch.allclose(got, want, rtol=2e-6, atol=1e-12), (got / want)
    # the correction transfer_factor omits, quoted in harmonic.py's docstring
    assert float((om_tilde / om).min()) == pytest.approx(1.0 - 8.6e-4, abs=1e-4)


def test_wrong_source_grid_is_a_growing_phase_error():
    """
    The size of the mistake, so the 6% figure quoted in `harmonic.py` and
    `fdtd_elastic.source_spectrum` cannot drift away from the code.
    """
    om = H.omegas_tensor()
    good = source_spectrum(om, cfg.DT, cfg.NT)
    bad = dft_at_freqs(_burst(cfg.NT, 0.0).unsqueeze(0), om, cfg.DT, cfg.NT,
                       t_offset=0.5).squeeze(0)
    rel = float(((good / bad) - 1.0).abs().max())
    assert rel == pytest.approx(2.0 * math.sin(0.25 * cfg.OMEGAS[-1] * cfg.DT),
                                rel=1e-9)
    assert 0.05 < rel < 0.08


# ---------------------------------------------------------------------------
# The envelope convention, which is what solver check 2 measures against
# ---------------------------------------------------------------------------
def _two_phase(nt: int, d: float, nu: float, n_cycles: int,
               a_s: float) -> tuple[torch.Tensor, float, float]:
    """
    A synthetic vector trace with *exactly known* arrivals: a Hann burst at d/c_p
    and another, `a_s` times larger, at d/c_s, in fixed non-aligned polarisations.
    Returns [1, 1, 2, nt] and the two true envelope-peak times.
    """
    t = (torch.arange(nt, dtype=torch.float64) + 0.5) * cfg.DT
    group = 0.5 * n_cycles / cfg.FC
    tp, ts = d / cfg.CP, d / cfg.cs_over_cp(nu)
    p = tone_burst(t - tp, cfg.FC, n_cycles)
    s = tone_burst(t - ts, cfg.FC, n_cycles) * a_s
    vx = 0.9 * p - 0.3 * s
    vy = 0.4 * p + 0.95 * s
    return torch.stack([vx, vy]).reshape(1, 1, 2, nt), tp + group, ts + group


def _peak_err(env: torch.Tensor, pred: float, half: float) -> float:
    """Envelope-peak error in time steps, searched in pred +- half, as check 2 does."""
    lo = int(max(0, (pred - half) / cfg.DT))
    hi = int(min(env.shape[-1], (pred + half) / cfg.DT))
    k = int(env[lo:hi].argmax()) + lo
    return abs((k + 0.5) * cfg.DT - pred) / cfg.DT


@pytest.mark.parametrize("d,a_s", [(5.0, 4.0), (5.0, 1.0), (3.0, 4.0)])
def test_vector_envelope_peaks_at_the_group_delay(d, a_s):
    """
    Both phases' envelope peaks land within the check-2 gate on a trace whose
    arrivals are known exactly, so a failure of check 2 is the solver's and not the
    measurement's.  This is the property `vector_envelope` exists for.
    """
    nc = 2
    x, tp, ts = _two_phase(cfg.NT, d, 1.0 / 3.0, nc, a_s)
    env = H.vector_envelope(x)[0, 0]
    half = 0.5 * nc / cfg.FC
    for pred in (tp, ts):
        assert _peak_err(env, pred, half) < cfg.GATE_ARRIVAL_STEPS


@pytest.mark.parametrize("d,a_s", [(5.0, 4.0), (3.0, 4.0)])
def test_rectifying_before_the_hilbert_transform_breaks_the_gate(d, a_s):
    """
    The bug this replaced, kept as a test so it cannot come back: taking the
    analytic signal of sqrt(vx^2 + vy^2) moves the spectrum to DC and 2 f_c, and the
    resulting "envelope" peaks more than an order of magnitude further out than the
    1-step gate -- on a trace where the true arrivals are known exactly.
    """
    nc = 2
    x, tp, ts = _two_phase(cfg.NT, d, 1.0 / 3.0, nc, a_s)
    bad = H.envelope(x[0, 0].pow(2).sum(dim=0).sqrt())
    half = 0.5 * nc / cfg.FC
    worst = max(_peak_err(bad, tp, half), _peak_err(bad, ts, half))
    assert worst > 10.0 * cfg.GATE_ARRIVAL_STEPS
