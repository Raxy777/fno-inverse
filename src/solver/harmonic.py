"""
Frequency-domain reduction of the time-domain solve (§5).

The solver runs a tone burst; the network learns a *time-harmonic* operator.  This
file is the bridge, and it is deliberately small, because the bridge is where the
subtle errors live.  Three of them, named so they can be checked:

1. **Quadrature mismatch.**  Deconvolution divides one discrete transform by
   another, and the two quantities do not live on the same time grid.  A leapfrog
   scheme stores velocities at the half levels t = (n+1/2)dt, so the running DFT of
   the A-scans uses that grid; the point force, however, is added to the velocity
   update that steps v^{n-1/2} -> v^{n+1/2}, and that update is centred on the
   *integer* level t = n dt, so the injected source is the sequence s(n dt).  The
   condition is therefore not that the two transforms share an offset -- they must
   not -- but that each uses the grid its own samples actually sit on.  Both paths
   go through `fdtd_elastic.dft_at_freqs`, whose `t_offset` argument makes the grid
   explicit: 1/2 for the recorded velocities, 0 for `source_spectrum`.

   Getting it wrong is invisible and expensive.  Computing the source spectrum on
   the half grid leaves exp(+i omega dt/2) on every deconvolved phasor: a pure
   phase, linear in omega, 0.072 rad at the top of the band, which is exactly what
   a uniform 0.85% wave-speed error looks like and the same size as the effect the
   inversion is trying to measure.  It was in fact present until measured against
   the analytic Green's function (`solver.validate.check_green_incident`, 6.2% ->
   1.05% on a domain with no absorber), and `tests/test_dft_consistency.py` now
   pins the source grid to the injection site in `fdtd_elastic.run`.

   One O(dt^2) refinement is deliberately *not* applied: the leapfrog differences a
   quantity sampled at the half levels, so its exact symbol is
   omega_tilde = (2/dt) sin(omega dt/2) rather than omega, and dividing by
   i*omega_tilde instead of i*omega would remove a further 0.086% at f_max.  That
   is far below the discretisation error of the fields themselves, and introducing
   it here would make the harmonic reduction depend on the solver's time stepping,
   which is the coupling this module exists to avoid.

2. **Band-edge conditioning.**  Deconvolution divides by s_hat(omega), and a
   5-cycle Hann burst has exact spectral nulls at f_c(1 +- 2/5) = 0.6 f_c and
   1.4 f_c.  The operating band is 0.66..1.34 f_c precisely to stay inside those
   nulls, but 1/|s_hat| still varies across the band and multiplies whatever noise
   is present.  `conditioning_report` prints the amplification per frequency and
   `assert_conditioned` refuses to proceed if the worst frequency is more than
   MAX_DECONV_AMPLIFICATION times the best.  A silent 100x amplification at m=19
   would show up much later as "the high frequencies just don't train".

3. **Wrap-around.**  A finite-window DFT is implicitly periodic.  If the wavefield
   has not left the domain by t_end, energy from the tail folds back onto the head
   and corrupts every phasor -- not just the late-arriving ones.
   `tail_energy_fraction` measures this on the A-scans, and it is exactly what
   sanity check 3 of §3.7 is protecting.

Convention throughout: X(omega) = integral x(t) exp(-i omega t) dt, so
d/dt <-> i omega and displacement is u_hat = v_hat / (i omega).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .. import config as cfg
from .fdtd_elastic import dft_at_freqs, source_spectrum

# Worst-to-best ratio of 1/|s_hat| tolerated across the operating band.  Chosen,
# not derived: at 30 dB input SNR a 25x spread means the weakest frequency still
# carries ~4 dB, which the multi-frequency misfit can average over.  Beyond that
# the band should be narrowed rather than the tolerance widened.
MAX_DECONV_AMPLIFICATION: float = 25.0


# ---------------------------------------------------------------------------
# Frequencies and the source transfer factor
# ---------------------------------------------------------------------------
def omegas_tensor(device=None, dtype=torch.float64) -> Tensor:
    """The M_FREQ operating angular frequencies as a tensor, [M]."""
    return torch.tensor(cfg.OMEGAS, device=device, dtype=dtype)


def band_slice(stage: int) -> slice:
    """
    Frequency subset for a frequency-continuation stage (§8.4).

    Stage 1 uses the lowest 6 frequencies because the half-period cycle-skipping
    criterion is most forgiving there: a position error that would be more than
    half a period at f_c is still under half a period at 0.66 f_c, so the low band
    has a wider basin of attraction around the true defect.
    """
    return {1: cfg.BAND_STAGE1, 2: cfg.BAND_STAGE2, 3: cfg.BAND_STAGE3}[stage]


def source_hat(omegas: Tensor | None = None, *, dt: float = cfg.DT,
               nt: int = cfg.NT, device=None) -> Tensor:
    """s_hat(omega_m), complex128 [M].  Same quadrature as the running DFT."""
    if omegas is None:
        omegas = omegas_tensor(device)
    return source_spectrum(omegas, dt, nt)


def transfer_factor(omegas: Tensor | None = None, *, dt: float = cfg.DT,
                    nt: int = cfg.NT, device=None) -> Tensor:
    """
    1 / (i omega s_hat(omega)), complex128 [M].

    Multiplying a velocity phasor by this does two things at once: it removes the
    source waveform (so the result is the medium's response to a unit impulse of
    force, independent of the burst we happened to fire) and it integrates
    velocity to displacement.  Doing both in one factor means there is exactly one
    place where a sign or a factor of omega can be wrong.
    """
    if omegas is None:
        omegas = omegas_tensor(device)
    om = omegas.to(torch.complex128)
    return 1.0 / (1j * om * source_hat(omegas, dt=dt, nt=nt))


def conditioning_report(omegas: Tensor | None = None, *, dt: float = cfg.DT,
                        nt: int = cfg.NT) -> dict:
    """
    Per-frequency noise amplification of the deconvolution.

    Returns `amp` = |1/(i omega s_hat)| normalised to its minimum over the band,
    so amp[m] = 3 means frequency m amplifies measurement noise three times more
    than the best-conditioned frequency does.
    """
    if omegas is None:
        omegas = omegas_tensor()
    g = transfer_factor(omegas, dt=dt, nt=nt).abs()
    amp = (g / g.min()).to(torch.float64)
    return {
        "freqs": [w / (2.0 * math.pi) for w in omegas.tolist()],
        "s_hat_abs": source_hat(omegas, dt=dt, nt=nt).abs().tolist(),
        "amplification": amp.tolist(),
        "worst": float(amp.max()),
        "worst_index": int(amp.argmax()),
    }


def assert_conditioned(omegas: Tensor | None = None, *, dt: float = cfg.DT,
                       nt: int = cfg.NT,
                       limit: float = MAX_DECONV_AMPLIFICATION) -> dict:
    r = conditioning_report(omegas, dt=dt, nt=nt)
    if r["worst"] > limit:
        raise AssertionError(
            f"deconvolution amplifies noise {r['worst']:.1f}x at f = "
            f"{r['freqs'][r['worst_index']]:.3f} f_c (limit {limit:.0f}x). "
            "Narrow the band in config.F_START/DF, or raise N_CYCLES to move the "
            "Hann nulls outward -- do not raise the limit.")
    return r


# ---------------------------------------------------------------------------
# Velocity phasors -> displacement phasors
# ---------------------------------------------------------------------------
def displacement_from_field(vhat: Tensor, *, omegas: Tensor | None = None,
                            dt: float = cfg.DT, nt: int = cfg.NT) -> Tensor:
    """
    [B, 2, M, ny, nx] velocity phasors -> displacement phasors, same shape.

    Complex64 out: the phasors are already a lossy summary of the solve and the
    network trains in float32, so keeping complex128 here would only make the
    dataset twice as large for digits nothing downstream can use.
    """
    if omegas is None:
        omegas = omegas_tensor(vhat.device)
    g = transfer_factor(omegas.to(vhat.device), dt=dt, nt=nt)
    return (vhat.to(torch.complex128) * g.view(1, 1, -1, 1, 1)).to(torch.complex64)


def displacement_from_ascans(ascans: Tensor, *, omegas: Tensor | None = None,
                             dt: float = cfg.DT, nt: int | None = None) -> Tensor:
    """
    [B, R, 2, nt] time-domain velocity -> [B, R, 2, M] displacement phasors.

    This is the only route from a measurement to the inversion's data vector, and
    it is byte-identical in structure to `displacement_from_field` so that
    "predicted at the receiver ring" and "observed at the receiver ring" are the
    same kind of number.  Note that the solver samples A-scans from the
    downsampled network grid for exactly this reason (see ElasticFDTD2D.run).
    """
    if nt is None:
        nt = ascans.shape[-1]
    if omegas is None:
        omegas = omegas_tensor(ascans.device)
    om = omegas.to(ascans.device)
    vhat = dft_at_freqs(ascans, om, dt, nt)            # [B, R, 2, M] complex128
    g = transfer_factor(om, dt=dt, nt=nt)
    return (vhat * g.view(1, 1, 1, -1)).to(torch.complex64)


# ---------------------------------------------------------------------------
# Incident / scattered decomposition
# ---------------------------------------------------------------------------
def scattered(u_total: Tensor, u_incident: Tensor) -> Tensor:
    """u_s = u - u_inc.  Trivial, but named so the sign is written down once."""
    return u_total - u_incident


def incident_scale(u_incident: Tensor, *, eps: float = 1e-20) -> Tensor:
    """
    Per-(sample, frequency) amplitude scale from the *incident* field:
    max over the domain and over both components, shape [B, 1, M, 1, 1].

    Normalising by the incident field rather than by the scattered field is a
    deliberate and load-bearing choice (§7.1).  The scattered amplitude scales
    roughly as R^2 in the Rayleigh regime, so dividing by its own norm removes
    exactly the information the inversion needs to recover R -- every void would
    look like a unit-amplitude scatterer.  The incident field depends only on the
    source position and Poisson ratio, never on the defect, so dividing by it
    rescales without destroying anything.
    """
    return u_incident.abs().amax(dim=(1, 3, 4), keepdim=True).clamp_min(eps)


def incident_scale_at_receivers(u_inc_r: Tensor, *, eps: float = 1e-20) -> Tensor:
    """Same idea for [B, R, 2, M] receiver data; returns [B, 1, 1, M]."""
    return u_inc_r.abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)


# ---------------------------------------------------------------------------
# Measurement noise and wrap-around diagnostics
# ---------------------------------------------------------------------------
def add_measurement_noise(ascans: Tensor, snr_db: float, *,
                          generator: torch.Generator | None = None) -> Tensor:
    """
    White Gaussian noise at a per-sample SNR, added in the *time* domain (§9.3).

    Time domain rather than adding complex noise to the phasors, because that is
    where a real transducer's noise lives and because the DFT then colours it
    correctly: a flat time-domain spectrum becomes flat in frequency, and the
    deconvolution amplifies it by the 1/|s_hat| profile that
    `conditioning_report` prints.  Injecting flat noise on the phasors instead
    would quietly understate the difficulty at the band edges.

    The SNR is set from the RMS over the whole gather, not per trace, so weak
    receivers stay weak -- shadowed receivers carry real information about where
    the void is, and per-trace normalisation would erase it.
    """
    rms = ascans.pow(2).mean(dim=tuple(range(1, ascans.ndim)), keepdim=True).sqrt()
    sigma = rms / (10.0 ** (snr_db / 20.0))
    noise = torch.randn(ascans.shape, device=ascans.device, dtype=ascans.dtype,
                        generator=generator)
    return ascans + sigma * noise


def tail_energy_fraction(ascans: Tensor, tail: float = 0.1) -> Tensor:
    """
    Energy in the last `tail` of the record, as a fraction of the total, [B].

    The DFT is implicitly periodic, so anything still ringing at t_end folds back
    onto t = 0.  Should be < 1e-3 once the absorber is working; if it is not,
    every phasor is contaminated and no amount of network capacity will fix it.
    """
    nt = ascans.shape[-1]
    n0 = int(round((1.0 - tail) * nt))
    total = ascans.pow(2).sum(dim=tuple(range(1, ascans.ndim))).clamp_min(1e-30)
    late = ascans[..., n0:].pow(2).sum(dim=tuple(range(1, ascans.ndim)))
    return late / total


# ---------------------------------------------------------------------------
# Arrival times, for the §11.2 step 2 and step 7 gates
# ---------------------------------------------------------------------------
def envelope(x: Tensor) -> Tensor:
    """
    |analytic signal| along the last axis, via the standard FFT Hilbert route.

    For *time-domain traces* only: arrival picking in solver check 2, through
    `vector_envelope`, which is what callers with a vector trace should use.  The
    inversion's envelope misfit does not come through here.  It cannot: the
    inversion holds 20 phasors, not a trace, and synthesising a trace only to
    Hilbert-transform it back would make the objective depend on a zero-padding
    choice.  `inverse.timedomain.Reconstruction` builds the band-limited analytic
    signal directly from the phasors instead, which is exact, differentiable, and
    applied identically to prediction and observation.

    Note what this function is *not* a substitute for either: |u_hat| is invariant
    under time shift (`inverse.timedomain.shift_invariance_demo`), so a
    magnitude-spectrum comparison is travel-time blind, while |analytic signal| is
    shift-equivariant and is not.
    """
    nt = x.shape[-1]
    X = torch.fft.fft(x.to(torch.float64), dim=-1)
    h = torch.zeros(nt, device=x.device, dtype=torch.float64)
    h[0] = 1.0
    if nt % 2 == 0:
        h[nt // 2] = 1.0
        h[1:nt // 2] = 2.0
    else:
        h[1:(nt + 1) // 2] = 2.0
    analytic = torch.fft.ifft(X * h.to(X.dtype), dim=-1)
    return analytic.abs().to(x.dtype)


def vector_envelope(x: Tensor, *, dim: int = -2) -> Tensor:
    """
    Orientation-free envelope of a *vector* trace: sqrt(sum_c |analytic(x_c)|^2).

    `x` has its components along `dim` and time along the last axis; the component
    axis is reduced.  For [B, R, 2, nt] ascans the default is right.

    Take the analytic signal of each **signed** component and *then* the norm, never
    the other way round.  `envelope(sqrt(vx^2 + vy^2))` is a different quantity and
    it is a wrong one: the rectified magnitude is non-negative, so its spectrum sits
    at DC and 2 f_c rather than at f_c, and the Hilbert transform of that is not the
    modulating envelope of anything.  Measured on a synthetic two-phase trace with
    exactly known arrivals -- a 2-cycle Hann burst at t = d/c_p and another at
    d/c_s -- rectify-then-analytic misplaces both envelope peaks by 16 to 18 time
    steps, while this function lands within 0.5, which is grid quantisation.  Solver
    check 2 gates at 1.0 step, so the convention was most of its failure.

    The componentwise analytic signal is also the right object for a physical reason
    and not only a numerical one: it is linear, so it commutes with the rotation that
    takes (vx, vy) to (radial, transverse), and the answer therefore does not depend
    on which frame the receiver reports in.  `tests/test_dft_consistency.py` pins the
    peak against the burst's analytic group delay.
    """
    a = torch.stack([envelope(c) ** 2 for c in x.unbind(dim)], dim=dim)
    return a.sum(dim=dim).clamp_min(0.0).sqrt()


def first_arrival(ascans: Tensor, *, dt: float = cfg.DT,
                  threshold: float = 0.05) -> Tensor:
    """
    Time at which the envelope first exceeds `threshold` x its own peak, [B, R].

    A fractional-of-peak threshold rather than an absolute one because the
    amplitude varies by orders of magnitude between a receiver facing the source
    and one in the shadow, while the *shape* of the leading edge does not.
    Combines the two velocity components through `vector_envelope`, so the pick is
    orientation-free -- and, unlike the rectify-then-Hilbert version this used to
    do, actually an envelope.
    """
    env = vector_envelope(ascans)                       # [B, R, nt]
    peak = env.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    over = env >= threshold * peak
    idx = torch.where(over.any(dim=-1),
                      over.to(torch.int64).argmax(dim=-1),
                      torch.zeros_like(over[..., 0], dtype=torch.int64))
    return (idx.to(torch.float64) + 0.5) * dt


__all__ = [
    "MAX_DECONV_AMPLIFICATION",
    "add_measurement_noise",
    "assert_conditioned",
    "band_slice",
    "conditioning_report",
    "displacement_from_ascans",
    "displacement_from_field",
    "envelope",
    "first_arrival",
    "incident_scale",
    "incident_scale_at_receivers",
    "omegas_tensor",
    "scattered",
    "source_hat",
    "tail_energy_fraction",
    "transfer_factor",
    "vector_envelope",
]
