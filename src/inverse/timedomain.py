"""
Time-domain objectives reconstructed from the band-limited phasors (§8.4, v2.1).

Why this module exists
----------------------
Up to v2.0 the plan and the code treated a time-domain envelope misfit and a
frequency-magnitude misfit as interchangeable.  They are not, and the reason is one
line of Fourier algebra.  For a trace delayed by tau,

    g_tau(t) = g(t - tau)        =>      g_tau_hat(omega) = e^{-i omega tau} g_hat(omega)

so

    |g_tau_hat| = |g_hat|                                     (identity A)

*exactly*, for every tau.  A misfit built on magnitude spectra is therefore blind to
travel-time shifts -- not approximately blind, not blind up to a smoothing, but
identically blind.  The old docstring's claim that being "phase-free" supplies the
relevant envelope behaviour had it backwards: phase is precisely where the arrival
time lives, and discarding it discards the quantity the screen is trying to
localise.  `shift_invariance_demo` below turns identity A into an executable check
so the point cannot drift back into prose.

A time-domain analytic-signal envelope does not have this defect.  With the
reconstruction defined below, delaying the input phasors by tau translates the
reconstructed analytic signal by exactly tau (`shift_equivariance_demo`), so its
modulus moves with the packet and retains arrival-time information while still
being free of the carrier oscillation that causes cycle skipping.

What amplitude information is still worth: |u_s| across receivers and across
frequency does constrain position, because the scattered amplitude pattern is
geometry-dependent.  That is a real but *weaker* statement than "envelope basin",
and it is now made in those terms, with a measured capture rate
(`inverse.invert.screen_capture_rate`) instead of a guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .. import config as cfg

# ---------------------------------------------------------------------------
# The reconstruction, written down rather than implied (§5.3-§5.4, v2.1)
# ---------------------------------------------------------------------------
# Convention, identical to solver.harmonic:
#
#     X(w) = int x(t) e^{-i w t} dt,     x(t) = (1/2pi) int X(w) e^{+i w t} dw
#
# so d/dt <-> i w and u_hat = v_hat / (i w).  The analytic signal keeps positive
# frequencies and doubles them, and we hold samples of X only at the M = 20 band
# frequencies, so a rectangle rule at spacing dw = 2 pi df defines
#
#     z(t) = (dw / pi) sum_m  w_m  X(w_m)  e^{+i w_m t}                (definition R)
#
# with w_m a spectral taper.  Three properties of R matter, and "it recovers the
# true trace" is not one of them:
#
#   1. Band-limited by construction.  Twenty samples spanning 0.66-1.34 f_c carry
#      the in-band part of the burst and nothing else; the physical 5-cycle Hann
#      burst has out-of-band energy that R cannot contain.  `reconstruction_report`
#      measures the shortfall instead of calling it negligible.
#   2. Periodic with period 1/df = 27.9 T_p.  T_end = 24 T_p is shorter, which is
#      the entire reason config asserts df <= 1/T_end: later scattered energy would
#      alias onto the early window.
#   3. Exactly shift-equivariant.  Multiplying every X(w_m) by e^{-i w_m tau}
#      yields z(t - tau) with no approximation, because the factor comes out of the
#      sum.  This is what makes |z| a legitimate envelope and the reason a
#      magnitude-spectrum misfit is not one.
#
# R is applied identically to prediction and to observation, so the comparison is
# between two consistently band-limited reconstructions.  That is well posed even
# though neither reconstruction equals the physical trace -- and it is a different,
# weaker claim than "we compare envelopes of the measured waveforms".

TAPERS: tuple[str, ...] = ("hann", "none")


def spectral_taper(m_freq: int = cfg.M_FREQ, kind: str = "hann", *,
                   device=None, dtype: torch.dtype = torch.float64) -> Tensor:
    """
    Per-frequency weight w_m in definition R.  [M]

    "none" is the rectangle rule.  Its transfer function is a Dirichlet kernel with
    -13 dB sidelobes decaying like 1/|t|, so a hard band edge rings for many carrier
    periods and the envelope acquires precursors that look like early arrivals.
    "hann" tapers the band edges to (but not through) zero -- w_m = sin^2(pi (m+1) /
    (M+1)) -- trading a 1.5x wider main lobe for sidelobes that fall like 1/|t|^3.
    The main lobe is what sets envelope resolution and the sidelobes are what create
    spurious arrivals, so on this problem the trade is worth making; `kind` is
    exposed so the choice can be measured rather than argued.
    """
    if kind not in TAPERS:
        raise ValueError(f"taper must be one of {TAPERS}, got {kind!r}")
    m = torch.arange(m_freq, device=device, dtype=dtype)
    if kind == "none":
        return torch.ones_like(m)
    return torch.sin(math.pi * (m + 1.0) / (m_freq + 1.0)) ** 2


@dataclass
class Reconstruction:
    """
    Definition R, frozen into an object so a reported envelope misfit is traceable
    to the band, taper and time sampling that produced it.

    band     which prefix of cfg.FREQS is in play (the continuation stages use 6,
             10 and 20 samples, and the reconstruction is *different* in each --
             a 6-sample reconstruction has a main lobe 3.3x wider, which is exactly
             the smoothing that makes an early stage forgiving)
    n_t      time samples over one period 1/df.  128 is not a resolution choice:
             with at most 20 nonzero spectral components and a sample rate of
             128 df = 4.58 f_c against a highest frequency of 1.34 f_c, R is
             *exactly* representable on this grid and no aliasing occurs.
    """
    band: slice = cfg.BAND_STAGE3
    n_t: int = 128
    taper: str = "hann"
    device: torch.device | None = None
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        f = torch.tensor(cfg.FREQS[self.band], device=self.device, dtype=self.dtype)
        self._freqs = f
        self._w = spectral_taper(len(f), self.taper, device=self.device,
                                dtype=self.dtype)
        # one period of the reconstruction; df is the *nominal* spacing, so this is
        # the same window at every stage and traces from different bands are
        # directly comparable
        self.period = 1.0 / cfg.DF
        self.dt = self.period / self.n_t
        self._t = torch.arange(self.n_t, device=self.device,
                               dtype=self.dtype) * self.dt
        self._basis: dict[torch.dtype, Tensor] = {}

    @property
    def times(self) -> Tensor:
        """[n_t] reconstruction times, in carrier periods."""
        return self._t

    @property
    def freqs(self) -> Tensor:
        return self._freqs

    def basis(self, cdtype: torch.dtype) -> Tensor:
        """
        [M, n_t] complex: (dw/pi) w_m exp(+i w_m t).  Cached per output dtype.

        Cached because the inverse loop calls this once per objective evaluation --
        of order 10^4 times per inversion -- and rebuilding a 20 x 128 complex
        exponential each time is pure overhead.  Built in the Reconstruction's own
        (double) precision and cast down, so a float32 inversion still gets a basis
        whose phases were computed exactly.
        """
        b = self._basis.get(cdtype)
        if b is None:
            dw = 2.0 * math.pi * cfg.DF
            ph = 2.0 * math.pi * self._freqs.view(-1, 1) * self._t.view(1, -1)
            b = ((dw / math.pi) * torch.polar(self._w.view(-1, 1).expand_as(ph), ph)
                 ).to(cdtype)
            self._basis[cdtype] = b
        return b

    def analytic(self, phasors: Tensor) -> Tensor:
        """
        [..., M] complex phasors -> [..., n_t] complex analytic signal.

        The frequency axis must be last and must match `band`; that is the layout
        `SurrogateForward.predict` and `InverseCase.d_obs` already use, so no
        permutation happens here and none can be got wrong.
        """
        assert phasors.is_complex(), "definition R consumes phasors, not traces"
        assert phasors.shape[-1] == self._freqs.numel(), (
            f"got {phasors.shape[-1]} frequencies, band has {self._freqs.numel()}")
        b = self.basis(phasors.dtype)
        return phasors.to(b.dtype) @ b

    def envelope(self, phasors: Tensor, *, eps: float = 1e-30) -> Tensor:
        """|z(t)|, the analytic-signal envelope.  [..., n_t] real."""
        z = self.analytic(phasors)
        return (z.real.pow(2) + z.imag.pow(2) + eps).sqrt()

    def trace(self, phasors: Tensor) -> Tensor:
        """Re z(t), the band-limited real trace.  [..., n_t] real."""
        return self.analytic(phasors).real

    def delay(self, phasors: Tensor, tau: float | Tensor) -> Tensor:
        """
        Apply a pure time shift in the phasor domain: X -> e^{-i w tau} X.

        Exact, not approximate, and the operation both shift demonstrations below
        are built on.
        """
        w = 2.0 * math.pi * self._freqs
        t = torch.as_tensor(tau, device=w.device, dtype=w.dtype)
        ph = -w * t.unsqueeze(-1) if t.dim() else -w * t
        return phasors * torch.polar(torch.ones_like(ph), ph).to(phasors.dtype)


_RECON_CACHE: dict = {}


def reconstruction(band: slice = cfg.BAND_STAGE3, *, n_t: int = 128,
                   taper: str = "hann", device=None,
                   dtype: torch.dtype = torch.float64) -> Reconstruction:
    """Cached Reconstruction; the basis is a [M, n_t] matrix worth reusing."""
    key = (band.start, band.stop, band.step, n_t, taper, str(device), dtype)
    r = _RECON_CACHE.get(key)
    if r is None:
        r = Reconstruction(band=band, n_t=n_t, taper=taper, device=device,
                           dtype=dtype)
        _RECON_CACHE[key] = r
    return r


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def _trace_weights(obs_env: Tensor, eps: float) -> Tensor:
    """
    Per-(receiver, component) weights proportional to observed energy.  [B,R,2,1]

    Unweighted travel-time picking on a near-silent trace returns the lag of the
    noise, which is uniform on the search window and therefore contributes a large
    spurious squared time to the objective.  Energy weighting is not cosmetic here:
    on this ring the shadow-side receivers are 20-30 dB down on the illuminated
    ones, so roughly half the traces would otherwise be pure noise votes.
    """
    e = obs_env.pow(2).sum(dim=-1, keepdim=True)
    return e / e.sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)


def envelope_misfit(pred: Tensor, obs: Tensor, *, recon: Reconstruction | None = None,
                    band: slice = cfg.BAND_STAGE1, scale_invariant: bool = False,
                    eps: float = 1e-30) -> Tensor:
    """
    Relative squared error between analytic-signal envelopes.  [B]

    pred [B,R,2,F], obs [1 or B,R,2,F] complex, same units, F matching `band`.

    This is the objective §8.4 always meant.  Unlike the magnitude-spectrum screen
    it *is* sensitive to travel time (identity A does not apply to |z(t)|), and
    unlike the complex misfit it has no carrier in it, so a half-period error costs
    almost nothing and cycle skipping is suppressed rather than merely hidden.

    When `scale_invariant` is true, fit the nonnegative least-squares amplitude of
    each predicted envelope to the observed envelope before scoring.  The screen
    uses this because it evaluates a fixed trial radius while the true radius varies;
    the final amplitude-sensitive stages leave it false so size information remains.

    The basin is wider than the waveform basin by roughly the ratio of the envelope
    width to the carrier period -- an empirical factor to be *measured* by
    `misfit.envelope_basin_width`, not the N_c/2 factor v2.0 asserted.  No claim of
    global monotonicity is made or intended: elastic multipath and P-to-S conversion
    give the envelope its own secondary maxima.
    """
    r = recon or reconstruction(band, device=pred.device)
    ep, eo = r.envelope(pred), r.envelope(obs)
    eo = eo.expand_as(ep)
    if scale_invariant:
        dims = (1, 2, 3)
        amp = ((ep * eo).sum(dims, keepdim=True)
               / ep.pow(2).sum(dims, keepdim=True).clamp_min(eps))
        ep = amp.clamp_min(0.0) * ep
    num = (ep - eo).pow(2).sum(dim=(1, 2, 3))
    den = eo.pow(2).sum(dim=(1, 2, 3)).clamp_min(eps)
    return num / den


def _correlation(pred: Tensor, obs: Tensor, r: Reconstruction, *,
                 max_lag: float, eps: float
                 ) -> tuple[Tensor, Tensor, Tensor]:
    """
    Circular normalised cross-correlation of the band-limited real traces.

    Returns (c_hat [B,R,2,n_t] in [-1,1], lag [n_t] signed and in carrier periods,
    mask [n_t] bool for |lag| <= max_lag).

    Circular is correct rather than convenient: definition R is *genuinely* periodic
    with period 1/df, so a wrap-around correlation is the exact correlation of the
    signals being compared.  A linear (zero-padded) correlation would instead be the
    exact correlation of two different signals -- R truncated -- and would taper the
    coefficient towards zero at large lag, biasing the pick towards small shifts.
    """
    p = r.trace(pred)
    o = r.trace(obs).expand_as(p)
    n = p.shape[-1]
    P, O = torch.fft.rfft(p, dim=-1), torch.fft.rfft(o, dim=-1)
    c = torch.fft.irfft(P * O.conj(), n=n, dim=-1)      # c[l] = sum_t p[t] o[t-l]
    nrm = (p.pow(2).sum(-1, keepdim=True) * o.pow(2).sum(-1, keepdim=True)
           ).clamp_min(eps).sqrt()
    idx = torch.arange(n, device=p.device)
    lag = (((idx + n // 2) % n) - n // 2).to(p.dtype) * r.dt
    return c / nrm, lag, lag.abs() <= max_lag


def traveltime_shift(pred: Tensor, obs: Tensor, *,
                     recon: Reconstruction | None = None,
                     band: slice = cfg.BAND_STAGE1,
                     max_lag: float | None = None, beta: float = 40.0,
                     eps: float = 1e-30) -> tuple[Tensor, Tensor]:
    """
    Differentiable travel-time shift per trace.  ([B,R,2] tau, [B,R,2,1] weights)

    tau > 0 means the prediction arrives *late*.  The pick is a soft-argmax --
    tau = sum_l softmax(beta c_hat)_l lag_l -- because a hard argmax has zero
    gradient almost everywhere and an undefined one where the winning lag changes,
    and the whole purpose of this objective is to be differentiated.

    `max_lag` bounds the search, default N_c / f_c = one burst length.  The bound is
    load-bearing: over the full 27.9 T_p period the correlation of a 5-cycle burst
    has of order 28 near-equal local maxima one carrier period apart, and a
    soft-argmax over all of them returns their centroid, which is neither a travel
    time nor a useful gradient.  Restricting to one burst length keeps the pick
    inside a single packet, at the cost of being unable to see a shift larger than
    that -- which is what the amplitude screen and the envelope stage are for.

    beta = 40 on a coefficient in [-1, 1] gives an effective window of about
    1/40 = 0.025 in correlation, i.e. the lags within a few percent of the peak.
    """
    r = recon or reconstruction(band, device=pred.device)
    ml = max_lag if max_lag is not None else cfg.N_CYCLES / cfg.FC
    c, lag, mask = _correlation(pred, obs, r, max_lag=ml, eps=eps)
    s = torch.softmax(beta * c.masked_fill(~mask, -float("inf")), dim=-1)
    tau = (s * lag).sum(-1)
    w = _trace_weights(r.envelope(obs), eps).squeeze(-1)        # [1 or B, R, 2]
    return tau, w


def traveltime_misfit(pred: Tensor, obs: Tensor, *,
                      recon: Reconstruction | None = None,
                      band: slice = cfg.BAND_STAGE1,
                      max_lag: float | None = None, beta: float = 40.0,
                      eps: float = 1e-30) -> Tensor:
    """
    Energy-weighted mean squared travel-time error, in carrier periods.  [B]

        J_tt(theta) = sum_{r,c} w_rc (tau_rc(theta) / T_p)^2,   sum w_rc = 1

    Dimensionless and O(1) when the error is one carrier period, so it is directly
    comparable to the relative waveform misfit and a single Tikhonov weight serves
    both.  This is the "explicitly defined travel-time objective" of the review's
    alternative: it measures arrival time and nothing else, which makes it immune to
    amplitude errors in the surrogate but also blind to radius, so it is a *position*
    objective and is only ever used as one.
    """
    tau, w = traveltime_shift(pred, obs, recon=recon, band=band, max_lag=max_lag,
                              beta=beta, eps=eps)
    return (w * (tau / cfg.T_P).pow(2)).sum(dim=(1, 2))


def correlation_misfit(pred: Tensor, obs: Tensor, *,
                       recon: Reconstruction | None = None,
                       band: slice = cfg.BAND_STAGE1,
                       max_lag: float | None = None, beta: float = 40.0,
                       eps: float = 1e-30) -> Tensor:
    """
    Energy-weighted mean of (1 - peak normalised correlation).  [B]

    The travel-time misfit's complement: it ignores *when* the best alignment occurs
    and scores how well the shapes match once aligned, so it is sensitive to radius
    and to waveform distortion but not to position.  Used together with
    `traveltime_misfit` these two decompose the complex misfit into the part that
    cycle-skips and the part that does not, which is why both are exposed instead of
    only their sum.

    The peak is taken with a log-sum-exp soft maximum over the same lag window as
    the travel-time pick, so the two objectives cannot disagree about which lag they
    are talking about.
    """
    r = recon or reconstruction(band, device=pred.device)
    ml = max_lag if max_lag is not None else cfg.N_CYCLES / cfg.FC
    c, _, mask = _correlation(pred, obs, r, max_lag=ml, eps=eps)
    peak = torch.logsumexp(beta * c.masked_fill(~mask, -float("inf")), dim=-1) / beta
    w = _trace_weights(r.envelope(obs), eps).squeeze(-1)
    return (w * (1.0 - peak).clamp_min(0.0)).sum(dim=(1, 2))


# ---------------------------------------------------------------------------
# Making the review's two claims executable
# ---------------------------------------------------------------------------
@torch.no_grad()
def shift_invariance_demo(phasors: Tensor, taus: Sequence[float] = (0.1, 0.5, 1.0),
                          *, band: slice = cfg.BAND_STAGE1) -> dict:
    """
    Identity A, measured: a magnitude spectrum cannot see a time shift.

    Returns, for each tau, the largest relative change in |X| (which is zero to
    round-off) beside the envelope misfit the same shift produces (which is not).
    The ratio is the whole argument for replacing the screening objective, and it is
    reported rather than asserted so the numbers appear in the notebook.
    """
    r = reconstruction(band, device=phasors.device)
    x = phasors[..., band] if phasors.shape[-1] == cfg.M_FREQ else phasors
    mag = x.abs()
    rows = []
    for tau in taus:
        xd = r.delay(x, float(tau))
        d_mag = (xd.abs() - mag).abs().max() / mag.abs().max().clamp_min(1e-30)
        d_env = float(envelope_misfit(xd, x, recon=r)[0])
        tt, _ = traveltime_shift(xd, x, recon=r)
        rows.append(dict(tau=float(tau), rel_change_in_magnitude=float(d_mag),
                         envelope_misfit=d_env,
                         traveltime_pick=float(tt.abs().max())))
    return dict(band=(band.start, band.stop), rows=rows)


@torch.no_grad()
def shift_equivariance_demo(phasors: Tensor, n_shift: int = 3, *,
                            band: slice = cfg.BAND_STAGE1) -> dict:
    """
    The complementary fact: |z(t)| *does* move with the packet, exactly.

    Delaying the phasors by an integer number of reconstruction samples must equal
    rolling the envelope by the same number of samples, to round-off.  If this ever
    fails, definition R has picked up a taper or a normalisation that depends on
    absolute time and the envelope is no longer an envelope.
    """
    r = reconstruction(band, device=phasors.device)
    x = phasors[..., band] if phasors.shape[-1] == cfg.M_FREQ else phasors
    env = r.envelope(x)
    shifted = r.envelope(r.delay(x, n_shift * r.dt))
    rolled = torch.roll(env, shifts=n_shift, dims=-1)
    err = (shifted - rolled).abs().max() / env.abs().max().clamp_min(1e-30)
    return dict(n_shift=n_shift, dt=r.dt, max_relative_error=float(err))


# ---------------------------------------------------------------------------
# How good is the reconstruction?  (§5.3-§5.4, the number v2.0 never gave)
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruction_report(band: slice = cfg.BAND_STAGE3, *, n_t: int = 128,
                          taper: str = "hann") -> dict:
    """
    Round-trip the source burst through definition R and measure what is lost.

    The burst is the one signal in the problem whose exact form is known, so it is
    the honest test of the reconstruction: take its phasors with the solver's own
    quadrature, run them back through R, and compare against the analytic burst
    evaluated at the same times.  Reported quantities:

      rel_l2                relative L2 error of Re z against the true burst
      energy_in_band        fraction of the burst's energy the band contains
      peak_time_error       envelope-peak time minus the true envelope-peak time,
                            in carrier periods -- a bias here would be a systematic
                            travel-time error, i.e. a position bias
      psf_width_3db         -3 dB width of R's point-spread function, in carrier
                            periods.  This is the *measured* envelope resolution
                            that v2.0 asserted as N_c/2 = 2.5 periods.
      psf_sidelobe_db       peak sidelobe of the same kernel, in dB

    Deliberately a report and not an assertion: the numbers depend on the band and
    the taper, and the point of exposing them is to choose those, not to pass.
    """
    from ..solver.fdtd_elastic import dft_at_freqs, tone_burst

    r = reconstruction(band, n_t=n_t, taper=taper)
    t_fine = (torch.arange(cfg.NT, dtype=torch.float64) + 0.5) * cfg.DT
    s = tone_burst(t_fine)
    om = 2.0 * math.pi * r.freqs
    s_hat = dft_at_freqs(s.unsqueeze(0), om, cfg.DT, cfg.NT).squeeze(0)

    z = r.analytic(s_hat)
    truth = tone_burst(r.times)
    num = (z.real - truth).pow(2).sum().sqrt()
    den = truth.pow(2).sum().sqrt().clamp_min(1e-30)

    # Parseval, in this module's convention: for real x,
    #   int |x|^2 dt = (1/pi) int_0^inf |X|^2 dw,  and dw = 2 pi df
    # so the one-sided rectangle rule for the band's share is 2 df sum_m |X_m|^2.
    e_tot = s.pow(2).sum() * cfg.DT
    e_band = 2.0 * cfg.DF * s_hat.abs().pow(2).sum()

    # the true analytic envelope of a Hann-windowed burst is the window itself
    dur = cfg.N_CYCLES / cfg.FC
    win = 0.5 * (1.0 - torch.cos(2.0 * math.pi * cfg.FC * r.times / cfg.N_CYCLES))
    env_true = torch.where((r.times >= 0) & (r.times <= dur), win,
                           torch.zeros_like(win))
    env = z.abs()
    scale = env.max().clamp_min(1e-30) / env_true.max().clamp_min(1e-30)
    env_err = ((env - scale * env_true).pow(2).sum().sqrt()
               / (scale * env_true).pow(2).sum().sqrt().clamp_min(1e-30))
    t_peak = float(r.times[int(env.argmax())])
    t_peak_true = 0.5 * dur                      # the Hann window is symmetric

    # point-spread function: R applied to a flat spectrum
    k = r.analytic(torch.ones_like(s_hat)).abs()
    k = k / k.max()
    half = (k >= 10 ** (-3.0 / 20.0)).to(torch.float64).sum() * r.dt
    # sidelobes: everything outside the first zero-crossing region around t = 0
    centre = int(k.argmax())
    guard = max(1, int(round(0.5 * cfg.N_CYCLES / cfg.FC / r.dt)))
    mask = torch.ones_like(k, dtype=torch.bool)
    idx = (torch.arange(k.numel()) - centre) % k.numel()
    mask[(idx <= guard) | (idx >= k.numel() - guard)] = False
    side = float(k[mask].max()) if bool(mask.any()) else 0.0

    return dict(
        band=(band.start, band.stop), n_freq=int(r.freqs.numel()), taper=taper,
        n_t=n_t, dt=r.dt, period=r.period,
        rel_l2=float(num / den),
        envelope_rel_l2=float(env_err),
        energy_in_band=float(e_band / e_tot.clamp_min(1e-30)),
        peak_time_error=t_peak - t_peak_true,
        psf_width_3db=float(half),
        psf_sidelobe_db=(20.0 * math.log10(side) if side > 0 else float("-inf")),
    )


__all__ = [
    "Reconstruction",
    "correlation_misfit",
    "envelope_misfit",
    "reconstruction",
    "reconstruction_report",
    "shift_equivariance_demo",
    "shift_invariance_demo",
    "spectral_taper",
    "traveltime_misfit",
    "traveltime_shift",
]

