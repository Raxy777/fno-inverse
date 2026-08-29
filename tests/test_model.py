"""
The operator itself.

Four things in here are load-bearing.

`test_masked_modes_are_zero_at_init_and_stay_dead_in_the_forward` is the test the
module docstring of fno2d.py promises.  The radial truncation is applied by
multiplying the weights *inside* `forward`; if someone "optimises" that away and
applies the mask once at construction, the discarded coefficients become ordinary
live parameters at the first optimiser step and the truncation silently stops being a
truncation.  Nothing else would notice: the loss would go down slightly faster, the
saved checkpoint would still load, and the isotropy argument of §6.4 would quietly be
false.  So the test perturbs the masked entries to a huge value after construction
and demands that the output not move.

`test_spectral_conv_is_exactly_discretisation_invariant` is the §4.3 claim, in the
one setting where it is exactly true rather than approximately: a band-limited input
through the spectral convolution alone.  With `norm="ortho"` on both transforms the
composed map is y = sum_k R_k c_k e^{ikx}, in which the grid appears nowhere -- so at
two resolutions the outputs must agree to floating-point at coincident points, not to
a tolerance.  Everything approximate about the full network's invariance comes from
GELU generating content above the coarse grid's Nyquist, and that is measured
separately and honestly in the test below it.

`test_spectral_conv_row_layout_is_signed_wavenumber` pins the index convention that
`SpectralConv2d`'s docstring states in words: row iy of the `[-kmax:]` block is
k_y = -(kmax - iy), *not* k_y = -iy.  Getting it backwards leaves every shape and
count intact and reverses the y direction of the learned kernel.

`test_to_double_converts_complex_parameters` guards the reason `to_double` exists at
all.  `nn.Module.double()` skips complex tensors, so the spectral weights stay
complex64 while everything around them becomes float64, and the gradient check of
§11.2 step 9 would then fail for a precision reason -- the worst kind of failure,
because the response is to relax the tolerance.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src import config as cfg
from src.geometry.sdf import Circle
from src.models.cnn_regressor import RING_CHANNELS, RingCNN, pack_ring
from src.models.fno2d import (FNO2d, FourierBlock, SpectralConv2d, band_in_modes,
                              build, to_double)


# ---------------------------------------------------------------------------
# Parameter counts: the model against the sizing table
# ---------------------------------------------------------------------------
def test_effective_params_matches_the_sizing_table():
    """
    `config.total_params` is what the document quotes; `FNO2d.effective_params` is
    what the optimiser actually sees.  They are computed by completely different
    routes -- one counts lattice points in a loop, the other sums a buffer -- so
    agreement is evidence, not tautology.
    """
    net = build()
    assert net.effective_params() == cfg.total_params() == 10_276_452
    assert "MISMATCH" not in net.summary()

    layers = [m for m in net.modules() if isinstance(m, SpectralConv2d)]
    assert len(layers) == cfg.N_BLOCKS == 4
    for sc in layers:
        assert sc.effective_params() == cfg.spectral_params(cfg.D_V, cfg.KMAX, True)
        assert sc.effective_params() == 2_566_144
        # The two blocks are not symmetric: w1 owns the k_y = 0 row and w2 does not.
        assert int(sc.m1.sum().item()) == 640
        assert int(sc.m2.sum().item()) == 613


def test_allocated_exceeds_effective_by_exactly_the_masked_modes():
    """
    The masked coefficients still occupy memory and still land in the checkpoint;
    only their gradient is dead.  Reporting both numbers is honesty about the file
    size, and the difference is exactly the square-minus-radial count.
    """
    net = build()
    alloc, eff = net.allocated_params(), net.effective_params()
    assert alloc == 12_856_932
    per_layer_gap = (cfg.spectral_params(cfg.D_V, cfg.KMAX, False)
                     - cfg.spectral_params(cfg.D_V, cfg.KMAX, True))
    assert alloc - eff == cfg.N_BLOCKS * per_layer_gap == 4 * 645_120


def test_square_mask_keeps_the_whole_block():
    sc_sq = SpectralConv2d(4, 4, kmax=6, radial=False)
    sc_ra = SpectralConv2d(4, 4, kmax=6, radial=True)
    assert sc_sq.m1.shape == (1, 1, 6, 6)
    assert sc_sq.m1.min().item() == 1.0 and sc_sq.m2.min().item() == 1.0
    assert sc_sq.effective_params() == cfg.spectral_params(4, 6, False) == 2_304
    assert sc_ra.effective_params() == cfg.spectral_params(4, 6, True)
    assert sc_ra.effective_params() < sc_sq.effective_params()


def test_variant_capacities_are_ordered_and_buildable():
    counts = {}
    for name in cfg.VARIANTS:
        net = build(name)
        counts[name] = net.effective_params()
        assert net.effective_params() == cfg.total_params(**cfg.VARIANTS[name])
    assert counts["primary"] > counts["small"] > counts["tiny"]
    with pytest.raises(AssertionError, match="unknown variant"):
        build("enormous")


# ---------------------------------------------------------------------------
# The mask, and whether it is actually a truncation
# ---------------------------------------------------------------------------
def test_masked_modes_are_zero_at_init_and_stay_dead_in_the_forward():
    sc = SpectralConv2d(4, 4, kmax=6)
    off1 = (sc.m1 == 0).expand_as(sc.w1)
    off2 = (sc.m2 == 0).expand_as(sc.w2)
    assert off1.any() and off2.any(), "kmax=6 must have corner modes to discard"

    # Zeroed at init, so the parameter file and effective_params() agree.
    assert sc.w1.detach()[off1].abs().max().item() == 0.0
    assert sc.w2.detach()[off2].abs().max().item() == 0.0

    x = torch.randn(1, 4, 32, 32)
    y0 = sc(x).detach().clone()

    # Now make the discarded coefficients enormous.  If the mask were applied only at
    # construction this would change the output by three orders of magnitude.
    with torch.no_grad():
        sc.w1.add_((sc.m1 == 0).to(sc.w1.dtype) * complex(1e3, -1e3))
        sc.w2.add_((sc.m2 == 0).to(sc.w2.dtype) * complex(1e3, -1e3))
    assert sc.w1.detach()[off1].abs().min().item() > 1e2
    assert torch.allclose(y0, sc(x), atol=1e-6)

    # ... whereas a *kept* coefficient moves the output, so the test above is not
    # passing because the layer is inert.
    with torch.no_grad():
        sc.w1[:, :, 0, 0] += 1.0
    assert not torch.allclose(y0, sc(x), atol=1e-6)


def test_masked_modes_receive_exactly_zero_gradient():
    """
    The optimiser's view of the truncation.  Multiplying by the mask in the forward
    makes d/dw of a masked entry identically zero -- not small, zero -- so weight
    decay is the only thing that could ever move them, and they start at zero.

    The loss is a random projection of the output rather than its sum.  A plain
    `.sum()` would be a trap: summing a real field picks out its DC coefficient alone,
    so the gradient would be nonzero at exactly one weight entry and the test would
    pass whether or not the mask did anything.
    """
    sc = SpectralConv2d(3, 3, kmax=6)
    y = sc(torch.randn(2, 3, 32, 32))
    (y * torch.randn_like(y)).sum().backward()

    off1 = (sc.m1 == 0).expand_as(sc.w1)
    off2 = (sc.m2 == 0).expand_as(sc.w2)
    assert sc.w1.grad is not None and sc.w2.grad is not None
    assert sc.w1.grad[off1].abs().max().item() == 0.0
    assert sc.w2.grad[off2].abs().max().item() == 0.0
    # Not vacuous: the kept modes do move.  Checked on both blocks, since w2 is the
    # one whose row indexing is easy to get wrong.
    assert sc.w1.grad[~off1].abs().min().item() > 0.0
    assert sc.w2.grad[~off2].abs().min().item() > 0.0


def test_radial_mask_geometry_at_the_production_kmax():
    """
    The lattice counts behind the pi/4 claim, read off the buffers rather than
    recomputed.  m2's k_y = -kmax row keeps exactly one mode (k_x = 0), because
    784 = 28^2 is not a sum of two nonzero squares -- which is also why no float
    comparison in `_masks` is ambiguous: nothing sits infinitesimally off the circle.
    """
    sc = SpectralConv2d(1, 1, kmax=cfg.KMAX)
    k = cfg.KMAX
    assert int(sc.m1[0, 0, 0].sum().item()) == k        # k_y = 0: whole row kept
    assert int(sc.m2[0, 0, 0].sum().item()) == 1        # k_y = -kmax: only k_x = 0
    assert sc.m2[0, 0, 0, 0].item() == 1.0
    # m1 row iy and m2 row iy describe |k_y| = iy and kmax - iy respectively, so the
    # two blocks together cover |k_y| = 0 .. kmax with kmax + 1 distinct rows.
    assert int((sc.m1.sum() + sc.m2.sum()).item()) == 1253
    frac = 1253 / (2 * k * k)
    assert frac == pytest.approx(0.79911, abs=1e-5)
    assert frac > math.pi / 4.0        # lattice boundary rounds the disc outward


# ---------------------------------------------------------------------------
# Index layout
# ---------------------------------------------------------------------------
def _single_mode_field(n: int, ky: int, kx: int) -> torch.Tensor:
    """
    A real [1, 1, n, n] field whose rfft2 is a single coefficient at (ky, kx).

    k_x is required positive so that the coefficient has no self-conjugacy
    constraint: `irfft2` supplies the (-ky, -kx) partner and the result is exactly
    real, so `rfft2` of it returns the spectrum that was asked for.
    """
    assert kx > 0, "k_x = 0 would need its own conjugate at -k_y"
    Y = torch.zeros(1, 1, n, n // 2 + 1, dtype=torch.complex128)
    Y[0, 0, ky % n, kx] = 1.0 + 0.0j
    return torch.fft.irfft2(Y, s=(n, n), norm="ortho")


def test_spectral_conv_row_layout_is_signed_wavenumber():
    """
    Row iy of the [-kmax:] block is k_y = -(kmax - iy).  Probed by giving the layer a
    single-mode input at k_y = -2 and checking which weight entry it responds to.
    """
    n, kmax = 24, 6
    x = _single_mode_field(n, ky=-2, kx=3)
    sc = to_double(SpectralConv2d(1, 1, kmax=kmax))

    def response(block: str, iy: int, ix: int) -> float:
        with torch.no_grad():
            sc.w1.zero_()
            sc.w2.zero_()
            getattr(sc, block)[0, 0, iy, ix] = 1.0
        return sc(x).abs().max().item()

    # k_y = -2 lives in w2 at slice row kmax - 2 = 4.
    assert response("w2", kmax - 2, 3) > 1e-3
    # The naive reading (row = |k_y|) would put it at row 2, which is k_y = -4.
    assert response("w2", 2, 3) < 1e-12
    # And w1 handles k_y = +2 only, so the same row index there sees nothing.
    assert response("w1", 2, 3) < 1e-12
    # Sanity: w1 row 2 does respond to the mirror-image input at k_y = +2.
    with torch.no_grad():
        sc.w1.zero_()
        sc.w2.zero_()
        sc.w1[0, 0, 2, 3] = 1.0
    assert sc(_single_mode_field(n, ky=2, kx=3)).abs().max().item() > 1e-3


def test_forward_rejects_a_grid_that_cannot_hold_the_band():
    """
    2 kmax <= ny is not a convenience check.  Below it the [0:kmax] and [-kmax:] row
    slices overlap, so some modes would be written twice with different weights and
    the layer would silently stop being a Fourier multiplier.
    """
    net = FNO2d(d_v=8, kmax=6)
    with pytest.raises(AssertionError, match="does not fit"):
        net(torch.randn(1, cfg.C_IN, 10, 10))
    net(torch.randn(1, cfg.C_IN, 12, 12))          # exactly 2 kmax: allowed


def test_forward_rejects_the_wrong_channel_count():
    net = FNO2d(d_v=8, kmax=6)
    with pytest.raises(AssertionError, match="pack_inputs"):
        net(torch.randn(1, 3, 32, 32))


# ---------------------------------------------------------------------------
# Discretisation invariance (section 4.3)
# ---------------------------------------------------------------------------
def _band_limited_pair(n: int, factor: int, c: int, *, gen,
                       modes=((0, 1), (1, 2), (-1, 1), (2, 3), (-2, 2), (-3, 1))
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One band-limited real field, sampled on an n-grid and on a (factor*n)-grid.

    Built in the spectral domain rather than by evaluating a formula at cell centres,
    because the FFT's implied sample points are x_j = j L / N and the cell centres
    used everywhere else in this codebase are offset by half a cell.  Sampling a
    formula at cell centres would put a resolution-dependent phase e^{-i pi k / N}
    into the spectrum and the two grids would disagree for a reason that has nothing
    to do with the operator.

    With norm="ortho" the coefficient of a fixed continuous mode scales as sqrt(N),
    so the fine spectrum carries the same values multiplied by `factor`.
    """
    vals = torch.randn(c, len(modes), 2, generator=gen, dtype=torch.float64)

    def spectrum(N: int, scale: float) -> torch.Tensor:
        Y = torch.zeros(1, c, N, N // 2 + 1, dtype=torch.complex128)
        for j, (ky, kx) in enumerate(modes):
            Y[0, :, ky % N, kx] = torch.complex(vals[:, j, 0], vals[:, j, 1]) * scale
        return Y

    xc = torch.fft.irfft2(spectrum(n, 1.0), s=(n, n), norm="ortho")
    m = factor * n
    xf = torch.fft.irfft2(spectrum(m, float(factor)), s=(m, m), norm="ortho")
    return xc, xf


def test_band_limited_pair_helper_is_self_consistent(gen):
    """The helper's own precondition: the fine field restricted to coincident points
    is the coarse field, exactly.  If this fails the invariance tests below are
    measuring the helper, not the operator."""
    xc, xf = _band_limited_pair(24, 2, 3, gen=gen)
    assert xc.shape == (1, 3, 24, 24) and xf.shape == (1, 3, 48, 48)
    assert torch.allclose(xf[..., ::2, ::2], xc, atol=1e-12)
    assert xc.abs().max() > 1e-3


def test_spectral_conv_is_exactly_discretisation_invariant(gen):
    """
    The same weights, two grids, agreement to floating point.

    This is the strong form of §4.3 and it holds because ortho normalisation cancels
    the grid size out of the composed transform: y(x) = sum_k R_k c_k e^{ikx} refers
    to no grid at all.  The tolerance is 1e-11 in float64, i.e. round-off, not
    modelling error -- there is no approximation being made here.
    """
    n, factor, c = 24, 2, 3
    xc, xf = _band_limited_pair(n, factor, c, gen=gen)
    sc = to_double(SpectralConv2d(c, c, kmax=6))
    with torch.no_grad():
        yc, yf = sc(xc), sc(xf)
    assert yc.abs().max().item() > 1e-4, "the probe modes were all masked off"
    assert torch.allclose(yf[..., ::factor, ::factor], yc, atol=1e-11)


def test_full_network_is_approximately_discretisation_invariant(gen):
    """
    The same statement for the whole operator, where it is only approximate.

    GELU turns a band-limited field into one that is not, and the content it creates
    above the coarse grid's Nyquist frequency folds back differently at the two
    resolutions.  That error is real and there is no honest exact tolerance to assert,
    so the claim is made comparatively: the two resolutions must agree far better than
    two *different* inputs do.  A network that had accidentally become
    resolution-dependent -- through a hardcoded grid size, a coordinate channel built
    at the wrong n, or a mask indexed by fraction of Nyquist rather than by absolute
    mode number -- would fail this by orders of magnitude.

    Note the caveat worth stating out loud in a viva: the real input includes the
    x_norm / y_norm ramps of §6.2, which are not band-limited on a periodic grid, so
    the *deployed* invariance is weaker than what is measured here.  It is measured
    here on band-limited inputs because that isolates the operator from the input
    representation, which is the thing being claimed.
    """
    n, factor = 32, 2
    xc, xf = _band_limited_pair(n, factor, cfg.C_IN, gen=gen)
    other, _ = _band_limited_pair(n, factor, cfg.C_IN, gen=gen)
    net = to_double(FNO2d(d_v=8, kmax=6)).eval()

    with torch.no_grad():
        yc, yf, yo = net(xc), net(xf), net(other)
    assert yf.shape == (1, cfg.C_OUT, factor * n, factor * n)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    r_grid = rel(yf[..., ::factor, ::factor], yc)
    r_input = rel(yo, yc)
    assert r_input > 1e-2, "the control inputs were too similar to be a control"
    assert r_grid < 0.05 * r_input
    assert r_grid < 0.1


# ---------------------------------------------------------------------------
# Structure: no normalisation, last block linear
# ---------------------------------------------------------------------------
def test_no_normalisation_layers_anywhere():
    """
    BatchNorm would make the inversion's gradient depend on the other samples in the
    batch, and the inversion runs one sample at a time; Layer/Instance/Group norm
    would divide the field by its own amplitude and destroy the radius information the
    same way per-sample target normalisation would (§7.1).  The name check catches
    norm classes that do not exist yet.
    """
    known = (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm,
             nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm)
    for m in build("tiny").modules():
        assert not isinstance(m, known), f"normalisation layer {type(m).__name__}"
        assert "norm" not in type(m).__name__.lower(), type(m).__name__


def test_last_block_is_linear_and_the_others_are_not():
    net = build("tiny")
    assert [b.act for b in net.blocks] == [True] * (cfg.N_BLOCKS - 1) + [False]
    assert all(isinstance(b, FourierBlock) for b in net.blocks)


def test_the_block_stack_cannot_blow_up_before_the_first_update():
    """
    The stated reason no normalisation is needed (§6.3): the spectral weights start at
    scale 1/(d_in d_out), so the residual branch is a modest perturbation and a
    four-block stack is close to the identity in norm.

    Stated as growth across the whole stack rather than per block, because that is the
    claim that matters -- a single block's residual is not tiny (the pointwise 1x1
    convolution at PyTorch's default init contributes ~0.4 of the input norm), but the
    branch is uncorrelated with what it is added to, so the growth compounds like
    sqrt(1 + r^2) rather than like (1 + r).  Without the residual, or with the
    spectral scale set to 1, this ratio would run away with depth and the stack would
    need a norm layer to survive -- which the inversion cannot afford (§6.3).
    """
    torch.manual_seed(cfg.SEED)
    net = FNO2d(d_v=16, kmax=6)
    v = net.lift(torch.randn(2, cfg.C_IN, 32, 32))
    ratios = []
    with torch.no_grad():
        for b in net.blocks:
            before = v.norm()
            v = b(v)
            ratios.append((v.norm() / before).item())
    assert all(0.5 < r < 2.0 for r in ratios), ratios
    total = math.prod(ratios)
    assert 0.3 < total < 4.0, f"stack gain {total:.2f} at initialisation"


def test_output_channel_count_is_the_scattered_phasor():
    net = build("tiny")
    y = net(torch.randn(1, cfg.C_IN, cfg.N_NET, cfg.N_NET))
    assert y.shape == (1, cfg.C_OUT, cfg.N_NET, cfg.N_NET) == (1, 4, 128, 128)


# ---------------------------------------------------------------------------
# to_double
# ---------------------------------------------------------------------------
def test_to_double_converts_complex_parameters():
    net = FNO2d(d_v=8, kmax=6)
    sc = net.blocks[0].spectral
    assert sc.w1.dtype == torch.complex64
    before = sc.w1.detach().clone()

    # The regression this function exists for: Module.double() converts a parameter
    # only when `t.is_floating_point()`, and that predicate is False for complex, so
    # the spectral weights are left behind.  Asserted as "did not become complex128"
    # rather than "is still complex64", so that if PyTorch ever fixes this the test
    # reports it as news instead of as a crash -- at which point `to_double` can go.
    net.double()
    assert net.lift[0].weight.dtype == torch.float64
    assert sc.w1.dtype != torch.complex128, (
        "nn.Module.double() now handles complex parameters; to_double is redundant")

    assert to_double(net) is net
    assert sc.w1.dtype == torch.complex128
    assert sc.w2.dtype == torch.complex128
    assert sc.m1.dtype == torch.float64          # buffers too, or the mask promotes
    assert net.lift[0].weight.dtype == torch.float64
    assert net.project[-1].bias.dtype == torch.float64

    # Lossless: complex64 -> complex128 -> complex64 is exact.
    assert torch.equal(sc.w1.detach().to(torch.complex64), before)
    assert sc.w1.requires_grad


def test_doubled_model_runs_and_backpropagates_in_float64():
    net = to_double(FNO2d(d_v=8, kmax=6))
    x = torch.randn(1, cfg.C_IN, 32, 32, dtype=torch.float64, requires_grad=True)
    y = net(x)
    assert y.dtype == torch.float64
    y.pow(2).sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.float64
    assert torch.isfinite(x.grad).all()
    w1 = net.blocks[0].spectral.w1
    assert w1.grad is not None and w1.grad.dtype == torch.complex128


# ---------------------------------------------------------------------------
# band_in_modes: the KMAX headroom claim
# ---------------------------------------------------------------------------
def test_band_in_modes_and_the_headroom_it_reports():
    """
    KMAX = 28 against what the band actually needs -- and a warning about which nu
    the default argument uses.

    `band_in_modes()` defaults to nu = min(NU_LIST) = 0.25, which is the *fastest*
    shear wave and therefore the *least* demanding material.  At the top of the band
    it reports 18.6 modes, giving an apparent 1.51x headroom.  The number to quote is
    the worst case, nu = 0.37: 23.6 modes, headroom 1.19x.  Both are above the
    requirement, but 1.19 and 1.51 support rather different sentences, and the
    __main__ block of fno2d.py prints the optimistic one.

    Also checked: at the Hann main-lobe edge (BURST_SPREAD = 1.4 f_c, where the
    deconvolution stops being conditioned at all) the requirement is exactly
    config.K_REQUIRED, which is where KMAX was chosen from.
    """
    f_top = max(cfg.FREQS) / cfg.FC

    worst = band_in_modes(cfg.NU_WORST, f_top)
    assert worst == pytest.approx(cfg.K_CARRIER_WORST * f_top, rel=1e-12)
    assert worst == pytest.approx(23.602, abs=0.02)
    assert cfg.KMAX / worst == pytest.approx(1.186, abs=0.01)

    default = band_in_modes()
    assert default == pytest.approx(18.570, abs=0.02)
    assert cfg.KMAX / default == pytest.approx(1.508, abs=0.01)
    assert default < worst, "the default nu is the optimistic end of the range"

    edge = band_in_modes(cfg.NU_WORST, cfg.BURST_SPREAD)
    assert edge == pytest.approx(cfg.K_REQUIRED, rel=1e-12)
    assert cfg.K_REQUIRED <= cfg.KMAX


# ---------------------------------------------------------------------------
# The antagonist (section 9.2)
# ---------------------------------------------------------------------------
def test_pack_ring_channel_layout():
    """
    Channel c*M + m: one component's whole spectrum is contiguous, because frequency
    is the axis the scattered field is smooth along and a kernel of width 5 should see
    a smooth strip rather than five unrelated components.
    """
    B, R, M = 2, cfg.N_RECV, cfg.M_FREQ
    d = torch.zeros(B, R, 2, M, dtype=torch.complex64)
    for m in range(M):
        d[:, :, 0, m] = complex(m + 1, -(m + 1))
        d[:, :, 1, m] = complex(100 + m, -(100 + m))

    x = pack_ring(d, src_idx=torch.tensor([0, 1]), nu=torch.tensor([0.25, 0.37]))
    assert x.shape == (B, RING_CHANNELS, R) == (2, 4 * M + 3, 32)
    for m in range(M):
        assert x[0, 0 * M + m].unique().tolist() == [float(m + 1)]
        assert x[0, 1 * M + m].unique().tolist() == [float(-(m + 1))]
        assert x[0, 2 * M + m].unique().tolist() == [float(100 + m)]
        assert x[0, 3 * M + m].unique().tolist() == [float(-(100 + m))]

    # Conditioning: normalised source coordinates, then nu.  Position rather than a
    # one-hot index, so the held-out sources of §11.2 step 11 are representable and
    # the baseline gets a fair shot at that test.
    sx, sy = cfg.SOURCE_XY[0]
    assert x[0, 4 * M + 0].unique().item() == pytest.approx(
        2.0 * sx / cfg.L_DOMAIN - 1.0, rel=1e-6)
    assert x[0, 4 * M + 1].unique().item() == pytest.approx(
        2.0 * sy / cfg.L_DOMAIN - 1.0, rel=1e-6)
    assert x[0, 4 * M + 2].unique().item() == pytest.approx(-1.0, abs=1e-6)
    assert x[1, 4 * M + 2].unique().item() == pytest.approx(1.0, abs=1e-6)


def test_pack_ring_divides_by_the_receiver_space_scale():
    B, R, M = 1, cfg.N_RECV, cfg.M_FREQ
    d = torch.full((B, R, 2, M), 3.0 + 4.0j, dtype=torch.complex64)
    scale = torch.full((B, M), 2.0)
    raw = pack_ring(d, src_idx=torch.tensor([0]), nu=torch.tensor([0.33]))
    scaled = pack_ring(d, src_idx=torch.tensor([0]), nu=torch.tensor([0.33]),
                       scale=scale)
    n_data = 4 * M
    assert torch.allclose(scaled[:, :n_data], 0.5 * raw[:, :n_data], atol=1e-6)
    assert torch.allclose(scaled[:, n_data:], raw[:, n_data:])


def test_pack_ring_rejects_real_input():
    with pytest.raises(AssertionError, match="complex"):
        pack_ring(torch.zeros(1, cfg.N_RECV, 2, cfg.M_FREQ),
                  src_idx=torch.tensor([0]), nu=torch.tensor([0.33]))


def test_ring_cnn_is_invariant_to_rotating_the_receiver_ring():
    """
    The receivers form a closed loop, so `padding_mode='circular'` plus symmetric
    pooling makes the network *exactly* invariant to relabelling receiver 0.  With a
    zero pad it would not be: the network would learn an edge between receiver 31 and
    receiver 0 that does not exist, and the artefact would sit at a fixed place on the
    ring and be learned as a feature of the domain rather than of the defect.
    """
    net = RingCNN(c_in=8, width=16, depth=2).eval()
    x = torch.randn(2, 8, cfg.N_RECV)
    with torch.no_grad():
        a, b = net(x), net(x.roll(5, dims=-1))
    assert a.shape == (2, 3)
    assert torch.allclose(a, b, atol=1e-5)


def test_ring_cnn_predict_theta_lands_in_the_per_nu_box():
    """
    lambda_s varies 13% across the four Poisson ratios, so the feasible box does too.
    `predict_theta` maps per distinct nu; using one box for all of them would put a
    systematic radius bias of that size into the baseline's estimates -- and the
    baseline is the number the thesis is stated against, so a bias in its favour or
    against it both matter.
    """
    net = RingCNN(c_in=RING_CHANNELS, width=16, depth=2).eval()
    x = torch.randn(3, RING_CHANNELS, cfg.N_RECV)
    nu = torch.tensor([0.25, 0.37, 0.37])
    theta = net.predict_theta(x, nu)
    assert theta.shape == (3, 3)
    fam = Circle()
    for i, v in enumerate(nu.tolist()):
        lo, hi = fam.bounds(cfg.cs_over_cp(v) / cfg.FC)
        assert (theta[i] >= lo).all() and (theta[i] <= hi).all()
    # The two nu = 0.37 rows share a box; the nu = 0.25 row has a wider one.
    lo25, hi25 = fam.bounds(cfg.cs_over_cp(0.25) / cfg.FC)
    lo37, hi37 = fam.bounds(cfg.cs_over_cp(0.37) / cfg.FC)
    assert hi25[2] > hi37[2], "a faster shear wave means a larger maximal radius"
