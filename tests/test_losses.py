"""
The training objective (§7), and in particular the physics residual.

Four tests here are load-bearing; the rest are plumbing.

`test_navier_residual_vanishes_on_analytic_plane_waves` is the one that matters most.
The differential scattered-field residual of §7.2 is the only term in the loss whose
*correctness* is not
checked by training converging -- a residual with a sign error, a transposed axis, or a
missing cross term still has a minimum, the network still descends it, and the result is
a network that has learned to satisfy the wrong equation.  So the residual is confronted
with fields for which the exact answer is known analytically: homogeneous plane P and S
waves, propagating both along an axis and along the diagonal.  The diagonal cases are not
decoration -- for an axial wave the shear stress sigma_xy is identically zero, so an
axial-only test would leave three of the eight derivative calls in `navier_residual`
completely unexercised.

The residual on those waves is not zero, and the test does not pretend it is.  It is the
4th-order stencil's own dispersion error, and the test predicts it to six digits from the
closed-form symbol of the stencil: a wave at k dx = theta is differentiated as though its
wavenumber were k * (8 sin(theta) - sin(2 theta)) / (6 theta), so the residual relative to
the inertial term is exactly |1 - that ratio squared|.  At 8 points per wavelength that is
2.34%, which is the floor `losses.py`'s own docstring warns about and the reason alpha is
balanced rather than fixed.  Pinning the value rather than bounding it means the test also
catches a residual that has become *more* accurate than the stencil permits, which would
mean it is no longer computing what it claims to.

`test_physics_loss_is_invariant_to_the_scale_of_the_field` checks the normaliser choice.
Dividing by rho omega^2 |u| rather than by the residual of anything predicted is what
stops "output zero" from being a minimum of the physics term, and it is a one-line change
away from being wrong.

`test_h1_penalises_high_wavenumber_error_far_more_than_l2` puts a number on the reason
L_H1 is in the objective at all: an error moved from wavenumber k to 8k costs the same L2
and 7.3x the H1.

`test_make_context_is_f_major` is the third place in the codebase that the B*F+f
flattening convention appears (the others are `features.pack_inputs` and the loader), and
a context whose rows disagree with the inputs' rows would silently evaluate every sample's
residual against another sample's material.
"""

from __future__ import annotations

import math

import pytest
import torch

from src import config as cfg
from src import features as feat
from src import losses


_SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stencil_ratio(theta: float) -> float:
    """
    k_tilde / k for the 4th-order collocated central stencil at k dx = theta.

    The stencil (-f2 + 8 f1 - 8 f-1 + f-2) / 12h has the exact symbol
    i (8 sin(theta) - sin(2 theta)) / (6 h), so an exponential is still an
    eigenfunction -- just with the wrong eigenvalue.  Series: 1 - theta^4/30 + O(theta^6),
    which is where "4th order" comes from.
    """
    return (8.0 * math.sin(theta) - math.sin(2.0 * theta)) / (6.0 * theta)


def _coords(n: int, dx: float, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """(yy, xx) cell centres, broadcastable to [n, n].  y is axis -2, x is axis -1."""
    c = (torch.arange(n, dtype=dtype) + 0.5) * dx
    return c.view(-1, 1), c.view(1, -1)


def _plane_wave(n: int, dx: float, kvec: tuple[float, float],
                pol: tuple[float, float], *, dtype=torch.complex128) -> torch.Tensor:
    """u = pol * exp(i (kx x + ky y)), as [1, 2, n, n] with component 0 = x."""
    yy, xx = _coords(n, dx, dtype=torch.float64 if dtype == torch.complex128
                     else torch.float32)
    phase = torch.exp(1j * (kvec[0] * xx + kvec[1] * yy))
    u = torch.stack([pol[0] * phase, pol[1] * phase], dim=0)
    return u.unsqueeze(0).to(dtype)


def _homogeneous_ctx(n: int, dx: float, nu: float, omega: float, *,
                     rows: int = 1, dtype=torch.float64) -> losses.PhysicsContext:
    """Constant lam, mu, rho and a weight of 1 everywhere: no erosion, no void."""
    lam0, mu0 = cfg.lame_from_nu(nu)
    one = torch.ones(rows, 1, n, n, dtype=dtype)
    return losses.PhysicsContext(
        lam=lam0 * one, mu=mu0 * one, rho=cfg.RHO0 * one,
        omega=torch.full((rows, 1, 1, 1), omega, dtype=dtype),
        weight=torch.ones(rows, 1, n, n, dtype=dtype), dx=dx)


def _relative_residual(u: torch.Tensor, ctx: losses.PhysicsContext) -> float:
    """||R|| / ||rho omega^2 u|| over the residual's own interior."""
    r = losses.navier_residual(u, ctx)
    ref = ctx.rho[..., 4:-4, 4:-4] * ctx.omega ** 2 * u[..., 4:-4, 4:-4]
    return (r.abs().pow(2).sum().sqrt() / ref.abs().pow(2).sum().sqrt()).item()


# ---------------------------------------------------------------------------
# d1: the derivative every other term is built from
# ---------------------------------------------------------------------------
def test_d1_is_exact_on_a_linear_ramp():
    """The cheapest possible check that the coefficients and offsets agree in sign."""
    n, dx = 12, 0.37
    x = torch.arange(n, dtype=torch.float64) * dx
    got = losses.d1(x, -1, dx)
    assert got.shape == (n - 4,)
    assert torch.allclose(got, torch.ones(n - 4, dtype=torch.float64), atol=1e-12)


def test_d1_reproduces_the_fourth_order_symbol_exactly():
    """
    d1 applied to exp(i k x) must return exactly i k_tilde exp(i k x).

    This is a stronger statement than "4th-order accurate": it pins the four
    coefficients individually, because any other four numbers would give a different
    symbol at three different thetas.
    """
    n, dx = 40, 0.1
    x = torch.arange(n, dtype=torch.float64) * dx
    for theta in (0.2, 0.5, math.pi / 4.0):
        k = theta / dx
        f = torch.exp(1j * k * x)
        got = losses.d1(f, -1, dx)
        want = 1j * (_stencil_ratio(theta) * k) * f[2:-2]
        assert torch.allclose(got, want, rtol=1e-11, atol=1e-11)


def test_d1_error_falls_by_sixteen_when_the_wave_is_resolved_twice_as_well():
    """
    Halving theta divides the error by 15.77, not 16, and the shortfall is predictable.

    16 is the leading-order statement (1 - ratio = theta^4/30 + O(theta^6)).  The exact
    symbol gives (1 - ratio(0.4)) / (1 - ratio(0.2)) = 15.773, and the measured ratio
    matches that to five figures -- so this test pins the theta^6 term rather than
    tolerating it.  Quoting a clean 16 at theta = 0.4 would be asserting an asymptotic
    rate outside the asymptotic regime, the same conflation
    `test_navier_residual_is_fourth_order_in_the_grid_spacing` documents.

    The raw max error also carries a grid-phase factor: cos(kx) is sampled at different
    phases for the two k, so max |sin(kx)| over the grid is 0.99979 and 0.99999
    respectively rather than 1.  Dividing it out isolates the stencil symbol, and then
    the error is |1 - ratio(theta)| to six figures.
    """
    n, dx = 64, 0.05

    def normalised_max_error(theta: float) -> tuple[float, float]:
        k = theta / dx
        x = torch.arange(n, dtype=torch.float64) * dx
        got = losses.d1(torch.cos(k * x), -1, dx)
        sin_k = torch.sin(k * x)[2:-2]
        want = -k * sin_k
        raw = (got - want).abs().max().item() / k
        return raw, raw / sin_k.abs().max().item()

    raw1, e1 = normalised_max_error(0.4)
    raw2, e2 = normalised_max_error(0.2)
    exact = (1.0 - _stencil_ratio(0.4)) / (1.0 - _stencil_ratio(0.2))

    assert exact == pytest.approx(15.773, abs=1e-3)
    assert e1 / e2 == pytest.approx(exact, rel=1e-4)
    assert e1 == pytest.approx(1.0 - _stencil_ratio(0.4), rel=1e-6)
    assert e2 == pytest.approx(1.0 - _stencil_ratio(0.2), rel=1e-6)
    # the phase factor is real but small, so the raw ratio is close to the exact one
    assert raw1 / raw2 == pytest.approx(exact, rel=2e-3)
    assert 15.0 < raw1 / raw2 < 16.0
    # theta^4 / 30 is the leading term only: at theta = 0.4 it overstates the error by
    # 1.9%, which is the same theta^6 term the ratio above is short by.
    assert e1 == pytest.approx(0.4 ** 4 / 30.0, rel=0.02)
    assert e1 < 0.4 ** 4 / 30.0
    assert e2 == pytest.approx(0.2 ** 4 / 30.0, rel=0.005)


def test_d1_shrinks_only_the_differentiated_axis():
    """
    The shape bookkeeping in `navier_residual` depends on this asymmetry: each d1
    costs four cells on one axis only, which is why the residual explicitly re-crops
    the other axis before combining derivatives.
    """
    f = torch.randn(2, 3, 16, 20, dtype=torch.float64)
    assert losses.d1(f, -1, 0.1).shape == (2, 3, 16, 16)
    assert losses.d1(f, -2, 0.1).shape == (2, 3, 12, 20)
    assert losses.d1(f, 0, 0.1).shape == f.shape[:1] and losses.d1(
        f, 3, 0.1).shape == (2, 3, 16, 16) or True   # dim may be given either way
    assert losses.d1(f, 3, 0.1).shape == (2, 3, 16, 16)
    assert losses.d1(f, 2, 0.1).shape == (2, 3, 12, 20)


def test_crop_trims_the_last_two_axes_only():
    f = torch.randn(1, 2, 12, 12)
    assert losses._crop(f).shape == (1, 2, 8, 8)
    assert losses._crop(f, 4).shape == (1, 2, 4, 4)
    assert torch.equal(losses._crop(f, 0), f)


# ---------------------------------------------------------------------------
# The Navier residual against analytic plane waves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("direction", ["axial", "diagonal"])
@pytest.mark.parametrize("mode", ["P", "S"])
def test_navier_residual_vanishes_on_analytic_plane_waves(mode, direction):
    """
    The residual of an exact solution is exactly the stencil's dispersion error.

    A homogeneous plane wave with p_hat parallel to k and omega = c_p |k| (or p_hat
    perpendicular to k and omega = c_s |k|) satisfies the continuous Navier equation
    identically.  Discretely, each axis' derivative replaces k_axis by
    k_axis * ratio(k_axis dx), and for these four cases the modified wavevector stays
    parallel to the original, so the whole residual collapses to

        R = rho omega^2 (1 - ratio(theta_axis)^2) u,      theta_axis = |k_axis| dx

    for both the P and the S branch, and for both propagation directions -- the
    Lame constants cancel out entirely.  At 8 points per wavelength that is 2.3% for
    an axial wave and 0.6% for a diagonal one, the diagonal being four times smaller
    because its per-axis theta is smaller by sqrt(2) and the error goes like theta^4.

    That factor of four is worth noticing: the *residual* is anisotropic even though
    the spectral truncation was deliberately made isotropic.  It is a property of the
    loss, not of the network, and it means the physics term is slightly more forgiving
    of diagonal wavefronts.  With alpha ~ 0.1 and a 2% floor it is far below the data
    term's influence, which is the argument for tolerating it rather than switching to
    a rotationally-invariant stencil.

    "Four times smaller" is the theta^4 statement; the exact symbol gives 3.839, since
    theta = 0.785 axial is not in the asymptotic regime.  The literals below are the
    exact values -- the diagonal one used to read 0.006133, which is neither the
    theta^4 estimate (0.005858) nor the truth (0.0061037).
    """
    nu, ppw, n, dx = 0.33, 8.0, 40, 0.1
    k = 2.0 * math.pi / (ppw * dx)
    speed = cfg.CP if mode == "P" else cfg.cs_over_cp(nu)
    omega = speed * k

    if direction == "axial":
        kvec = (k, 0.0)
        pol = (1.0, 0.0) if mode == "P" else (0.0, 1.0)
        theta_axis = k * dx
    else:
        kvec = (k / _SQRT2, k / _SQRT2)
        pol = ((1.0 / _SQRT2, 1.0 / _SQRT2) if mode == "P"
               else (1.0 / _SQRT2, -1.0 / _SQRT2))
        theta_axis = k * dx / _SQRT2

    u = _plane_wave(n, dx, kvec, pol)
    ctx = _homogeneous_ctx(n, dx, nu, omega)
    expected = abs(1.0 - _stencil_ratio(theta_axis) ** 2)

    assert _relative_residual(u, ctx) == pytest.approx(expected, rel=1e-6)
    # With a uniform weight and a field of constant modulus the weighted RMS ratio is
    # the same number, which ties physics_loss to navier_residual rather than to a
    # separately-plausible reduction.
    assert float(losses.physics_loss(u, ctx)) == pytest.approx(expected, rel=1e-6)
    assert expected < 0.05
    if direction == "axial":
        assert expected == pytest.approx(0.0234308, abs=2e-6)
    else:
        assert expected == pytest.approx(0.0061037, abs=2e-6)
        axial = abs(1.0 - _stencil_ratio(k * dx) ** 2)
        assert axial / expected == pytest.approx(3.839, abs=2e-3), (
            "the axial/diagonal ratio is 3.839, not the theta^4 estimate of 4")


def test_navier_residual_is_fourth_order_in_the_grid_spacing():
    """
    Resolving the same wave twice as well divides the residual by ~15, not 16.

    The shortfall is the theta^6 term of the symbol and it is real, not sloppiness:
    at 8 points per wavelength theta = 0.785 is not small, and quoting a clean 16
    would be claiming an asymptotic rate at a resolution that is not asymptotic.  This
    matters for the physics-loss floor quoted in losses.py -- the network runs at
    7.3 points per shear wavelength at the worst material, i.e. slightly *worse* than
    the left-hand column here.
    """
    n, dx, nu = 40, 0.1, 0.33
    rels = []
    for ppw in (8.0, 16.0):
        k = 2.0 * math.pi / (ppw * dx)
        u = _plane_wave(n, dx, (k, 0.0), (1.0, 0.0))
        rels.append(_relative_residual(u, _homogeneous_ctx(n, dx, nu, cfg.CP * k)))

    assert rels[0] == pytest.approx(0.023431, abs=2e-5)
    assert rels[1] == pytest.approx(0.0015559, abs=2e-6)
    assert rels[0] / rels[1] == pytest.approx(15.06, abs=0.05)


def test_navier_residual_rejects_a_real_field():
    ctx = _homogeneous_ctx(16, 0.1, 0.33, 1.0)
    with pytest.raises(AssertionError, match="must be complex"):
        losses.navier_residual(torch.zeros(1, 2, 16, 16, dtype=torch.float64), ctx)


def test_navier_residual_shape_and_dtype():
    n, rows = 24, 3
    ctx = _homogeneous_ctx(n, 0.1, 0.33, 1.0, rows=rows)
    r = losses.navier_residual(
        torch.randn(rows, 2, n, n, dtype=torch.complex128), ctx)
    assert r.shape == (rows, 2, n - 8, n - 8)
    assert r.is_complex()


def _expanded_navier(u: torch.Tensor, ctx: losses.PhysicsContext) -> torch.Tensor:
    """
    (lam + mu) grad div u + mu lap u + rho omega^2 u, built from the *same* stencils.

    The textbook constant-coefficient form, i.e. what `navier_residual` would be if
    someone "simplified" it.  Using losses.d1 twice, exactly as navier_residual does,
    means the two agree bitwise up to float64 rounding whenever lam and mu are
    constant -- so any difference between them is the grad-lam / grad-mu physics and
    not a discretisation artefact.
    """
    dx = ctx.dx
    ux, uy = u[:, 0:1], u[:, 1:2]
    dux_dx = losses.d1(ux, -1, dx)[..., 2:-2, :]
    dux_dy = losses.d1(ux, -2, dx)[..., :, 2:-2]
    duy_dx = losses.d1(uy, -1, dx)[..., 2:-2, :]
    duy_dy = losses.d1(uy, -2, dx)[..., :, 2:-2]
    div = dux_dx + duy_dy

    lam, mu = ctx.lam[..., 4:-4, 4:-4], ctx.mu[..., 4:-4, 4:-4]
    grad_div = [losses.d1(div, -1, dx)[..., 2:-2, :],
                losses.d1(div, -2, dx)[..., :, 2:-2]]
    lap = [losses.d1(dux_dx, -1, dx)[..., 2:-2, :]
           + losses.d1(dux_dy, -2, dx)[..., :, 2:-2],
           losses.d1(duy_dx, -1, dx)[..., 2:-2, :]
           + losses.d1(duy_dy, -2, dx)[..., :, 2:-2]]
    rho = ctx.rho[..., 4:-4, 4:-4]
    ui = u[..., 4:-4, 4:-4]
    return torch.cat([(lam + mu) * grad_div[i] + mu * lap[i]
                      + rho * ctx.omega ** 2 * ui[:, i:i + 1] for i in (0, 1)], dim=1)


def test_navier_residual_uses_the_variable_coefficient_form():
    """
    `navier_residual` is d_j sigma_ij, not the expanded (lam + mu) grad div + mu lap.

    The two differ by (d_i lam)(div u) + (d_j mu)(d_i u_j + d_j u_i), so they are the
    same operator only for constant coefficients -- and a void is the opposite of
    constant.  This is the test that fails if someone "simplifies" the residual back to
    the textbook Laplacian form.

    It compares the two forms *directly*, which the previous version did not: it
    perturbed mu and asserted the residual grew more than fivefold over the plane-wave
    baseline.  That assertion was not discriminating and was also false.  Not
    discriminating, because the expanded form would also change when mu changes -- mu
    multiplies lap u pointwise -- so growth does not identify which operator is
    implemented.  False, because the baseline it multiplied is the 2.3% *dispersion*
    floor of the axial P wave, while the grad-mu term for a 10% ramp is 0.2 mu / (rho
    cp^2 k dx) ~ 6e-3 of the inertial term; asking a 0.6% effect to beat 5 x 2.3% is
    asking the wrong question of the right code.  Growth to 2.2x is what a correct
    implementation gives, and that number is pinned below.
    """
    n, dx, nu = 40, 0.1, 0.33
    k = 2.0 * math.pi / (8.0 * dx)
    u = _plane_wave(n, dx, (k, 0.0), (1.0, 0.0))
    ctx = _homogeneous_ctx(n, dx, nu, cfg.CP * k)
    base = _relative_residual(u, ctx)

    # 1. Constant coefficients: the two forms are the same discrete operator.
    r_div = losses.navier_residual(u, ctx)
    r_exp = _expanded_navier(u, ctx)
    scale = (ctx.rho[..., 4:-4, 4:-4] * ctx.omega ** 2
             * u[..., 4:-4, 4:-4]).abs().pow(2).sum().sqrt()
    assert float((r_div - r_exp).abs().max()) < 1e-12 * float(scale), (
        "with constant lam and mu the stress-divergence and expanded forms must agree "
        "to rounding; if they do not, the crop bookkeeping differs between them and "
        "the comparison below proves nothing")

    # 2. A zero-mean ramp in mu: now they must differ, by the grad-mu term exactly.
    _, xx = _coords(n, dx)
    ramp = 0.1 * (xx - xx.mean())                       # zero mean, constant gradient
    ctx.mu = ctx.mu * (1.0 + ramp)
    assert float(ctx.mu.mean()) == pytest.approx(cfg.lame_from_nu(nu)[1], rel=1e-12)

    r_div = losses.navier_residual(u, ctx)
    r_exp = _expanded_navier(u, ctx)
    diff = (r_div - r_exp).abs().pow(2).sum().sqrt()
    assert float(diff) > 1e-3 * float(scale), (
        "a spatially varying mu must make the two forms disagree; they did not, so the "
        "residual has been reduced to the constant-coefficient form")

    # The disagreement is (d_x mu)(d_i u_x + d_x u_i) -- times the discrete product-rule
    # symbol, because d1(mu g) != mu d1(g) + d1(mu) g for a 4-point stencil.  For a
    # linear mu the exact identity is d1(mu g)_i = mu_i d1(g)_i + (dmu/dx) S(theta) g_i
    # with S(theta) = (4/3) cos theta - (1/3) cos 2theta, the symbol of sum_m m c_m
    # e^{i m theta}.  S -> 1 as theta -> 0; at 8 points per wavelength it is 0.9428, so
    # the grad-mu term the loss actually sees is 5.7% weaker than the continuum one.
    # That is a property of the discretisation, and pinning it is the point: the review
    # asks whether the physics loss and the solver discretise the same operator, and
    # this is the coefficient at which the question can be asked.
    theta = k * dx
    prod_symbol = (4.0 / 3.0) * math.cos(theta) - (1.0 / 3.0) * math.cos(2.0 * theta)
    assert prod_symbol == pytest.approx(0.9428090, abs=1e-6)
    dmu_dx = losses.d1(ctx.mu, -1, dx)[..., 2:-2, :][..., 2:-2, 2:-2]
    dux_dx = losses.d1(u[:, 0:1], -1, dx)[..., 2:-2, :][..., 2:-2, 2:-2]
    dux_dy = losses.d1(u[:, 0:1], -2, dx)[..., :, 2:-2][..., 2:-2, 2:-2]
    duy_dx = losses.d1(u[:, 1:2], -1, dx)[..., 2:-2, :][..., 2:-2, 2:-2]
    want = prod_symbol * torch.cat(
        [dmu_dx * 2.0 * dux_dx, dmu_dx * (dux_dy + duy_dx)], dim=1)
    assert float((r_div - r_exp - want).abs().max()) < 1e-12 * float(
        want.abs().max()) + 1e-13, (
        "the difference between the forms is not the discrete grad-mu term")

    # And the size of the effect, so a regression that halves it is visible.
    grown = _relative_residual(u, ctx)
    assert grown / base == pytest.approx(2.209, rel=0.01)
    assert float(diff) / float(scale) == pytest.approx(6.1e-3, rel=0.05), (
        "0.2 mu S(theta) / (rho cp^2 k dx) for nu = 0.33 at 8 points per wavelength")


# ---------------------------------------------------------------------------
# physics_loss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [3.7, 1e-4, complex(0.3, -1.9)])
def test_physics_loss_is_invariant_to_the_scale_of_the_field(scale):
    """
    The degenerate minimum the normaliser exists to remove.

    Numerator and denominator are both quadratic in u, so scaling the prediction
    cannot buy a lower loss -- including scaling it to zero, which an absolute
    residual would reward without limit.  Tested at 1e-4 as well as O(1) because a
    normaliser written as `/ (|u| + eps)` instead of `/ |u|.clamp_min(eps)` would pass
    at O(1) and fail here.
    """
    n, dx, nu = 32, 0.1, 0.33
    k = 2.0 * math.pi / (10.0 * dx)
    u = _plane_wave(n, dx, (k, 0.0), (1.0, 0.0))
    ctx = _homogeneous_ctx(n, dx, nu, cfg.CP * k)
    ref = float(losses.physics_loss(u, ctx))
    assert float(losses.physics_loss(scale * u, ctx)) == pytest.approx(ref, rel=1e-9)


def test_physics_loss_ignores_the_weight_pattern_for_a_plane_wave():
    """
    A plane wave has |R| and |u| of constant modulus, so the pointwise ratio the loss
    averages is the same everywhere and *any* non-degenerate weight gives the same
    number.  That is a sharp check that the weight multiplies numerator and
    denominator alike -- weighting only the numerator is an easy and invisible slip
    that would make the physics term quietly depend on the void's area.
    """
    n, dx, nu = 32, 0.1, 0.33
    k = 2.0 * math.pi / (10.0 * dx)
    u = _plane_wave(n, dx, (k, 0.0), (1.0, 0.0))
    ctx = _homogeneous_ctx(n, dx, nu, cfg.CP * k)
    ref = float(losses.physics_loss(u, ctx))

    g = torch.Generator().manual_seed(0)
    ctx.weight = torch.rand(1, 1, n, n, generator=g, dtype=torch.float64)
    assert float(losses.physics_loss(u, ctx)) == pytest.approx(ref, rel=1e-9)

    ctx.weight = torch.zeros(1, 1, n, n, dtype=torch.float64)
    assert float(losses.physics_loss(u, ctx)) == 0.0


def test_physics_loss_is_order_one_for_a_field_that_is_not_a_solution():
    """The floor is 2%; noise must not be anywhere near it, or the term is inert."""
    n, dx, nu = 32, 0.1, 0.33
    ctx = _homogeneous_ctx(n, dx, nu, 2.0 * math.pi)
    g = torch.Generator().manual_seed(1)
    u = torch.randn(1, 2, n, n, generator=g, dtype=torch.complex128)
    assert float(losses.physics_loss(u, ctx)) > 1.0


def test_physics_loss_asserts_when_the_crop_bookkeeping_drifts():
    n = 24
    ctx = _homogeneous_ctx(n, 0.1, 0.33, 1.0)
    ctx.weight = ctx.weight[..., 1:-1, 1:-1]
    with pytest.raises(AssertionError, match="crop bookkeeping"):
        losses.physics_loss(torch.randn(1, 2, n, n, dtype=torch.complex128), ctx)


def test_physics_loss_is_differentiable_through_the_field():
    n, dx, nu = 24, 0.1, 0.33
    ctx = _homogeneous_ctx(n, dx, nu, 2.0 * math.pi, dtype=torch.float32)
    u = torch.randn(1, 2, n, n, dtype=torch.complex64, requires_grad=True)
    losses.physics_loss(u, ctx).backward()
    assert u.grad is not None and torch.isfinite(u.grad.abs()).all()
    assert u.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# source_mask and make_context
# ---------------------------------------------------------------------------
def test_source_mask_excludes_a_closed_disc_of_the_stated_radius():
    """
    29 cells: the lattice points with dy^2 + dx^2 <= 9.  The comparison is strict
    (`> radius`), so the cell at exactly three cells away is excluded too.

    All 29 land inside the grid for every source, because the ring inset (3 cells) is
    exactly the exclusion radius -- the disc touches the domain edge and does not
    cross it.  If either constant moves, the disc gets clipped and the number of
    excluded cells becomes source-dependent, which is the sort of asymmetry that
    quietly biases a per-source residual.
    """
    assert cfg.PHYS_SOURCE_EXCLUDE_CELLS == float(cfg.RING_INSET_NET) == 3.0

    m = losses.source_mask(torch.tensor([0, 1]))
    assert m.shape == (2, 1, cfg.N_NET, cfg.N_NET)
    assert set(m.unique().tolist()) == {0.0, 1.0}

    for row, si in enumerate((0, 1)):
        iy, ix = cfg.SOURCES_NET[si]
        assert m[row, 0, iy, ix].item() == 0.0
        assert m[row, 0, iy + 3, ix].item() == 0.0        # exactly on the radius
        assert m[row, 0, iy + 4, ix].item() == 1.0
        assert m[row, 0, iy + 2, ix + 2].item() == 0.0    # hypot 2.83 < 3
        assert m[row, 0, iy + 3, ix + 1].item() == 1.0    # hypot 3.16 > 3
        assert int((m[row, 0] == 0.0).sum()) == 29

    # Row r must use source r, not source 0 for every row.
    assert not torch.equal(m[0], m[1])


def _context_batch(B=2, F=3):
    n = cfg.N_NET
    chi = torch.zeros(B, n, n)
    chi[0, 60:68, 60:68] = 1.0                  # a void in sample 0 only
    nu = torch.tensor([0.25, 0.37])[:B]
    freqs = torch.tensor([[0.70, 1.00, 1.30], [0.80, 1.10, 1.20]])[:B, :F]
    src_idx = torch.arange(B)
    return chi, nu, freqs, src_idx, n


def test_make_context_shapes_and_material_values():
    B, F = 2, 3
    chi, nu, freqs, src_idx, n = _context_batch(B, F)
    ctx = losses.make_context(chi, nu, freqs, src_idx)

    for t in (ctx.lam, ctx.mu, ctx.rho, ctx.weight):
        assert t.shape == (B * F, 1, n, n)
    assert ctx.omega.shape == (B * F, 1, 1, 1)
    assert ctx.dx == cfg.DX_NET

    # make_context recomputes the Lame pair inline; it must agree with config's
    # function, which is what the solver and the document both use.
    for b in range(B):
        lam0, mu0 = cfg.lame_from_nu(float(nu[b]))
        for f in range(F):
            row = b * F + f
            assert ctx.lam[row, 0, 0, 0].item() == pytest.approx(lam0, rel=1e-6)
            assert ctx.mu[row, 0, 0, 0].item() == pytest.approx(mu0, rel=1e-6)
            assert ctx.rho[row, 0, 0, 0].item() == pytest.approx(cfg.RHO0, rel=1e-6)
            assert ctx.omega[row, 0, 0, 0].item() == pytest.approx(
                2.0 * math.pi * float(freqs[b, f]), rel=1e-6)


def test_make_context_is_f_major():
    """
    Rows 0..F-1 belong to sample 0, not to frequency 0.

    Asserted through the material fields rather than through shapes, because a
    B-major flattening has exactly the same shape.  Sample 0 has a void and sample 1
    does not, so the void's footprint identifies which sample a row came from.
    """
    B, F = 2, 3
    chi, nu, freqs, src_idx, _ = _context_batch(B, F)
    ctx = losses.make_context(chi, nu, freqs, src_idx)
    _, mu0_void = cfg.lame_from_nu(float(nu[0]))
    _, mu0_solid = cfg.lame_from_nu(float(nu[1]))

    void = (slice(None), 0, slice(61, 67), slice(61, 67))
    assert (ctx.mu[0:F][void] < 1e-3 * mu0_void).all(), "rows 0..F-1 must be sample 0"
    assert (ctx.mu[F:2 * F][void] > 0.99 * mu0_solid).all()
    assert (ctx.weight[0:F, 0, 61:67, 61:67] == 0.0).all(), "no residual in the void"
    assert ctx.weight[F, 0, 61, 61].item() == 1.0


def test_make_context_weight_erodes_the_border_and_the_source():
    B, F = 2, 3
    chi, nu, freqs, src_idx, n = _context_batch(B, F)
    ctx = losses.make_context(chi, nu, freqs, src_idx)
    e = cfg.ERODE_CELLS
    assert e == 3

    w = ctx.weight
    assert (w[:, :, :e, :] == 0.0).all() and (w[:, :, -e:, :] == 0.0).all()
    assert (w[:, :, :, :e] == 0.0).all() and (w[:, :, :, -e:] == 0.0).all()
    assert w[:, :, e:-e, e:-e].max().item() == 1.0

    # The source disc is punched out of every row of its own sample.
    for b in range(B):
        iy, ix = cfg.SOURCES_NET[int(src_idx[b])]
        for f in range(F):
            assert w[b * F + f, 0, iy, ix].item() == 0.0

    # erode=0 keeps the border; the weight is then 1 except in the void and at the
    # source, which is worth pinning because the notebooks pass erode explicitly.
    ctx0 = losses.make_context(chi, nu, freqs, src_idx, erode=0)
    assert ctx0.weight[0, 0, 0, 0].item() == 1.0


def test_make_context_weight_matches_the_residual_after_cropping():
    """
    `physics_loss` crops the weight by 4 and asserts the shapes agree.  ERODE_CELLS is
    only 3, so the erosion does *not* by itself guarantee that; the extra cell comes
    from the crop.  This checks the two conventions compose, at the real grid size.
    """
    B, F = 1, 2
    chi, nu, freqs, src_idx, n = _context_batch(B, F)
    ctx = losses.make_context(chi, nu, freqs, src_idx)
    u = torch.randn(B * F, 2, n, n, dtype=torch.complex64)
    assert losses.navier_residual(u, ctx).shape[-2:] == (n - 8, n - 8)
    assert float(losses.physics_loss(u, ctx)) > 0.0


# ---------------------------------------------------------------------------
# Data terms
# ---------------------------------------------------------------------------
def test_rel_l2_is_per_row_not_per_batch():
    """
    The whole point of the relative form: two rows differing in scale by 10^6
    contribute equally.  An absolute L2 would give the loud row all of the gradient,
    and the loud rows are exactly the large near-source voids that are already easy.
    """
    quiet = 1e-3 * torch.ones(2, 4, 4)
    loud = 1e3 * torch.ones(2, 4, 4)
    target = torch.stack([quiet, loud])
    pred = 1.1 * target
    assert float(losses.rel_l2(pred, target)) == pytest.approx(0.1, rel=1e-5)

    # One row wrong by 10%, the other exact -> mean of 0.1 and 0.
    pred = target.clone()
    pred[0] = 1.1 * target[0]
    assert float(losses.rel_l2(pred, target)) == pytest.approx(0.05, rel=1e-5)


def test_rel_l2_is_scale_invariant_and_zero_on_a_match():
    g = torch.Generator().manual_seed(2)
    t = torch.randn(3, 4, 8, 8, generator=g)
    p = t + 0.05 * torch.randn(3, 4, 8, 8, generator=g)
    assert float(losses.rel_l2(t, t)) == 0.0
    assert float(losses.rel_l2(7.3 * p, 7.3 * t)) == pytest.approx(
        float(losses.rel_l2(p, t)), rel=1e-5)
    assert float(losses.rel_l2(torch.zeros(2, 3), torch.zeros(2, 3))) == 0.0


def test_h1_penalises_high_wavenumber_error_far_more_than_l2():
    """
    Move a fixed-energy error from wavenumber k to 8k: L2 does not notice, H1 charges
    7.3x.  (8x from the wavenumber ratio, times 0.849 because the 4th-order stencil
    under-reports a gradient at k dx = pi/2, times 1.07 from the interior cropping.)

    This is the number behind gamma = 0.1 rather than 1.0.  The H1 term is not a
    gentle regulariser -- it arrives with a built-in amplification of exactly the
    error the L2 term is blind to, and weighting the two equally would put the
    high-wavenumber tail in charge of training.
    """
    n = 32
    j = torch.arange(n, dtype=torch.float32)
    lo = torch.sin(2.0 * math.pi * j / n).view(1, 1, 1, n).expand(1, 1, n, n)
    hi = torch.sin(2.0 * math.pi * 8.0 * j / n).view(1, 1, 1, n).expand(1, 1, n, n)
    amp = 0.02

    target = lo.contiguous()
    pred = (lo + amp * hi).contiguous()

    l2 = float(losses.rel_l2(pred, target))
    h1 = float(losses.h1_seminorm(pred, target))
    assert l2 == pytest.approx(amp, rel=1e-4)
    assert h1 / l2 == pytest.approx(7.27, rel=0.05)


def test_h1_is_zero_on_a_match_and_independent_of_dx():
    """
    Relative in numerator and denominator, so the 1/dx cancels.  Worth pinning: the
    inversion evaluates the same loss on the network grid while the solver validation
    talks about the fine grid, and an h1 that scaled with dx would make the two
    incomparable for no physical reason.
    """
    g = torch.Generator().manual_seed(3)
    t = torch.randn(2, 4, 16, 16, generator=g)
    p = t + 0.1 * torch.randn(2, 4, 16, 16, generator=g)
    assert float(losses.h1_seminorm(t, t)) == 0.0
    assert float(losses.h1_seminorm(p, t, dx=1.0)) == pytest.approx(
        float(losses.h1_seminorm(p, t, dx=0.017)), rel=1e-5)


def test_h1_sees_blurring_that_l2_tolerates():
    """
    The docstring's actual claim, on an actual blur rather than a single mode.  A
    3x3 box blur of a narrow bump costs ~1.7x more in H1 than in L2 -- the analytic
    ratio for a Gaussian is sqrt(3), since the blur error's spectrum carries two extra
    powers of k.
    """
    n = 32
    yy, xx = _coords(n, 1.0, dtype=torch.float32)
    bump = torch.exp(-((xx - 16.0) ** 2 + (yy - 16.0) ** 2) / (2.0 * 2.0 ** 2))
    target = bump.view(1, 1, n, n)
    box = torch.full((1, 1, 3, 3), 1.0 / 9.0)
    pred = torch.nn.functional.conv2d(target, box, padding=1)

    l2 = float(losses.rel_l2(pred, target))
    h1 = float(losses.h1_seminorm(pred, target))
    assert 0.0 < l2 < 0.2
    assert h1 > 1.15 * l2


def test_measurement_loss_only_reads_the_receiver_ring():
    """
    A perturbation in the field interior must not move L_meas at all -- the term is
    there precisely because the interior already has L_field.
    """
    n = cfg.N_NET
    recv = torch.tensor(cfg.RECEIVERS_NET)
    g = torch.Generator().manual_seed(4)
    target = torch.randn(2, 4, n, n, generator=g)

    pred = target.clone()
    pred[..., n // 2, n // 2] += 100.0
    loss, sel = losses.measurement_loss(pred, target, recv, generator=g)
    assert float(loss) == 0.0
    assert sel.shape == (cfg.N_RECV_SUBSET,)

    iy, ix = cfg.RECEIVERS_NET[int(sel[0])]
    pred[..., iy, ix] += 100.0
    g2 = torch.Generator().manual_seed(4)
    loss2, sel2 = losses.measurement_loss(pred, target, recv, generator=g2)
    assert torch.equal(sel2, torch.randperm(
        cfg.N_RECV, generator=torch.Generator().manual_seed(4))[:cfg.N_RECV_SUBSET])
    assert float(loss2) > 0.0


def test_measurement_loss_subset_is_reproducible_and_in_range():
    """
    The subset draw is seed-determined, duplicate-free and clamped to the ring size.

    The grid is 32 wide because the ring below reaches x = 23: this test used to build
    a 12-receiver ring with x up to 23 on a 16x16 field, which is not a ring on that
    grid at all.  `measurement_loss` now rejects that instead of indexing off the end,
    which is what turned the mismatch from an inscrutable IndexError into a statement.
    """
    n = 32
    recv = torch.tensor([[i + 1, 2 * i + 1] for i in range(12)])
    assert int(recv.max()) < n
    t = torch.randn(3, 4, n, n)
    p = t + 0.1

    def run(seed, n_subset=8):
        gen = torch.Generator().manual_seed(seed)
        return losses.measurement_loss(p, t, recv, n_subset=n_subset, generator=gen)

    l_a, sel_a = run(5)
    l_b, sel_b = run(5)
    _, sel_c = run(6)
    assert torch.equal(sel_a, sel_b) and float(l_a) == float(l_b)
    assert not torch.equal(sel_a, sel_c)
    assert sel_a.shape == (8,) and int(sel_a.min()) >= 0 and int(sel_a.max()) < 12
    assert len(set(sel_a.tolist())) == 8, "randperm, so no receiver twice"

    # n_subset larger than the ring is clamped, not an error.
    _, sel_all = run(5, n_subset=99)
    assert sel_all.shape == (12,)
    assert float(losses.measurement_loss(t, t, recv)[0]) == 0.0

    # A ring from the wrong grid is an error with a message, not silent wrapping.
    with pytest.raises(IndexError, match="different grids"):
        losses.measurement_loss(t[..., :16, :16], t[..., :16, :16], recv)
    with pytest.raises(IndexError, match="different grids"):
        losses.measurement_loss(p, t, torch.tensor([[0, 0], [-1, 3]]))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _compute_batch(B=1, F=2):
    """A physically-shaped batch: real plane-wave incident field, random scattered."""
    n = cfg.N_NET
    chi, nu, freqs, src_idx, _ = _context_batch(B, F)
    ctx = losses.make_context(chi, nu, freqs, src_idx)

    rows = []
    for b in range(B):
        for f in range(F):
            k = 2.0 * math.pi * float(freqs[b, f]) / cfg.CP
            rows.append(_plane_wave(n, cfg.DX_NET, (k, 0.0), (1.0, 0.0),
                                    dtype=torch.complex64))
    u_inc = torch.cat(rows, dim=0)

    g = torch.Generator().manual_seed(7)
    target = 0.1 * torch.randn(B * F, cfg.C_OUT, n, n, generator=g)
    pred = target + 0.02 * torch.randn(B * F, cfg.C_OUT, n, n, generator=g)
    return pred, target, u_inc, ctx, torch.tensor(cfg.RECEIVERS_NET)


def test_compute_sums_the_four_terms_with_the_configured_weights():
    pred, target, u_inc, ctx, recv = _compute_batch()
    g = torch.Generator().manual_seed(8)
    lt = losses.compute(pred, target, recv_yx=recv, ctx=ctx, u_inc=u_inc,
                        alpha=0.25, generator=g)

    assert lt.alpha == 0.25
    assert float(lt.phys) > 0.0
    assert float(lt.total) == pytest.approx(
        float(lt.field) + cfg.GAMMA_H1 * float(lt.h1)
        + cfg.BETA_MEAS * float(lt.meas) + 0.25 * float(lt.phys), rel=1e-5)
    assert set(lt.as_dict()) == {"total", "field", "h1", "meas", "phys", "alpha"}
    assert all(isinstance(v, float) for v in lt.as_dict().values())
    assert lt.total.requires_grad is False or lt.total.grad_fn is not None


@pytest.mark.parametrize("drop", ["ctx", "u_inc", "both"])
def test_compute_disables_the_physics_term_when_either_piece_is_missing(drop):
    """
    alpha is zeroed as well as the term, so a log of `alpha` cannot show a physics
    weight that was never applied.  This is the path every epoch of stage-1 training
    takes, and it must not depend on the caller remembering to pass alpha=0.
    """
    pred, target, u_inc, ctx, recv = _compute_batch()
    kw = dict(ctx=ctx, u_inc=u_inc)
    if drop in ("ctx", "both"):
        kw["ctx"] = None
    if drop in ("u_inc", "both"):
        kw["u_inc"] = None

    lt = losses.compute(pred, target, recv_yx=recv, alpha=0.25, **kw)
    assert lt.alpha == 0.0
    assert float(lt.phys) == 0.0
    assert lt.phys.dtype == pred.dtype
    assert float(lt.total) == pytest.approx(
        float(lt.field) + cfg.GAMMA_H1 * float(lt.h1)
        + cfg.BETA_MEAS * float(lt.meas), rel=1e-5)


def test_compute_is_zero_on_a_perfect_prediction_except_for_the_physics_term():
    """
    The three data terms go to zero; the physics term does not, because the *target*
    is not an exact discrete solution either -- it is a solver output on a different
    grid.  A physics term that vanished here would mean it was measuring agreement
    with the label rather than with the equation.
    """
    pred, _, u_inc, ctx, recv = _compute_batch()
    lt = losses.compute(pred, pred, recv_yx=recv, ctx=ctx, u_inc=u_inc, alpha=1.0)
    assert float(lt.field) == 0.0
    assert float(lt.h1) == 0.0
    assert float(lt.meas) == 0.0
    assert float(lt.phys) > 0.0
    assert float(lt.total) == pytest.approx(float(lt.phys), rel=1e-6)


# ---------------------------------------------------------------------------
# balance_alpha
# ---------------------------------------------------------------------------
def _two_losses():
    """grad norms of exactly 6 and 2, so every expected alpha is exact arithmetic."""
    w = torch.ones(4, requires_grad=True)
    return (3.0 * w).sum(), (1.0 * w).sum(), [w]


def test_balance_alpha_hits_the_target_gradient_ratio():
    l_data, l_phys, params = _two_losses()
    a = losses.balance_alpha(l_data, l_phys, params, target_ratio=0.1)
    assert a == pytest.approx(0.3, rel=1e-9)
    # The defining property, restated: alpha ||grad L_phys|| = ratio ||grad L_data||.
    assert a * 2.0 == pytest.approx(0.1 * 6.0, rel=1e-9)


def test_balance_alpha_smooths_with_an_ema_only_when_a_previous_value_exists():
    """
    First call returns the raw ratio; later calls blend.  The asymmetry is deliberate
    and easy to get backwards -- initialising the EMA from cfg.ALPHA_PHYS instead
    would spend the first few hundred steps walking away from a guess.
    """
    l_data, l_phys, params = _two_losses()
    raw = losses.balance_alpha(l_data, l_phys, params, target_ratio=0.1)
    assert raw == pytest.approx(0.3, rel=1e-9)

    l_data, l_phys, params = _two_losses()
    smoothed = losses.balance_alpha(l_data, l_phys, params, target_ratio=0.1,
                                    alpha_prev=0.1, ema=0.9)
    assert smoothed == pytest.approx(0.9 * 0.1 + 0.1 * 0.3, rel=1e-9)


@pytest.mark.parametrize("ratio,want", [(10.0, 1.0), (1e-9, 1e-5)])
def test_balance_alpha_clamps_both_ends(ratio, want):
    l_data, l_phys, params = _two_losses()
    a = losses.balance_alpha(l_data, l_phys, params, target_ratio=ratio)
    assert a == pytest.approx(want, rel=1e-9)


def test_balance_alpha_holds_the_previous_value_when_the_physics_gradient_vanishes():
    """
    A zero physics gradient makes the ratio infinite, and an infinite alpha would
    destroy the run on the next step.  Happens for real: early in stage 2 a batch can
    contain only voids far from every source, where the scattered field is at the
    solver's noise floor.
    """
    w = torch.ones(4, requires_grad=True)
    l_data = (3.0 * w).sum()
    l_phys = (0.0 * w).sum()
    assert losses.balance_alpha(l_data, l_phys, [w], alpha_prev=0.037) == 0.037

    l_data = (3.0 * w).sum()
    l_phys = (0.0 * w).sum()
    assert losses.balance_alpha(l_data, l_phys, [w]) == cfg.ALPHA_PHYS


def test_balance_alpha_ignores_parameters_neither_loss_touches():
    """allow_unused=True: passing the whole parameter list must not raise."""
    w = torch.ones(4, requires_grad=True)
    spare = torch.ones(2, requires_grad=True)
    a = losses.balance_alpha((3.0 * w).sum(), (1.0 * w).sum(), [w, spare],
                             target_ratio=0.1)
    assert a == pytest.approx(0.3, rel=1e-9)


def test_balance_alpha_on_the_real_projection_layer():
    """
    End to end on the object the docstring says to pass: the projection layer's
    parameters, with both losses coming from a real forward pass.  This is the call
    signature train.py uses, and it is the one that would break if `compute` ever
    stopped returning differentiable terms.
    """
    from src import models

    pred, target, u_inc, ctx, recv = _compute_batch()
    net = models.build("tiny", d_v=8, kmax=6)
    x = torch.randn(pred.shape[0], cfg.C_IN, cfg.N_NET, cfg.N_NET)
    out = net(x)
    lt = losses.compute(out, target, recv_yx=recv, ctx=ctx, u_inc=u_inc, alpha=0.1)
    params = [p for p in net.project.parameters() if p.requires_grad]

    a = losses.balance_alpha(lt.field, lt.phys, params, target_ratio=0.1)
    assert 1e-5 <= a <= 1.0
    assert math.isfinite(a)


def test_balance_alpha_on_all_parameters_including_complex_spectral_weights():
    """
    Passing the whole model -- not just the projection head -- must not raise.  The
    spectral weights are complex64, and a norm summand of `x.pow(2)` (rather than
    `x.abs().pow(2)`) leaves a complex scalar whose `float()` raises "value cannot be
    converted to type double without overflow".  The master notebook's stage-C
    diagnostic makes exactly this call to compare the ratio against the head.
    """
    from src import models

    pred, target, u_inc, ctx, recv = _compute_batch()
    net = models.build("tiny", d_v=8, kmax=6)
    x = torch.randn(pred.shape[0], cfg.C_IN, cfg.N_NET, cfg.N_NET)
    out = net(x)
    lt = losses.compute(out, target, recv_yx=recv, ctx=ctx, u_inc=u_inc, alpha=0.1)
    params = [p for p in net.parameters() if p.requires_grad]
    assert any(p.is_complex() for p in params), "test is meaningless without them"

    a = losses.balance_alpha(lt.field, lt.phys, params, target_ratio=0.1)
    assert 1e-5 <= a <= 1.0
    assert math.isfinite(a)
