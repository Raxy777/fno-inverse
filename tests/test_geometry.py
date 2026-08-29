"""
Differentiable geometry.

Two tests here are load-bearing rather than decorative.

`test_dchi_dtheta_matches_autograd` checks the hand-derived expressions of section
8.2 against autograd.  Those expressions are not used in the training path -- autograd
handles it -- so a mistake in them would sit undetected until someone quoted them in
the viva or used them to sanity-check a gradient.  Deriving them and never checking
them would be worse than not deriving them.

`test_interface_width_is_physical_not_grid_relative` checks that the sigmoid width is
a length, not a number of cells of whatever grid happens to be in play.  The solver
runs at dx_fine and the network at dx_net = 2 dx_fine, so "1.5 cells" is ambiguous by
a factor of two, and a factor of two in the interface width changes the effective
scatterer size by ~5% of a radius.  The training labels would then be generated for a
slightly different void than the one the network is told about, and the inversion
would inherit a small systematic radius bias -- which is the hardest kind of error to
find, because everything is internally consistent.
"""

from __future__ import annotations

import math

import pytest
import torch

from src import config as cfg
from src.geometry import sdf as G


LAMBDA_S = cfg.LAMBDA_S_MIN


# ---------------------------------------------------------------------------
# Coordinate grids
# ---------------------------------------------------------------------------
def test_grid_coords_are_cell_centres():
    yy, xx = G.grid_coords(4, 0.25)
    assert xx[0].tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875])
    assert torch.allclose(yy, xx.T)


def test_grid_coords_offset_shifts_the_origin():
    """offset = n_pml puts x = 0 at the left edge of the first physical cell."""
    n_pml = 3
    _, xx = G.grid_coords(10, 0.5, offset=n_pml)
    assert xx[0, n_pml].item() == pytest.approx(0.25)     # first physical centre
    assert xx[0, n_pml - 1].item() == pytest.approx(-0.25)  # last absorber cell


def test_fine_and_net_grids_are_pool_aligned():
    """
    The centroid of each 2x2 block of fine physical cells is exactly a network cell
    centre.  This is why receivers can be read off the pooled grid with no
    interpolation -- and it is also why sources cannot (see
    test_source_position_exposes_the_quarter_cell_offset in test_config.py).
    """
    yyn, xxn = G.net_coords()
    yyf, xxf = G.fine_coords()
    p = cfg.N_PML_FINE
    core_x = xxf[p:p + cfg.N_FINE, p:p + cfg.N_FINE]
    core_y = yyf[p:p + cfg.N_FINE, p:p + cfg.N_FINE]
    pooled_x = core_x.view(cfg.N_NET, 2, cfg.N_NET, 2).mean(dim=(1, 3))
    pooled_y = core_y.view(cfg.N_NET, 2, cfg.N_NET, 2).mean(dim=(1, 3))
    assert torch.allclose(pooled_x, xxn, atol=1e-6)
    assert torch.allclose(pooled_y, yyn, atol=1e-6)


# ---------------------------------------------------------------------------
# Circle
# ---------------------------------------------------------------------------
def test_circle_sdf_is_the_exact_distance():
    fam = G.Circle()
    n = 128
    dx = cfg.L_DOMAIN / n
    theta = torch.tensor([[3.0, 5.0, 0.7]], dtype=torch.float64)
    yy, xx = G.grid_coords(n, dx, dtype=torch.float64)
    phi = fam.sdf(theta, yy, xx)
    exact = torch.sqrt((xx - 3.0) ** 2 + (yy - 5.0) ** 2) - 0.7
    assert torch.allclose(phi[0], exact, atol=1e-12)
    # |grad phi| = 1 away from the centre: it is a *signed distance*, not just some
    # level set with the right zero contour.  The physics loss never differentiates
    # phi, but the network is handed phi/dx as a channel and a non-unit gradient
    # would make "3 cells from the boundary" mean different things in different
    # directions.  Checked only outside the void, since the distance function has a
    # genuine cusp at the centre where a finite difference is meaningless.
    gy = (phi[0, 2:, 1:-1] - phi[0, :-2, 1:-1]) / (2 * dx)
    gx = (phi[0, 1:-1, 2:] - phi[0, 1:-1, :-2]) / (2 * dx)
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    far = phi[0, 1:-1, 1:-1] > 1.0
    assert far.any()
    assert torch.allclose(mag[far], torch.ones_like(mag[far]), atol=1e-3)


def test_circle_area():
    fam = G.Circle()
    theta = torch.tensor([[1.0, 1.0, 0.5], [2.0, 2.0, 1.0]])
    assert fam.area(theta).tolist() == pytest.approx(
        [math.pi * 0.25, math.pi])


def test_dchi_dtheta_matches_autograd():
    """The section 8.2 expressions, against autograd, in float64."""
    fam = G.Circle()
    n = 48
    dx = cfg.L_DOMAIN / n
    eps = cfg.EPS_INTERFACE_CELLS * dx
    yy, xx = G.grid_coords(n, dx, dtype=torch.float64)

    theta = torch.tensor([[3.7, 4.3, 0.62], [5.1, 2.9, 0.45]],
                         dtype=torch.float64, requires_grad=True)
    chi = G.soft_indicator(fam.sdf(theta, yy, xx), eps)
    w = torch.randn(chi.shape, dtype=torch.float64)
    g_auto, = torch.autograd.grad(chi, theta, grad_outputs=w)

    d = fam.dchi_dtheta_analytic(theta.detach(), yy, xx, eps)
    assert d.shape == (2, 3, n, n)
    g_ana = torch.einsum("byx,bpyx->bp", w, d)
    assert torch.allclose(g_auto, g_ana, rtol=1e-10, atol=1e-12)


def test_dchi_dtheta_is_an_annulus_with_the_right_multipoles():
    """
    Sensitivity lives in a ring of width ~eps around the boundary; the R-derivative
    is a monopole over that ring and the position derivatives are dipoles.  This is
    the geometric reason all three parameters are separately identifiable from one
    data vector, so it is worth asserting rather than assuming.
    """
    fam = G.Circle()
    n = 96
    dx = cfg.L_DOMAIN / n
    eps = cfg.EPS_INTERFACE_CELLS * dx
    yy, xx = G.grid_coords(n, dx, dtype=torch.float64)
    theta = torch.tensor([[4.0, 4.0, 0.8]], dtype=torch.float64)

    d = fam.dchi_dtheta_analytic(theta, yy, xx, eps)[0]
    phi = fam.sdf(theta, yy, xx)[0]

    # Support: sigma' decays like e^{-|phi|/eps}, so at 12 eps the sensitivity is
    # down by ~4 orders of magnitude.  (At 6 eps it is only down by 2 -- the tail is
    # exponential but not abrupt, which is the whole reason eps has to be chosen
    # rather than assumed negligible.)
    far = phi.abs() > 12.0 * eps
    assert far.any()
    assert d.abs()[:, far].max() < 1e-4 * d.abs().max()

    # Monopole: dR is strictly positive (growing R always adds void).
    assert (d[2] >= 0).all()
    assert d[2].sum() > 0

    # Dipoles: the position derivatives integrate to ~0 by symmetry.
    assert abs(d[0].sum().item()) < 1e-6 * d[2].sum().item()
    assert abs(d[1].sum().item()) < 1e-6 * d[2].sum().item()

    # ... and they are 90 degrees apart: dxc is odd in x, dyc is odd in y.
    assert torch.allclose(d[0], -d[0].flip(-1), atol=1e-9)
    assert torch.allclose(d[1], -d[1].flip(-2), atol=1e-9)


# ---------------------------------------------------------------------------
# Bounds and the sigmoid reparameterisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["circle", "ellipse", "two_circle"])
def test_bounds_shape_matches_n_params(name):
    fam = G.FAMILIES[name]
    lo, hi = fam.bounds(LAMBDA_S)
    assert lo.shape == hi.shape == (fam.n_params,)
    assert (hi > lo).all()
    assert len(fam.param_names) == fam.n_params


@pytest.mark.parametrize("name", ["circle", "ellipse", "two_circle"])
def test_reparameterisation_round_trip(name):
    fam = G.FAMILIES[name]
    lo, hi = fam.bounds(LAMBDA_S)
    # Strictly interior, so the clamp in to_unconstrained is not engaged.
    theta = (lo + (hi - lo) * torch.tensor([0.3, 0.7, 0.5, 0.4, 0.6, 0.45]
                                           )[:fam.n_params]).unsqueeze(0)
    z = fam.to_unconstrained(theta, LAMBDA_S)
    back = fam.to_physical(z, LAMBDA_S)
    assert torch.allclose(back, theta, rtol=1e-5, atol=1e-6)


def test_reparameterisation_never_leaves_the_box():
    """
    The point of the sigmoid: no matter how large the unconstrained step, theta stays
    feasible and the gradient stays defined.  A clip would give exactly zero gradient
    at the boundary, and L-BFGS reads a zero gradient as convergence -- so a run that
    walked into the wall would report success while sitting on it.
    """
    fam = G.Circle()
    lo, hi = fam.bounds(LAMBDA_S)
    theta = fam.to_physical(torch.tensor([[-1e3, 1e3, 40.0]]), LAMBDA_S)
    assert (theta >= lo).all() and (theta <= hi).all()

    # Saturated but not clipped: at a plausible optimiser excursion the gradient is
    # small and still strictly positive, so a step back off the wall exists.
    z = torch.tensor([[-20.0, 20.0, 15.0]], requires_grad=True)
    fam.to_physical(z, LAMBDA_S).sum().backward()
    assert (z.grad.abs() > 0).all()


def test_inversion_box_is_wider_than_the_training_support():
    """
    `Circle.bounds` is centre-based; the generator's keep-out is boundary-based.

    So the feasible box the inversion explores is strictly larger than the region any
    training sample came from.  At the corner of the box a maximal void reaches
    within (1.5 - 1.2) lambda_s = 0.3 lambda_s of the wall -- which is *inside* the
    receiver ring at 3 network cells -- whereas every training void keeps its whole
    boundary 1.5 lambda_s clear of every wall.

    generate.py says this gap is intentional, and it is worth pinning rather than
    closing: if the surrogate misbehaves outside its training support, that is a real
    failure mode of the method and it should be visible.  Clamping the optimiser into
    the training support would hide it and would also make the reported success rate
    meaningless, because the constraint would be doing the work.
    """
    lo, hi = G.Circle().bounds(LAMBDA_S)
    wall_clearance_at_box_corner = lo[0].item() - hi[2].item()
    ring_x = cfg.RING_INSET_NET * cfg.DX_NET

    assert wall_clearance_at_box_corner > 0.0, "the void must stay inside the domain"
    assert wall_clearance_at_box_corner < ring_x, (
        "the box is expected to reach past the receiver ring; if this is now false "
        "the keep-out has been tightened and the transfer claim needs revisiting")

    # The training support, by contrast, keeps the boundary a full keep-out clear.
    from src.data.generate import SRC_KEEPOUT_LS
    training_clearance = cfg.BOUNDARY_KEEPOUT_LS * LAMBDA_S
    assert training_clearance > ring_x
    assert 0.0 < SRC_KEEPOUT_LS < cfg.BOUNDARY_KEEPOUT_LS


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------
def test_soft_indicator_limits_and_monotonicity():
    assert G.soft_indicator(torch.zeros(1), 0.05).item() == pytest.approx(0.5)
    phi = torch.linspace(-1.0, 1.0, 201)
    chi = G.soft_indicator(phi, 0.05)
    assert chi[0].item() > 0.999          # deep inside the void
    assert chi[-1].item() < 0.001         # deep in the solid
    assert (chi.diff() < 0).all()         # strictly decreasing in phi
    assert (chi > 0).all() and (chi < 1).all()


def test_phi_tilde_is_clipped_in_cell_units():
    phi = torch.tensor([-100.0, -0.5, 0.0, 0.25, 100.0]) * cfg.DX_NET
    pt = G.phi_tilde(phi, cfg.DX_NET)
    assert pt.tolist() == pytest.approx(
        [-cfg.SDF_CLIP_CELLS, -0.5, 0.0, 0.25, cfg.SDF_CLIP_CELLS])
    assert pt.abs().max() <= cfg.SDF_CLIP_CELLS


def test_interface_width_is_physical_not_grid_relative():
    """
    chi at a fixed physical eps is the same field however finely it is sampled.

    The refinement factor is 3, not 2, so that coarse cell centres coincide exactly
    with every third fine cell centre -- the same odd-factor alignment trick the
    solver's grid-convergence check uses.  With an even factor there is no common
    point and the comparison would need interpolation, which would blur exactly the
    interface being tested.
    """
    fam = G.Circle()
    theta = torch.tensor([[4.13, 3.71, 0.55]], dtype=torch.float64)
    n = 64
    dx_c = cfg.L_DOMAIN / n
    eps = cfg.EPS_INTERFACE_CELLS * cfg.DX_NET          # a *length*

    yc_, xc_ = G.grid_coords(n, dx_c, dtype=torch.float64)
    yf_, xf_ = G.grid_coords(3 * n, dx_c / 3, dtype=torch.float64)
    chi_c = G.soft_indicator(fam.sdf(theta, yc_, xc_), eps)
    chi_f = G.soft_indicator(fam.sdf(theta, yf_, xf_), eps)
    assert torch.allclose(chi_c, chi_f[:, 1::3, 1::3], atol=1e-12)

    # Contrast: `geometry_channels` takes eps in *cells* of the dx it is given, so
    # calling it with the fine dx halves the physical width.  That is legitimate
    # for building network inputs on the network grid, and wrong for building
    # solver material fields -- which is precisely why validate.py defines
    # EPS_LEN_PHYS = EPS_INTERFACE_CELLS * DX_NET and passes the length around.
    _, chi_net = G.geometry_channels(theta, fam, dx=cfg.DX_NET, yy=yc_, xx=xc_)
    _, chi_fine = G.geometry_channels(theta, fam, dx=cfg.DX_FINE, yy=yc_, xx=xc_)
    assert not torch.allclose(chi_net, chi_fine, atol=1e-3)

    from src.solver import validate as V
    assert V.EPS_LEN_PHYS == pytest.approx(cfg.EPS_INTERFACE_CELLS * cfg.DX_NET)


def test_geometry_channels_shapes_and_defaults():
    fam = G.Circle()
    theta = torch.tensor([[4.0, 4.0, 0.6], [3.0, 5.0, 0.5]])
    pt, chi = G.geometry_channels(theta, fam)
    assert pt.shape == chi.shape == (2, cfg.N_NET, cfg.N_NET)
    assert chi.max() > 0.99 and chi.min() < 0.01
    pt_f, chi_f = G.geometry_channels(theta, fam, dx=cfg.DX_FINE)
    assert pt_f.shape == (2, cfg.N_FINE_TOTAL, cfg.N_FINE_TOTAL)


def test_material_fields_void_is_soft_but_not_lighter():
    """
    VOID_DENSITY_SCALE = 1.0: the moduli collapse, the density does not.  A void's
    physics is mu -> 0 (traction-free); nulling rho as well makes the local wave
    speed a 0/0 whose numerical value can exceed c_p and break the CFL condition
    that the whole time step was sized against.
    """
    lam0, mu0 = cfg.lame_from_nu(0.33)
    chi = torch.tensor([[0.0, 0.5, 1.0]])
    lam, mu, rho = G.material_fields(chi, lam0, mu0)

    assert lam[0, 0].item() == pytest.approx(lam0)
    assert mu[0, 0].item() == pytest.approx(mu0)
    assert lam[0, 2].item() == pytest.approx(lam0 * cfg.VOID_STIFFNESS_FLOOR)
    assert mu[0, 2].item() == pytest.approx(mu0 * cfg.VOID_STIFFNESS_FLOOR)
    assert mu[0, 1].item() == pytest.approx(0.5 * mu0)
    assert torch.allclose(rho, torch.full_like(rho, cfg.RHO0))

    # The floor keeps the local speed finite and *below* c_p, so no cell in the
    # void ever violates the global CFL condition.
    c_void = math.sqrt((lam[0, 2].item() + 2 * mu[0, 2].item()) / cfg.RHO0)
    assert 0.0 < c_void < cfg.CP


def test_material_fields_are_differentiable_in_chi():
    chi = torch.rand(1, 8, 8, requires_grad=True)
    lam, mu, rho = G.material_fields(chi, 1.0, 0.5)
    (lam.sum() + mu.sum()).backward()
    assert chi.grad is not None and chi.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# smooth_min and the transfer families
# ---------------------------------------------------------------------------
def test_smooth_min_approaches_min_from_below():
    a = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    b = torch.tensor([2.0, 3.0, 0.5], dtype=torch.float64)
    hard = torch.minimum(a, b)
    for k, tol in [(0.5, 0.4), (0.05, 5e-2), (1e-3, 1e-3)]:
        s = G.smooth_min(a, b, k)
        assert (s <= hard + 1e-12).all(), "smooth_min must under-estimate min"
        assert torch.allclose(s, hard, atol=tol)


def test_smooth_min_has_a_gradient_at_the_crossing():
    """A hard min puts a kink where the two SDFs cross, and L-BFGS's curvature
    estimate is meaningless across a kink.  At a = b the smooth version splits the
    gradient evenly instead of picking a branch arbitrarily."""
    a = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    G.smooth_min(a, b, 0.1).backward()
    assert a.grad.item() == pytest.approx(0.5)
    assert b.grad.item() == pytest.approx(0.5)


def test_ellipse_reduces_to_the_first_order_circle():
    """
    With a = b = A the normalised implicit function is exactly
    (r^2 - A^2) / (2 r), which vanishes on the boundary and agrees with the true
    distance to first order there.  Exactness *on* the boundary is all that is
    needed, because that is the only place chi has support.
    """
    fam = G.Ellipse()
    A = 0.7
    theta = torch.tensor([[4.0, 4.0, A, A, 0.0]], dtype=torch.float64)
    yy, xx = G.grid_coords(48, cfg.L_DOMAIN / 48, dtype=torch.float64)
    phi = fam.sdf(theta, yy, xx)[0]
    r = torch.sqrt((xx - 4.0) ** 2 + (yy - 4.0) ** 2)
    assert torch.allclose(phi, (r ** 2 - A ** 2) / (2.0 * r), atol=1e-9)
    near = (r - A).abs() < 0.05
    assert torch.allclose(phi[near], (r - A)[near], atol=3e-3)


def test_ellipse_rotation_is_a_rotation():
    """alpha = pi/2 with (a, b) swapped is the same shape."""
    fam = G.Ellipse()
    yy, xx = G.grid_coords(48, cfg.L_DOMAIN / 48, dtype=torch.float64)
    t1 = torch.tensor([[4.0, 4.0, 0.9, 0.5, 0.0]], dtype=torch.float64)
    t2 = torch.tensor([[4.0, 4.0, 0.5, 0.9, math.pi / 2]], dtype=torch.float64)
    assert torch.allclose(fam.sdf(t1, yy, xx), fam.sdf(t2, yy, xx), atol=1e-9)


def test_two_circle_is_the_union_away_from_the_crossing():
    fam = G.TwoCircle()
    one = G.Circle()
    theta = torch.tensor([[3.0, 4.0, 0.5, 5.2, 4.0, 0.5]], dtype=torch.float64)
    yy, xx = G.grid_coords(64, cfg.L_DOMAIN / 64, dtype=torch.float64)
    p1 = one.sdf(theta[:, 0:3], yy, xx)
    p2 = one.sdf(theta[:, 3:6], yy, xx)
    phi = fam.sdf(theta, yy, xx)
    # Away from where the two distances are comparable the blend is inactive.  The
    # threshold is 25 blend widths, not a handful: the correction decays like
    # k exp(-|p1-p2|/k), so "clearly separated" for a 1e-9 tolerance means ~20 k.
    away = (p1 - p2).abs() > 25.0 * fam.blend
    assert away.any()
    assert torch.allclose(phi[away], torch.minimum(p1, p2)[away], atol=1e-9)
    assert (phi <= torch.minimum(p1, p2) + 1e-12).all()
    # Two separate voids, so the shape the network never trained on really is a
    # different topology rather than a deformed circle: a line through both centres
    # crosses the chi = 0.5 contour four times.
    chi = G.soft_indicator(phi, cfg.EPS_INTERFACE_CELLS * cfg.DX_NET)
    assert chi.max() > 0.99
    row = chi[0, 32]
    inside = (row > 0.5).to(torch.int8)
    transitions = int(inside.diff().abs().sum().item())
    assert transitions == 4, "expected two separate voids along the centre row"


def test_family_registry_is_complete():
    assert set(G.FAMILIES) == {"circle", "ellipse", "two_circle"}
    for name, fam in G.FAMILIES.items():
        assert fam.name == name
        assert isinstance(fam, G.ShapeFamily)
