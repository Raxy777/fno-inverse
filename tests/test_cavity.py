"""
The analytic cavity reference, verified against calculus (§2.4, §3.6-3.7).

`solver/cavity.py` is the thing that decides whether the labels are cavity
scattering, so it is the one piece of the project that cannot be checked by
comparing it against something else in the project.  Everything here is therefore a
self-contained argument: the Green's tensor is substituted into the Navier operator,
the traction matrix is compared against differentiating the field it claims to be
the traction of, and the matched total field is checked to have no traction left on
the cavity wall.  A reference that passes all three satisfies the PDE, satisfies the
boundary condition and is outgoing, which by uniqueness leaves nothing else it could
be.

No FDTD here and no torch: these run in about a second, so they run every time.  The
comparison against the solver is `solver/validate.py`'s job and is slow.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import config as cfg
from src.solver import cavity as CAV


@pytest.fixture(scope="module")
def report():
    return CAV.self_test()


# ---------------------------------------------------------------------------
# The Green's tensor solves the equation it is supposed to solve
# ---------------------------------------------------------------------------
def test_navier_residual_converges_second_order(report):
    res = report["navier_residual"]
    assert res[0] < 0.1, f"residual already large at the coarsest h: {res}"
    assert res == sorted(res, reverse=True)
    # 2nd-order FD of a smooth field: halving h must buy a factor of ~4
    assert report["navier_order"] == pytest.approx(2.0, abs=0.1), report["navier_order"]
    assert res[-1] < 5e-3


def test_swapped_green_ordering_is_not_a_solution(report):
    """`(g_p - g_s)` is the plausible-looking ordering and it is wrong.

    Pinned so that "simplifying" the sign later fails loudly instead of producing a
    reference that quietly disagrees with the solver by a constant factor.
    """
    flipped = report["navier_flipped"]
    assert min(flipped) > 1.0
    assert abs(flipped[0] - flipped[1]) / flipped[0] < 0.05   # not converging at all


def test_symmetry_and_far_field(report):
    assert report["symmetry"] < 1e-12
    assert report["far_field"] < 1e-3


def test_graf_expansion_reproduces_direct_evaluation(report):
    """Two unrelated computations of the incident field must agree."""
    assert report["graf_vs_direct"] < 1e-5


# ---------------------------------------------------------------------------
# The boundary condition
# ---------------------------------------------------------------------------
def test_traction_matrix_matches_differentiated_field(report):
    fd = report["traction_fd"]
    for kind in ("regular", "outgoing"):
        coarse, fine = fd[f"{kind}_h0.001"], fd[f"{kind}_h0.00025"]
        assert coarse < 1e-3, (kind, coarse)
        ratio = coarse / fine
        assert 8.0 < ratio < 32.0, f"{kind}: expected ~16x for h/4, got {ratio:.1f}"


def test_matched_field_has_no_traction_on_the_wall(report):
    assert report["traction_bc_residual"] < 1e-10


def test_scattering_vanishes_with_the_cavity(report):
    ratios = report["scattered_over_incident"]
    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[-1] < 0.1 * ratios[0]


# ---------------------------------------------------------------------------
# Discrete-operator symbols (repository finding (d))
# ---------------------------------------------------------------------------
def test_symbol_formulas_match_the_stencils(report):
    s = report["symbols"]
    for name in ("collocated", "staggered", "product"):
        assert s[f"{name}_formula"] == pytest.approx(s[f"{name}_stencil"], rel=1e-12), name


def test_operator_defect_is_reported_at_the_production_grid(report):
    d = report["operator_defect"]
    assert d["points_per_wavelength"] == pytest.approx(
        cfg.cs_over_cp(1 / 3) * cfg.CP / (cfg.FREQS[-1] * cfg.DX_NET), rel=1e-9)
    # the loss operator is the less accurate of the two, and the product rule is
    # the dominant defect.  Both are the point of the check, so both are pinned.
    assert d["collocated"] < d["staggered"] < 1.0
    assert d["product_rule"] < 0.75
    assert 0.0 < d["derivative_mismatch"] < 1.0


def test_product_rule_symbol_is_fourth_order():
    """The t^2 term has to cancel, or the operator is not 4th order."""
    small = np.array([0.05, 0.025])
    err = 1.0 - CAV.product_rule_symbol(small)
    assert err[0] / err[1] == pytest.approx(16.0, rel=0.05), err


# ---------------------------------------------------------------------------
# Geometry and API contracts
# ---------------------------------------------------------------------------
def test_green_displacement_layout_and_force_column():
    """`green_displacement` must be the `+y` column of `green_tensor`, in [R,2,M]."""
    src = cfg.source_position(0)
    recv = [cfg.receiver_position(i) for i in (0, 7, 19)]
    freqs = cfg.FREQS[:3]
    u = CAV.green_displacement(src, recv, freqs, nu=1 / 3)
    assert u.shape == (3, 2, 3)
    _, cs, _, _, _ = CAV._speeds(1 / 3)
    d = np.asarray(recv[1]) - np.asarray(src)
    g = CAV.green_tensor(d[0], d[1], 2 * np.pi * float(freqs[2]), cs=cs)
    assert u[1, 0, 2] == pytest.approx(g[0, 1], rel=1e-12)
    assert u[1, 1, 2] == pytest.approx(g[1, 1], rel=1e-12)


def test_cavity_field_rejects_interior_points():
    with pytest.raises(ValueError):
        CAV.cavity_field((2.0, 0.0), [(0.1, 0.0)], [1.0], radius=0.5, nu=1 / 3)


def test_cavity_series_is_truncated_far_past_its_tail():
    f = CAV.cavity_field(cfg.source_position(0),
                         [cfg.receiver_position(i) for i in range(0, 32, 8)],
                         cfg.FREQS[::8], radius=cfg.R_MAX_LS * cfg.LAMBDA_S_MIN,
                         centre=(cfg.L_DOMAIN / 2, cfg.L_DOMAIN / 2), nu=1 / 3)
    assert f.tail < 1e-6, f"series truncated too early: tail {f.tail:.2e}"
    assert f.n_modes >= 12


def test_accuracy_report_separates_calibration_from_shape():
    ref = CAV.cavity_field(cfg.source_position(0),
                           [cfg.receiver_position(i) for i in range(0, 32, 4)],
                           cfg.FREQS[:4], radius=0.4,
                           centre=(cfg.L_DOMAIN / 2, cfg.L_DOMAIN / 2), nu=1 / 3)
    scale = 1.3 * np.exp(0.21j)
    rep = CAV.cavity_accuracy_report(scale * ref.scattered, ref.scattered)
    assert rep["amp_ratio"] == pytest.approx(1.3, rel=1e-9)
    assert rep["phase_rad"] == pytest.approx(0.21, rel=1e-9)
    assert rep["rel_l2_cal"] < 1e-12
    assert rep["rel_l2"] > 0.3
    # a conjugated reference must not look better than the true one
    assert rep["rel_l2_conj"] > rep["rel_l2"]


def test_accuracy_report_rejects_mismatched_shapes():
    a = np.zeros((2, 2, 2), dtype=complex)
    with pytest.raises(ValueError):
        CAV.cavity_accuracy_report(a, np.zeros((2, 2, 3), dtype=complex))
