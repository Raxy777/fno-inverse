"""
config.py is the single source of truth, so it gets the strictest tests.

Most of this file pins *exact integers*.  That is deliberate: config.py already
contains `self_check`, which asserts the internal consistency of every derived
quantity, and re-testing that here would only prove that assert statements assert.
What `self_check` cannot do is notice that a number disagrees with the architecture
document -- it has no access to the document.  So the job of this file is to be the
place where the document's claims and the code's arithmetic are confronted, and where
any discrepancy is written down with a verdict rather than quietly tolerated.

Three discrepancies are recorded below, in `test_parameter_count_corrects_the_document`
and `test_dispersion_budget`.  In each case the code is right and the document's
quoted figure is a rounding or algebra slip; the test pins the code's value so that
propagating the fix into the .md cannot silently change the implementation.
"""

from __future__ import annotations

import math

import pytest

from src import config as cfg


# ---------------------------------------------------------------------------
# The built-in self-check
# ---------------------------------------------------------------------------
def test_self_check_passes():
    """Every internal consistency claim in config.py, in one call."""
    cfg.self_check(verbose=False)


# ---------------------------------------------------------------------------
# Grids and time stepping
# ---------------------------------------------------------------------------
def test_grid_ratios():
    assert cfg.N_FINE == 2 * cfg.N_NET
    assert cfg.DOWNSAMPLE == 2
    assert cfg.DX_NET == pytest.approx(cfg.LAMBDA_P / 16)
    assert cfg.DX_FINE == pytest.approx(cfg.LAMBDA_P / 32)
    assert cfg.N_FINE_TOTAL == cfg.N_FINE + 2 * cfg.N_PML_FINE == 316
    # The absorber is a whole number of network cells thick, so a network-grid
    # index maps to a fine index without a half-cell fudge.
    assert cfg.N_PML_FINE % cfg.DOWNSAMPLE == 0


def test_cfl_honours_the_stated_safety_factor():
    """
    The document's section 3.3 quotes both a 0.9 safety factor and dt = 0.6 dx/c_p.
    Those are inconsistent -- 0.6 / 0.606092 = 0.990, i.e. no margin at all -- and
    the implementation honours the safety factor, which moves n_t from the
    document's 1280 to 1408.  This test pins that choice so it cannot drift back.
    """
    assert cfg.CFL_LIMIT_4TH == pytest.approx(6.0 / (7.0 * math.sqrt(2.0)))
    assert cfg.CFL_LIMIT_4TH == pytest.approx(0.606092, abs=1e-6)
    assert cfg.NT == 1408
    assert cfg.CFL_NUMBER == pytest.approx(0.545455, abs=1e-6)
    assert cfg.CFL_NUMBER / cfg.CFL_LIMIT_4TH == pytest.approx(0.9, abs=0.002)
    # dt divides T_end exactly, so the DFT window is exactly n_t samples long and
    # the midpoint quadrature has no ragged final step.
    assert cfg.DT * cfg.NT == pytest.approx(cfg.T_END, rel=1e-14)


def test_frame_saving_divides_evenly():
    assert cfg.SAVE_EVERY * cfg.N_SAVED_FRAMES == cfg.NT
    assert cfg.SAVE_EVERY == 22


def test_temporal_sampling_of_the_carrier():
    """~59 steps per carrier period; the burst is resolved, not just stable."""
    steps_per_period = cfg.T_P / cfg.DT
    assert steps_per_period > 40.0


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def test_shear_speed_and_lame():
    for nu in cfg.NU_LIST:
        r = cfg.cs_over_cp(nu)
        lam, mu = cfg.lame_from_nu(nu)
        # c_s^2 / c_p^2 = mu / (lam + 2 mu), with rho = 1
        assert r ** 2 == pytest.approx(mu / (lam + 2.0 * mu), rel=1e-12)
        # c_p = 1 by construction
        assert lam + 2.0 * mu == pytest.approx(cfg.RHO0 * cfg.CP ** 2, rel=1e-12)
        # nu recovered from the Lame pair
        assert lam / (2.0 * (lam + mu)) == pytest.approx(nu, rel=1e-12)


def test_worst_case_material_is_the_largest_nu():
    """
    Larger nu means a softer shear modulus, a slower S wave, a shorter S
    wavelength, and therefore the tightest resolution and mode-truncation demands.
    Everything sized "worst case" in config.py must be sized on nu = 0.37.
    """
    assert cfg.NU_WORST == max(cfg.NU_LIST) == 0.37
    speeds = [cfg.cs_over_cp(nu) for nu in cfg.NU_LIST]
    assert speeds == sorted(speeds, reverse=True)
    assert cfg.CS_MIN == pytest.approx(0.454256, abs=1e-6)
    assert cfg.LAMBDA_S_MIN == pytest.approx(0.454256, abs=1e-6)


def test_material_table():
    m = cfg.material(0.37)
    assert m.ppw_s_net == pytest.approx(7.268, abs=1e-3)
    assert m.ppw_s_fine == pytest.approx(14.536, abs=1e-3)
    assert m.domain_in_lambda_s == pytest.approx(17.611, abs=1e-3)
    assert m.cs == pytest.approx(0.454256, abs=1e-6)
    assert len(cfg.MATERIALS) == len(cfg.NU_LIST)


def test_physical_units_report():
    """1 lambda_p = 25.2 mm in 250 kHz aluminium, so a void is 11..27 mm across."""
    assert cfg.to_mm(1.0) == pytest.approx(25.2, rel=1e-9)
    d_min = cfg.to_mm(2.0 * cfg.R_MIN_LS * cfg.LAMBDA_S_MIN)
    d_max = cfg.to_mm(2.0 * cfg.R_MAX_LS * cfg.LAMBDA_S_MIN)
    assert 8.0 < d_min < 12.0
    assert 24.0 < d_max < 30.0


# ---------------------------------------------------------------------------
# Numerical dispersion (section 3.4)
# ---------------------------------------------------------------------------
def test_dispersion_budget():
    """
    Why 4th order on a doubly-refined grid, in three numbers.

    All three are accumulated phase error for the *shortest* wave in the problem
    (S at nu = 0.37) crossing the whole 8 lambda_p domain, i.e. 17.6 shear
    wavelengths.  The budget is pi/4 = 0.785 rad; beyond that the phase error is a
    sizeable fraction of the cycle-skipping half-period and the inversion would be
    fitting the solver's dispersion rather than the defect.

        2nd order, network spacing (7.27 ppw)  -> 3.44 rad   ~ pi, unusable
        4th order, network spacing (7.27 ppw)  -> 0.290 rad  passes, 2.7x margin
        4th order, fine    spacing (14.5 ppw)  -> 0.018 rad  production, 43x margin

    The document's section 3.4 quotes the first of these as the motivation, and
    3.44 rad is the arithmetic behind that claim.  The middle row is the one worth
    staring at: even 4th order on the network grid would pass, and it is the
    combination of the two refinements that buys the margin to spend elsewhere
    (on the void-interface averaging, which is where the accuracy actually goes).
    """
    m = cfg.material(cfg.NU_WORST)
    n_lam = m.domain_in_lambda_s

    phi_2_net = cfg.accumulated_phase(m.ppw_s_net, 2, n_lam)
    phi_4_net = cfg.accumulated_phase(m.ppw_s_net, 4, n_lam)
    phi_4_fine = cfg.accumulated_phase(m.ppw_s_fine, 4, n_lam)

    assert phi_2_net == pytest.approx(3.445, abs=5e-3)
    assert phi_4_net == pytest.approx(0.2897, abs=5e-4)
    assert phi_4_fine == pytest.approx(0.01810, abs=5e-5)

    budget = math.pi / 4.0
    assert phi_2_net > budget, "2nd order should blow the budget; that is the point"
    assert phi_4_fine < budget / 20.0

    # The order-4 error must fall like dx^4: halving dx divides it by 16.
    assert phi_4_net / phi_4_fine == pytest.approx(16.0, rel=1e-9)


def test_dispersion_error_rejects_other_orders():
    with pytest.raises(ValueError):
        cfg.dispersion_error(10.0, 3)


# ---------------------------------------------------------------------------
# Frequency band vs the Hann nulls
# ---------------------------------------------------------------------------
def test_band_sits_inside_the_hann_main_lobe():
    """
    A Hann-windowed N_c-cycle burst has exact spectral nulls at
    f_c (1 +- 2/N_c) = 0.6 and 1.4 f_c.  The operating band must stay strictly
    inside them, because the deconvolution divides by |s_hat| and a null is an
    infinity.  Note how little headroom the upper edge has: 1.340 against 1.4 is
    0.06 f_c, which is exactly why harmonic.MAX_DECONV_AMPLIFICATION exists and is
    checked rather than assumed.
    """
    null_lo = cfg.FC * (1.0 - 2.0 / cfg.N_CYCLES)
    null_hi = cfg.FC * (1.0 + 2.0 / cfg.N_CYCLES)
    assert null_lo == pytest.approx(0.6)
    assert null_hi == pytest.approx(1.4)

    f_lo, f_hi = cfg.FREQS[0], cfg.FREQS[-1]
    assert f_lo == pytest.approx(0.66)
    assert f_hi == pytest.approx(1.3402, abs=1e-4)
    assert null_lo + 0.04 < f_lo
    assert f_hi < null_hi - 0.04
    assert len(cfg.FREQS) == cfg.M_FREQ == 20
    assert cfg.OMEGAS[0] == pytest.approx(2.0 * math.pi * cfg.FREQS[0])


def test_continuation_bands_are_nested_prefixes():
    """
    Frequency continuation only works if each stage's band *contains* the previous
    one; a stage that swapped in a disjoint set would throw away the low-frequency
    constraint that made the previous stage's minimum trustworthy.
    """
    s1, s2, s3 = cfg.BAND_STAGE1, cfg.BAND_STAGE2, cfg.BAND_STAGE3
    for s in (s1, s2, s3):
        assert s.start == 0 and s.step in (None, 1)
    assert s1.stop < s2.stop < s3.stop == cfg.M_FREQ
    assert (s1.stop, s2.stop, s3.stop) == (6, 10, 20)


# ---------------------------------------------------------------------------
# Mode truncation (section 6.4)
# ---------------------------------------------------------------------------
def test_kmax_covers_the_burst_and_stays_under_nyquist():
    assert cfg.K_CARRIER_WORST == pytest.approx(17.611, abs=1e-3)
    assert cfg.K_REQUIRED == pytest.approx(24.655, abs=1e-3)
    assert cfg.K_REQUIRED <= cfg.KMAX <= cfg.K_NYQUIST
    # Two half-spectrum quadrants of width kmax must fit the rfft2 layout.
    assert 2 * cfg.KMAX <= cfg.N_NET
    assert cfg.KMAX <= cfg.N_NET // 2 + 1


def test_parameter_count_corrects_the_document():
    """
    Exact lattice counts, not the document's rounded figures.

    Section 6.4 quotes 2.52 M parameters per spectral layer and 10.1 M in total for
    the radial mask at d_v = 32, kmax = 28.  The exact counts are 2,566,144 and
    10,276,452 -- the document is low by about 1.8%, which is a rounding slip, not
    a design disagreement.  Pinned here as integers so that fixing the .md cannot
    accidentally become a change to the model.

    The lattice counts themselves are worth writing down because they are the only
    place the geometry of the mask is visible: 640 kept modes in the ky >= 0
    quadrant and 613 in the ky < 0 one.  They differ by 27 because the two blocks
    are not symmetric -- w1 includes the ky = 0 row, w2 does not, and w2's
    ky = -kmax row keeps exactly one mode (kx = 0), since 784 = 28^2 has no
    representation as a sum of two nonzero squares.
    """
    kept_radial = cfg.spectral_params(cfg.D_V, cfg.KMAX, True) // (2 * cfg.D_V ** 2)
    assert kept_radial == 640 + 613 == 1253

    per_layer = cfg.spectral_params(cfg.D_V, cfg.KMAX, True)
    assert per_layer == 2_566_144

    per_block = cfg.block_params(cfg.D_V, cfg.KMAX, True)
    assert per_block == per_layer + cfg.D_V ** 2 + cfg.D_V == 2_567_200

    total = cfg.total_params()
    assert total == 10_276_452
    lift = cfg.C_IN * cfg.LIFT_HIDDEN + cfg.LIFT_HIDDEN \
        + cfg.LIFT_HIDDEN * cfg.D_V + cfg.D_V
    proj = cfg.D_V * cfg.PROJ_HIDDEN + cfg.PROJ_HIDDEN \
        + cfg.PROJ_HIDDEN * cfg.C_OUT + cfg.C_OUT
    assert (lift, proj) == (2912, 4740)
    assert total == cfg.N_BLOCKS * per_block + lift + proj


def test_radial_mask_keeps_pi_over_four():
    """
    The radial mask keeps the disc |k| <= kmax out of a 2 kmax x kmax rectangle,
    so the asymptotic kept fraction is (pi kmax^2 / 2) / (2 kmax^2) = pi/4 = 0.785.
    At kmax = 28 the lattice boundary rounds that up to 0.799.

    The 20% it discards is not free: those are the corner modes out to
    sqrt(2) kmax, and dropping them is what makes the truncation isotropic.  A
    square mask resolves diagonal features about 41% better than axial ones, which
    on this problem means the network would be measurably better at finding voids
    on the diagonals than on the axes -- a direction-dependent detector, which is
    much worse than a uniformly slightly-blurrier one.
    """
    radial = cfg.spectral_params(cfg.D_V, cfg.KMAX, True)
    square = cfg.spectral_params(cfg.D_V, cfg.KMAX, False)
    assert square == 2 * cfg.D_V ** 2 * 2 * cfg.KMAX ** 2 == 3_211_264
    frac = radial / square
    assert frac == pytest.approx(0.79911, abs=1e-5)
    assert frac == pytest.approx(math.pi / 4.0, abs=0.02)
    assert math.sqrt(2.0) == pytest.approx(1.41, abs=0.01)  # the 41% of the docstring


def test_variants_are_ordered_by_capacity():
    """VARIANTS holds kwargs dicts, because `models.build` splats them into FNO2d."""
    p = {k: cfg.total_params(**v) for k, v in cfg.VARIANTS.items()}
    assert p["primary"] > p["small"] > p["tiny"]
    assert cfg.VARIANTS["primary"] == dict(d_v=cfg.D_V, kmax=cfg.KMAX)
    assert set(cfg.VARIANTS) == {"primary", "small", "tiny"}


# ---------------------------------------------------------------------------
# Ring geometry
# ---------------------------------------------------------------------------
def test_ring_positions_are_distinct_and_inset():
    recv = cfg.RECEIVERS_NET
    src = cfg.SOURCES_NET
    assert len(recv) == cfg.N_RECV == 32
    assert len(src) == cfg.N_SRC == 8
    assert len(set(recv)) == len(recv), "duplicate receiver -- a corner was hit"
    assert len(set(src)) == len(src)
    # No source sits on a receiver: a co-located pair would make one column of the
    # data vector the incident field itself, which the scattered-field subtraction
    # would then zero.
    assert not (set(recv) & set(src))
    lo, hi = cfg.RING_INSET_NET, cfg.N_NET - 1 - cfg.RING_INSET_NET
    for iy, ix in recv + src:
        assert lo <= iy <= hi and lo <= ix <= hi
        assert iy in (lo, hi) or ix in (lo, hi), "ring point left the ring"


def test_held_out_sources_are_excluded_from_training_pool():
    assert set(cfg.SRC_HELDOUT) == {3, 6}
    assert set(cfg.SRC_TRAIN) | set(cfg.SRC_HELDOUT) == set(range(cfg.N_SRC))
    assert not (set(cfg.SRC_TRAIN) & set(cfg.SRC_HELDOUT))
    assert len(cfg.SRC_TRAIN) == 6


def test_net_to_fine_lands_inside_the_physical_region():
    for iy, ix in cfg.RECEIVERS_NET + cfg.SOURCES_NET:
        fy, fx = cfg.net_to_fine(iy, ix)
        assert cfg.N_PML_FINE <= fy < cfg.N_PML_FINE + cfg.N_FINE
        assert cfg.N_PML_FINE <= fx < cfg.N_PML_FINE + cfg.N_FINE


def test_source_position_exposes_the_quarter_cell_offset():
    """
    With DOWNSAMPLE = 2 no fine cell is centred on a network cell centre, so the
    injected force sits dx_net/4 = 0.0156 lambda_p below and left of nominal.
    Receivers do not have this problem because a 2x2 average *is* centred.

    This test exists because the offset is the kind of thing that gets "cleaned up"
    by someone recomputing (ix + 0.5) * dx_net at a call site, at which point the
    physics loss and the inversion disagree with the solver by a quarter cell and
    the recovered positions acquire a small fixed bias -- in the quantity being
    measured, and in a direction that looks like a real result.
    """
    offset = cfg.DX_NET / 4.0
    for i in range(cfg.N_SRC):
        iy, ix = cfg.SOURCES_NET[i]
        x, y = cfg.source_position(i)
        nominal_x, nominal_y = (ix + 0.5) * cfg.DX_NET, (iy + 0.5) * cfg.DX_NET
        assert nominal_x - x == pytest.approx(offset, rel=1e-12)
        assert nominal_y - y == pytest.approx(offset, rel=1e-12)
    assert offset == pytest.approx(0.015625)

    for i in range(cfg.N_RECV):
        iy, ix = cfg.RECEIVERS_NET[i]
        x, y = cfg.receiver_position(i)
        assert x == pytest.approx((ix + 0.5) * cfg.DX_NET)
        assert y == pytest.approx((iy + 0.5) * cfg.DX_NET)

    assert cfg.SOURCE_XY[0] == cfg.source_position(0)
    assert cfg.RECEIVER_XY[0] == cfg.receiver_position(0)
    assert all(0.0 < v < cfg.L_DOMAIN for xy in cfg.SOURCE_XY for v in xy)


# ---------------------------------------------------------------------------
# Void size, keep-out, and the screening grid
# ---------------------------------------------------------------------------
def test_void_radius_range_is_resolvable_and_scattering():
    """R from 0.4 to 1.2 lambda_s: above Rayleigh, below the domain scale."""
    r_min = cfg.R_MIN_LS * cfg.LAMBDA_S_MIN
    r_max = cfg.R_MAX_LS * cfg.LAMBDA_S_MIN
    assert r_min / cfg.DX_NET > 2.5, "smallest void must span several network cells"
    assert 2.0 * r_max < cfg.L_DOMAIN / 4.0, "largest void must not fill the domain"
    assert cfg.EPS_INTERFACE_CELLS * cfg.DX_NET < r_min / 3.0, (
        "interface smoothing must be small against the smallest radius")


def test_screen_grid_is_finer_than_the_envelope_basin_but_not_the_waveform_basin():
    """
    The 16 x 16 screen steps 0.41 lambda_p across the interior.  Two basin scales
    matter and the screen sits between them, which is the whole argument for Stage 1
    using a phase-free misfit:

        waveform basin   lambda_s / 4          = 0.11    screen step is 3.7x too coarse
        envelope basin   N_c lambda_s / 2      = 1.14    screen step is 2.7x finer

    So a screen scored on the ordinary complex misfit would step clean over the true
    minimum -- 256 candidates and none of them in the basin -- and the failure would
    present as a bad forward model rather than as a sampling artefact.  Scored on the
    amplitude (envelope) misfit, whose basin is N_c/2 times wider because it is blind
    to carrier phase, the coverage is guaranteed by a factor of nearly three.

    config.self_check asserts the second inequality against the nu = 1/3 reference
    material; this test asserts it at the *worst* material, where lambda_s is 9%
    shorter and the basin correspondingly tighter.
    """
    interior = cfg.L_DOMAIN - 2.0 * cfg.BOUNDARY_KEEPOUT_LS * cfg.LAMBDA_S_MIN
    step = interior / cfg.SCREEN_GRID
    basin_waveform = cfg.LAMBDA_S_MIN / 4.0
    basin_envelope = cfg.N_CYCLES * cfg.LAMBDA_S_MIN / 2.0

    assert step == pytest.approx(0.4148, abs=1e-3)
    assert step > basin_waveform, (
        "if this ever becomes false the phase-free Stage 1 misfit is no longer "
        "load-bearing and §8.4 should say so")
    assert step < basin_envelope / 2.0
    assert cfg.N_SURVIVORS < cfg.SCREEN_GRID ** 2


def test_void_material_scaling():
    """
    VOID_DENSITY_SCALE = 1.0 is a documented deviation: section 7.2 writes the
    contrast as delta-rho = -rho_0 chi, but scaling density to zero inside the void
    makes the explicit time step's local CFL condition blow up (c = sqrt(mu/rho)
    with both going to zero is a 0/0 whose numerical value is set by the
    stiffness floor).  Keeping rho fixed and softening only the moduli gives a
    traction-free hole with a stable step; the physics of a void is in mu -> 0.
    """
    assert cfg.VOID_DENSITY_SCALE == 1.0
    assert 0.0 < cfg.VOID_STIFFNESS_FLOOR <= 1e-3
    assert cfg.ERODE_CELLS == cfg.STENCIL_HALF_WIDTH + 1 == 3


def test_gates_are_all_positive_and_ordered():
    """Every acceptance gate must exist and be a number, not a feeling."""
    gates = {k: v for k, v in vars(cfg).items() if k.startswith("GATE_")}
    assert len(gates) >= 9
    assert all(isinstance(v, (int, float)) and v > 0 for v in gates.values())
    assert cfg.GATE_GRAD_SIGFIGS == 3
    assert cfg.GATE_POSITION_LS == 0.10
    assert cfg.GATE_SUCCESS_RATE == 0.90
    # The label noise floor must be tighter than the accuracy claimed against it.
    assert cfg.GATE_GRID_CONVERGENCE < cfg.GATE_REL_L2
