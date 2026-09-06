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
    assert cfg.N_FINE_TOTAL == cfg.N_FINE + 2 * cfg.N_PML_FINE == 376
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


def test_source_force_position_adds_the_y_face_offset():
    """
    The force enters `vy`, which lives on y-faces, so the point force is half a
    *fine* cell above `source_position` -- a different offset from the quarter net
    cell above, and on top of it.

    Worth its own test because the two are a plausible-looking pair to collapse into
    one, and 0.5 dx_fine is 0.26 rad of shear phase at the top of the band, larger
    than GATE_CAVITY_PHASE_RAD.  `solver.validate.check_green_incident` compares the
    solver against an analytic Green's function evaluated here, and would fail for
    the wrong reason if this drifted.
    """
    for i in range(cfg.N_SRC):
        x, y = cfg.source_position(i)
        fx, fy = cfg.source_force_position(i)
        assert fx == pytest.approx(x, rel=1e-12)
        assert fy - y == pytest.approx(0.5 * cfg.DX_FINE, rel=1e-12)
    assert cfg.SOURCE_FORCE_XY[0] == cfg.source_force_position(0)
    assert cfg.source_force_position(0) == pytest.approx((2.078125, 0.21875))


def test_absorber_is_labelled_as_a_sponge():
    """
    Finding (c) of the architecture review: the boundary treatment is a graded
    damping sponge, not a split-field PML, and the write-up used to call it a PML.
    The name is a config constant so that generated captions read it instead of
    asserting it, and `ABSORBER_R_TARGET` is a *design input* to `absorber_d0`,
    not an achieved reflection -- `solver.validate.check_absorber_reflection`
    measures what the layer actually does.

    The thickness rule asserted below is the review's other half, "validate the sponge
    honestly", turned into a constraint.  It is measured, not conventional: against a
    padded open domain the artefact on the incident ring field is 8.1% at 0.74
    lambda_p(f_lo), 4.4% at 0.91 and 1.0% at 1.24, and at the two thin thicknesses no
    order in 4..6 and no R_target in 1e-2..1e-1 reached GATE_ABSORBER_REFLECTION.  So
    the binding quantity is thickness measured in the *longest* wavelength in the band,
    and one of those is the floor.  The v2.0 layer was 0.62 and failed the gate 11x.
    """
    assert cfg.ABSORBER_KIND == "graded sponge"
    assert "pml" not in cfg.ABSORBER_KIND.lower()
    # the deprecated aliases still point at the primary constants
    assert cfg.N_PML_FINE == cfg.N_ABSORBER_FINE
    assert cfg.N_PML_NET == cfg.N_ABSORBER_NET
    assert cfg.N_ABSORBER_FINE == cfg.DOWNSAMPLE * cfg.N_ABSORBER_NET
    assert cfg.PML_ORDER == cfg.ABSORBER_ORDER
    assert cfg.PML_R_TARGET == cfg.ABSORBER_R_TARGET
    # more than one wavelength thick at the *lowest* frequency in the band
    lam_p_lo = cfg.CP / min(cfg.FREQS)
    assert lam_p_lo == pytest.approx(1.5151515, rel=1e-6)
    thickness = cfg.N_ABSORBER_FINE * cfg.DX_FINE
    assert thickness / lam_p_lo > 1.0, "below one lambda_p(f_lo) the gate is unreachable"
    assert thickness / lam_p_lo == pytest.approx(1.2375, rel=1e-4)
    # R_target sits at the measured interior minimum, not at the smallest value: d0
    # grows like -ln(R_target) and it is the gradient of d that an unmatched layer
    # reflects, so the v2.0 reasoning "smaller is better" had the sign wrong.
    assert cfg.ABSORBER_R_TARGET == 3.0e-2
    assert cfg.absorber_d0() == pytest.approx(4.67541, rel=1e-4)
    assert cfg.absorber_d0() < cfg.absorber_d0(r_target=1.0e-4)
    assert cfg.absorber_d0() * cfg.DT < 0.2, "damping must be gentle per time step"
    # the padded grid follows the thickness
    assert cfg.N_FINE_TOTAL == cfg.N_FINE + 2 * cfg.N_ABSORBER_FINE == 376
    cost = (cfg.N_FINE_TOTAL / (cfg.N_FINE + 2 * 30)) ** 2
    assert cost == pytest.approx(1.4159, rel=1e-3), "cost multiplier vs the v2.0 layer"


# ---------------------------------------------------------------------------
# Void size, keep-out, and the screening grid
# ---------------------------------------------------------------------------
def test_void_radius_range_is_resolvable_and_scattering():
    """R from 0.4 to 1.2 lambda_s: above Rayleigh, below the domain scale."""
    r_min = cfg.R_MIN_LS * cfg.LAMBDA_S_MIN
    r_max = cfg.R_MAX_LS * cfg.LAMBDA_S_MIN
    assert r_min / cfg.DX_NET > 2.5, "smallest void must span several network cells"
    assert 2.0 * r_max < cfg.L_DOMAIN / 4.0, "largest void must not fill the domain"


def test_interface_transition_is_narrower_than_the_smallest_void():
    """
    The former largest physical discrepancy, now closed by measurement.

    An older version of this test asserted `EPS_INTERFACE_CELLS * DX_NET < r_min / 3`,
    i.e. "the interface smoothing is small against the smallest radius", and it passed
    -- on both sides of a unit error.  `eps` is the sigmoid's *scale*, not its width:
    chi = sigmoid(-phi/eps) crosses from 10% to 90% over 2 ln(9) eps = 4.39 eps.  At
    the v2.0 width of 1.5 network cells the honest comparison read

        eps                        1.5 net cells = 0.094
        10-90% transition          6.59 cells    = 0.412      4.39x larger
        transition / lambda_s      0.91                       worst-case material
        transition / r_min         2.27                       NOT < 1/3.  Not < 1.

    so at the small end of the radius range the model was not a slightly smoothed
    cavity: the transition was more than twice as wide as the defect was across.  That
    is a question about the exterior scattered field -- the only place the network and
    the inversion ever look -- and therefore a measurement, which
    `solver.validate.check_cavity_scattering` now makes against GATE_CAVITY_REL_L2 and
    GATE_CAVITY_PHASE_RAD.  It came back 84% wrong, and the interface was one of the
    two reasons.

    The width is now set from that measurement rather than argued for: the cavity error
    as a function of the 10-90% width, in *fine* cells, is a U with a broad flat
    bottom (see `config.EPS_INTERFACE_FINE_CELLS` and
    `solver.validate.check_interface_width`), and the production value sits on it.  The
    left branch is the physical objection above; the right branch is a sigmoid narrower
    than the cell that samples it, which is a staircase and over-scatters.  Both bounds
    are pinned below, because the constant is no longer free: moving it in either
    direction makes the labels less like cavity scattering.

    The three knobs that collide here are still the same three, and the resolution is
    now on the record:

      - R_MIN_LS stays at 0.4 lambda_s -- the transition is 0.28 of it, so the small
        end of the radius range no longer has to be given up,
      - eps is as *large* as the flat bottom allows, because `dchi/dtheta` has support
        eps and that is the inversion's entire shape gradient,
      - dx refines independently, because EPS_INTERFACE_PHYS is a length; that
        separation is `check_interface_width` part B, measured flat to 0.2 points
        across a 3x refinement.
    """
    r = cfg.interface_report()

    assert cfg.EPS_TRANSITION_FACTOR == pytest.approx(4.3944, abs=1e-4)
    assert r["transition_fine_cells"] == pytest.approx(1.648, abs=1e-3)
    assert r["transition_cells"] == pytest.approx(0.824, abs=1e-3)
    assert r["transition_phys"] == pytest.approx(0.05150, abs=1e-5)
    assert r["transition_over_lambda_s_worst"] == pytest.approx(0.1134, abs=1e-4)
    assert r["ratio_to_r_min"] == pytest.approx(0.2834, abs=1e-4)
    assert r["ratio_to_r_min"] < 1.0 / 3.0, (
        "the interface must be narrow against the smallest defect; this is the "
        "assertion the v2.0 test claimed to make and did not")
    assert r["transition_fine_cells"] > 1.0, (
        "a 10-90% transition narrower than one *fine* cell is a staircase, not a "
        "smooth interface, and the measured cavity error turns back up: 0.99 cells "
        "reads 12.1% against 10.4% at the production width")

    # The impedance the "void" actually presents: sqrt(stiffness * density), both
    # floors together.  1e-3 of the host, against 1e-2 when density was retained.
    assert r["impedance_ratio"] == pytest.approx(1e-3, rel=1e-12)

    # EPS_INTERFACE_PHYS is the quantity a grid-refinement study holds fixed, so it
    # must be a length.  It is now defined in *fine* cells, because the sampling that
    # decides the right-hand branch of the U is the solver's, not the network's, and
    # EPS_INTERFACE_CELLS is derived from it for the network grid.
    assert cfg.EPS_INTERFACE_PHYS == pytest.approx(
        cfg.EPS_INTERFACE_FINE_CELLS * cfg.DX_FINE, rel=1e-12)
    assert cfg.EPS_INTERFACE_CELLS == pytest.approx(
        cfg.EPS_INTERFACE_PHYS / cfg.DX_NET, rel=1e-12)
    assert cfg.interface_report(eps_cells=2.0 * cfg.EPS_INTERFACE_CELLS,
                                dx=cfg.DX_NET / 2.0)["eps_phys"] == (
        pytest.approx(cfg.EPS_INTERFACE_PHYS, rel=1e-12)), (
        "doubling the resolution at fixed physical interface width must double the "
        "cell count; if this identity breaks, grid convergence and interface "
        "convergence are being measured as one thing")


def test_screen_grid_is_coarser_than_the_waveform_basin():
    """
    The 16 x 16 screen steps 0.44 = 0.97 lambda_s across the interior, and the
    complex-waveform basin is about lambda_s / 4 = 0.11.  The screen is therefore
    8.6x too coarse to be scored on the ordinary complex misfit: it would step clean
    over the true minimum, 256 candidates and none of them in the basin, and the
    failure would present as a bad forward model rather than as a sampling artefact.

    That inequality is the entire reason Stage 1 uses a different objective from
    Stage 3, so it is asserted rather than assumed.  If it ever becomes false the
    screen has become fine enough to score on the waveform misfit directly, stages 1
    and 2 are redundant, and §8.4 should say so.

    The step is `interior / (SCREEN_GRID - 1)`, not `interior / SCREEN_GRID`.  That is
    an off-by-one worth spelling out because both versions of this test had the second
    form and it flatters the screen: `inverse.invert.screen_candidates` builds the
    lattice with `torch.linspace(lo, hi, n_grid)`, which is endpoint-inclusive and so
    has 15 intervals, not 16.  Endpoint-inclusive is the right choice -- a void may
    legitimately sit exactly on the keepout boundary, and a cell-centred lattice would
    leave the corners of the feasible box further from any candidate than the interior
    is -- but it means the real node spacing is 7% larger than the cell width, and the
    error is in the unsafe direction.

    config.self_check asserts the same inequality at the nu = 1/3 reference material and
    over the full domain span; this test uses the interior span reachable by
    `screen_candidates` at the *worst* material, where lambda_s is 9% shorter and the
    basin correspondingly tighter.  The tighter basin is the one that has to lose.
    """
    interior = cfg.L_DOMAIN - 2.0 * cfg.BOUNDARY_KEEPOUT_LS * cfg.LAMBDA_S_MIN
    step = interior / (cfg.SCREEN_GRID - 1)
    basin_waveform = cfg.LAMBDA_S_MIN / 4.0

    assert step == pytest.approx(0.4425, abs=1e-3)
    assert step / cfg.LAMBDA_S_MIN == pytest.approx(0.974, abs=1e-3)
    assert step > basin_waveform
    assert cfg.N_SURVIVORS < cfg.SCREEN_GRID ** 2


def test_envelope_basin_scale_narrows_as_the_band_widens():
    """
    The screen's basin, done with the corrected arithmetic (§8.4).

    v2.0 claimed the Stage 1 basin was `N_c lambda_s / 2` = 1.14 because the misfit
    was "phase-free".  It was phase-free in the fatal sense: a magnitude-spectrum
    misfit satisfies |g_hat(w) e^{-i w tau}| = |g_hat(w)| exactly, so it does not see
    a travel-time shift at all and has no basin in the shift direction -- it has a
    flat plateau.  `inverse.timedomain.shift_invariance_demo` exhibits that identity
    numerically.  The number was arithmetic about the wrong object.

    A band-limited *envelope*, reconstructed over the stage's frequencies, does have
    a real basin, and its width is set by the reconstruction bandwidth rather than by
    the burst length: an envelope of bandwidth B decorrelates over a delay ~1/B, and
    a travel-time error tau maps to a position error c_s tau, so

        full basin width ~ c_s / B = (f_c / B) lambda_s.

    Hence the corollary that catches people out, and the reason this is a test: the
    envelope basin *narrows* as the band widens.  Stage 1's six frequencies span
    0.179 f_c and give 5.6 lambda_s; the full 20 span 0.680 f_c and give 1.5.  A
    well-meant change that widened Stage 1's band to "use more data" would shrink its
    basin by 3.8x and break the screen, silently.

    These are scale estimates for a smooth single-scatterer response, not bounds --
    elastic multipath and mode conversion put structure inside the envelope.  The
    coverage claim they used to support is now measured, not asserted, in two places:
    `inverse.misfit.envelope_basin_width` maps the actual basin of the actual
    objective, and `inverse.invert.screen_capture_rate` measures the fraction of
    screens whose survivor set contains the true basin, against GATE_SCREEN_CAPTURE.
    """
    basin_ls = {}
    for stage, band in ((1, cfg.BAND_STAGE1), (2, cfg.BAND_STAGE2),
                        (3, cfg.BAND_STAGE3)):
        f = cfg.FREQS[band]
        bandwidth = f[-1] - f[0]
        assert bandwidth > 0.0
        basin_ls[stage] = 1.0 / bandwidth          # in lambda_s, material-free

    assert basin_ls[1] == pytest.approx(5.59, abs=0.02)
    assert basin_ls[2] == pytest.approx(3.10, abs=0.02)
    assert basin_ls[3] == pytest.approx(1.47, abs=0.02)
    assert basin_ls[1] > basin_ls[2] > basin_ls[3], (
        "the band-limited envelope basin must narrow as the band widens; if this "
        "ordering breaks, the frequency continuation of §8.4 is running backwards")

    # Necessary condition for the screen: the worst point of an n x n lattice cell is
    # sqrt(2)/2 of a step from the nearest node, and that has to sit inside the Stage
    # 1 basin's half-width with room to spare.  Both sides scale with lambda_s, so the
    # margin is material-independent -- and it is a necessary condition only, which is
    # why GATE_SCREEN_CAPTURE exists.  The step is interior / (SCREEN_GRID - 1): the
    # lattice `screen_candidates` builds is endpoint-inclusive, so it has 15 intervals.
    interior_ls = (cfg.L_DOMAIN
                   - 2.0 * cfg.BOUNDARY_KEEPOUT_LS * cfg.LAMBDA_S_MIN) / cfg.LAMBDA_S_MIN
    worst_node_ls = math.sqrt(2.0) / 2.0 * interior_ls / (cfg.SCREEN_GRID - 1)
    margin = 0.5 * basin_ls[1] / worst_node_ls

    assert worst_node_ls == pytest.approx(0.689, abs=1e-3)
    assert margin == pytest.approx(4.06, abs=0.02)
    assert margin > 2.0, (
        f"screen leaves points {worst_node_ls:.2f} lambda_s from the nearest "
        f"candidate against a Stage 1 basin half-width of {0.5 * basin_ls[1]:.2f}: "
        "raise SCREEN_GRID or narrow BAND_STAGE1")


def test_void_material_scaling():
    """
    The void is soft *and* light, and both floors were set by measurement.

    v2.0 kept the full density (VOID_DENSITY_SCALE = 1.0) and called it a documented
    deviation from §7.2's delta-rho = -rho_0 chi, on the grounds that nulling density
    makes c = sqrt(mu/rho) a 0/0 limit whose value is set by the stiffness floor.  That
    reason was wrong: with both floors active the cell-centred speed is
    sqrt(1e-4/1e-2) = 0.1 c_p, and it is bounded by c_p for every chi in between, so
    there is no CFL blow-up at cell centres at all.

    The real constraint is that the solver does not evaluate either field at a cell
    centre alone -- `rho_vy = avg_plus(rho, -2)` averages neighbouring densities while
    the 4th-order stress stencil reaches two cells, so a face just inside a *sharp*
    void combines a small density with full-strength stiffness from two cells away.  At
    1e-4 with a sub-cell interface that diverges (step 200 of 1408).  1e-2 is stable at
    the production interface width, and leaves a residual impedance of
    sqrt(VOID_STIFFNESS_FLOOR * VOID_DENSITY_SCALE) = 1e-3 of the host.

    Retaining the full density is not a small deviation, which is why this test no
    longer records it as one.  The void interior becomes a bag of nearly free masses
    whose first layer rides on the interface as an added mass, loading it by k_s dx =
    0.43 at f_max.  Measured against the analytic traction-free cavity at the
    production interface width: 77.6% relative error with rho retained, 10.4% without
    (`solver.validate.check_cavity_scattering`, which can reproduce both rows).
    """
    assert cfg.VOID_DENSITY_SCALE == 1e-2
    assert 0.0 < cfg.VOID_DENSITY_SCALE < 1.0, (
        "section 7.2 asks for delta-rho = -rho_0 chi; this is the nearest stable "
        "thing to it, and 1.0 is not it")
    assert 0.0 < cfg.VOID_STIFFNESS_FLOOR <= 1e-3
    assert math.sqrt(cfg.VOID_STIFFNESS_FLOOR * cfg.VOID_DENSITY_SCALE) < 1e-2, (
        "residual impedance of the 'void' relative to the host")
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
