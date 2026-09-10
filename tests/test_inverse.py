"""
Tests for the inverse package: objectives, candidate generation, scoring.

These cover the four things the architecture review found unverified and the code
already promises: that the objective registry and the stage assignments agree and that
the *named* objectives behave as their names claim (identity A -- a magnitude-spectrum
misfit is travel-time blind -- is the reason this matters); that `screen_candidates`
works for every family rather than only the circle it was written for; that the
success metrics are shape-appropriate and permutation-invariant, so a two-void recovery
is not scored against a labelling accident; and that the lack-of-fit threshold is
frozen on calibration data, since a threshold chosen after seeing the test set reports
the best rate available in hindsight rather than a rate anything can reproduce.

`from src.inverse import invert` binds the *function* -- src/inverse/__init__.py
re-exports it -- so the module is imported by path.
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from importlib import import_module

import pytest
import torch

from src import config as cfg
from src.geometry.sdf import FAMILIES, Circle, Ellipse, TwoCircle
from src.inverse import misfit as M
from src.inverse import timedomain as TD
from src.solver import harmonic as H
from src import training

INV = import_module("src.inverse.invert")
LAMBDA_S = cfg.LAMBDA_S_MIN


# ---------------------------------------------------------------------------
# The objective registry
# ---------------------------------------------------------------------------
def test_every_configured_objective_exists_and_is_named_honestly():
    """
    config's stage assignments must resolve, and the reconstructed set must be right.

    Import of `misfit` already runs `check_objective` over the config, so a mismatch is
    an ImportError rather than a mid-inversion KeyError; this asserts the same thing
    where a reader will find it, plus the split between objectives that need a
    time-domain reconstruction and those that do not.
    """
    for stage, name in cfg.STAGE_OBJECTIVE.items():
        assert name in M.OBJECTIVES, f"stage {stage} wants {name!r}"
    assert cfg.SCREEN_OBJECTIVE in M.OBJECTIVES
    assert cfg.SCREEN_FALLBACK_OBJECTIVE in M.OBJECTIVES
    assert M.RECONSTRUCTED <= set(M.OBJECTIVES)
    assert M.RECONSTRUCTED == {"envelope", "traveltime", "correlation"}
    assert "spectral_magnitude" not in M.RECONSTRUCTED, (
        "spectral_magnitude is a frequency-domain screen, not an envelope; the whole "
        "point of separating them is that it never touches a reconstruction")
    with pytest.raises(KeyError, match="unknown objective"):
        M.check_objective("envelop")


def test_the_screen_does_not_use_the_travel_time_blind_objective():
    """
    Identity A: |ghat(w)| is unchanged by a time shift, so a magnitude-spectrum misfit
    cannot see a travel-time error -- which is the *only* error a position screen has to
    see.  This pins that the screen's configured objective is not that one.
    """
    assert cfg.SCREEN_OBJECTIVE != "spectral_magnitude", (
        "a magnitude-spectrum objective is identically blind to the shifts the screen "
        "exists to detect; see misfit.spectral_magnitude_misfit's docstring")
    assert cfg.STAGE_OBJECTIVE[2] in M.RECONSTRUCTED
    assert cfg.STAGE_OBJECTIVE[3] == "complex", (
        "the final stage is the one that uses carrier phase")


def test_identity_a_holds_numerically():
    """
    A shifted trace has the same magnitude spectrum to numerical precision, and a
    different complex spectrum.  The demo helpers in `timedomain` exist to show this;
    this is the assertion form, on the [B, R, 2, M] layout the objectives take.
    """
    r = TD.reconstruction(cfg.BAND_STAGE3, n_t=128)
    m = r.freqs.numel()
    torch.manual_seed(0)
    g = torch.randn(1, 4, 2, m, dtype=torch.complex128)
    shifted = r.delay(g, 0.37)

    assert torch.allclose(shifted.abs(), g.abs(), atol=1e-12), "identity A"
    assert not torch.allclose(shifted, g, atol=1e-6)
    assert float(M.spectral_magnitude_misfit(shifted, g)) < 1e-20, (
        "the magnitude misfit of a pure shift must be zero, not merely small")
    assert float(M.complex_misfit(shifted, g)) > 1e-2, (
        "the complex misfit must see the same shift")
    # ... and the objective the screen actually uses does see it.
    assert float(TD.envelope_misfit(shifted, g, recon=r)) > 1e-2, (
        "an envelope objective is shift-equivariant, so a delay is visible to it")


def test_envelope_scale_invariance_fits_candidate_amplitude():
    """The screen can ignore a fixed radius mismatch without changing its final stages."""
    r = TD.reconstruction(cfg.BAND_STAGE1, n_t=128)
    torch.manual_seed(4)
    obs = torch.randn(1, 4, 2, r.freqs.numel(), dtype=torch.complex128)
    pred = 3.7 * obs
    fixed = float(TD.envelope_misfit(pred, obs, recon=r))
    fitted = float(TD.envelope_misfit(pred, obs, recon=r, scale_invariant=True))
    assert fixed > 1.0
    assert fitted < 1e-20
    assert float(M.OBJECTIVES["envelope"](
        pred, obs, band=cfg.BAND_STAGE1, scale_invariant=True)) < 1e-20



def test_screen_requests_scale_invariance_but_final_complex_objective_does_not():
    """The deployed screen and final complex stage retain their distinct semantics."""
    assert INV.screen.__name__ == "screen"
    assert cfg.SCREEN_OBJECTIVE == "envelope"
    assert cfg.STAGE_OBJECTIVE[3] == "complex"
    r = TD.reconstruction(cfg.BAND_STAGE3, n_t=128)
    obs = torch.ones(1, 2, 2, r.freqs.numel(), dtype=torch.complex128)
    pred = 2.0 * obs
    assert float(TD.envelope_misfit(pred, obs, recon=r)) > 0.1
    assert float(M.complex_misfit(pred, obs)) > 0.1




# ---------------------------------------------------------------------------
# Candidate generation, for every family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_screen_candidates_are_in_bounds_for_every_family(name):
    """
    What made the screen circle-only was three literal 3s; `_blob_slots` reads the
    parameter names instead.  So this must hold for a 3-, a 5- and a 6-parameter family.
    """
    fam = FAMILIES[name]
    th = INV.screen_candidates(fam, LAMBDA_S)
    lo, hi = fam.bounds(LAMBDA_S)
    assert th.dim() == 2 and th.shape[1] == fam.n_params
    assert th.shape[0] >= cfg.SCREEN_GRID ** 2
    assert (th >= lo).all() and (th <= hi).all(), "every candidate must be feasible"
    # Position-only scan: every non-centre parameter is bit-identical across rows.
    # Compared as max - min, not std: over 256 identical float32 values the two-pass
    # std returns one ulp (3e-8) because the mean does not round back to the value.
    slots = {i for pair in INV._blob_slots(fam) for i in pair}
    for i in range(fam.n_params):
        if i not in slots:
            spread = float(th[:, i].max() - th[:, i].min())
            assert spread == 0.0, (
                f"parameter {fam.param_names[i]!r} varies by {spread} across the "
                "screen; the scan is meant to move centres only")
    # ... and the centres do move, over the whole feasible span in x and y.
    for ix, iy in INV._blob_slots(fam):
        for j in (ix, iy):
            assert float(th[:, j].max() - th[:, j].min()) > 0.5 * float(hi[j] - lo[j])


def test_screen_candidate_count_matches_the_family_geometry():
    """One blob scans a grid; two blobs also scan a separation and an orientation."""
    n1 = INV.screen_candidates(Circle(), LAMBDA_S).shape[0]
    n2 = INV.screen_candidates(Ellipse(), LAMBDA_S).shape[0]
    n3 = INV.screen_candidates(TwoCircle(), LAMBDA_S).shape[0]
    assert n1 == cfg.SCREEN_GRID ** 2
    assert n2 == n1, "an ellipse screen starts circular, so it is the same lattice"
    assert n3 == 4 * n1, "a two-void screen also scans four orientations of the pair"


def test_screen_step_is_the_number_the_basin_argument_uses():
    """
    The lattice step in lambda_s, which `test_config` compares against the measured
    envelope basin.  Pinned here because it is a property of `screen_candidates` and
    not only of the constants -- and because the arithmetic has an off-by-one that is
    easy to get wrong in the flattering direction: the lattice is an endpoint-inclusive
    `linspace(lo, hi, 16)`, so it has 15 intervals and steps `interior / 15`, not
    `interior / 16`.  The real step is 7% coarser than the cell width.
    """
    th = INV.screen_candidates(Circle(), LAMBDA_S)
    xs = torch.unique(th[:, 0])
    lo, hi = Circle().bounds(LAMBDA_S)
    interior = cfg.L_DOMAIN - 2.0 * cfg.BOUNDARY_KEEPOUT_LS * LAMBDA_S

    assert len(xs) == cfg.SCREEN_GRID
    assert float(hi[0] - lo[0]) == pytest.approx(interior, rel=1e-6), (
        "the lattice spans the keepout-trimmed interior")
    step = float(xs.diff().median())
    assert step == pytest.approx(interior / (cfg.SCREEN_GRID - 1), rel=1e-6)
    assert step / LAMBDA_S == pytest.approx(0.974, abs=0.005)
    assert float(xs.min()) == pytest.approx(float(lo[0]), rel=1e-6)
    assert float(xs.max()) == pytest.approx(float(hi[0]), rel=1e-6)


# ---------------------------------------------------------------------------
# Shape-appropriate scoring (§9)
# ---------------------------------------------------------------------------
def _result(fam_name: str, theta, theta_true, truth_fam: str | None = None
            ) -> "INV.InversionResult":
    return INV.InversionResult(
        theta=torch.tensor(theta, dtype=torch.float64), misfit=1e-3,
        theta_true=torch.tensor(theta_true, dtype=torch.float64),
        lambda_s=LAMBDA_S, family=FAMILIES[fam_name],
        truth_family=None if truth_fam is None else FAMILIES[truth_fam])


def test_two_void_scoring_is_permutation_invariant():
    """
    A two-void ground truth is a *set*.  Reporting the recovery of the same two voids
    listed in the other order as a 6-lambda_s position error -- which indexing theta[:2]
    does -- is reporting a labelling accident.
    """
    truth = [4.0, 4.0, 0.5, 6.0, 6.0, 0.7]
    swapped = _result("two_circle", [6.0, 6.0, 0.7, 4.0, 4.0, 0.5], truth)
    assert swapped.position_error_ls == pytest.approx(0.0, abs=1e-12)
    assert swapped.radius_error_ls == pytest.approx(0.0, abs=1e-12)
    assert swapped.iou() == pytest.approx(1.0, abs=1e-9)
    assert swapped.success

    # ... and it is still the *worst* blob that is reported, not the mean.
    half = _result("two_circle", [4.0, 4.0, 0.5, 6.0, 6.0 + 0.3 * LAMBDA_S, 0.7], truth)
    assert half.position_error_ls == pytest.approx(0.3, abs=1e-6)
    assert not half.success, "one void found and one missed is not a recovery"

    doubled = _result("two_circle", [4.0, 4.0, 0.5, 4.0, 4.0, 0.5], truth)
    assert doubled.position_error_ls > 1.0
    assert doubled.iou() < 0.5
    assert not doubled.success, "finding one void twice must fail"


def test_ellipse_scoring_separates_size_shape_and_orientation():
    """
    (a, b, alpha) is two-to-one: (a, b, alpha) == (b, a, alpha + pi/2).  Differencing
    the raw parameters calls a pure 90-degree rotation a large axis-ratio error and a
    zero orientation error, i.e. exactly backwards.  Canonicalising to the major axis
    first makes the three numbers independent and each one true.
    """
    truth = [4.0, 4.2, 0.8, 0.5, 0.4]
    perfect = _result("ellipse", truth, truth)
    assert perfect.axis_ratio_error == pytest.approx(0.0, abs=1e-12)
    assert perfect.orientation_error_deg == pytest.approx(0.0, abs=1e-9)
    assert perfect.iou() == pytest.approx(1.0, abs=1e-9)

    rot90 = _result("ellipse", [4.0, 4.2, 0.5, 0.8, 0.4], truth)
    assert rot90.axis_ratio_error == pytest.approx(0.0, abs=1e-12), (
        "a rotation is not a change of aspect ratio")
    assert rot90.orientation_error_deg == pytest.approx(90.0, abs=1e-6)
    assert not rot90.success, "the right shape in the wrong orientation is not recovery"

    rot30 = _result("ellipse", [4.0, 4.2, 0.8, 0.5, 0.4 + math.radians(30)], truth)
    assert rot30.orientation_error_deg == pytest.approx(30.0, abs=1e-6)

    # A circle fitted to an ellipse: no orientation to be right or wrong about.
    circ = _result("ellipse", [4.0, 4.2, 0.632, 0.632, 0.0], truth)
    assert math.isnan(circ.orientation_error_deg)
    assert circ.axis_ratio_error == pytest.approx(1.6 - 1.0, abs=0.01)
    assert circ.radius_error_ls < 0.01, "equal-area radius is right; the shape is not"
    assert not circ.success


def test_success_needs_the_shape_gate_not_only_the_centre():
    """
    GATE_POSITION_LS scores a centre and is not size-normalised.  A perfectly centred
    void of 1.4x the true radius passed v2.0's gate; the IoU gate is what fails it.
    """
    R = 0.6 * LAMBDA_S
    truth = [4.0, 4.0, R]
    assert _result("circle", truth, truth).success

    fat = _result("circle", [4.0, 4.0, 1.4 * R], truth)
    assert fat.position_error_ls == pytest.approx(0.0, abs=1e-12)
    assert fat.position_error_ls < cfg.GATE_POSITION_LS      # v2.0 said PASS here
    assert fat.iou() < cfg.GATE_IOU
    assert not fat.success

    # The two gates nearly coincide for a pure displacement, by construction.
    near = _result("circle", [4.0 + 0.099 * LAMBDA_S, 4.0, R], truth)
    far = _result("circle", [4.0 + 0.15 * LAMBDA_S, 4.0, R], truth)
    assert near.success and not far.success
    assert near.iou() > cfg.GATE_IOU > far.iou()


def test_iou_is_defined_across_families_and_bounded():
    """
    The transfer experiment compares an ellipse truth against whatever the optimiser
    returns, so the primary metric has to be a field overlap rather than a parameter
    difference.  Self-overlap is 1, disjoint shapes are near 0 (not exactly 0: the soft
    indicator has exponential tails).
    """
    R = 0.6 * LAMBDA_S
    same = _result("circle", [4.0, 4.0, R], [4.0, 4.0, R])
    assert same.iou() == pytest.approx(1.0, abs=1e-9)
    apart = _result("circle", [2.0, 2.0, R], [6.0, 6.0, R])
    assert 0.0 <= apart.iou() < 1e-6
    for v in (same.iou(), apart.iou(), _result(
            "ellipse", [4.0, 4.0, R, 0.5 * R, 0.3], [4.0, 4.0, R, R, 0.0]).iou()):
        assert 0.0 <= v <= 1.0

    # A zero-eccentricity ellipse and the circle it equals score identically, which is
    # only true because Ellipse.sdf normalises s - 1 rather than s^2 - 1.
    ell = _result("ellipse", [4.1, 4.0, R, R, 0.0], [4.0, 4.0, R, R, 0.0])
    cir = _result("circle", [4.1, 4.0, R], [4.0, 4.0, R])
    assert ell.iou() == pytest.approx(cir.iou(), abs=1e-9)


def test_cross_family_scoring_reads_each_theta_with_its_own_family():
    """
    The transfer experiment fits a *circle* to an ellipse or a pair of voids, so the two
    thetas have different lengths and different meanings per column.  Applying the
    prediction's family to both -- which is what happened before `truth_family` existed
    -- read an ellipse truth's first three columns as (xc, yc, R) and scored its
    semi-major axis as a radius.  Nothing raised; the number was simply wrong.
    """
    R = 0.6 * LAMBDA_S
    a, b = R, 0.4 * R                                   # 2.5:1, equal-area sqrt(ab)
    eq = math.sqrt(a * b)
    fit = _result("circle", [4.0, 4.0, eq], [4.0, 4.0, a, b, 0.3], "ellipse")

    # The equal-area circle of the truth is, by construction, a zero position and size
    # error -- and a decidedly non-unit overlap, which is the whole point of the
    # experiment: a circle cannot represent a 2.5:1 ellipse however well it is placed.
    assert fit.position_error_ls == pytest.approx(0.0, abs=1e-12)
    assert fit.radius_error_ls == pytest.approx(0.0, abs=1e-12)
    assert 0.4 < fit.iou() < 0.75
    assert not fit.success, "a good centre with a wrong shape is not a recovery"
    assert fit.axis_ratio_error is None, "a circle fit has no axis ratio to be wrong"

    # Reading the truth with the *prediction's* family is the old behaviour, and it
    # differs -- it compares against a circle of radius a rather than the ellipse.
    stale = _result("circle", [4.0, 4.0, eq], [4.0, 4.0, a])
    assert abs(stale.iou() - fit.iou()) > 0.05

    # Two voids: the equal-area circle straddles both, and the overlap says so.
    pair = [4.0 - 0.5 * R, 4.0, R, 4.0 + 0.5 * R, 4.0, R]
    both = _result("circle", [4.0, 4.0, R * math.sqrt(2.0)], pair, "two_circle")
    assert both.position_error_ls == pytest.approx(0.0, abs=1e-9)
    assert both.iou() < 0.8 and both.iou() > 0.0
    assert "equal-area circles" in both.summary()

    # And the other direction, which used to index off the end of a 3-vector.
    wide = _result("ellipse", [4.0, 4.0, R, R, 0.0], [4.0, 4.0, R], "circle")
    assert wide.position_error_ls == pytest.approx(0.0, abs=1e-12)
    assert wide.iou() == pytest.approx(1.0, abs=1e-9)
    assert wide.axis_ratio_error is None and wide.orientation_error_deg is None


def test_equivalent_circle_matches_each_family_area():
    """
    The one reduction used for cross-family position and size errors.  It has to agree
    with each family's own `area`, or the two halves of the comparison disagree about how
    big the truth is.
    """
    from src.geometry.sdf import equivalent_circle

    for name, theta in (("circle", [4.0, 4.0, 0.5]),
                        ("ellipse", [4.0, 4.0, 0.6, 0.24, 0.3]),
                        ("two_circle", [3.5, 4.0, 0.3, 4.5, 4.2, 0.4])):
        fam = FAMILIES[name]
        t = torch.tensor([theta], dtype=torch.float64)
        eq = equivalent_circle(t, fam)
        assert eq.shape == (1, 3)
        assert float(math.pi * eq[0, 2] ** 2) == pytest.approx(float(fam.area(t)),
                                                              rel=1e-12)
    # Area-weighted, not midpoint: the bigger void pulls the centre.
    eq = equivalent_circle(torch.tensor([[0.0, 0.0, 1.0, 4.0, 0.0, 2.0]],
                                        dtype=torch.float64), FAMILIES["two_circle"])
    assert float(eq[0, 0]) == pytest.approx(3.2, rel=1e-12)


def test_metrics_degrade_gracefully_without_a_family_or_a_truth():
    """An unpickled v2.0 result has no `family`; a real inversion has no truth."""
    r = INV.InversionResult(theta=torch.tensor([4.0, 4.0, 0.5]), misfit=1.0)
    assert r.position_error_ls is None and r.radius_error_ls is None
    assert r.iou() is None and r.wall_saturation is None
    assert not r.success
    assert "theta" in r.summary()

    no_fam = INV.InversionResult(theta=torch.tensor([4.0, 4.0, 0.5]), misfit=1.0,
                                 theta_true=torch.tensor([4.0, 4.0, 0.5]),
                                 lambda_s=LAMBDA_S)
    assert no_fam.position_error_ls == pytest.approx(0.0), "circle interpretation"
    assert no_fam.axis_ratio_error is None


def test_summarise_reports_both_success_definitions_and_wall_saturation():
    """
    The IoU gate makes `success_rate` incomparable with a v2.0 number, so the
    position-only rate is carried alongside it rather than silently replaced.
    """
    R = 0.6 * LAMBDA_S
    truth = [4.0, 4.0, R]
    good = _result("circle", truth, truth)
    ok_ish = _result("circle", [4.0 + 0.05 * LAMBDA_S, 4.0, R], truth)
    centred_but_fat = _result("circle", [4.0, 4.0, 1.4 * R], truth)
    s = INV.summarise([good, ok_ish, centred_but_fat])

    assert s["n"] == 3
    assert s["success_rate"] == pytest.approx(2.0 / 3.0)
    assert s["success_rate_position_only"] == pytest.approx(1.0)
    assert not s["gate_pass"], "2/3 is below GATE_SUCCESS_RATE"
    assert s["iou_median"] == pytest.approx(ok_ish.iou(), abs=1e-6)
    assert s["gates"] == dict(position_ls=cfg.GATE_POSITION_LS, iou=cfg.GATE_IOU,
                              success_rate=cfg.GATE_SUCCESS_RATE)
    assert s["n_wall_saturated"] == 0

    # torch.median takes the *lower* of the two central values on an even count, so an
    # even-n summary is mildly pessimistic rather than interpolated.  Pinned because a
    # reader comparing against numpy.median would otherwise see a discrepancy and
    # assume one of the two was computing the wrong thing.
    two = INV.summarise([good, centred_but_fat])
    assert two["iou_median"] == pytest.approx(centred_but_fat.iou(), abs=1e-6)

    lo, hi = Circle().bounds(LAMBDA_S)
    on_wall = _result("circle", [float(lo[0]), 4.0, R], truth)
    assert on_wall.wall_saturation == pytest.approx(cfg.WALL_Z_CLAMP, rel=1e-6)
    assert cfg.WALL_Z_CLAMP == pytest.approx(math.log(9999.0), rel=1e-9)
    assert cfg.WALL_Z_WARN < cfg.WALL_Z_CLAMP, (
        "the warn threshold has to be reachable: `to_unconstrained` clamps at "
        "log(9999) = 9.2103, so a threshold above that counts nothing, ever")
    with pytest.warns(RuntimeWarning, match="edge of the feasible box"):
        assert INV.summarise([on_wall])["n_wall_saturated"] == 1

    # A run comfortably inside the box must not be flagged.
    assert good.wall_saturation < cfg.WALL_Z_WARN
    assert INV.summarise([good])["n_wall_saturated"] == 0


# ---------------------------------------------------------------------------
# The reconstruction the envelope objectives are built on
# ---------------------------------------------------------------------------
def test_reconstruction_is_shift_equivariant_and_periodic_past_the_record():
    """
    Definition R is exactly shift-equivariant, which is what makes an envelope
    objective able to see a travel-time error at all.  Its period is 1 / DF, which
    must exceed T_END or a late arrival wraps onto an early one.
    """
    r = TD.reconstruction(cfg.BAND_STAGE3, n_t=128)
    period = 1.0 / cfg.DF
    assert period > cfg.T_END, (
        f"reconstruction period {period:.2f} must exceed the record {cfg.T_END:.2f}")

    g = torch.randn(2, 3, r.freqs.numel(), dtype=torch.complex128)
    k = 3
    tau = k * r.dt
    z0 = r.analytic(g)
    z1 = r.analytic(r.delay(g, tau))
    assert torch.allclose(z1[..., k:], z0[..., :-k], atol=1e-9), (
        "a phase ramp must move the reconstruction by exactly that many samples")

    # The reconstruction is only periodic up to the carrier phase e^{2 pi i f0 P}
    # (F_START / DF = 18.436 is not an integer), so it is the *envelope* that wraps
    # exactly -- which is all an envelope objective needs.
    e0, e1 = z0.abs(), r.analytic(r.delay(g, r.period)).abs()
    assert torch.allclose(e1, e0, atol=1e-9), "the envelope has period 1 / DF"


def test_reconstruction_report_is_a_report_with_the_fields_it_documents():
    """
    Not a gate: the numbers depend on the band and the taper, and exposing them is how
    the band and taper get chosen.  But the *keys* are promised, and v2.0 asserted a
    2.5-period envelope resolution that this measures instead.
    """
    rep = TD.reconstruction_report(cfg.BAND_STAGE3, n_t=256)
    for key in ("rel_l2", "energy_in_band", "peak_time_error", "psf_width_3db",
                "psf_sidelobe_db"):
        assert key in rep, key
        assert math.isfinite(rep[key]), f"{key} = {rep[key]}"
    assert 0.0 < rep["energy_in_band"] <= 1.0
    assert rep["psf_width_3db"] > 0.0
    assert rep["psf_sidelobe_db"] < 0.0, "a sidelobe is below the peak, so negative"
    assert abs(rep["peak_time_error"]) < 0.5, (
        "a systematic envelope-peak bias larger than half a carrier period would be a "
        "position bias in every inversion")


# ---------------------------------------------------------------------------
# The lack-of-fit indicator (§9.3)
# ---------------------------------------------------------------------------
ETA = float(torch.tensor(H.conditioning_report()["amplification"]).pow(2).mean())


def test_the_statistic_normalises_by_noise_and_a_nonzero_model_floor():
    """
    T = J / (eta * 10^(-SNR/10) + floor), and each piece of that has to earn its place.

    The floor is why a high-SNR case is not reported as catastrophic lack of fit: at 60
    dB the noise term is 3.2e-5, twenty times below the surrogate's own accuracy, and
    without a floor the denominator would be measuring a noise level the surrogate
    cannot resolve.  `snr_db=None` is the noiseless limit, which needs the floor simply
    to avoid dividing by zero.
    """
    assert cfg.LOF_MODEL_FLOOR == pytest.approx(cfg.GATE_REL_L2 ** 2)
    j = 1.2e-3
    assert INV.lack_of_fit_statistic(j, snr_db=None) == pytest.approx(
        j / cfg.LOF_MODEL_FLOOR)

    # Monotone in the residual at fixed SNR -- the only property a threshold needs.
    ts = [INV.lack_of_fit_statistic(m, snr_db=30.0, amplification=ETA)
          for m in (1e-4, 1e-3, 1e-2, 1e-1)]
    assert ts == sorted(ts) and ts[0] < ts[-1]

    # Noisier data explains more residual, so the same misfit is less damning.
    loud = INV.lack_of_fit_statistic(j, snr_db=40.0, amplification=ETA)
    quiet = INV.lack_of_fit_statistic(j, snr_db=20.0, amplification=ETA)
    assert quiet < loud, "a lower SNR must lower T for the same residual"

    # Hoisting `amplification` out of a per-case loop -- which both stage-F notebooks
    # do, to avoid recomputing the conditioning report once per inversion -- must not
    # change the number.
    assert INV.lack_of_fit_statistic(j, snr_db=30.0) == pytest.approx(
        INV.lack_of_fit_statistic(j, snr_db=30.0, amplification=ETA), rel=1e-12)
    assert ETA == pytest.approx(31.9, abs=0.5), (
        "eta is the *squared* amplification profile's mean, because the misfit is "
        "quadratic; the un-squared mean is a different and smaller number")

    # At 60 dB the floor dominates, which is the whole reason it exists.
    noise60 = ETA * 10.0 ** (-6.0)
    assert noise60 < 0.1 * cfg.LOF_MODEL_FLOOR
    assert INV.lack_of_fit_statistic(j, snr_db=60.0, amplification=ETA) == (
        pytest.approx(j / (noise60 + cfg.LOF_MODEL_FLOOR)))


def test_the_threshold_is_set_by_in_family_data_alone():
    """
    The methodological claim, as an assertion: `stats_out` cannot move the threshold.

    This is the property that makes the reported FPR mean anything, and it is one line
    away from being lost -- any future edit that picks the threshold by maximising
    `tpr - fpr` over both classes would still pass every other test in this file.
    """
    cal = [0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 2.0]
    a = INV.calibrate_lack_of_fit(cal)
    b = INV.calibrate_lack_of_fit(cal, [50.0, 60.0, 70.0])
    c = INV.calibrate_lack_of_fit(cal, [1.05] * 3)     # nearly overlapping
    assert a.threshold == b.threshold == c.threshold, (
        "the threshold is the (1 - target_fpr) quantile of the in-family statistics; "
        "stats_out only reports the TPR that threshold happens to achieve")
    assert math.isnan(a.calibration_tpr) and b.calibration_tpr == 1.0
    assert b.n_out == 3 and a.n_out == 0

    # It is the quantile it says it is, and it achieves the FPR it targets on the data
    # it was fitted to (up to the granularity of ten samples).
    assert a.target_fpr == cfg.GATE_LOF_FPR
    assert a.threshold == pytest.approx(
        float(torch.tensor(cal, dtype=torch.float64).quantile(1.0 - cfg.GATE_LOF_FPR)))
    assert a.calibration_fpr <= a.target_fpr + 1e-12

    # Frozen in the dataclass sense too, so a threshold cannot be edited after the fact.
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.threshold = 0.0                                   # type: ignore[misc]
    with pytest.raises(ValueError, match="needs in-family statistics"):
        INV.calibrate_lack_of_fit([])


def test_evaluate_applies_the_frozen_threshold_and_refits_nothing():
    """Held-out scoring: the same threshold, whatever the test data turns out to be."""
    lof = INV.calibrate_lack_of_fit([0.5] * 9 + [3.0], note="unit test")
    # torch.quantile interpolates: q = 0.9 over 10 sorted values lands at index
    # 0.9 * (10 - 1) = 8.1, i.e. a tenth of the way from 0.5 to 3.0.  Worth pinning
    # because "the 90th percentile of ten numbers" reads as if it were the 9th of them,
    # and here that would be 0.5 rather than 0.75 -- a threshold six times tighter.
    assert lof.threshold == pytest.approx(0.75)
    assert lof.flag(lof.threshold * 1.001) and not lof.flag(lof.threshold * 0.999)

    ev = lof.evaluate([0.5, 0.6, 0.7, 0.74], [10.0, 20.0, 30.0])
    assert ev["threshold"] == lof.threshold
    assert ev["fpr"] == 0.0 and ev["tpr"] == 1.0
    assert ev["gate_pass"] and ev["gate"] == cfg.GATE_LOF_FPR
    assert ev["n_in"] == 4 and ev["n_out"] == 3

    # Making the alternative class easier or harder cannot change the FPR, because the
    # FPR is a property of the frozen threshold and the in-family data only.
    assert lof.evaluate([0.5, 0.6, 0.7, 9.9], [1.0])["fpr"] == 0.25
    assert lof.evaluate([0.5, 0.6, 0.7, 9.9], [99.0])["fpr"] == 0.25
    assert not lof.evaluate([9.9] * 4, [99.0])["gate_pass"], "an FPR of 1.0 must fail"

    # Empty is nan rather than a silent 0.0: no data is not the same as no failures.
    assert math.isnan(lof.evaluate([], [1.0])["fpr"])
    assert math.isnan(lof.evaluate([1.0], [])["tpr"])


def test_calibration_warns_when_the_residual_is_the_surrogate_not_the_noise():
    """
    An in-family median far above 1 means the denominator is understated: the cases the
    threshold was calibrated on are not noise-limited, they are surrogate-limited.  The
    threshold still separates the classes, so this is a warning and not an error -- but
    quoting it as a calibrated false-positive rate at a stated SNR would be wrong.
    """
    with pytest.warns(RuntimeWarning, match="floor is understated"):
        INV.calibrate_lack_of_fit([40.0, 45.0, 50.0, 60.0])
    # Comfortably explained residuals stay quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        INV.calibrate_lack_of_fit([0.4, 0.5, 0.6, 0.7])


def test_the_roc_is_a_diagnostic_with_an_exact_auc():
    """
    AUC = P(stat_out > stat_in) with ties at half, computed by the rank identity rather
    than by integrating the curve -- at ten-per-class sample sizes a trapezoidal AUC of
    a step function is off by a visible amount, and the number is quoted to 3 decimals.
    """
    r = INV.lack_of_fit_roc([1.0, 2.0], [1.5, 3.0])
    assert r["auc"] == pytest.approx(0.75), "3 of the 4 pairs are correctly ordered"
    assert INV.lack_of_fit_roc([1.0], [1.0])["auc"] == pytest.approx(0.5), "a tie is 0.5"
    assert INV.lack_of_fit_roc([1.0, 2.0], [8.0, 9.0])["auc"] == pytest.approx(1.0)
    assert INV.lack_of_fit_roc([8.0, 9.0], [1.0, 2.0])["auc"] == pytest.approx(0.0), (
        "the AUC is one-sided, so a reversed statistic scores 0 rather than 1")

    # The curve spans both classes and both rates are monotone non-increasing in the
    # threshold, which is what makes the frozen operating point plottable against it.
    assert len(r["tpr"]) == len(r["fpr"]) == len(r["threshold"]) == 200
    assert r["threshold"][0] == pytest.approx(1.0)
    assert r["threshold"][-1] == pytest.approx(3.0)
    for k in ("tpr", "fpr"):
        v = r[k]
        assert all(v[i] >= v[i + 1] for i in range(len(v) - 1)), f"{k} not monotone"
    assert r["median_in"] == pytest.approx(1.0), "torch.median takes the lower of two"
    assert r["median_out"] == pytest.approx(1.5)


def test_the_old_detector_name_still_works_and_says_it_is_deprecated():
    """`detector_roc` is kept so an existing record can be reproduced, not recommended."""
    with pytest.warns(DeprecationWarning, match="lack-of-fit indicator"):
        old = INV.detector_roc([1.0, 2.0], [1.5, 3.0])
    assert old["auc"] == INV.lack_of_fit_roc([1.0, 2.0], [1.5, 3.0])["auc"]
