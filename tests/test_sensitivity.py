"""
The two gradient gates and the solver re-check (§11.2 steps 9a/9b/10b).

What is testable without a trained network and without minutes of FDTD is the part
most likely to rot: the h-sweep logic that turns two gradients into a number of
digits, the direction metric that decides whether two complex sensitivity columns
point the same way, and the physical-width parameterisation that lets 9b be asked of a
family the dataset never contained.  Those are tested here with stubs and closed forms.

The parts that need the reference solver are marked `slow`.  They are not the point of
this file -- notebook 05 runs the real gates -- but the plumbing between
`solver_receivers` and `InverseCase.d_obs` has one asymmetry in it (A-scan route
versus field-phasor route) that is worth pinning, because getting it wrong biases the
scattered field rather than breaking it.
"""

from __future__ import annotations

import math

import pytest
import torch

from src import config as cfg
from src.geometry.sdf import Circle, Ellipse
from src.inverse import sensitivity as SENS
from src.inverse.invert import InversionResult
from src.inverse.misfit import InverseCase


# ---------------------------------------------------------------------------
# A stand-in for the Objective, so 9a can be tested against a closed form
# ---------------------------------------------------------------------------
class _StubForward:
    def __init__(self, dtype=torch.float64):
        self.dtype = dtype
        self.device = torch.device("cpu")


class _StubObjective:
    """
    J(theta) = sum_k c_k * (theta_k - t_k)^4, whose gradient is 4 c_k (theta_k - t_k)^3.

    A quartic rather than a quadratic on purpose: central differences are *exact* for
    a quadratic, so a quadratic would report infinite digits at every h and the tent
    -- the thing the sweep exists to find -- would not exist.
    """

    def __init__(self, c, t, dtype=torch.float64, bug=1.0):
        self.forward = _StubForward(dtype)
        self.case = InverseCase(d_obs=torch.zeros(1), src_idx=0, nu_idx=0)
        self.family = Circle()
        self.c = torch.tensor(c, dtype=torch.float64)
        self.t = torch.tensor(t, dtype=torch.float64)
        self.bug = bug          # 1.0 = a correct graph; anything else is a wrong one

    def residual(self, theta):
        d = theta[0].to(torch.float64) - self.t
        j = (self.c * d.pow(4)).sum()
        # `bug` scales the *differentiated* path only, which is what a real graph error
        # looks like: the value is right, so no forward test catches it.
        return (j * self.bug + j.detach() * (1.0 - self.bug)).reshape(())

    def grad_exact(self, theta):
        d = torch.as_tensor(theta, dtype=torch.float64) - self.t
        return 4.0 * self.c * d.pow(3)


def _stub(**kw):
    return _StubObjective(c=[1.0, 0.5, 2.0], t=[0.1, -0.2, 0.4], **kw)


THETA = torch.tensor([[0.25, -0.05, 0.55]], dtype=torch.float64)


# ---------------------------------------------------------------------------
# 9a: the autodiff graph
# ---------------------------------------------------------------------------
def test_the_gradient_gate_measures_digits_and_finds_the_tent_peak():
    """The sweep is the measurement: one h cannot tell a bad gradient from a bad h."""
    obj = _stub()
    r = SENS.autodiff_vs_fd_surrogate(obj, THETA)

    exact = obj.grad_exact(THETA[0])
    assert torch.allclose(r["grad_autodiff"], exact, rtol=1e-12), "stub sanity"
    assert r["gate_pass"] and r["worst_digits"] > 6.0
    assert r["param_names"] == ["xc", "yc", "R"]
    assert r["gate"] == cfg.GATE_GRAD_SIGFIGS

    # The tent: truncation error falls as h^2 on the left flank, cancellation grows as
    # 1/h on the right, so the peak is interior to the sweep and both ends are worse.
    curve = r["digits_curve"].min(dim=1).values
    k = int(curve.argmax())
    assert 0 < k < len(curve) - 1, "the peak should be interior, not at an endpoint"
    assert curve[0] < curve[k] and curve[-1] < curve[k]
    assert r["best_h"] == pytest.approx(r["best_h_rel"] * obj.case.lambda_s)


def test_the_gradient_gate_catches_a_graph_that_is_wrong_by_a_factor():
    """
    The check has teeth, and this is the failure mode it is for.

    `bug=1.02` leaves the objective's *value* exactly right and scales only the
    backward path -- a 2% error of the kind a misplaced normalisation or a wrong
    broadcast produces.  No forward test can see it; two digits of agreement is all it
    costs, and the gate asks for three.
    """
    r = SENS.autodiff_vs_fd_surrogate(_stub(bug=1.02), THETA)
    assert not r["gate_pass"]
    assert r["worst_digits"] == pytest.approx(math.log10(1 / 0.02), abs=0.05)

    # A sign error is the same test with the loudest possible answer.
    r = SENS.autodiff_vs_fd_surrogate(_stub(bug=-1.0), THETA)
    assert not r["gate_pass"] and r["worst_digits"] < 0.0


def test_the_gradient_gate_says_when_it_is_measuring_float32():
    """A float32 forward cannot reach three digits, and silence there is a wrong pass."""
    with pytest.warns(RuntimeWarning, match="float32"):
        SENS.autodiff_vs_fd_surrogate(_stub(dtype=torch.float32), THETA)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SENS.autodiff_vs_fd_surrogate(_stub(), THETA)      # float64: no warning


# ---------------------------------------------------------------------------
# The direction metric 9b is quoted in
# ---------------------------------------------------------------------------
def test_the_direction_metric_is_real_linear_not_hermitian():
    """
    A phase rotation is a physical disagreement, and |<a,b>| would score it as perfect.

    This is the whole reason `_rel_and_cosine` spells out `Re<a, b>`: theta is real, so
    a sensitivity column rotated in phase predicts a different arrival, and the
    Hermitian cosine -- the reflex when both vectors are complex -- cannot see it.
    """
    g = torch.Generator().manual_seed(cfg.SEED)
    b = torch.randn(64, generator=g, dtype=torch.float64).to(torch.complex128)
    b = b + 1j * torch.randn(64, generator=g, dtype=torch.float64)

    assert SENS._rel_and_cosine(b, b) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert SENS._rel_and_cosine(-b, b) == pytest.approx((2.0, -1.0), abs=1e-12)
    assert SENS._rel_and_cosine(3.0 * b, b) == pytest.approx((2.0, 1.0), abs=1e-12)

    for phi in (0.1, 0.5, 1.0, math.pi / 2):
        rel, cos = SENS._rel_and_cosine(b * math.e ** (1j * phi), b)
        assert cos == pytest.approx(math.cos(phi), abs=1e-12)
        hermitian = abs(complex((b.conj() * b * math.e ** (1j * phi)).sum())) / float(
            b.abs().pow(2).sum())
        assert hermitian == pytest.approx(1.0, abs=1e-12), "what we are not measuring"
        assert rel == pytest.approx(2 * abs(math.sin(phi / 2)), abs=1e-12)

    # Real input takes the real branch and gives the same answers.
    x = torch.tensor([3.0, 4.0], dtype=torch.float64)
    assert SENS._rel_and_cosine(x, x) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert SENS._rel_and_cosine(torch.zeros(2), x)[1] == 0.0, "no division by zero"


# ---------------------------------------------------------------------------
# The interface width, in physical units and separable from the grid (§3)
# ---------------------------------------------------------------------------
def _width_10_90(chi_row: torch.Tensor, x: torch.Tensor) -> float:
    """
    Distance between the chi = 0.9 and chi = 0.1 crossings.

    Interpolated in logit space, not in chi: chi = sigmoid(-phi/eps), so the logit is
    *linear* wherever phi is, and a linear interpolation of chi itself is off by ~0.3%
    of the width at this sampling -- enough to swamp the quantity being measured.
    """
    lg = torch.logit(chi_row.to(torch.float64).clamp(1e-12, 1 - 1e-12))
    out = []
    for level in (math.log(9.0), -math.log(9.0)):
        i = int((lg > level).nonzero()[-1])
        f = (float(lg[i]) - level) / (float(lg[i]) - float(lg[i + 1]))
        out.append(float(x[i]) + f * float(x[i + 1] - x[i]))
    return out[1] - out[0]


def test_fine_chi_takes_a_physical_width_and_is_family_general():
    """
    Two things `data.generate.fine_chi` cannot do, both required by the review.

    Family-generality, because 9b has to be askable of an ellipse -- the transfer claim
    is about shapes outside the training family, so a sensitivity gate that only accepts
    circles cannot test it.  And a width in physical units, because §3 asks that grid
    refinement and interface refinement be separable: quote eps in cells and halving dx
    silently halves the interface too, so the two effects can never be told apart.
    """
    yy, xx = SENS.fine_coords()
    mid = cfg.N_FINE_TOTAL // 2
    x = xx[mid]
    th = torch.tensor([[0.0, float(yy[mid, 0]), 0.9]])       # on the sampled row

    chi = SENS.fine_chi(th, Circle(), eps_phys=cfg.EPS_INTERFACE_PHYS)
    assert tuple(chi.shape) == (1, cfg.N_FINE_TOTAL, cfg.N_FINE_TOTAL)
    # chi saturates at both ends, and that is the fix rather than an accident.  This
    # test used to assert `chi.max() < 1.0`, which was true at the v2.0 interface width
    # and was precisely the defect: a sigmoid 6.6 network cells wide across a void 3.2
    # cells in radius never reaches 1, so the "void" was a soft dimple with a stiffness
    # floor it never touched.  At the measured width it reaches 1 in float32 well inside
    # the smallest radius in the range, so the material floors mean what they say.
    assert float(chi.min()) == 0.0 and float(chi.max()) == 1.0

    w = _width_10_90(chi[0, mid], x)
    assert w == pytest.approx(cfg.EPS_TRANSITION_FACTOR * cfg.EPS_INTERFACE_PHYS,
                              rel=1e-6)
    assert w / cfg.DX_NET == pytest.approx(0.8240, rel=1e-4), "0.82 network cells"
    assert w / cfg.DX_FINE == pytest.approx(1.6479, rel=1e-4), "1.65 fine cells"

    # Halving the physical width halves the transition on the *same* grid.
    w2 = _width_10_90(SENS.fine_chi(th, Circle(),
                                    eps_phys=cfg.EPS_INTERFACE_PHYS / 2)[0, mid], x)
    assert w2 == pytest.approx(w / 2, rel=1e-6)

    # An ellipse: five parameters, and -- to eight figures -- the same width.
    #
    # Not a coincidence, and worth recording next to review finding (g): `Ellipse.sdf`
    # is only an approximate distance (it returns (r^2 - R^2)/(2r) rather than r - R at
    # a = b = R), but that approximation is second-order *at* the interface, where its
    # gradient is exactly 1.  So the diffuse interface has the physical width claimed
    # for it even for the inexact family; what finding (g) actually distorts is the SDF
    # away from the boundary, i.e. the `phi_tilde` input channel and not chi.
    the = torch.tensor([[0.0, float(yy[mid, 0]), 1.1, 0.6, 0.0]])
    che = SENS.fine_chi(the, Ellipse(), eps_phys=cfg.EPS_INTERFACE_PHYS)
    assert tuple(che.shape) == tuple(chi.shape)
    assert _width_10_90(che[0, mid], x) == pytest.approx(w, rel=1e-6)


# ---------------------------------------------------------------------------
# The reference forward and 10b (these run the FDTD)
# ---------------------------------------------------------------------------
NT_SHORT = 64          # enough for the source to be injected and read; not physics


def _incident(nt: int) -> dict:
    """A zero incident cache, so `solver_receivers` returns the total field."""
    return dict(ascans=torch.zeros(cfg.N_SRC, len(cfg.NU_LIST), cfg.N_RECV, 2, nt))


@pytest.mark.slow
def test_solver_receivers_returns_the_observation_vector_shape_and_units():
    """
    [B, R, 2, M] complex, the same layout and the same route as `InverseCase.d_obs`.

    The route is the point.  `load_inversion_case` builds d_obs from *A-scans* through
    `displacement_from_ascans`; taking the incident from `incident["phasors"]` instead
    would look equivalent -- both are displacement phasors at the ring, both apply the
    same transfer factor -- and would leave a residue that is roundoff on the total
    field and a systematic bias on the scattered field, which is two orders of
    magnitude smaller.  So subtracting a *zero* incident here must return exactly the
    total field, with no cross-route mixing.
    """
    from src.solver import harmonic as H

    th = torch.tensor([[0.2, -0.3, 0.4], [0.2, -0.3, 0.5]])
    d = SENS.solver_receivers(th, Circle(), src_idx=0, nu_idx=0,
                              incident=_incident(NT_SHORT), nt=NT_SHORT)
    assert tuple(d.shape) == (2, cfg.N_RECV, 2, cfg.M_FREQ)
    assert d.is_complex() and torch.isfinite(d.abs()).all()
    assert float(d.abs().max()) > 0.0

    # Two different radii give two different observations: the map is not degenerate.
    assert float((d[0] - d[1]).abs().max()) > 1e-6 * float(d.abs().max())

    # And a nonzero incident subtracts exactly, in the units of the observation.
    inc = _incident(NT_SHORT)
    inc["ascans"][0, 0] += 1.0
    d2 = SENS.solver_receivers(th, Circle(), src_idx=0, nu_idx=0, incident=inc,
                               nt=NT_SHORT)
    om = H.omegas_tensor()
    off = H.displacement_from_ascans(inc["ascans"][0, 0].unsqueeze(0), omegas=om)
    assert torch.allclose(d2, d - off, atol=1e-6 * float(d.abs().max()))


@pytest.mark.slow
def test_verify_with_solver_scores_the_answer_and_refuses_to_pass_without_truth():
    """
    10b: the reference solver's opinion, and what it can and cannot certify.

    The final misfit is not a certificate (§8.5) -- a cycle-skipped inversion leaves a
    residual comparable to a correct one -- so the recovered geometry gets re-solved
    with a model the optimiser could not have exploited.  Where there is no ground
    truth there is also no position error, and `gate_pass` is then None rather than
    True: a None that reads as a pass is how a suite comes to certify data it never
    checked.

    The residual *ratio* is a separate matter: it is a measurement only when the truth's
    own residual is above the model-error floor, so it is reported for data carrying an
    error and withheld -- None, not a huge number -- for data this solver made itself.
    """
    fam, inc = Circle(), _incident(NT_SHORT)
    th_true = torch.tensor([0.2, -0.3, 0.45])
    d_obs = SENS.solver_receivers(th_true.reshape(1, -1), fam, src_idx=0, nu_idx=0,
                                  incident=inc, nt=NT_SHORT)
    case = InverseCase(d_obs=d_obs, src_idx=0, nu_idx=0, theta_true=th_true)

    def result_at(th, **kw):
        return InversionResult(theta=th, misfit=1e-4, theta_true=kw.pop("truth",
                               th_true), lambda_s=case.lambda_s, family=fam)

    # A successful answer: close but not exact, which is the only interesting case --
    # at theta_hat == theta_true the two solves are bit-identical and every ratio is
    # 0/0.  0.02 lambda_s is well inside the gate.
    hat = th_true + torch.tensor([0.02 * case.lambda_s, 0.0, 0.0])
    good = SENS.verify_with_solver(result_at(hat), case, incident=inc, nt=NT_SHORT)
    assert good["n_solves"] == 2
    assert good["misfit_solver_truth"] < 1e-10, "the truth reproduces its own data"
    assert good["position_error_ls"] == pytest.approx(0.02, rel=1e-6)
    assert good["gate_pass"] is True and good["gate"] == cfg.GATE_SOLVER_VERIFY_LS

    # ...and that is exactly why the ratio is withheld here.  d_obs came from this same
    # solver at this same nt, so the truth's residual is a numerical zero rather than a
    # small number, and J(theta_hat)/J(truth) is then unbounded for *any* estimate --
    # a large number that says nothing.  The ratio becomes a measurement when the data
    # carries noise or model error, which is the case it exists for; see below.
    assert good["truth_at_floor"] is True
    assert good["residual_ratio"] is None
    assert good["residual_floor"] == cfg.LOF_MODEL_FLOOR

    # A cycle-skip-sized miss: the reported misfit still says 1e-4, and the solver
    # disagrees.  This is the whole value of the check.
    bad_th = th_true + torch.tensor([0.75 * case.lambda_s, 0.0, 0.0])
    bad = SENS.verify_with_solver(result_at(bad_th), case, incident=inc, nt=NT_SHORT)
    assert bad["misfit_reported"] == 1e-4
    assert bad["misfit_solver"] > 100 * good["misfit_solver"]
    assert bad["residual_ratio"] is None and bad["truth_at_floor"] is True
    assert bad["position_error_ls"] == pytest.approx(0.75, rel=1e-6)
    assert bad["gate_pass"] is False

    # With a model error the solver cannot reproduce -- halving the data is the cheapest
    # one, and puts the truth's own residual at exactly 1.0 -- the floor is clear and
    # the ratio is reported.  This is the branch a noisy or transferred case takes.
    noisy = InverseCase(d_obs=d_obs * 0.5, src_idx=0, nu_idx=0, theta_true=th_true)
    meas = SENS.verify_with_solver(result_at(hat), noisy, incident=inc, nt=NT_SHORT)
    assert meas["truth_at_floor"] is False
    assert meas["misfit_solver_truth"] == pytest.approx(1.0, rel=1e-6)
    assert meas["residual_ratio"] == pytest.approx(
        meas["misfit_solver"] / meas["misfit_solver_truth"], rel=1e-9)

    # No truth: one solve, a residual, and no verdict.
    blind = InverseCase(d_obs=d_obs, src_idx=0, nu_idx=0)
    r = SENS.verify_with_solver(result_at(bad_th, truth=None), blind, incident=inc,
                                nt=NT_SHORT)
    assert r["n_solves"] == 1
    assert r["gate_pass"] is None and r["position_error_ls"] is None
    assert r["residual_ratio"] is None
    assert r["misfit_solver"] == pytest.approx(bad["misfit_solver"], rel=1e-6)
    assert r["misfit_surrogate"] is None, "no forward= given, so nothing to compare"
    assert r["surrogate_optimism"] is None


@pytest.mark.slow
def test_the_physics_gate_compares_two_models_and_fails_an_untrained_one():
    """
    9b end to end, at one perturbation size, with a tiny untrained surrogate.

    An untrained network *must* fail: its receiver sensitivities are whatever its
    initialisation happens to produce, and a gate that passes them is measuring
    nothing.  That is the assertion here -- the numbers are plumbing, the failure is
    the point -- so this test says the gate has teeth without needing a trained model
    or the minutes of FDTD that the real check in notebook 05 costs.

    It runs twice, for the two branches of the residual-floor guard: once at the truth,
    where the solver reproduces its own data and the residual-weighted scalar gradient
    is withheld, and once at a perturbed geometry against data with a model error in it,
    which is where the notebook evaluates it and where that comparison means something.
    """
    from src.models.fno2d import FNO2d
    from src.inverse.misfit import SurrogateForward

    fam = Circle()
    band = slice(0, 2)                       # two frequencies: this is a plumbing test
    m, ny = cfg.M_FREQ, cfg.N_NET
    inc = dict(
        ascans=torch.zeros(cfg.N_SRC, len(cfg.NU_LIST), cfg.N_RECV, 2, NT_SHORT),
        phasors=(1e-3 * torch.randn(cfg.N_SRC, len(cfg.NU_LIST), 2, m, ny, ny)
                 ).to(torch.complex64),
        scale=torch.ones(cfg.N_SRC, len(cfg.NU_LIST), m))

    th = torch.tensor([[0.2, -0.3, 0.45]])
    d_obs = SENS.solver_receivers(th, fam, src_idx=0, nu_idx=0, incident=inc,
                                  nt=NT_SHORT)
    case = InverseCase(d_obs=d_obs, src_idx=0, nu_idx=0, theta_true=th[0])
    fwd = SurrogateForward(FNO2d(d_v=4, n_blocks=1, kmax=6, lift_hidden=8,
                                 proj_hidden=8), inc)

    r = SENS.surrogate_vs_solver_sensitivity(
        fwd, case, fam, incident=inc, h_rel=(0.05,), band=band, nt=NT_SHORT)

    assert r["n_solves"] == 1 + 2 * 3, "one baseline plus two per parameter"
    assert len(r["rows"]) == 3 and r["param_names"] == ["xc", "yc", "R"]
    assert r["band"] == (0, 2) and r["eps_cells"] == pytest.approx(
        cfg.EPS_INTERFACE_CELLS)
    assert r["gate_rel"] == cfg.GATE_SENSITIVITY_REL
    assert r["gate_cosine"] == cfg.GATE_SENSITIVITY_COSINE
    assert not r["gate_pass"], "an untrained network must not pass a physics gate"
    assert -1.0 <= r["cosine_worst"] <= 1.0 and math.isfinite(r["rel_worst"])
    # Evaluated at the truth, on data this solver made: dJ/dtheta = 2 Re<r, J_k> is
    # roundoff divided by roundoff, so the scalar comparison is withheld.  The Jacobian
    # columns above carry no residual factor and are compared either way.
    assert r["residual_at_floor"] is True
    assert r["dj_rel"] is None and r["dj_cosine"] is None
    assert r["residual_floor"] == cfg.LOF_MODEL_FLOOR
    assert r["misfit_solver"] < cfg.LOF_MODEL_FLOOR
    # With a single h the stability clause reduces to "did that h pass", and it did not.
    assert r["n_h_pass"] == 0 and r["gate_pass_stable"] is False

    # And where notebook 05 measures it: a perturbed geometry, against data carrying a
    # model error the solver cannot reproduce.  Halving the data is the cheapest such
    # error, and at this nt it dominates the mismatch -- the residual is the missing half
    # of the data, near 1.0 and far clear of the floor -- so the scalar gradients are
    # compared and each side reports its own magnitude.
    off = th[0] + torch.tensor([0.2, -0.15, 0.1]) * case.lambda_s
    r2 = SENS.surrogate_vs_solver_sensitivity(
        fwd, InverseCase(d_obs=d_obs * 0.5, src_idx=0, nu_idx=0, theta_true=th[0]),
        fam, incident=inc, theta=off.reshape(1, -1), h_rel=(0.05,), band=band,
        nt=NT_SHORT)
    assert r2["theta"].tolist() == pytest.approx(off.tolist(), rel=1e-6)
    assert r2["residual_at_floor"] is False
    assert r2["misfit_solver"] == pytest.approx(1.0, rel=1e-3)
    assert r2["misfit_solver"] > 100 * cfg.LOF_MODEL_FLOOR
    assert math.isfinite(r2["dj_rel"]) and -1.0 <= r2["dj_cosine"] <= 1.0
    assert r2["dj_dtheta"][0]["norm_solver"] > 0.0
    assert r2["dj_dtheta"][0]["norm_surrogate"] > 0.0


# ---------------------------------------------------------------------------
# The eps anneal, licensed or not
# ---------------------------------------------------------------------------
def test_eps_transfer_report_licenses_the_anneal_only_if_every_width_passes(monkeypatch):
    """
    The aggregation, tested without the 57 FDTD runs the real thing costs.

    `EPS_INVERT_ANNEAL` is off because the surrogate saw one interface width in
    training, and the schedule's endpoints -- 2.0 cells and 1.0 -- are both
    extrapolations.  What this function must get right is that bracketing the trained
    width does not count as validating it: a pass at 1.5 and failures at the endpoints
    is exactly the situation the anneal would walk into, and `anneal_licensed` has to
    be False there.
    """
    seen = []

    def fake(forward, case, family, *, incident, theta=None,
             eps_phys=cfg.EPS_INTERFACE_PHYS, **kw):
        seen.append(eps_phys)
        trained = abs(eps_phys - cfg.EPS_INTERFACE_PHYS) < 1e-12
        return dict(rel_worst=0.05 if trained else 0.4,
                    cosine_worst=0.99 if trained else 0.5,
                    dj_rel=0.1, dj_cosine=0.9, gate_pass=trained)

    monkeypatch.setattr(SENS, "surrogate_vs_solver_sensitivity", fake)
    rep = SENS.eps_transfer_report(None, None, Circle(), incident={})

    assert rep["eps_cells"] == (cfg.EPS_INVERT_START, cfg.EPS_INTERFACE_CELLS,
                                cfg.EPS_INVERT_END)
    assert seen == [e * cfg.DX_NET for e in rep["eps_cells"]], "physical widths, in order"
    assert [x["trained_width"] for x in rep["per_eps"]] == [False, True, False]
    assert rep["anneal_licensed"] is False, "two of three widths fail"
    assert rep["anneal_enabled"] is cfg.EPS_INVERT_ANNEAL is False
    assert rep["degradation"] == pytest.approx(0.4 / 0.05)

    # And the other branch: everything passes, so the anneal is licensed.  The widths
    # have to be given relative to `EPS_INTERFACE_CELLS` rather than as literals -- the
    # trained width is now 0.1875 net cells, and `degradation` is defined as a ratio
    # against the trained width, so a pair that does not contain it is None by design.
    monkeypatch.setattr(SENS, "surrogate_vs_solver_sensitivity",
                        lambda *a, **k: dict(rel_worst=0.05, cosine_worst=0.99,
                                             dj_rel=0.1, dj_cosine=0.9,
                                             gate_pass=True))
    ok = SENS.eps_transfer_report(None, None, Circle(), incident={},
                                 eps_cells=(cfg.EPS_INTERFACE_CELLS,
                                            cfg.EPS_INVERT_END))
    assert ok["anneal_licensed"] is True
    assert ok["degradation"] == pytest.approx(1.0)
    assert len(ok["per_eps"]) == 2

    # A schedule that never visits the trained width cannot report a degradation.
    none = SENS.eps_transfer_report(None, None, Circle(), incident={},
                                    eps_cells=(1.5, 1.2))
    assert none["degradation"] is None








