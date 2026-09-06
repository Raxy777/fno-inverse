"""
Two gradient gates, and the solver's opinion of the answer (§11.2 steps 9a/9b/10b).

The headline claim of this project is that giving the surrogate the incident *field*
rather than a source label is what lets it transfer to shapes outside its training
family.  Field inputs make that possible; they do not establish it.  Two separate
things have to be true, and passing either one alone proves nothing about the other:

    9a  the autodiff gradient of the surrogate equals a finite difference *of the
        surrogate*.  This is a statement about the graph -- `soft_indicator`, the
        frequency-folded batch axis, the complex spectral weights, the F-major scale
        broadcast -- and it is easy to get wrong and easy to check.  It can pass
        perfectly while every number in it is physically meaningless.

    9b  the surrogate's receiver sensitivities equal a finite difference of the
        *validated reference solver*, over several perturbation sizes.  This is a
        statement about physics, and it is the one the thesis rests on: an inversion
        driven by a correctly differentiated but physically wrong sensitivity
        converges confidently to the wrong geometry, reports a small residual, and
        looks like a success from the outside.

`verify_with_solver` is the third leg, after the fact rather than before: re-solve at
the recovered geometry with the reference solver and see whether the fit survives a
model the optimiser could not have exploited.  §8.5's cycle-skipping demonstration is
the reason it exists -- a converged cycle-skipped inversion leaves a residual
comparable to a correct one, so the final misfit is not a certificate.

`eps_transfer_report` is 9b asked at the interface widths `cfg.EPS_INVERT_ANNEAL`
would visit.  The anneal is off by default because the surrogate saw exactly one
interface width in training; this is the measurement that would license turning it on.

Cost, stated plainly because it decides how these are used: every solver sensitivity
column costs two full FDTD runs on the 376^2 padded grid, so 9b at one perturbation
size with three parameters is 6 solves plus a baseline.  These are validation
functions run on a handful of cases, not part of an inversion.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from .. import config as cfg
from ..geometry.sdf import (ShapeFamily, fine_coords, material_fields,
                            soft_indicator)
from ..solver import harmonic as H
from ..solver.fdtd_elastic import ElasticFDTD2D
from .invert import InversionResult
from .misfit import InverseCase, Objective, SurrogateForward


# ---------------------------------------------------------------------------
# The reference forward, in the observation's own units
# ---------------------------------------------------------------------------
def fine_chi(theta: Tensor, family: ShapeFamily, *,
             eps_phys: float = cfg.EPS_INTERFACE_PHYS,
             device=None, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Soft void indicator on the padded fine solver grid, [B, 376, 376].

    Family-general, and parameterised by a *physical* interface width rather than a
    cell count.  `data.generate.fine_chi` hard-codes `Circle()` and
    `EPS_LEN_PHYS`, which is correct for dataset generation (the training set is
    circles at one width) and wrong here: 9b has to be askable for an ellipse, and
    §3 of the review asks that grid refinement and interface refinement be separable,
    which they are not while eps is quoted in cells.
    """
    yy, xx = fine_coords(device=device, dtype=dtype)
    phi = family.sdf(theta, yy, xx)
    return soft_indicator(phi, eps_phys)


def solver_receivers(theta: Tensor, family: ShapeFamily, *, src_idx: int,
                     nu_idx: int, incident: dict,
                     eps_phys: float = cfg.EPS_INTERFACE_PHYS,
                     nt: int = cfg.NT, device=None,
                     progress=None) -> Tensor:
    """
    theta [B, P] -> scattered displacement phasors at the ring, [B, R, 2, M] complex.

    The same quantity as `InverseCase.d_obs`, by the same route: total A-scans
    through `displacement_from_ascans`, minus the cached *incident A-scans* through
    the same function.  Taking the incident from `incident["phasors"]` instead would
    be a plausible-looking mistake -- those are field phasors from the running DFT
    and the difference of the two routes is roundoff on the total field but a
    systematic bias on the scattered field, which is two orders of magnitude smaller.

    No `torch.no_grad()` decoration is needed and none would help: `ElasticFDTD2D.run`
    is already `@torch.no_grad()`, which is exactly why the solver has to be
    finite-differenced rather than differentiated.
    """
    B = theta.shape[0]
    om = H.omegas_tensor(device)
    chi = fine_chi(theta, family, eps_phys=eps_phys, device=device)
    lam0, mu0 = cfg.lame_from_nu(float(cfg.NU_LIST[nu_idx]))
    lam, mu, rho = material_fields(chi, lam0, mu0)

    sim = ElasticFDTD2D(lam, mu, rho)
    src = [cfg.net_to_fine(*cfg.SOURCES_NET[src_idx])] * B
    res = sim.run(src, nt=nt, recv_yx=cfg.RECEIVERS_NET, omegas=om,
                  progress=progress)

    u_tot = H.displacement_from_ascans(res.ascans, omegas=om)          # [B,R,2,M]
    a_inc = incident["ascans"][src_idx, nu_idx].unsqueeze(0).to(u_tot.device)
    u_inc = H.displacement_from_ascans(a_inc, omegas=om)               # [1,R,2,M]
    return u_tot - u_inc


# ---------------------------------------------------------------------------
# 9a: the autodiff graph (surrogate against itself)
# ---------------------------------------------------------------------------
DEFAULT_H_SWEEP: tuple[float, ...] = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4,
                                      3e-5, 1e-5, 3e-6, 1e-6, 3e-7)


def autodiff_vs_fd_surrogate(obj: Objective, theta: Tensor, *,
                             h_rel: tuple[float, ...] = DEFAULT_H_SWEEP,
                             gate: int = cfg.GATE_GRAD_SIGFIGS) -> dict:
    """
    dJ/dtheta by the chain rule against dJ/dtheta by central differences, in digits.

    Both numbers differentiate the surrogate.  Agreement says the graph is right and
    says nothing about the physics -- that is `surrogate_vs_solver_sensitivity`.

    The sweep over `h_rel` (in units of lambda_s) is not decoration.  A single step
    size cannot distinguish a wrong gradient from a badly chosen h: truncation error
    falls as h^2 while cancellation error grows as 1/h, so the digits-versus-h curve
    is a tent whose peak is the real measurement and whose two flanks are artefacts of
    the estimator.  Quoting the peak, and keeping the whole curve so the tent shape is
    visible, is the difference between a check and a coincidence.

    `theta` is [P] or [B, P]; only the first row is differentiated.
    """
    if obj.forward.dtype != torch.float64:
        warnings.warn(
            "autodiff_vs_fd_surrogate on a float32 forward model: central "
            "differences cancel to ~1e-7 relative, so the tent peaks near 3-4 "
            "digits and a gate of "
            f"{gate} is measuring float32 rather than the graph.  Build the "
            "SurrogateForward with dtype=torch.float64 (see nb05).",
            RuntimeWarning, stacklevel=2)

    th = theta if theta.dim() == 2 else theta.unsqueeze(0)
    th = th[:1].detach().to(obj.forward.device)
    lam_s = obj.case.lambda_s
    names = list(obj.family.param_names)
    p = th.shape[1]

    t_ad = th.clone().requires_grad_(True)
    j0 = obj.residual(t_ad)
    j0.backward()
    grad_ad = t_ad.grad[0].detach().clone()

    def fd(h: float) -> Tensor:
        out = torch.zeros(p, dtype=grad_ad.dtype, device=grad_ad.device)
        with torch.no_grad():
            for i in range(p):
                e = torch.zeros_like(th)
                e[0, i] = h
                out[i] = (obj.residual(th + e) - obj.residual(th - e)) / (2.0 * h)
        return out

    hs = torch.tensor([r * lam_s for r in h_rel], dtype=torch.float64)
    digits = torch.zeros(len(hs), p, dtype=torch.float64)
    grads = []
    for k, h in enumerate(hs):
        g = fd(float(h))
        grads.append(g)
        rel = (g - grad_ad).abs().to(torch.float64) / grad_ad.abs().to(
            torch.float64).clamp_min(1e-300)
        digits[k] = -torch.log10(rel.clamp_min(1e-300))

    best = int(digits.min(dim=1).values.argmax())
    worst_digits = float(digits[best].min())
    return dict(
        grad_autodiff=grad_ad.cpu(), grad_fd=grads[best].cpu(),
        digits=digits[best].cpu(), digits_curve=digits.cpu(),
        h=hs.cpu(), h_rel=tuple(h_rel), best_h=float(hs[best]),
        best_h_rel=float(h_rel[best]), worst_digits=worst_digits,
        param_names=names, j=float(j0.detach()),
        gate=int(gate), gate_pass=bool(worst_digits >= gate),
        dtype=str(obj.forward.dtype),
    )


# ---------------------------------------------------------------------------
# 9b: the physics (surrogate against the reference solver)
# ---------------------------------------------------------------------------
def _rel_and_cosine(a: Tensor, b: Tensor) -> tuple[float, float]:
    """
    Relative L2 error and direction agreement between two complex tensors.

    The cosine is the *real-linear* one: C^N is treated as R^2N, so the inner product
    is Re<a, b>.  This is the right notion here and the Hermitian |<a,b>|/(|a||b|) is
    not: theta is real, so a sensitivity column that is correct in magnitude but
    rotated in phase describes a different physical response, and an absolute value
    would score that rotation as perfect agreement.
    """
    af, bf = a.reshape(-1), b.reshape(-1)
    nb = float(bf.abs().pow(2).sum().sqrt())
    na = float(af.abs().pow(2).sum().sqrt())
    rel = float((af - bf).abs().pow(2).sum().sqrt()) / max(nb, 1e-300)
    dot = float(torch.real(af.conj() * bf).sum()) if af.is_complex() else float(
        (af * bf).sum())
    return rel, dot / max(na * nb, 1e-300)


def _solver_columns(thetas: Tensor, family: ShapeFamily, *, src_idx: int,
                    nu_idx: int, incident: dict, eps_phys: float, nt: int,
                    device, batch: int, progress=None) -> Tensor:
    """`solver_receivers` over a long list of geometries, in chunks of `batch`."""
    out = []
    lo_range = range(0, thetas.shape[0], batch)
    it = progress(lo_range) if progress is not None else lo_range
    for lo in it:
        out.append(solver_receivers(thetas[lo:lo + batch], family,
                                    src_idx=src_idx, nu_idx=nu_idx,
                                    incident=incident, eps_phys=eps_phys,
                                    nt=nt, device=device))
    return torch.cat(out, dim=0)


SOLVER_H_SWEEP: tuple[float, ...] = (0.05, 0.02, 0.01)


def surrogate_vs_solver_sensitivity(
        forward: SurrogateForward, case: InverseCase, family: ShapeFamily, *,
        incident: dict, theta: Tensor | None = None,
        h_rel: tuple[float, ...] = SOLVER_H_SWEEP,
        band: slice = cfg.BAND_STAGE3,
        eps_phys: float = cfg.EPS_INTERFACE_PHYS,
        nt: int = cfg.NT, batch: int = cfg.GEN_BATCH,
        gate_rel: float = cfg.GATE_SENSITIVITY_REL,
        gate_cos: float = cfg.GATE_SENSITIVITY_COSINE,
        device=None, progress=None) -> dict:
    """
    Do the surrogate's receiver sensitivities agree with the reference solver's?

    The gate the thesis rests on (§1.2, §8.5, §11.2 step 9b).  For each parameter and
    each perturbation size it forms one column of the receiver Jacobian,

        J_k = d u_s(theta) / d theta_k     in C^{R x 2 x F},

    from the *same* central difference applied to two different models, and reports
    magnitude agreement (`GATE_SENSITIVITY_REL`) and direction agreement
    (`GATE_SENSITIVITY_COSINE`).

    Central differences on both sides rather than autodiff on the surrogate side, and
    that choice is deliberate: an identical estimator makes the comparison isolate the
    physics instead of confounding it with differentiation error, and 9a
    (`autodiff_vs_fd_surrogate`) is what licenses the transitive step to the gradient
    the optimiser actually descends.  The solver *cannot* be differentiated in any
    case -- `ElasticFDTD2D.run` is `@torch.no_grad()`, and a 3000-step tape over
    376^2 fields is not a thing to make lightly.

    Relative error rather than significant figures because two different models are
    never going to agree to 3 s.f.; 20% is a claim that the surrogate has the right
    physics, not that it is the solver.

    **Where to evaluate it.** Pass `theta=` a perturbed geometry, not the truth.  The
    Jacobian columns are meaningful anywhere, but the scalar `dJ/dtheta` reported
    alongside them is proportional to the residual, and on noise-free data from this
    same solver the residual at `theta_true` is a numerical zero -- so that comparison
    would divide one roundoff by another.  `residual_at_floor` says when this has
    happened and `dj_rel`/`dj_cosine` are then None.  Notebook 05 uses the same
    `theta_true + delta` as step 9a, which makes the two gates a matched pair.

    Cost: `1 + 2 * P * len(h_rel)` FDTD runs, batched `batch` at a time.
    """
    dev = device or forward.device
    th0 = (case.theta_true if theta is None else theta).detach()
    th0 = (th0 if th0.dim() == 2 else th0.unsqueeze(0))[:1].to(dev)
    assert th0 is not None, "pass theta= when the case has no ground truth"
    lam_s, p = case.lambda_s, th0.shape[1]
    names = list(family.param_names)

    hs = [r * lam_s for r in h_rel]
    thetas = [th0]
    for h in hs:
        for k in range(p):
            e = torch.zeros_like(th0)
            e[0, k] = h
            thetas += [th0 + e, th0 - e]
    thetas = torch.cat(thetas, dim=0)                       # [1 + 2*p*len(hs), P]

    d_sol = _solver_columns(thetas, family, src_idx=case.src_idx,
                            nu_idx=case.nu_idx, incident=incident,
                            eps_phys=eps_phys, nt=nt, device=dev,
                            batch=batch, progress=progress)[..., band]
    eps_cells = eps_phys / cfg.DX_NET
    with torch.no_grad():
        d_sur = torch.cat([
            forward.predict(thetas[lo:lo + batch], family, src_idx=case.src_idx,
                            nu_idx=case.nu_idx, band=band, eps_cells=eps_cells)
            for lo in range(0, thetas.shape[0], batch)], dim=0)

    obs = case.d_obs[..., band].to(d_sur.device, d_sur.dtype)
    n_obs2 = max(float(obs.abs().pow(2).sum()), 1e-300)
    r_sol = (d_sol[:1].to(d_sur.dtype) - obs)
    r_sur = (d_sur[:1] - obs)
    j_sol = float(r_sol.abs().pow(2).sum()) / n_obs2
    # The scalar dJ/dtheta = 2 Re<r, J_k> is proportional to the residual, so at a point
    # where the solver reproduces its own noise-free data it is *identically* zero and
    # the comparison below divides one roundoff by another.  Evaluate 9b at a perturbed
    # geometry -- exactly the reason 9a is measured at theta_true + delta -- and this
    # flag says whether that was done.  The Jacobian columns themselves do not involve
    # the residual and stay meaningful either way.
    at_floor = j_sol < cfg.LOF_MODEL_FLOOR

    rows, rel_h, cos_h, gj_h = [], [], [], []
    for hi, h in enumerate(hs):
        rel_k, cos_k, g_sol, g_sur = [], [], [], []
        for k in range(p):
            base = 1 + 2 * (hi * p + k)
            js = (d_sol[base] - d_sol[base + 1]) / (2.0 * h)
            jn = (d_sur[base] - d_sur[base + 1]) / (2.0 * h)
            r, c = _rel_and_cosine(jn.to(torch.complex128),
                                   js.to(torch.complex128))
            rel_k.append(r)
            cos_k.append(c)
            # dJ/dtheta_k for J = ||pred - obs||^2 / ||obs||^2, each model scored
            # against the same observation with the same functional.
            g_sol.append(2.0 * float(torch.real(r_sol[0].conj()
                                                * js.to(r_sol.dtype)).sum()) / n_obs2)
            g_sur.append(2.0 * float(torch.real(r_sur[0].conj() * jn).sum()) / n_obs2)
            rows.append((names[k], h / lam_s, r, c))

        gs = torch.tensor(g_sol, dtype=torch.float64)
        gn = torch.tensor(g_sur, dtype=torch.float64)
        gj_h.append(dict(
            h_rel=h / lam_s, solver=gs.tolist(), surrogate=gn.tolist(),
            norm_solver=float(gs.norm()), norm_surrogate=float(gn.norm()),
            rel=(None if at_floor else
                 float((gn - gs).norm() / gs.norm().clamp_min(1e-300))),
            cosine=(None if at_floor else
                    float(torch.dot(gn, gs)
                          / (gn.norm() * gs.norm()).clamp_min(1e-300)))))
        rel_h.append(max(rel_k))          # worst parameter at this h
        cos_h.append(min(cos_k))

    # Best h is the peak of the tent, as in 9a: the largest h that has stopped being a
    # secant, the smallest that has not yet drowned in the solver's own discretisation
    # noise.  Chosen by the direction metric, because the cosine is bounded and so
    # comparable across h while the relative error is not.
    best = int(torch.tensor(cos_h).argmax())
    n_pass = sum(1 for i in range(len(hs))
                 if rel_h[i] <= gate_rel and cos_h[i] >= gate_cos)
    # A pass at exactly one perturbation size out of three is a coincidence, not a
    # measurement, so with a real sweep the agreement has to survive more than one h.
    stable = n_pass >= (2 if len(hs) >= 3 else 1)
    return dict(
        rows=rows, per_h=[dict(h_rel=h / lam_s, rel_worst=rel_h[i],
                               cosine_worst=cos_h[i]) for i, h in enumerate(hs)],
        rel_worst=rel_h[best], cosine_worst=cos_h[best],
        best_h=hs[best], best_h_rel=h_rel[best], h_rel=tuple(h_rel),
        dj_dtheta=gj_h, dj_rel=gj_h[best]["rel"], dj_cosine=gj_h[best]["cosine"],
        residual_at_floor=bool(at_floor), residual_floor=cfg.LOF_MODEL_FLOOR,
        misfit_solver=j_sol,
        misfit_surrogate=float(r_sur.abs().pow(2).sum()) / n_obs2,
        theta=th0[0].cpu(), param_names=names, family=family.name,
        eps_phys=float(eps_phys), eps_cells=float(eps_cells),
        band=(band.start, band.stop), n_solves=int(thetas.shape[0]),
        gate_rel=float(gate_rel), gate_cosine=float(gate_cos),
        n_h_pass=int(n_pass), gate_pass_stable=bool(stable),
        gate_pass_rel=bool(rel_h[best] <= gate_rel),
        gate_pass_cosine=bool(cos_h[best] >= gate_cos),
        gate_pass=bool(rel_h[best] <= gate_rel and cos_h[best] >= gate_cos
                       and stable),
    )


# ---------------------------------------------------------------------------
# 10b: the solver's opinion of the recovered geometry
# ---------------------------------------------------------------------------
def verify_with_solver(result: InversionResult, case: InverseCase, *,
                       incident: dict, family: ShapeFamily | None = None,
                       forward: SurrogateForward | None = None,
                       band: slice = cfg.BAND_STAGE3,
                       eps_phys: float = cfg.EPS_INTERFACE_PHYS,
                       nt: int = cfg.NT, device=None,
                       gate: float = cfg.GATE_SOLVER_VERIFY_LS) -> dict:
    """
    Re-solve at the recovered geometry with the reference solver (§11.2 step 10b).

    Why this exists, in one sentence from §8.5: a converged cycle-skipped inversion
    leaves a residual comparable to a correct one, so the final misfit is not a
    certificate.  A surrogate error the optimiser has learned to exploit does not
    survive an independent forward solve, which makes the solver residual evidence
    that the surrogate's own residual cannot be.

    What is gated is `GATE_SOLVER_VERIFY_LS`, the position error of the recovered
    geometry -- so this is a check available only where ground truth is.  On data with
    no truth (`theta_true is None`) `gate_pass` is None, `residual_ratio` is None, and
    `misfit_solver` is the only number: an absolute residual with nothing to compare it
    against.  Said explicitly because a None that reads as a pass is exactly how a
    validation suite comes to certify real data it never checked.

    `residual_ratio` is also None when the truth's own residual sits at the numerical
    floor (`truth_at_floor`), which is what happens on noise-free data generated by
    this same solver: dividing by a numerical zero makes the ratio unbounded for every
    estimate, correct or not.  There `misfit_solver` and `surrogate_optimism` are the
    measurements, and comparing two candidate answers to each other is what the ratio
    was for.

    Cost: two FDTD runs when there is truth to compare against, one when there is not.
    """
    fam = family or result.family
    assert fam is not None, "pass family= : the result carries no family"
    fam_true = case.truth_family or result.truth_family or fam
    dev = device or (forward.device if forward is not None else None)
    th_hat = result.theta.detach().reshape(1, -1).to(dev)
    th_true = case.theta_true if case.theta_true is not None else result.theta_true

    # One batched solve when the estimate and the truth share a parameterisation, two
    # when they do not -- the transfer case, where the truth is an ellipse or a pair of
    # voids and the fit is a circle.  Concatenating those into one batch used to raise a
    # shape error, which meant the one experiment that most needs an independent forward
    # solve at the *true* geometry was the one experiment that could not have it.
    if th_true is None:
        d = solver_receivers(th_hat, fam, src_idx=case.src_idx, nu_idx=case.nu_idx,
                             incident=incident, eps_phys=eps_phys, nt=nt, device=dev)
    elif fam_true.name == fam.name:
        d = solver_receivers(torch.cat([th_hat, th_true.detach().reshape(1, -1).to(dev)],
                                       dim=0),
                             fam, src_idx=case.src_idx, nu_idx=case.nu_idx,
                             incident=incident, eps_phys=eps_phys, nt=nt, device=dev)
    else:
        d = torch.cat([
            solver_receivers(th_hat, fam, src_idx=case.src_idx, nu_idx=case.nu_idx,
                             incident=incident, eps_phys=eps_phys, nt=nt, device=dev),
            solver_receivers(th_true.detach().reshape(1, -1).to(dev), fam_true,
                             src_idx=case.src_idx, nu_idx=case.nu_idx,
                             incident=incident, eps_phys=eps_phys, nt=nt, device=dev)],
            dim=0)
    d = d[..., band]
    obs = case.d_obs[..., band].to(d.device, d.dtype)
    n_obs2 = max(float(obs.abs().pow(2).sum()), 1e-300)
    j = [float((d[i:i + 1] - obs).abs().pow(2).sum()) / n_obs2
         for i in range(d.shape[0])]

    j_sur = None
    if forward is not None:
        with torch.no_grad():
            pred = forward.predict(th_hat, fam, src_idx=case.src_idx,
                                   nu_idx=case.nu_idx, band=band,
                                   eps_cells=eps_phys / cfg.DX_NET)
        j_sur = float((pred - obs.to(pred.dtype)).abs().pow(2).sum()) / n_obs2

    pos = result.position_error_ls
    # On noise-free data produced by this same solver at this same nt, the residual at
    # the truth is a numerical zero rather than a small number, so J(theta_hat)/J(truth)
    # is unbounded for *any* estimate and says nothing.  The ratio becomes a measurement
    # exactly when the data carries noise or model error -- which is the case it is for.
    truth_at_floor = (th_true is not None and j[1] < cfg.LOF_MODEL_FLOOR)
    return dict(
        theta=th_hat[0].cpu(), theta_true=None if th_true is None else th_true.cpu(),
        misfit_solver=j[0], misfit_solver_truth=(None if th_true is None else j[1]),
        misfit_surrogate=j_sur, misfit_reported=float(result.misfit),
        # The solver's residual at the answer, relative to its residual at the truth.
        # Reported as a ratio and not as a "solver prefers the truth" flag: with
        # noise-free data from this same solver the truth fits better than *any*
        # non-exact estimate, so the boolean would be True for every successful
        # inversion ever run.  The magnitude is the signal -- close to 1 for a good
        # answer once there is a real noise floor to compare against, orders of
        # magnitude above it for a cycle-skipped one either way.
        residual_ratio=(None if th_true is None or truth_at_floor
                        else j[0] / j[1]),
        truth_at_floor=bool(truth_at_floor), residual_floor=cfg.LOF_MODEL_FLOOR,
        # The exploitation signature: the surrogate scores the recovered geometry far
        # better than the solver does, i.e. the optimiser found a hole in the model
        # rather than a defect in the specimen.
        surrogate_optimism=(None if j_sur is None else j[0] / max(j_sur, 1e-300)),
        position_error_ls=pos, gate=float(gate),
        gate_pass=(None if pos is None else bool(pos <= gate)),
        eps_phys=float(eps_phys), band=(band.start, band.stop),
        family=fam.name, n_solves=int(d.shape[0]),
    )


# ---------------------------------------------------------------------------
# The interface-width anneal, licensed or not
# ---------------------------------------------------------------------------
def eps_transfer_report(forward: SurrogateForward, case: InverseCase,
                        family: ShapeFamily, *, incident: dict,
                        eps_cells: tuple[float, ...] | None = None,
                        theta: Tensor | None = None, **kw) -> dict:
    """
    9b asked again at each interface width `cfg.EPS_INVERT_ANNEAL` would visit.

    `EPS_INVERT_ANNEAL` is False because training used one interface width and
    nothing else, so a schedule from 2.0 cells down to 1.0 asks the surrogate three
    questions it was never asked and calls the answers gradients.  Bracketing the
    training value is not validation of the endpoints -- and the endpoints are where
    the anneal spends its first and last iterations.

    This is the measurement that would license turning it on: agreement with the
    solver, at each width, to the same 9b gates.  `anneal_licensed` is the conjunction.
    """
    if eps_cells is None:
        eps_cells = (cfg.EPS_INVERT_START, cfg.EPS_INTERFACE_CELLS,
                     cfg.EPS_INVERT_END)

    per_eps = []
    for e in eps_cells:
        r = surrogate_vs_solver_sensitivity(
            forward, case, family, incident=incident, theta=theta,
            eps_phys=e * cfg.DX_NET, **kw)
        per_eps.append(dict(
            eps_cells=float(e), eps_phys=float(e * cfg.DX_NET),
            trained_width=bool(abs(e - cfg.EPS_INTERFACE_CELLS) < 1e-9),
            rel_worst=r["rel_worst"], cosine_worst=r["cosine_worst"],
            dj_cosine=r["dj_cosine"], dj_rel=r["dj_rel"],
            gate_pass=r["gate_pass"], report=r))

    licensed = all(x["gate_pass"] for x in per_eps)
    trained = [x for x in per_eps if x["trained_width"]]
    return dict(
        per_eps=per_eps, eps_cells=tuple(float(e) for e in eps_cells),
        anneal_licensed=bool(licensed),
        anneal_enabled=bool(cfg.EPS_INVERT_ANNEAL),
        # Degradation away from the trained width, which is the quantity the anneal is
        # betting against: 1.0 would mean the width does not matter at all.
        degradation=(None if not trained else
                     max(x["rel_worst"] for x in per_eps)
                     / max(trained[0]["rel_worst"], 1e-300)),
        gate_rel=float(kw.get("gate_rel", cfg.GATE_SENSITIVITY_REL)),
        gate_cosine=float(kw.get("gate_cos", cfg.GATE_SENSITIVITY_COSINE)),
        family=family.name,
    )


__all__ = [
    "fine_chi", "solver_receivers",
    "autodiff_vs_fd_surrogate", "surrogate_vs_solver_sensitivity",
    "verify_with_solver", "eps_transfer_report",
    "DEFAULT_H_SWEEP", "SOLVER_H_SWEEP",
]

