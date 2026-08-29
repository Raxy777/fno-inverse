"""
The four-stage inversion (§8.4, §11.2 steps 10-12).

    Stage 0  CNN regressor, one forward pass            -> one candidate
    Stage 1  256-candidate amplitude screen, m = 1..6   -> 16 survivors
    Stage 2  Adam, 200 steps, m = 1..10, eps = 2.0 dx   -> best survivor
    Stage 3  L-BFGS, full band, eps annealed 2.0 -> 1.0 -> the answer

Why four stages and not one Adam run from a random start: the misfit is non-convex
with a basin about lambda_s/4 across (§11.3's landscape figure), so a start further
away than that converges to a cycle-skipped minimum -- one where the predicted
arrival is a whole period out and the residual is locally minimal but globally wrong.
Every stage exists to hand the next one a starting point inside its basin.

Two implementation choices worth defending:

**The 16 survivors are optimised as one batch, not in a loop.**  They share the
incident field, the observation and the network, so a batch of 16 candidate
geometries is one forward pass of batch 16 x F instead of 16 passes of F.  The
objective is summed across candidates before `backward()`; since the candidates are
independent, row i's gradient depends only on row i, so the sum is exactly the same
as 16 separate backward passes and about an order of magnitude cheaper.

**eps is annealed by restarting L-BFGS, not by changing it mid-run.**  L-BFGS
approximates curvature from differences of gradients taken at different iterates; if
the objective changes underneath it, those differences describe two different
functions and the approximation is not merely stale but wrong -- it will confidently
step in a direction that was never a descent direction.  So Stage 3 is three short
L-BFGS runs at eps = 2.0, 1.5, 1.0 cells, each with a fresh history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .. import config as cfg
from ..geometry.sdf import Circle, ShapeFamily
from .misfit import InverseCase, Objective, SurrogateForward, tikhonov


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class InversionResult:
    theta: Tensor                       # [P] final estimate, physical
    misfit: float                       # final full-band complex misfit
    stages: dict = field(default_factory=dict)
    theta_true: Tensor | None = None
    lambda_s: float = 1.0
    seconds: float = 0.0
    n_forward: int = 0

    # -- scoring ----------------------------------------------------------
    @property
    def position_error_ls(self) -> float | None:
        if self.theta_true is None:
            return None
        d = (self.theta[:2] - self.theta_true[:2]).pow(2).sum().sqrt()
        return float(d) / self.lambda_s

    @property
    def radius_error_ls(self) -> float | None:
        if self.theta_true is None:
            return None
        return float((self.theta[2] - self.theta_true[2]).abs()) / self.lambda_s

    @property
    def success(self) -> bool:
        e = self.position_error_ls
        return e is not None and e < cfg.GATE_POSITION_LS

    def summary(self) -> str:
        rows = [f"theta = {[round(float(v), 4) for v in self.theta]}",
                f"final misfit {self.misfit:.4e}   "
                f"{self.n_forward} forward evaluations   {self.seconds:.1f} s"]
        if self.theta_true is not None:
            rows.append(
                f"position error {self.position_error_ls:.4f} lambda_s   "
                f"radius error {self.radius_error_ls:.4f} lambda_s   "
                f"{'PASS' if self.success else 'FAIL'} "
                f"(gate < {cfg.GATE_POSITION_LS})")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Stage 1: screening
# ---------------------------------------------------------------------------
@torch.no_grad()
def screen(forward: SurrogateForward, case: InverseCase, family: ShapeFamily, *,
           n_grid: int = cfg.SCREEN_GRID, n_keep: int = cfg.N_SURVIVORS,
           chunk: int = cfg.SCREEN_CHUNK, radius: float | None = None
           ) -> tuple[Tensor, Tensor]:
    """
    Amplitude-misfit screen over an n_grid x n_grid position grid.  (theta, J)

    The grid spans the feasible box, so its spacing is about
    (L - 2 x 1.5 lambda_s) / 16 ~ 0.3 lambda_s.  That is slightly coarser than the
    lambda_s/4 basin, which is why 16 survivors are kept rather than one: the true
    optimum may sit between grid nodes, and the nearest few nodes are then all
    mediocre and nearly tied.  Keeping one would be a coin flip; keeping 16 makes it
    a near-certainty that at least one lands in the basin.
    """
    lam_s = case.lambda_s
    lo, hi = family.bounds(lam_s)
    dev = forward.device
    if radius is None:
        radius = 0.5 * (cfg.R_MIN_LS + cfg.R_MAX_LS) * lam_s

    xs = torch.linspace(float(lo[0]), float(hi[0]), n_grid)
    ys = torch.linspace(float(lo[1]), float(hi[1]), n_grid)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    theta = torch.stack([gx.reshape(-1), gy.reshape(-1),
                         torch.full((n_grid * n_grid,), radius)], dim=-1).to(dev)

    obj = Objective(forward, case, family, band=cfg.BAND_STAGE1,
                    eps_cells=cfg.EPS_INVERT_START, amplitude=True,
                    scale_invariant=True)
    j = torch.cat([obj.residual(theta[i:i + chunk]).cpu()
                   for i in range(0, theta.shape[0], chunk)])
    order = torch.argsort(j)[:n_keep]
    return theta[order.to(dev)], j[order]


# ---------------------------------------------------------------------------
# Stage 2: Adam on the survivors
# ---------------------------------------------------------------------------
def refine_adam(forward: SurrogateForward, case: InverseCase, family: ShapeFamily,
                theta0: Tensor, *, steps: int = cfg.ADAM_STEPS_STAGE2,
                lr: float = cfg.ADAM_LR_STAGE2, band: slice = cfg.BAND_STAGE2,
                eps_cells: float = cfg.EPS_INVERT_START,
                log_every: int = 0) -> tuple[Tensor, Tensor, list[float]]:
    """
    Adam on all candidates at once.  Returns (theta, J, trace of mean J).

    Adam rather than L-BFGS here because the iterate is still far from the optimum
    and the objective is not yet locally quadratic; a curvature model fitted to a
    non-quadratic region is worse than no curvature model.  lr = 5e-2 is in the
    unconstrained coordinates, where the feasible range of every parameter is O(1),
    so one step moves a position by at most a few percent of the domain.
    """
    lam_s = case.lambda_s
    z = family.to_unconstrained(theta0, lam_s).clone().requires_grad_(True)
    obj = Objective(forward, case, family, band=band, eps_cells=eps_cells)
    opt = torch.optim.Adam([z], lr=lr)
    trace: list[float] = []
    for it in range(steps):
        j = obj.of_z(z)
        loss = j.sum()          # candidates are independent; see module docstring
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append(float(j.mean()))
        if log_every and it % log_every == 0:
            print(f"    adam {it:4d}  mean J {trace[-1]:.4e}  "
                  f"best {float(j.min()):.4e}")
    with torch.no_grad():
        theta = family.to_physical(z.detach(), lam_s)
        j = obj.residual(theta)
    return theta, j, trace


# ---------------------------------------------------------------------------
# Stage 3: L-BFGS with an interface-width anneal
# ---------------------------------------------------------------------------
def refine_lbfgs(forward: SurrogateForward, case: InverseCase,
                 family: ShapeFamily, theta0: Tensor, *,
                 steps: int = cfg.LBFGS_STEPS_STAGE3,
                 band: slice = cfg.BAND_STAGE3,
                 eps_schedule: tuple[float, ...] | None = None,
                 mu: float = cfg.TIKHONOV_MU,
                 log: bool = False) -> tuple[Tensor, float, list[float]]:
    """
    Short L-BFGS runs at successively sharper interfaces.  (theta, J, trace)

    theta0 is a single candidate, [1, P].  Strong-Wolfe line search is on: without a
    line search L-BFGS can take a step that overshoots into a region where the
    surrogate has never been evaluated (a void overlapping the domain edge, say) and
    the returned "descent" direction is then based on a garbage gradient.
    """
    if eps_schedule is None:
        a, b = cfg.EPS_INVERT_START, cfg.EPS_INVERT_END
        eps_schedule = (a, 0.5 * (a + b), b)
    lam_s = case.lambda_s
    z = family.to_unconstrained(theta0, lam_s).clone()
    trace: list[float] = []
    per = max(1, steps // len(eps_schedule))

    for eps in eps_schedule:
        z = z.detach().clone().requires_grad_(True)
        obj = Objective(forward, case, family, band=band, eps_cells=eps)
        opt = torch.optim.LBFGS([z], max_iter=per, history_size=10,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = (obj.residual(family.to_physical(z, lam_s)) + tikhonov(z, mu)).sum()
            loss.backward()
            trace.append(float(loss))
            return loss

        opt.step(closure)
        if log:
            print(f"    lbfgs eps={eps:.2f} cells  J {trace[-1]:.4e}")

    with torch.no_grad():
        theta = family.to_physical(z.detach(), lam_s)
        obj = Objective(forward, case, family, band=band,
                        eps_cells=eps_schedule[-1])
        j = float(obj.residual(theta)[0])
    return theta, j, trace


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def invert(forward: SurrogateForward, case: InverseCase, *,
           family: ShapeFamily | None = None, theta_init: Tensor | None = None,
           n_keep: int = cfg.N_SURVIVORS, skip_screen: bool = False,
           log: bool = False) -> InversionResult:
    """
    Run stages 0-3 and return the estimate with its per-stage trace.

    `theta_init` is the Stage-0 CNN guess, or None.  It is *added to* the screen's
    survivors rather than replacing them: if the CNN is right the extra candidate
    costs one row in a batch of 17, and if the defect is out of distribution -- which
    is the case the whole thesis is about -- the CNN's guess can be badly wrong and
    must not be the only starting point.  `skip_screen=True` trusts it alone, which
    is only sensible for timing comparisons.
    """
    family = family or Circle()
    t0 = time.perf_counter()
    stages: dict = {}
    lam_s = case.lambda_s
    dev = forward.device
    n_fwd = 0

    # -- stages 0 and 1 ---------------------------------------------------
    cands: list[Tensor] = []
    if theta_init is not None:
        cands.append(theta_init.reshape(1, -1).to(dev))
        stages["stage0_theta"] = theta_init.reshape(-1).tolist()
    if not (skip_screen and cands):
        th, j = screen(forward, case, family, n_keep=n_keep)
        n_fwd += cfg.SCREEN_GRID ** 2
        cands.append(th)
        stages["stage1_best_J"] = float(j[0])
        stages["stage1_J"] = j.tolist()
    theta = torch.cat(cands, dim=0)
    if log:
        print(f"  stage 1: {theta.shape[0]} candidates, "
              f"best J {stages.get('stage1_best_J', float('nan')):.4e}")

    # -- stage 2 ----------------------------------------------------------
    theta, j, trace2 = refine_adam(forward, case, family, theta,
                                   log_every=50 if log else 0)
    n_fwd += cfg.ADAM_STEPS_STAGE2 * theta.shape[0]
    k = int(j.argmin())
    stages["stage2_trace"] = trace2
    stages["stage2_J"] = float(j[k])
    stages["stage2_theta"] = theta[k].tolist()
    if log:
        print(f"  stage 2: J {float(j[k]):.4e} at "
              f"{[round(float(v), 3) for v in theta[k]]}")

    # -- stage 3 ----------------------------------------------------------
    theta_f, j_f, trace3 = refine_lbfgs(forward, case, family,
                                        theta[k:k + 1], log=log)
    n_fwd += len(trace3)
    stages["stage3_trace"] = trace3
    stages["stage3_J"] = j_f

    res = InversionResult(theta=theta_f[0].detach().cpu(), misfit=j_f,
                          stages=stages,
                          theta_true=(None if case.theta_true is None
                                      else case.theta_true.detach().cpu()),
                          lambda_s=lam_s,
                          seconds=time.perf_counter() - t0, n_forward=n_fwd)
    if log:
        print(res.summary())
    return res


# ---------------------------------------------------------------------------
# Batch evaluation, for the §11.2 step 11 and step 12 statistics
# ---------------------------------------------------------------------------
def run_many(forward: SurrogateForward, cases: list[InverseCase], *,
             family: ShapeFamily | None = None,
             theta_inits: Tensor | None = None,
             progress=None) -> list[InversionResult]:
    """Invert a list of cases sequentially, returning every result."""
    it = range(len(cases))
    if progress is not None:
        it = progress(it)
    out = []
    for i in it:
        ti = None if theta_inits is None else theta_inits[i]
        out.append(invert(forward, cases[i], family=family, theta_init=ti))
    return out


def summarise(results: list[InversionResult]) -> dict:
    """
    Success rate and error statistics, plus the §11.2 step 11 gate.

    The *median* position error is reported alongside the mean because the failure
    mode here is bimodal, not heavy-tailed-continuous: an inversion either lands in
    the right basin (error ~ lambda_s/20) or cycle-skips into a neighbouring one
    (error ~ lambda_s/2 or worse).  A mean over a bimodal distribution describes
    neither mode.
    """
    pos = torch.tensor([r.position_error_ls for r in results
                        if r.position_error_ls is not None])
    rad = torch.tensor([r.radius_error_ls for r in results
                        if r.radius_error_ls is not None])
    ok = torch.tensor([float(r.success) for r in results])
    mis = torch.tensor([r.misfit for r in results])
    rate = float(ok.mean()) if len(ok) else float("nan")
    return dict(
        n=len(results),
        success_rate=rate,
        gate_pass=bool(rate >= cfg.GATE_SUCCESS_RATE),
        position_ls_mean=float(pos.mean()) if len(pos) else float("nan"),
        position_ls_median=float(pos.median()) if len(pos) else float("nan"),
        position_ls_p90=float(pos.quantile(0.9)) if len(pos) else float("nan"),
        radius_ls_median=float(rad.median()) if len(rad) else float("nan"),
        misfit_median=float(mis.median()) if len(mis) else float("nan"),
        seconds_mean=float(sum(r.seconds for r in results) / max(len(results), 1)),
    )


# ---------------------------------------------------------------------------
# Model-mismatch detector (§11.2 step 12)
# ---------------------------------------------------------------------------
def detector_roc(misfits_in: list[float], misfits_out: list[float],
                 n_thresh: int = 200) -> dict:
    """
    ROC for "is this data explained by the family I inverted with?".

    The statistic is the *final* relative misfit.  The logic: after convergence, a
    circle fitted to a circle's data leaves only surrogate error and measurement
    noise, while a circle fitted to an ellipse's or a two-void's data leaves
    structured residual it cannot represent.  The residual is therefore a model-
    mismatch detector for free, which matters because a defect-localisation tool that
    silently reports a confident wrong answer on an unmodelled defect is worse than
    one that says it does not know.

    `misfits_in` come from in-family cases, `misfits_out` from out-of-family ones.
    Returns the ROC points and the AUC, computed by the rank identity rather than by
    trapezoidal integration so it is exact at the sample size involved.
    """
    a = torch.tensor(misfits_in, dtype=torch.float64)
    b = torch.tensor(misfits_out, dtype=torch.float64)
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    th = torch.linspace(lo, hi, n_thresh, dtype=torch.float64)
    tpr = [(b > t).double().mean().item() for t in th]      # out-of-family flagged
    fpr = [(a > t).double().mean().item() for t in th]
    # AUC = P(misfit_out > misfit_in), ties counted as half
    diff = b.view(-1, 1) - a.view(1, -1)
    auc = float((diff > 0).double().mean() + 0.5 * (diff == 0).double().mean())
    return dict(threshold=th.tolist(), tpr=tpr, fpr=fpr, auc=auc,
                median_in=float(a.median()), median_out=float(b.median()))


__all__ = [
    "InversionResult",
    "detector_roc",
    "invert",
    "refine_adam",
    "refine_lbfgs",
    "run_many",
    "screen",
    "summarise",
]
