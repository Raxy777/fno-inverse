"""
Training loop (§7.5, §11.2 step 7).

Deliberate omissions, each for a reason:

**No mixed precision.**  The spectral convolution is the bulk of the compute and it
runs in complex arithmetic; complex half precision is not well supported and, more to
the point, the phase of the predicted phasor is the quantity the inversion depends
on.  float16 carries about three decimal digits, and a phase error of 1e-3 rad is
already 1.6e-4 of a period -- fine on its own, but the gradient check of §11.2 step 9
asserts agreement to three significant figures and would fail for reasons that have
nothing to do with correctness.  bf16 would be worse still.

**No learning-rate warmup.**  The blocks start close to the identity (see
`fno2d.FourierBlock`), so there is no early instability for warmup to protect
against, and adding it would be one more knob nobody tuned.

**Cosine decay to LR_FINAL, not to zero.**  The last epochs at a nonzero rate keep
refining the high-|k| part of the spectrum, which is where the error concentrates
(§11.3's error-vs-|k| figure) and which is also the part that decays last.

Reported metrics are the ones the gates are stated in, not proxies for them:
relative L2 on the field, relative L2 restricted to the receiver ring, and
arrival-time error expressed in periods.  The last is measured as receiver *phase*
error divided by 2 pi rather than by synthesising a time trace from 20 phasors: the
band is 0.66-1.34 f_c, so a synthesised trace has a time resolution of roughly
1/(0.68 f_c) = 1.5 periods and could not resolve a 0.05-period error even in
principle.  Phase error at the measured frequencies is the same quantity without the
synthesis, and it is what cycle skipping (§8.3) actually responds to.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from . import config as cfg
from . import features as feat
from . import losses as L
from .data.dataset import WaveDataset, batch_to_model, make_loader, to_device
from .data.generate import config_snapshot
from .models.fno2d import FNO2d, build


def receivers_tensor(device=None) -> Tensor:
    return torch.tensor(cfg.RECEIVERS_NET, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ring_rel_l2(pred: Tensor, target: Tensor, recv: Tensor, *,
                eps: float = 1e-12) -> Tensor:
    """Relative L2 over all 32 receivers -- the deterministic version of L_meas."""
    ry, rx = recv[:, 0], recv[:, 1]
    p, t = pred[..., ry, rx], target[..., ry, rx]
    dims = tuple(range(1, p.dim()))
    return ((p - t).pow(2).sum(dims).sqrt()
            / t.pow(2).sum(dims).sqrt().clamp_min(eps)).mean()


def per_sample_relative_errors(pred: Tensor, target: Tensor, recv: Tensor, *,
                               eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """
    Field and receiver-ring relative L2 for a batch of frequency-folded predictions.

    `pred` and `target` are `[B, F, C, ny, nx]`; the returned tensors are `[B]`.
    Keeping this reduction here, rather than duplicating it in a notebook, prevents a
    stale dimension tuple from silently turning a per-sample metric into a scalar.
    """
    dims = (1, 2, 3)
    field = ((pred - target).pow(2).sum(dims).sqrt()
             / target.pow(2).sum(dims).sqrt().clamp_min(eps))
    ry, rx = recv[:, 0], recv[:, 1]
    p_ring, t_ring = pred[..., ry, rx], target[..., ry, rx]
    ring = ((p_ring - t_ring).pow(2).sum(dims).sqrt()
            / t_ring.pow(2).sum(dims).sqrt().clamp_min(eps))
    return field, ring


def phase_error_periods(pred: Tensor, target: Tensor, recv: Tensor, *,
                        amp_floor: float = 0.05) -> Tensor:
    """
    Receiver phase error in periods: |arg(pred) - arg(true)| / (2 pi), wrapped.

    Receivers whose true scattered amplitude is below `amp_floor` of that row's
    maximum are excluded.  Phase is meaningless where there is no signal, and a
    shadowed receiver would otherwise contribute a uniformly distributed phase error
    of 0.25 periods on average and swamp the statistic.  Reporting the phase error
    over *illuminated* receivers is the number the cycle-skipping argument is about.
    """
    ry, rx = recv[:, 0], recv[:, 1]
    zp = feat.channels_to_complex(pred)[..., ry, rx]        # [N, 2, R]
    zt = feat.channels_to_complex(target)[..., ry, rx]
    amp = zt.abs()
    keep = amp >= amp_floor * amp.amax(dim=(1, 2), keepdim=True)
    d = (zp * zt.conj()).angle().abs() / (2.0 * math.pi)
    if keep.sum() == 0:
        return torch.zeros((), device=pred.device)
    return d[keep].mean()


@dataclass
class EvalResult:
    rel_l2: float
    ring_rel_l2: float
    phase_periods: float
    per_freq: list[float] = field(default_factory=list)

    def gates(self) -> dict[str, bool]:
        return {
            "rel_l2 < 5%": self.rel_l2 < cfg.GATE_REL_L2,
            "arrival < 0.05 periods": self.phase_periods < cfg.GATE_ARRIVAL_PERIODS,
        }

    def __str__(self) -> str:
        g = self.gates()
        rows = [f"rel-L2 field   {self.rel_l2:8.4f}",
                f"rel-L2 ring    {self.ring_rel_l2:8.4f}",
                f"phase error    {self.phase_periods:8.4f} periods"]
        rows += [f"  {'PASS' if v else 'FAIL'}  {k}" for k, v in g.items()]
        return "\n".join(rows)


@torch.no_grad()
def evaluate(model: FNO2d, loader, device, *, per_freq: bool = True) -> EvalResult:
    model.eval()
    recv = receivers_tensor(device)
    n = 0
    acc = torch.zeros(3, device=device)
    pf_num = torch.zeros(cfg.M_FREQ, device=device)
    pf_den = torch.zeros(cfg.M_FREQ, device=device)
    for batch in loader:
        batch = to_device(batch, device)
        x, y = batch_to_model(batch)
        p = model(x)
        b = x.shape[0]
        acc[0] += L.rel_l2(p, y) * b
        acc[1] += ring_rel_l2(p, y, recv) * b
        acc[2] += phase_error_periods(p, y, recv) * b
        n += b
        if per_freq:
            F_ = batch["freqs"].shape[1]
            pr = feat.unflatten_freq(p, F_)
            yr = feat.unflatten_freq(y, F_)
            dims = (2, 3, 4)
            # ratio of summed norms rather than mean of per-sample ratios: the point
            # of this breakdown is which frequencies the error lives at, and a mean
            # of ratios would let low-amplitude samples dominate a band they barely
            # contribute energy to
            num = (pr - yr).pow(2).sum(dims).sqrt().reshape(-1)
            den = yr.pow(2).sum(dims).sqrt().reshape(-1)
            # eval loaders use every frequency in order, so column j is freq j
            idx = torch.arange(F_, device=device).repeat(pr.shape[0])
            pf_num.index_add_(0, idx, num)
            pf_den.index_add_(0, idx, den)
    acc /= max(n, 1)
    pf = (pf_num / pf_den.clamp_min(1e-12)).tolist() if per_freq else []
    return EvalResult(rel_l2=float(acc[0]), ring_rel_l2=float(acc[1]),
                      phase_periods=float(acc[2]), per_freq=pf)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model: FNO2d, train_path: str, val_path: str, *, out_dir: str,
          device=None, epochs: int = cfg.EPOCHS, batch_size: int = cfg.BATCH_SIZE,
          lr: float = cfg.LR, lr_final: float = cfg.LR_FINAL,
          weight_decay: float = cfg.WEIGHT_DECAY, grad_clip: float = cfg.GRAD_CLIP,
          gamma: float = cfg.GAMMA_H1, beta: float = cfg.BETA_MEAS,
          alpha: float | None = cfg.ALPHA_PHYS, balance_every: int = 200,
          n_meas: int = 8, num_workers: int = 4, seed: int = cfg.SEED,
          log_every: int = 50, progress=None, resume_from: str | None = None) -> dict:
    """
    Train and checkpoint.  `alpha=None` disables the physics term entirely, which is
    the ablation §11.2 step 8 asks for -- the same code path, one flag, so the two
    arms of the ablation cannot differ in anything else.

    `resume_from` warm-starts from a checkpoint written by `save`, and it is a
    *warm-start*, not a bit-exact resume, on purpose: the checkpoint carries the
    weights and the balanced `alpha` but not AdamW's moment estimates, the scheduler
    position, or the RNG -- §7.5 treats the cosine schedule as part of the method, so
    there was never a state to save.  What continues: the loop picks up at the
    checkpoint's `epoch + 1`, `history.json` in `out_dir` is extended rather than
    overwritten (so the curves and the best-so-far survive), and a fresh cosine runs
    from the last logged LR down to `lr_final` over the remaining epochs.  What
    restarts from zero: Adam's momentum.  Use it to rescue a run that died mid-way,
    not to reproduce one.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    tr = WaveDataset(train_path, train=True, seed=seed)
    va = WaveDataset(val_path, train=False, seed=seed)
    tl = make_loader(tr, batch_size=batch_size, num_workers=num_workers)
    vl = make_loader(va, batch_size=max(1, batch_size // 4), shuffle=False,
                     num_workers=num_workers // 2)     # 0 stays 0 (in-process)

    model = model.to(device)

    hist: dict = dict(train=[], val=[], alpha=[], lr=[], config=config_snapshot())
    best = math.inf
    start_epoch = 0
    lr_start = lr
    if resume_from is not None:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        if alpha is not None and ck.get("alpha") is not None:
            alpha = float(ck["alpha"])                 # continue the balanced value
        hp = out / "history.json"
        if hp.exists():
            hist = json.loads(hp.read_text())
            for k in ("train", "val", "alpha", "lr"):
                hist.setdefault(k, [])
            if hist["val"]:
                best = min(r["rel_l2"] for r in hist["val"])   # don't clobber best.pt
            if hist["lr"]:
                lr_start = float(hist["lr"][-1])         # fresh cosine from where LR was
        if start_epoch >= epochs:
            raise ValueError(
                f"resume_from is at epoch {start_epoch} but epochs={epochs}; "
                f"nothing to train -- raise epochs")
        print(f"warm-start from {resume_from}: resuming at epoch {start_epoch}/{epochs}, "
              f"alpha {alpha}, lr {lr_start:.2e}, best-so-far rel-L2 {best:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay)
    # A fresh cosine over the *remaining* epochs (lr_start -> lr_final).  Adam's
    # moments restart regardless; see the resume note in the docstring.
    remaining = max(1, (epochs - start_epoch) * max(1, len(tl)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=remaining, eta_min=lr_final)
    recv = receivers_tensor(device)
    gen = torch.Generator().manual_seed(seed)
    # the two 1x1 convolutions of the projection head: the last point where the data
    # and physics gradients are still the same kind of quantity (see balance_alpha)
    balance_params = [p for p in model.project.parameters() if p.requires_grad]

    step = start_epoch * max(1, len(tl))
    t0 = time.perf_counter()

    for ep in range(start_epoch, epochs):
        model.train()
        tr.set_epoch(ep)
        run = dict(total=0.0, field=0.0, h1=0.0, meas=0.0, phys=0.0, n=0)
        it = progress(tl, leave=False) if progress is not None else tl
        for batch in it:
            batch = to_device(batch, device)
            x, y = batch_to_model(batch)
            pred = model(x)

            ctx = u_inc = None
            if alpha is not None:
                ctx = L.make_context(batch["chi"], batch["nu"], batch["freqs"],
                                     batch["src_idx"])
                u_inc = feat.flatten_freq(batch["u_inc"])
            terms = L.compute(pred, y, recv_yx=recv, ctx=ctx, u_inc=u_inc,
                              alpha=alpha or 0.0, beta=beta, gamma=gamma,
                              n_meas=n_meas, generator=gen)

            opt.zero_grad(set_to_none=True)
            due = alpha is not None and step % balance_every == 0
            terms.total.backward(retain_graph=due)
            # Before opt.step(), deliberately: balance_alpha re-differentiates the
            # retained graph, whose saved activations belong to the *current*
            # parameters.  Doing it after the step would measure the ratio of
            # gradients at a point the network is no longer at.
            if due:
                l_data = terms.field + gamma * terms.h1 + beta * terms.meas
                alpha = L.balance_alpha(l_data, terms.phys, balance_params,
                                        alpha_prev=alpha)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

            b = x.shape[0]
            # detach: these are logging accumulators, and terms.total still carries
            # the retained graph on balance steps -- float() on a requires_grad tensor
            # warns and pins the graph alive one iteration longer than intended.
            for k in ("total", "field", "h1", "meas", "phys"):
                run[k] += float(getattr(terms, k).detach()) * b
            run["n"] += b
            step += 1
            if log_every and step % log_every == 0:
                print(f"  ep {ep:3d} step {step:6d}  "
                      f"loss {float(terms.total.detach()):.4f}  "
                      f"field {float(terms.field.detach()):.4f}  "
                      f"phys {float(terms.phys.detach()):.4f}  alpha {terms.alpha:.2e}")

        n = max(run.pop("n"), 1)
        row = {k: v / n for k, v in run.items()}
        ev = evaluate(model, vl, device, per_freq=(ep % 10 == 0 or ep == epochs - 1))
        hist["train"].append(row)
        hist["val"].append(dict(rel_l2=ev.rel_l2, ring=ev.ring_rel_l2,
                                phase=ev.phase_periods, per_freq=ev.per_freq))
        hist["alpha"].append(float(alpha or 0.0))
        hist["lr"].append(sched.get_last_lr()[0])
        print(f"epoch {ep:3d}  train {row['total']:.4f}  "
              f"val rel-L2 {ev.rel_l2:.4f}  ring {ev.ring_rel_l2:.4f}  "
              f"phase {ev.phase_periods:.4f}  "
              f"[{(time.perf_counter()-t0)/60:.1f} min]")

        if ev.rel_l2 < best:
            best = ev.rel_l2
            save(model, out / "best.pt", epoch=ep, val=ev, alpha=alpha)
        save(model, out / "last.pt", epoch=ep, val=ev, alpha=alpha)
        (out / "history.json").write_text(json.dumps(hist, indent=1))

    # evaluate before closing: `vl`'s workers hold a copy of `va` and re-open the
    # file lazily, but the parent handle is what a num_workers=0 loader reads through
    final = evaluate(model, vl, device)
    tr.close()
    va.close()
    print("\nfinal gates")
    for k, v in final.gates().items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    hist["final"] = dict(rel_l2=final.rel_l2, ring=final.ring_rel_l2,
                         phase=final.phase_periods, per_freq=final.per_freq,
                         gates=final.gates())
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    return hist


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save(model: FNO2d, path, *, epoch: int, val: EvalResult | None = None,
         alpha: float | None = None) -> None:
    """
    The architecture hyperparameters travel with the weights.

    A state_dict alone is not enough to reconstruct this model: d_v and kmax
    determine the shape of every spectral weight, and loading a 'primary' checkpoint
    into a 'small' model raises a shape error, but loading a checkpoint trained with
    radial=False into radial=True does *not* -- the shapes match and the operator is
    silently different.  Storing the arguments removes that failure mode.
    """
    torch.save(dict(
        state_dict=model.state_dict(),
        arch=dict(c_in=model.c_in, c_out=model.c_out, d_v=model.d_v,
                  n_blocks=len(model.blocks), kmax=model.kmax,
                  radial=model.blocks[0].spectral.radial),
        epoch=epoch, alpha=alpha,
        val=None if val is None else dict(rel_l2=val.rel_l2,
                                          ring=val.ring_rel_l2,
                                          phase=val.phase_periods),
    ), path)


def load(path, device=None) -> tuple[FNO2d, dict]:
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = FNO2d(**ck["arch"]).to(device or "cpu")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


__all__ = ["EvalResult", "evaluate", "load", "per_sample_relative_errors",
           "phase_error_periods", "receivers_tensor", "ring_rel_l2", "save", "train"]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="train the FNO surrogate")
    ap.add_argument("--data", required=True, help="directory holding train/val .h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="primary", choices=list(cfg.VARIANTS))
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--no-physics", action="store_true",
                    help="ablation: drop L_phys (§11.2 step 8)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", default=None,
                    help="warm-start from this checkpoint (e.g. .../last.pt)")
    a = ap.parse_args()

    model = build(a.variant)
    print(model.summary())
    try:
        from tqdm.auto import tqdm as prog
    except ImportError:
        prog = None
    train(model, f"{a.data}/train.h5", f"{a.data}/val.h5", out_dir=a.out,
          epochs=a.epochs, alpha=None if a.no_physics else cfg.ALPHA_PHYS,
          num_workers=a.workers, progress=prog, resume_from=a.resume)


if __name__ == "__main__":
    main()
