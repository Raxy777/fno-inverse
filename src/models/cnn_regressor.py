"""
Direct regressor from receiver data to defect parameters (§9.2, §11.2 step 13).

This network plays two roles, and it is worth being clear that they pull in opposite
directions.

**As Stage 0 of the inversion** it supplies a starting guess, which saves the
screening stage most of its work when the defect is in-distribution.

**As the antagonist** it is the baseline the thesis is stated against.  The claim
(T0) is about adaptability: the operator + inversion pipeline keeps the shape family
*outside* the network, so an ellipse or a pair of voids can be inverted for without
retraining, while a direct regressor has the family baked into its output layer --
three numbers, xc, yc, R -- and cannot represent an ellipse at all, let alone fit one.
On in-distribution circles it is expected to be competitive or better, and saying so
plainly is the honest version of the comparison.  The interesting number is the gap
that opens on the transfer experiment of §8.5, not the in-distribution one.

Design notes:

**A circular 1-D convolution along the receiver ring.**  The 32 receivers form a
closed loop, so `padding_mode='circular'` is the geometrically correct choice; a zero
pad would tell the network there is an edge between receiver 31 and receiver 0, which
there is not, and the artefact would sit at a fixed place on the ring and be learned
as a feature of the domain.

**Conditioned on source position, not source index.**  A one-hot over the 8 sources
would be undefined for the held-out sources (`cfg.SRC_HELDOUT`), so the held-out-source
test of §11.2 step 11 could not even be run on the baseline -- which would make the
comparison unfair in the pipeline's favour.  Feeding the source's normalised
coordinates instead means a new source position is representable, so the baseline gets
a real chance and the comparison means something.

**Predicts the unconstrained parameters z, not theta.**  Same reparameterisation as
the inversion (§8.1), so its output is feasible by construction and can be handed to
Stage 2 without conversion, and so that both estimators are measured in the same
coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .. import config as cfg
from .. import features as feat
from ..geometry.sdf import Circle, ShapeFamily

# 4 real numbers per (component, Re/Im) per frequency, plus 3 conditioning channels
RING_CHANNELS: int = 4 * cfg.M_FREQ + 3


def pack_ring(d_obs: Tensor, *, src_idx: Tensor, nu: Tensor,
              scale: Tensor | None = None) -> Tensor:
    """
    [B, R, 2, M] complex receiver data -> [B, RING_CHANNELS, R] real.

    `scale` is the *receiver-space* incident amplitude [B, 1, 1, M] (or [B, M]);
    dividing by it is what makes the regressor's input the same normalised quantity
    the inversion's misfit works with.  The field-space scale must not be used here:
    it is set by the near-source singularity and is roughly two orders of magnitude
    larger than anything on the ring, so it would compress the entire input to the
    bottom of float32's useful range.
    """
    assert d_obs.is_complex(), "receiver data must be complex phasors"
    B, R, C, M = d_obs.shape
    assert C == 2, f"expected 2 components, got {C}"
    z = d_obs
    if scale is not None:
        z = z / scale.reshape(B, 1, 1, -1).clamp_min(1e-30)
    # [B, R, 2, M] -> [B, 4M, R], channel index c*M + m: all frequencies of Re_x,
    # then Im_x, then Re_y, then Im_y.  Grouping by component rather than by
    # frequency means a kernel sees one component's whole spectrum as a contiguous
    # block, which is the axis the physics is smooth along.
    parts = torch.stack([z[..., 0, :].real, z[..., 0, :].imag,
                         z[..., 1, :].real, z[..., 1, :].imag], dim=2)  # [B,R,4,M]
    x = parts.permute(0, 2, 3, 1).reshape(B, 4 * M, R).contiguous()

    sxy = torch.tensor([cfg.SOURCE_XY[int(i)] for i in src_idx],
                       device=d_obs.device, dtype=torch.float32)        # [B, 2]
    sxy = 2.0 * sxy / cfg.L_DOMAIN - 1.0
    nuc = torch.as_tensor(feat.nu_centred(nu), device=d_obs.device,
                          dtype=torch.float32).reshape(B, 1)
    cond = torch.cat([sxy, nuc], dim=1).unsqueeze(-1).expand(B, 3, R)
    return torch.cat([x.float(), cond], dim=1)


class RingCNN(nn.Module):
    """
    Circular 1-D CNN over the receiver ring -> 3 unconstrained parameters.

    Global average *and* max pooling are concatenated before the head.  The average
    carries the total scattered energy, which is the radius information; the max
    carries which part of the ring is brightest, which is the direction information.
    Average pooling alone measurably loses the second, and the two together cost 64
    extra numbers.
    """

    def __init__(self, c_in: int = RING_CHANNELS, width: int = 96,
                 n_out: int = 3, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        c = c_in
        for i in range(depth):
            layers += [nn.Conv1d(c, width, kernel_size=5, padding=2,
                                 padding_mode="circular"),
                       nn.GroupNorm(8, width),
                       nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * width, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_out))
        self.n_out = n_out

    def forward(self, x: Tensor) -> Tensor:
        h = self.body(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.head(pooled)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def predict_theta(self, x: Tensor, nu: Tensor,
                      family: ShapeFamily | None = None) -> Tensor:
        """
        [B, C, R] -> theta [B, n_out] physical, in `family`'s parameterisation.

        lambda_s depends on nu, so the bounds do too, and the mapping is done per
        distinct nu rather than with a single box.  Using one nu's box for all of
        them would put a systematic radius bias of up to
        (lambda_s(0.25)/lambda_s(0.37) - 1) ~ 13% into the estimate.
        """
        family = family or Circle()
        assert self.n_out == len(family.param_names), (
            f"head emits {self.n_out} numbers, {family.name!r} takes "
            f"{len(family.param_names)}; retrain with n_out=")
        z = self(x)
        out = torch.empty_like(z)
        for v in torch.unique(nu):
            m = nu == v
            lam_s = cfg.cs_over_cp(float(v)) / cfg.FC
            out[m] = family.to_physical(z[m], lam_s)
        return out


# ---------------------------------------------------------------------------
# Features straight from an HDF5 split
# ---------------------------------------------------------------------------
@dataclass
class RingData:
    x: Tensor          # [N, C, R] float32, ready for the network
    theta: Tensor      # [N, P] physical truth, P set by the *truth's* family
    nu: Tensor         # [N]
    src_idx: Tensor    # [N]

    def __len__(self) -> int:
        return self.x.shape[0]

    def to(self, device) -> "RingData":
        return RingData(self.x.to(device), self.theta.to(device),
                        self.nu.to(device), self.src_idx.to(device))

    def z(self, family: ShapeFamily | None = None) -> Tensor:
        """
        Targets in unconstrained coordinates, per-sample lambda_s.

        Training only, and therefore circle-only in practice.  The assertion is here
        because `theta` may hold an ellipse or two-void truth for the transfer
        experiment, and `to_unconstrained` would happily read the first three columns of
        one and return silently wrong targets.
        """
        family = family or Circle()
        assert self.theta.shape[1] == len(family.param_names), (
            f"theta has {self.theta.shape[1]} parameters, {family.name!r} takes "
            f"{len(family.param_names)}")
        out = torch.empty_like(self.theta)
        for v in torch.unique(self.nu):
            m = self.nu == v
            lam_s = cfg.cs_over_cp(float(v)) / cfg.FC
            out[m] = family.to_unconstrained(self.theta[m], lam_s)
        return out


def ring_features(path: str, *, snr_db: float | None = None, chunk: int = 64,
                  generator: torch.Generator | None = None,
                  device=None) -> RingData:
    """
    Read a split's A-scans and reduce them to ring features, once.

    The A-scans are the bulk of the file (0.36 MB per sample) and the reduction to
    [4M + 3, 32] is a factor of ~350, so this is done once up front and held in
    memory rather than per epoch in a DataLoader: 2000 samples come to about 21 MB.
    Doing the DFT inside `__getitem__` instead would spend a 1408-point transform per
    receiver per sample per epoch on a quantity that never changes.

    Noise, if requested, is added to the *total* A-scans before the incident field is
    subtracted, exactly as in `data.dataset.load_inversion_case`, so the baseline is
    trained and tested on the same noise model as the inversion.
    """
    from ..solver import harmonic as H

    om = H.omegas_tensor()
    xs, thetas, nus, srcs = [], [], [], []
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_samples"])
        theta_all = f["samples/theta"][:]
        src_all = f["samples/src_idx"][:]
        nu_all = f["samples/nu_idx"][:]
        inc_a = torch.from_numpy(f["incident/ascans"][:])       # [S,NU,R,2,nt]
        inc_scale_r = torch.from_numpy(f["incident/scale_recv"][:])
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            a = torch.from_numpy(f["samples/ascans"][lo:hi])
            s = torch.from_numpy(src_all[lo:hi].astype(np.int64))
            j = torch.from_numpy(nu_all[lo:hi].astype(np.int64))
            a_inc = inc_a[s, j]
            if snr_db is not None:
                a = H.add_measurement_noise(a, snr_db, generator=generator)
            u = (H.displacement_from_ascans(a, omegas=om)
                 - H.displacement_from_ascans(a_inc, omegas=om))
            nu = torch.tensor([cfg.NU_LIST[int(k)] for k in j],
                              dtype=torch.float32)
            xs.append(pack_ring(u, src_idx=s, nu=nu, scale=inc_scale_r[s, j]))
            thetas.append(torch.from_numpy(theta_all[lo:hi]).float())
            nus.append(nu)
            srcs.append(s)
    data = RingData(torch.cat(xs), torch.cat(thetas), torch.cat(nus),
                    torch.cat(srcs))
    return data.to(device) if device is not None else data


# ---------------------------------------------------------------------------
# Training and scoring
# ---------------------------------------------------------------------------
def train_regressor(train: RingData, val: RingData, *, device=None,
                    epochs: int = 200, batch_size: int = 64, lr: float = 1e-3,
                    weight_decay: float = 1e-4, seed: int = cfg.SEED,
                    width: int = 96, log_every: int = 25) -> tuple[RingCNN, dict]:
    """
    Plain supervised regression on z, MSE.

    Trained on z rather than on theta because the three parameters have different
    physical units and ranges (two positions spanning ~5 lambda_s, one radius
    spanning 0.8 lambda_s): an MSE on theta would weight position errors ~40x more
    than radius errors purely through the units, and the radius would never be
    learned.  In z both are O(1) by construction.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tr, va = train.to(device), val.to(device)
    ztr, zva = tr.z(), va.z()

    net = RingCNN(c_in=tr.x.shape[1], width=width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=lr / 100)
    hist: dict = dict(train=[], val=[])
    best, best_state = math.inf, None
    n = len(tr)

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for lo in range(0, n, batch_size):
            idx = perm[lo:lo + batch_size]
            loss = torch.nn.functional.mse_loss(net(tr.x[idx]), ztr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * idx.numel()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(net(va.x), zva))
        hist["train"].append(tot / n)
        hist["val"].append(v)
        if v < best:
            best = v
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        if log_every and ep % log_every == 0:
            print(f"  cnn ep {ep:4d}  train {tot/n:.5f}  val {v:.5f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, hist


@torch.no_grad()
def score(net: RingCNN, data: RingData, *, family: ShapeFamily | None = None,
          truth_family: ShapeFamily | None = None) -> dict:
    """
    The baseline's scores, computed by the code that scores the pipeline.

    Returns exactly `inverse.invert.summarise`'s dictionary, because it *is* that
    function: one `InversionResult` is built per sample and the list handed over.  The
    notebooks print the baseline's column beside the inversion's column in one table,
    and until now those two columns were two different definitions of "position error"
    -- this one differenced `theta[:, :2]` and `theta[:, 2]` literally, while the
    pipeline's matched blobs permutation-invariantly, reduced an ellipse to its
    equal-area radius and required IoU as well as position.  Comparing them was
    comparing measurements, not estimators, which is the specific failure §9 asks to be
    removed from the comparison.  It also means the baseline now gets an IoU, a p90 and
    a two-gate success rate for free.

    Errors are in lambda_s rather than absolute units because lambda_s varies by 13%
    across the four Poisson ratios and the gate (position error < lambda_s/10) is stated
    in wavelengths.

    `truth_family` is for the transfer experiment: `data.theta` then holds *that*
    family's parameters (5 columns for an ellipse, 6 for two voids) while the network
    still predicts a circle, and the reduction to comparable numbers happens inside
    `InversionResult` where it is documented, rather than in a notebook cell.  Callers
    previously pre-reduced the truth to an equal-area circle themselves; passing the
    real truth instead is what makes the IoU column meaningful.
    """
    from ..inverse.invert import InversionResult, summarise   # local: models <- inverse

    family = family or Circle()
    net.eval()
    theta = net.predict_theta(data.x, data.nu, family)
    exp = len((truth_family or family).param_names)
    assert data.theta.shape[1] == exp, (
        f"truth has {data.theta.shape[1]} parameters but "
        f"{(truth_family or family).name!r} takes {exp}; pass truth_family=")
    results = [
        InversionResult(theta=theta[i], misfit=float("nan"),
                        theta_true=data.theta[i],
                        lambda_s=cfg.cs_over_cp(float(data.nu[i])) / cfg.FC,
                        n_forward=1, family=family, truth_family=truth_family)
        for i in range(len(data))]
    return summarise(results)


def save(net: RingCNN, path) -> None:
    torch.save(dict(state_dict=net.state_dict(),
                    arch=dict(c_in=net.body[0].in_channels,
                              width=net.body[0].out_channels,
                              n_out=net.n_out,
                              depth=sum(1 for m in net.body
                                        if isinstance(m, nn.Conv1d)))), path)


def load(path, device=None) -> RingCNN:
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    net = RingCNN(**ck["arch"]).to(device or "cpu")
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


__all__ = [
    "RING_CHANNELS",
    "RingCNN",
    "RingData",
    "load",
    "pack_ring",
    "ring_features",
    "save",
    "score",
    "train_regressor",
]
