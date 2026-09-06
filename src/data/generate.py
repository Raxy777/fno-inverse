"""
Dataset generation (§3.7, §11.2 step 6).

Produces one HDF5 file per split.  What varies from sample to sample, and why:

* **defect** (x_c, y_c, R) -- the thing being learned.
* **source index** -- the change the document calls the highest-value one in it.
  With a single fixed source the network can memorise one illumination and the
  inversion's gradient then reflects that memorisation rather than the physics; with
  8 sources it has to represent the operator as a function of where the energy comes
  from.  Two source positions are withheld entirely (`cfg.SRC_HELDOUT`) so that
  §11.2 step 11 can be run on an illumination the network has never seen.
* **Poisson ratio** -- four values.  This is what makes the two wave speeds, and
  hence the mode-conversion physics, vary rather than being a fixed constant the
  network can absorb into its weights.

What is cached rather than recomputed: the incident field depends only on (source
index, nu), never on the defect, so there are 8 x 4 = 32 incident solves in total
rather than one per sample.  At 2800 samples that is the difference between 32 extra
solves and 2800 -- worth the small amount of bookkeeping.

Stored quantities and their conventions:

* `samples/us_phasors` -- *scattered displacement* phasors on the 128^2 network
  grid, complex64, i.e. the network's target, in raw physical units.  Raw and not
  normalised, deliberately: the normalisation choice (§7.1) is a modelling decision
  and storing it baked in would mean regenerating 16 GB to change one's mind.
* `samples/ascans` -- *total velocity* A-scans at the 32 receivers, full time
  resolution, float32.  Time domain and un-noised, so measurement noise can be added
  at any SNR later, in the domain where a real transducer's noise lives.
* `incident/*` -- the corresponding incident quantities, indexed [src, nu].

The interface width used by the solver is the *physical* width the network sees
(`cfg.EPS_INTERFACE_PHYS`), not a fixed number of cells of whichever grid is in front
of it.  Getting this wrong makes the label describe a slightly different void from the
one the input channels describe -- a small, systematic, and completely invisible
inconsistency.  The width itself was set by measuring the exterior scattered field
against an analytic traction-free cavity (`solver.validate.check_cavity_scattering`),
so it is part of the label definition and is snapshotted with the dataset.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch import Tensor

from .. import config as cfg
from ..geometry.sdf import Circle, fine_coords, material_fields, soft_indicator
from ..solver import harmonic as H
from ..solver.fdtd_elastic import ElasticFDTD2D

EPS_LEN_PHYS: float = cfg.EPS_INTERFACE_PHYS
SRC_KEEPOUT_LS: float = 1.0        # void boundary to source, in shear wavelengths


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------
@dataclass
class SampleTable:
    theta: np.ndarray        # [N, 3] float32: xc, yc, R
    src_idx: np.ndarray      # [N] int8
    nu_idx: np.ndarray       # [N] int8

    def __len__(self) -> int:
        return self.theta.shape[0]


def sample_parameters(n: int, rng: np.random.Generator, *,
                      src_pool: tuple[int, ...] = cfg.SRC_TRAIN,
                      nu_pool: tuple[float, ...] = cfg.NU_LIST) -> SampleTable:
    """
    Rejection-sample n (defect, source, nu) triples.

    Two constraints, both on the void *boundary* rather than its centre:

    1. at least BOUNDARY_KEEPOUT_LS shear wavelengths from every wall, so the void
       never interacts with the absorbing layer -- an absorber is only reflectionless
       for waves that are already propagating freely when they enter it;
    2. at least SRC_KEEPOUT_LS shear wavelengths from the source, so the source is
       never inside or touching the void, where the point-force injection would be
       dividing by a near-zero stiffness.

    Stating them on the boundary rather than the centre matters because R varies by a
    factor of three across the dataset; a centre-based keep-out would let large voids
    reach much closer to the wall than small ones, and the network would learn a
    spurious correlation between defect size and boundary proximity.

    Note that this support is strictly *inside* the feasible box that
    `Circle.bounds` gives the inversion.  That is intentional and worth being able
    to say out loud: the optimiser is allowed to wander into regions where no
    training data exists, and if the surrogate misbehaves there, that is a real
    failure mode of the method rather than something to be hidden by clamping.
    """
    theta = np.zeros((n, 3), dtype=np.float32)
    src_idx = np.zeros(n, dtype=np.int8)
    nu_idx = np.zeros(n, dtype=np.int8)

    i, tries = 0, 0
    while i < n:
        tries += 1
        assert tries < 200 * n, "rejection sampling is not converging; check keep-outs"
        j_nu = int(rng.integers(len(nu_pool)))
        nu = nu_pool[j_nu]
        lam_s = cfg.cs_over_cp(nu) / cfg.FC
        r = float(rng.uniform(cfg.R_MIN_LS, cfg.R_MAX_LS)) * lam_s
        wall = r + cfg.BOUNDARY_KEEPOUT_LS * lam_s
        if 2.0 * wall >= cfg.L_DOMAIN:
            continue
        xc = float(rng.uniform(wall, cfg.L_DOMAIN - wall))
        yc = float(rng.uniform(wall, cfg.L_DOMAIN - wall))

        s = int(rng.choice(src_pool))
        sxc, syc = cfg.SOURCE_XY[s]
        if math.hypot(xc - sxc, yc - syc) < r + SRC_KEEPOUT_LS * lam_s:
            continue

        theta[i] = (xc, yc, r)
        src_idx[i] = s
        nu_idx[i] = j_nu
        i += 1

    return SampleTable(theta=theta, src_idx=src_idx, nu_idx=nu_idx)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
def fine_chi(theta: Tensor, *, device=None, dtype=torch.float32) -> Tensor:
    """Soft void indicator on the padded fine solver grid, [B, 316, 316]."""
    yy, xx = fine_coords(device=device, dtype=dtype)
    phi = Circle().sdf(theta, yy, xx)
    return soft_indicator(phi, EPS_LEN_PHYS)


def _materials(chi: Tensor, nu_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Per-sample (lam, mu, rho) for a batch whose samples have different nu.

    Delegates the void rule to `material_fields` rather than repeating it: that
    function is also what the physics loss differentiates through (§7.2), and if
    the two disagreed about how a void is represented the residual would be
    penalising the network for reproducing the solver correctly.
    """
    mu0 = (1.0 - 2.0 * nu_values) / (2.0 * (1.0 - nu_values))     # c_s^2, rho=c_p=1
    mu0 = mu0.view(-1, 1, 1)
    lam0 = 1.0 - 2.0 * mu0
    return material_fields(chi, lam0, mu0)


# ---------------------------------------------------------------------------
# Incident field cache
# ---------------------------------------------------------------------------
def run_incident(device=None, *, progress=None) -> dict:
    """
    The 8 x 4 incident solves, as one batch of 32.

    Returns numpy arrays:
        phasors     [8, 4, 2, M, 128, 128] complex64  displacement
        ascans      [8, 4, 32, 2, nt]      float32    velocity
        scale       [8, 4, M]              float32    max|u_inc| over the domain
        scale_recv  [8, 4, M]              float32    max|u_inc| over the ring

    Both normalisation scales are stored because the field loss and the receiver
    misfit live on different amplitudes: the ring sits 3 cells from the edge where
    the incident field has already spread, while the domain max is dominated by the
    near-source singularity.  Using the domain scale for the receiver misfit would
    divide the data residual by a number two orders of magnitude too large and make
    the measurement term silently negligible.
    """
    n_src, n_nu = cfg.N_SRC, len(cfg.NU_LIST)
    B = n_src * n_nu
    chi = torch.zeros(B, cfg.N_FINE_TOTAL, cfg.N_FINE_TOTAL, device=device)
    nu_vals = torch.tensor([cfg.NU_LIST[j] for _ in range(n_src) for j in range(n_nu)],
                           device=device)
    lam, mu, rho = _materials(chi, nu_vals)
    sim = ElasticFDTD2D(lam, mu, rho)
    src = [cfg.net_to_fine(*cfg.SOURCES_NET[i]) for i in range(n_src) for _ in range(n_nu)]

    om = H.omegas_tensor(device)
    res = sim.run(src, nt=cfg.NT, recv_yx=cfg.RECEIVERS_NET, omegas=om,
                  progress=progress)
    u = H.displacement_from_field(res.phasors, omegas=om)      # [B,2,M,128,128]
    scale = H.incident_scale(u)                                # [B,1,M,1,1]
    u_r = H.displacement_from_ascans(res.ascans, omegas=om)    # [B,R,2,M]
    scale_r = H.incident_scale_at_receivers(u_r)

    shape = (n_src, n_nu)
    return dict(
        phasors=u.reshape(*shape, *u.shape[1:]).cpu().numpy(),
        ascans=res.ascans.reshape(*shape, *res.ascans.shape[1:]).cpu().numpy(),
        scale=scale.reshape(n_src, n_nu, -1).cpu().numpy().astype(np.float32),
        scale_recv=scale_r.reshape(n_src, n_nu, -1).cpu().numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Size and time projection
# ---------------------------------------------------------------------------
def projected_size(n: int, *, n_vis: int = cfg.N_VIS_SAMPLES) -> dict:
    """Byte counts, printed before generation rather than discovered after it."""
    npix = cfg.N_NET * cfg.N_NET
    per_phasor = 2 * cfg.M_FREQ * npix * 8                     # complex64
    per_ascan = cfg.N_RECV * 2 * cfg.NT * 4
    inc = cfg.N_SRC * len(cfg.NU_LIST) * (per_phasor + per_ascan)
    vis = n_vis * cfg.N_SAVED_FRAMES * 2 * npix * 4
    total = n * (per_phasor + per_ascan) + inc + vis
    return dict(per_sample=per_phasor + per_ascan, phasor=per_phasor,
                ascan=per_ascan, incident=inc, frames=vis, total=total)


def print_projection(n: int, *, throughput_cell_steps_per_s: float | None = None
                     ) -> None:
    p = projected_size(n)
    gb = 1e9
    print(f"projected size for {n} samples")
    print(f"  phasors  {p['phasor']/1e6:8.2f} MB/sample   "
          f"({2*cfg.M_FREQ} planes of {cfg.N_NET}^2 complex64)")
    print(f"  A-scans  {p['ascan']/1e6:8.2f} MB/sample   "
          f"({cfg.N_RECV} receivers x 2 x {cfg.NT} float32)")
    print(f"  incident {p['incident']/gb:8.3f} GB total    "
          f"({cfg.N_SRC} sources x {len(cfg.NU_LIST)} nu, cached)")
    print(f"  frames   {p['frames']/gb:8.3f} GB total     "
          f"({cfg.N_VIS_SAMPLES} visualisation samples)")
    print(f"  TOTAL    {p['total']/gb:8.3f} GB")
    cellsteps = n * cfg.N_FINE_TOTAL ** 2 * cfg.NT
    print(f"  compute  {cellsteps/1e9:8.1f} G cell-steps "
          f"({cfg.N_FINE_TOTAL}^2 x {cfg.NT} per sample)")
    if throughput_cell_steps_per_s:
        hrs = cellsteps / throughput_cell_steps_per_s / 3600
        print(f"           ~{hrs:.1f} GPU-hours at the measured "
              f"{throughput_cell_steps_per_s/1e9:.1f} G cell-steps/s")
    else:
        print("           run `calibrate()` for a wall-clock estimate on this GPU")
    if p["total"] > 20 * gb:
        print("  NOTE: over 20 GB.  The levers, in order of how little they cost:\n"
              "        M_FREQ 20 -> 12 (the top of the band is the least informative\n"
              "        part of it), then N_TRAIN.  Do not drop the A-scans: they are\n"
              "        the inversion's data and cannot be recovered from the phasors.")


def calibrate(device=None, nt: int = 50, batch: int = cfg.GEN_BATCH) -> float:
    """Measure solver throughput in cell-steps/s, for the time projection."""
    chi = torch.zeros(batch, cfg.N_FINE_TOTAL, cfg.N_FINE_TOTAL, device=device)
    nu = torch.full((batch,), cfg.NU_LIST[0], device=device)
    lam, mu, rho = _materials(chi, nu)
    sim = ElasticFDTD2D(lam, mu, rho)
    om = H.omegas_tensor(device)
    src = [cfg.net_to_fine(*cfg.SOURCES_NET[0])] * batch
    sim.run(src, nt=5, recv_yx=cfg.RECEIVERS_NET, omegas=om)     # warm up
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    sim.run(src, nt=nt, recv_yx=cfg.RECEIVERS_NET, omegas=om)
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    dt_wall = time.perf_counter() - t0
    return batch * cfg.N_FINE_TOTAL ** 2 * nt / dt_wall


# ---------------------------------------------------------------------------
# Config snapshot, so a dataset can be checked against the code that reads it
# ---------------------------------------------------------------------------
# Everything here changes what the *labels are*, as opposed to how they are used, so a
# mismatch has to be a hard refusal rather than a warning.  The interface and void
# constants are in the list because they were set by measurement against an analytic
# cavity and have moved once already: a dataset generated at the v2.0 width and full
# void density is a dataset of soft heavy inclusions, and nothing downstream can tell
# from the file.  `SOURCE_FORCE_XY` is in for the same reason one step further out --
# it encodes the y-face offset of the point force, which is the frame every analytic
# reference is evaluated in.
_SNAPSHOT_KEYS = (
    "L_DOMAIN", "N_NET", "N_FINE", "N_FINE_TOTAL", "N_PML_FINE", "DX_NET", "DX_FINE",
    "DOWNSAMPLE", "DT", "NT", "T_END", "CFL_NUMBER", "N_CYCLES", "M_FREQ", "F_START",
    "DF", "N_RECV", "N_SRC", "RING_INSET_NET", "EPS_INTERFACE_CELLS",
    "EPS_INTERFACE_FINE_CELLS", "EPS_INTERFACE_PHYS",
    "VOID_DENSITY_SCALE", "VOID_STIFFNESS_FLOOR", "R_MIN_LS", "R_MAX_LS",
    "BOUNDARY_KEEPOUT_LS", "DFT_EVERY", "ABSORBER_ORDER", "ABSORBER_R_TARGET",
    "N_ABSORBER_FINE",
)


def config_snapshot() -> dict:
    snap = {k: getattr(cfg, k) for k in _SNAPSHOT_KEYS}
    snap["NU_LIST"] = list(cfg.NU_LIST)
    snap["FREQS"] = list(cfg.FREQS)
    snap["EPS_LEN_PHYS"] = EPS_LEN_PHYS
    # The source convention, flattened: (x, y) of every point force, including the
    # half-fine-cell y-face offset.  Snapshotted as a vector rather than asserted in
    # prose because a dataset made before that offset was pinned down differs from
    # this one by 0.26 rad of shear phase at f_max, which reads as a wave-speed error.
    snap["SOURCE_FORCE_XY"] = [v for xy in cfg.SOURCE_FORCE_XY for v in xy]
    return snap


def assert_compatible(f: h5py.File) -> None:
    """Refuse to train on a dataset generated under a different configuration."""
    bad = []
    for k in _SNAPSHOT_KEYS:
        if k not in f.attrs:
            bad.append(f"{k}: missing from file")
            continue
        want, got = getattr(cfg, k), f.attrs[k]
        same = (abs(float(want) - float(got)) < 1e-9
                if isinstance(want, float) else want == got)
        if not same:
            bad.append(f"{k}: file has {got}, config has {want}")

    want_src = np.asarray([v for xy in cfg.SOURCE_FORCE_XY for v in xy], float)
    if "SOURCE_FORCE_XY" not in f.attrs:
        bad.append("SOURCE_FORCE_XY: missing from file (generated before the point "
                   "force's y-face offset was pinned down; the labels carry a "
                   "half-fine-cell source shift)")
    else:
        got_src = np.asarray(f.attrs["SOURCE_FORCE_XY"], float)
        if got_src.shape != want_src.shape or not np.allclose(got_src, want_src,
                                                              atol=1e-9):
            bad.append(f"SOURCE_FORCE_XY: file has {got_src.tolist()[:4]}..., "
                       f"config has {want_src.tolist()[:4]}...")
    if bad:
        raise AssertionError(
            "dataset was generated under a different configuration:\n  "
            + "\n  ".join(bad)
            + "\nRegenerate the dataset, or check out the config it was made with. "
              "Training on mismatched labels produces a network that is wrong in a "
              "way no validation metric computed from the same file can detect.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(path: str, n: int, *, split: str = "train", device=None,
             seed: int | None = None, batch: int = cfg.GEN_BATCH,
             src_pool: tuple[int, ...] | None = None,
             n_vis: int = cfg.N_VIS_SAMPLES,
             incident: dict | None = None,
             progress=None) -> str:
    """
    Write one split to `path`.

    `src_pool` defaults to cfg.SRC_TRAIN for train/val and to all sources for test,
    so that the test split contains the held-out illuminations and the train split
    provably does not.
    """
    if src_pool is None:
        src_pool = tuple(range(cfg.N_SRC)) if split == "test" else cfg.SRC_TRAIN
    if seed is None:
        seed = cfg.SEED + {"train": 0, "val": 1, "test": 2}.get(split, 3)
    rng = np.random.default_rng(seed)
    table = sample_parameters(n, rng, src_pool=src_pool)

    print_projection(n)
    if incident is None:
        print("running the 32 cached incident solves ...")
        incident = run_incident(device=device, progress=progress)

    om = H.omegas_tensor(device)
    npix = cfg.N_NET
    vis_index = sorted(rng.choice(n, size=min(n_vis, n), replace=False).tolist())
    vis_set = set(vis_index)

    with h5py.File(path, "w") as f:
        for k, v in config_snapshot().items():
            f.attrs[k] = v
        f.attrs["split"] = split
        f.attrs["seed"] = seed
        f.attrs["src_pool"] = list(src_pool)
        f.attrs["n_samples"] = n

        gi = f.create_group("incident")
        gi.create_dataset("phasors", data=incident["phasors"])
        gi.create_dataset("ascans", data=incident["ascans"])
        gi.create_dataset("scale", data=incident["scale"])
        gi.create_dataset("scale_recv", data=incident["scale_recv"])
        gi.create_dataset("positions", data=np.array(cfg.SOURCES_NET, dtype=np.int32))
        gi.create_dataset("xy", data=np.array(cfg.SOURCE_XY, dtype=np.float64))
        f.create_dataset("receivers", data=np.array(cfg.RECEIVERS_NET, dtype=np.int32))
        f.create_dataset("receiver_xy", data=np.array(cfg.RECEIVER_XY, dtype=np.float64))

        gs = f.create_group("samples")
        gs.create_dataset("theta", data=table.theta)
        gs.create_dataset("src_idx", data=table.src_idx)
        gs.create_dataset("nu_idx", data=table.nu_idx)
        d_ph = gs.create_dataset("us_phasors", shape=(n, 2, cfg.M_FREQ, npix, npix),
                                 dtype=np.complex64,
                                 chunks=(1, 1, 1, npix, npix))
        d_as = gs.create_dataset("ascans", shape=(n, cfg.N_RECV, 2, cfg.NT),
                                 dtype=np.float32, chunks=(1, cfg.N_RECV, 2, cfg.NT))
        gv = f.create_group("vis")
        gv.create_dataset("index", data=np.array(vis_index, dtype=np.int32))
        d_fr = gv.create_dataset(
            "frames", shape=(len(vis_index), cfg.N_SAVED_FRAMES, 2, npix, npix),
            dtype=np.float32) if vis_index else None

        inc_ph = torch.as_tensor(incident["phasors"], device=device)
        t0 = time.perf_counter()
        it = range(0, n, batch)
        if progress is not None:
            it = progress(it)
        for lo in it:
            hi = min(n, lo + batch)
            idx = list(range(lo, hi))
            th = torch.as_tensor(table.theta[lo:hi], device=device)
            nu_vals = torch.tensor([cfg.NU_LIST[j] for j in table.nu_idx[lo:hi]],
                                   device=device)
            chi = fine_chi(th, device=device)
            lam, mu, rho = _materials(chi, nu_vals)
            sim = ElasticFDTD2D(lam, mu, rho)
            src = [cfg.net_to_fine(*cfg.SOURCES_NET[s]) for s in table.src_idx[lo:hi]]
            want_frames = [i for i in idx if i in vis_set]
            res = sim.run(src, nt=cfg.NT, recv_yx=cfg.RECEIVERS_NET, omegas=om,
                          save_frames=cfg.N_SAVED_FRAMES if want_frames else 0)

            u_tot = H.displacement_from_field(res.phasors, omegas=om)
            u_inc = torch.stack([inc_ph[s, j] for s, j in
                                 zip(table.src_idx[lo:hi], table.nu_idx[lo:hi])])
            d_ph[lo:hi] = (u_tot - u_inc).cpu().numpy()
            d_as[lo:hi] = res.ascans.cpu().numpy()
            if want_frames and d_fr is not None:
                for i in want_frames:
                    d_fr[vis_index.index(i)] = res.frames[i - lo].cpu().numpy()

        wall = time.perf_counter() - t0
        f.attrs["generation_seconds"] = wall

    print(f"wrote {path}: {n} samples in {wall/60:.1f} min "
          f"({wall/max(n,1):.2f} s/sample)")
    return path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="generate the FNO training data")
    ap.add_argument("--out", required=True, help="directory for the .h5 files")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--n-train", type=int, default=cfg.N_TRAIN)
    ap.add_argument("--n-val", type=int, default=cfg.N_VAL)
    ap.add_argument("--n-test", type=int, default=cfg.N_TEST)
    ap.add_argument("--batch", type=int, default=cfg.GEN_BATCH)
    ap.add_argument("--device", default=None)
    ap.add_argument("--calibrate-only", action="store_true")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.calibrate_only:
        tp = calibrate(device=dev, batch=a.batch)
        print_projection(a.n_train + a.n_val + a.n_test,
                         throughput_cell_steps_per_s=tp)
        return

    try:
        from tqdm.auto import tqdm as _tqdm
        prog = _tqdm
    except ImportError:
        prog = None

    sizes = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
    # one incident cache, shared by every split -- the incident field does not
    # depend on the split
    incident = run_incident(device=dev, progress=prog)
    import os
    os.makedirs(a.out, exist_ok=True)
    for s in a.splits.split(","):
        generate(os.path.join(a.out, f"{s}.h5"), sizes[s], split=s, device=dev,
                 batch=a.batch, incident=incident, progress=prog)


if __name__ == "__main__":
    main()
