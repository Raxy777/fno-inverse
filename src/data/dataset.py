"""
Loading (§7.1, §11.2 step 7).

Three things here are less obvious than they look.

**Frequency subsetting.**  A sample carries M = 20 frequencies, and each is an
independent training example as far as the operator is concerned -- the network sees
one frequency at a time, conditioned on it.  Loading all 20 for every sample would
put 20 x BATCH_SIZE fields through the network per step; at d_v = 32 on 128^2 that is
about 3.4 GB of stored activations, which does not fit alongside the weights and the
optimiser state on a 16 GB card.  So each sample contributes a random subset of
N_FREQ_PER_SAMPLE frequencies per epoch, resampled every epoch.  This is not a
compromise on data: over 60 epochs every sample is seen at essentially every
frequency, and the gradient is *less* correlated within a batch than it would be if
one sample's 20 near-identical frequency slices dominated it.

**Normalisation.**  Input incident field and target scattered field are divided by
the *same* per-(sample, frequency) incident amplitude.  Dividing the target by its
own norm would be the obvious thing and is the one thing that must not happen: the
scattered amplitude carries the radius (§7.1), and normalising it away leaves the
inversion with nothing to recover R from.

**h5py and DataLoader workers.**  An open HDF5 handle cannot cross a fork.  The file
is therefore opened lazily, in whichever process first touches it, and the handle is
never created in `__init__`.  The failure mode if this is got wrong is not an
exception: it is silently corrupted reads, different in each worker.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .. import config as cfg
from .. import features as F
from ..geometry.sdf import Circle, geometry_channels
from .generate import assert_compatible


class WaveDataset(Dataset):
    """
    One HDF5 split.  `__getitem__` returns a dict of tensors:

        phi_t   [ny, nx]            clipped SDF in network cells
        chi     [ny, nx]            soft void indicator
        u_inc   [F, 2, ny, nx]      complex64, normalised
        u_s     [F, 2, ny, nx]      complex64, normalised   <- target
        freqs   [F]                 in units of f_c
        nu      scalar
        theta   [3]                 xc, yc, R (physical)
        src_idx scalar int64
        scale   [F]                 the divisor, kept so predictions can be un-scaled

    Collating is the default: `pack_inputs` is applied to the *batch*, in the
    training step, because the coordinate channels are shared across the whole batch
    and building them per sample would trivially triple the loader's work.
    """

    def __init__(self, path: str, *, n_freq: int = cfg.N_FREQ_PER_SAMPLE,
                 train: bool = True, seed: int = cfg.SEED,
                 with_ascans: bool = False, check_config: bool = True):
        self.path = str(path)
        self.train = train
        self.with_ascans = with_ascans
        self._f: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            if check_config:
                assert_compatible(f)
            self.n = int(f.attrs["n_samples"])
            self.theta = f["samples/theta"][:]
            self.src_idx = f["samples/src_idx"][:]
            self.nu_idx = f["samples/nu_idx"][:]
            self.inc_scale = f["incident/scale"][:]          # [S, NU, M]
            self.split = f.attrs["split"]
        self.n_freq = min(n_freq, cfg.M_FREQ) if train else cfg.M_FREQ
        self.epoch = 0
        self._seed = seed
        self._family = Circle()
        self._coords: tuple[Tensor, Tensor] | None = None

    # -- plumbing ---------------------------------------------------------
    def __len__(self) -> int:
        return self.n

    def set_epoch(self, epoch: int) -> None:
        """
        Reseeds the frequency subsetting so each epoch draws a fresh subset.

        Only has any effect because `make_loader` sets persistent_workers=False:
        with persistent workers the dataset object is copied into each worker once
        and never again, so mutating `self.epoch` in the parent would be invisible
        to them and every epoch would train on the same 4 frequencies per sample.
        The symptom would be a validation curve that plateaus early for no visible
        reason.  Re-forking workers costs about a second per epoch, against epochs
        that take minutes.
        """
        self.epoch = int(epoch)

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def _grid(self) -> tuple[Tensor, Tensor]:
        if self._coords is None:
            from ..geometry.sdf import net_coords
            self._coords = net_coords()
        return self._coords

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_f"] = None                 # never pickle an HDF5 handle
        return d

    # -- the sample -------------------------------------------------------
    def _freq_subset(self, i: int) -> np.ndarray:
        if not self.train:
            return np.arange(cfg.M_FREQ)
        # Deterministic given (seed, epoch, index): reproducible, and independent
        # across workers without any shared state.
        g = np.random.default_rng((self._seed, self.epoch, i))
        return np.sort(g.choice(cfg.M_FREQ, size=self.n_freq, replace=False))

    def __getitem__(self, i: int) -> dict:
        f = self._file()
        m = self._freq_subset(i)
        s, j = int(self.src_idx[i]), int(self.nu_idx[i])

        # h5py wants an increasing list for fancy indexing; m is sorted above.
        # Note h5py's fancy-indexing semantics, which differ from numpy's: h5py
        # keeps the dataset's axis order, so [i, :, m] returns [2, F, ny, nx],
        # whereas numpy would move the advanced-index axis to the front and return
        # [F, 2, ny, nx].  The assert below is the guard, because a silent swap of
        # the component and frequency axes would still train -- towards an operator
        # with the two transposed -- and only shows up as a mysteriously bad model.
        u_s = f["samples/us_phasors"][i, :, m]                # [2, F, ny, nx]
        u_i = f["incident/phasors"][s, j, :, m]               # [2, F, ny, nx]
        assert u_s.shape[:2] == (2, len(m)) and u_i.shape[:2] == (2, len(m)), (
            f"expected [2, {len(m)}, ...] from h5py, got {u_s.shape} / {u_i.shape}")
        u_s = np.ascontiguousarray(u_s.transpose(1, 0, 2, 3))  # [F, 2, ny, nx]
        u_i = np.ascontiguousarray(u_i.transpose(1, 0, 2, 3))

        scale = self.inc_scale[s, j, m].astype(np.float32)     # [F]
        div = scale[:, None, None, None]
        theta = torch.from_numpy(self.theta[i]).float()
        nu = float(cfg.NU_LIST[j])

        yy, xx = self._grid()
        phi_t, chi = geometry_channels(theta.unsqueeze(0), self._family,
                                       yy=yy, xx=xx)

        out = dict(
            phi_t=phi_t[0], chi=chi[0],
            u_inc=torch.from_numpy(u_i / div),
            u_s=torch.from_numpy(u_s / div),
            freqs=torch.tensor([cfg.FREQS[k] for k in m], dtype=torch.float32),
            nu=torch.tensor(nu, dtype=torch.float32),
            theta=theta,
            src_idx=torch.tensor(s, dtype=torch.long),
            nu_idx=torch.tensor(j, dtype=torch.long),
            scale=torch.from_numpy(scale),
            index=torch.tensor(i, dtype=torch.long),
        )
        if self.with_ascans:
            out["ascans"] = torch.from_numpy(f["samples/ascans"][i])
            out["ascans_inc"] = torch.from_numpy(f["incident/ascans"][s, j])
        return out


def _worker_start_method() -> str | None:
    """
    'forkserver' when workers are used on a fork-capable platform, else None.

    persistent_workers is off (see `WaveDataset.set_epoch`), so a fresh set of
    workers is forked every epoch.  The default `fork` start method forks the whole
    training process -- including whatever threads CUDA, the pin-memory copier and
    HDF5's global mutex are running -- and if a lock is held at the instant of the
    fork the child inherits it locked, owned by a thread that does not exist in the
    child, and deadlocks on its first HDF5 read.  It is intermittent by construction:
    it struck this run at epoch 88, having survived 87 forks.  `forkserver` forks
    each worker from a pristine single-threaded server process instead, so no
    application lock can be held across the fork.  The dataset is already pickled
    into workers (`__getstate__` drops the h5py handle), so nothing else changes --
    including the per-epoch reseed, which still rides the freshly pickled `self.epoch`.
    """
    import multiprocessing as mp
    methods = mp.get_all_start_methods()
    if "forkserver" in methods:
        return "forkserver"
    if "fork" in methods:
        return "fork"
    return None                                    # Windows/spawn-only: DataLoader default


def make_loader(dataset: WaveDataset, *, batch_size: int = cfg.BATCH_SIZE,
                shuffle: bool | None = None, num_workers: int = 4,
                pin_memory: bool = True) -> DataLoader:
    """
    persistent_workers is deliberately off; see `WaveDataset.set_epoch`.  Because
    that re-forks every epoch, the multiprocessing start method is pinned to
    'forkserver' when workers are used; see `_worker_start_method`.

    num_workers=4 rather than more because each worker holds its own HDF5 read
    buffer and the bottleneck here is chunk decompression of 128^2 complex planes,
    not Python overhead.  On Modal's smaller CPU allocations, drop it to 2.
    """
    if shuffle is None:
        shuffle = dataset.train
    # A start method may only be passed when workers are actually spawned; with
    # num_workers=0 the loader runs in-process and DataLoader rejects the argument.
    ctx = _worker_start_method() if num_workers > 0 else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      drop_last=dataset.train, persistent_workers=False,
                      multiprocessing_context=ctx)


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def batch_to_model(batch: dict, *, dx: float = cfg.DX_NET,
                   coords: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    (inputs, targets) for one batch, both [B*F, C, ny, nx] real.

    The two calls are deliberately adjacent so the frequency-flattening convention
    cannot drift between them.
    """
    x = F.pack_inputs(batch["phi_t"], batch["chi"], batch["u_inc"],
                      batch["freqs"], batch["nu"], dx=dx, coords=coords)
    y = F.flatten_freq(F.complex_to_channels(batch["u_s"]))
    assert x.shape[0] == y.shape[0], (
        f"input rows {x.shape[0]} != target rows {y.shape[0]}; the frequency axis "
        "was flattened inconsistently")
    return x, y


# ---------------------------------------------------------------------------
# Inversion data
# ---------------------------------------------------------------------------
def load_inversion_case(path: str, i: int, *, snr_db: float | None = None,
                        generator: torch.Generator | None = None) -> dict:
    """
    One test sample as the inversion sees it: receiver data only, plus the truth.

    Noise is added to the *total* velocity A-scans in the time domain and the
    incident field is subtracted afterwards, in that order.  Reversing them --
    subtracting first, then adding noise to the scattered trace -- would be
    convenient and wrong: a real instrument measures the total field, so the noise
    floor is set by the total amplitude, which near a source is far larger than the
    scattered signal.  Noising the residual instead would make the inversion look
    good at SNRs where it would in fact fail.
    """
    from ..solver import harmonic as H

    with h5py.File(path, "r") as f:
        theta = torch.from_numpy(f["samples/theta"][i]).float()
        s = int(f["samples/src_idx"][i])
        j = int(f["samples/nu_idx"][i])
        a_tot = torch.from_numpy(f["samples/ascans"][i]).unsqueeze(0)
        a_inc = torch.from_numpy(f["incident/ascans"][s, j]).unsqueeze(0)
        recv = torch.from_numpy(f["receivers"][:]).long()

    if snr_db is not None:
        a_tot = H.add_measurement_noise(a_tot, snr_db, generator=generator)

    om = H.omegas_tensor()
    u_tot = H.displacement_from_ascans(a_tot, omegas=om)       # [1, R, 2, M]
    u_inc = H.displacement_from_ascans(a_inc, omegas=om)
    return dict(
        theta_true=theta, src_idx=s, nu_idx=j, nu=float(cfg.NU_LIST[j]),
        d_obs=(u_tot - u_inc), u_inc_r=u_inc,
        scale=H.incident_scale_at_receivers(u_inc),
        ascans_tot=a_tot, ascans_inc=a_inc, receivers=recv,
        snr_db=snr_db,
    )


def load_incident(path: str, device=None) -> dict:
    """The whole incident cache as torch tensors, for the inversion's forward model."""
    with h5py.File(path, "r") as f:
        return dict(
            phasors=torch.from_numpy(f["incident/phasors"][:]).to(device),
            ascans=torch.from_numpy(f["incident/ascans"][:]).to(device),
            scale=torch.from_numpy(f["incident/scale"][:]).to(device),
            scale_recv=torch.from_numpy(f["incident/scale_recv"][:]).to(device),
        )


__all__ = [
    "WaveDataset",
    "batch_to_model",
    "load_incident",
    "load_inversion_case",
    "make_loader",
    "to_device",
]
