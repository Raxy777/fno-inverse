"""
Environment bootstrap.  Import this first in every notebook.

The notebooks are deliberately *Modal-agnostic*: they contain no `modal.App`,
no `.remote()`, no decorators.  All they need is a Python process with a GPU and
a writable directory that survives the container, and this module finds both
wherever it is run:

  * Modal Notebooks           -- GPU kernel, Volume mounted under /vol or /mnt
  * a Modal container         -- launched by `modal run modal_app.py::...`
  * Colab                     -- GPU runtime, /content/drive
  * a local machine           -- CPU or CUDA, ./data

That is the property worth having.  A notebook that hard-codes Modal cannot be
debugged locally, and a notebook that hard-codes Colab cannot be run on Modal.
`modal_app.py` is the optional headless launcher for the long jobs; it imports
exactly the same `src/` code these notebooks do.

Usage
-----
    import bootstrap
    E = bootstrap.setup()
    print(E.device, E.data)
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# (import name, pip spec) -- pinned so a Modal image and a local venv agree.
REQUIRED: tuple[tuple[str, str], ...] = (
    ("torch", "torch>=2.2"),
    ("numpy", "numpy>=1.26"),
    ("h5py", "h5py>=3.10"),
    ("scipy", "scipy>=1.11"),
    ("matplotlib", "matplotlib>=3.8"),
    ("tqdm", "tqdm>=4.66"),
)

_REPO_MARKER = Path("src") / "config.py"

_DATA_CANDIDATES = (
    "/vol/fno-data",
    "/mnt/fno-data",
    "/vol/fno-wave-inverse/data",
    "/mnt/fno-wave-inverse/data",
    "/root/data",
    "/content/drive/MyDrive/fno-wave-inverse",
)


@dataclass
class Env:
    repo: Path
    data: Path
    device: str
    platform: str
    gpu_name: str = ""
    persistent: bool = True
    notes: list[str] = field(default_factory=list)

    # Conventional subdirectories, created on demand.
    @property
    def datasets(self) -> Path:
        return _mk(self.data / "datasets")

    @property
    def checkpoints(self) -> Path:
        return _mk(self.data / "checkpoints")

    @property
    def figures(self) -> Path:
        return _mk(self.data / "figures")

    @property
    def results(self) -> Path:
        return _mk(self.data / "results")

    def __str__(self) -> str:
        lines = [
            f"platform : {self.platform}",
            f"repo     : {self.repo}",
            f"data     : {self.data}" + ("" if self.persistent else "   (EPHEMERAL)"),
            f"device   : {self.device}" + (f"  [{self.gpu_name}]" if self.gpu_name else ""),
        ]
        lines += [f"note     : {n}" for n in self.notes]
        return "\n".join(lines)


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_repo(start: Path | None = None) -> Path:
    """Locate the repository root by walking up for src/config.py."""
    explicit = os.environ.get("FNO_ROOT")
    if explicit and (Path(explicit) / _REPO_MARKER).exists():
        return Path(explicit).resolve()

    seeds: list[Path] = []
    if start is not None:
        seeds.append(Path(start))
    seeds.append(Path(__file__).resolve().parent)
    seeds.append(Path.cwd())

    for seed in seeds:
        here = seed.resolve()
        for cand in (here, *here.parents):
            if (cand / _REPO_MARKER).exists():
                return cand

    # Last resort: the usual mount points on Modal.
    for cand in ("/vol/fno-wave-inverse", "/mnt/fno-wave-inverse",
                 "/root/fno-wave-inverse", "/workspace/fno-wave-inverse"):
        if (Path(cand) / _REPO_MARKER).exists():
            return Path(cand)

    raise RuntimeError(
        "Could not find the repository root (looked for src/config.py).\n"
        "On Modal: attach the Volume holding the repo, or set FNO_ROOT to its path.\n"
        f"Searched from: {[str(s) for s in seeds]}"
    )


def detect_platform() -> str:
    if any(k in os.environ for k in ("MODAL_TASK_ID", "MODAL_IMAGE_ID",
                                     "MODAL_ENVIRONMENT", "MODAL_IS_REMOTE")):
        return "modal"
    if Path("/modal").exists() or Path("/pkg/modal").exists():
        return "modal"
    if "google.colab" in sys.modules or Path("/content").exists():
        return "colab"
    return "local"


def ensure_deps(quiet: bool = True) -> list[str]:
    """Install anything missing.  Returns the list of specs actually installed."""
    missing = []
    for mod, spec in REQUIRED:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(spec)
    if missing:
        cmd = [sys.executable, "-m", "pip", "install"]
        if quiet:
            cmd.append("--quiet")
        cmd += missing
        print(f"installing: {' '.join(missing)}")
        subprocess.check_call(cmd)
    return missing


def resolve_data_dir(repo: Path, platform: str) -> tuple[Path, bool]:
    """
    Pick a writable directory for datasets and checkpoints.

    Returns (path, persistent).  `persistent` is False when we had to fall back
    to container-local storage, which matters a great deal on Modal: an HDF5
    dataset written to a non-Volume path disappears when the container stops.
    """
    explicit = os.environ.get("FNO_DATA_DIR")
    if explicit:
        return _mk(Path(explicit)), True

    for cand in _DATA_CANDIDATES:
        p = Path(cand)
        # Only accept a candidate whose *parent* exists -- that means a Volume
        # really is mounted there, rather than us inventing a directory.
        if p.exists() or p.parent.exists():
            try:
                return _mk(p), True
            except OSError:
                continue

    return _mk(repo / "data"), platform == "local"


def setup(verbose: bool = True, install: bool = True) -> Env:
    repo = find_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    platform = detect_platform()
    notes: list[str] = []

    if install:
        installed = ensure_deps()
        if installed:
            notes.append(f"installed {len(installed)} package(s) at import time")

    import torch  # noqa: E402  (only safe after ensure_deps)

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        gpu_name = f"{gpu_name}, {gib:.0f} GiB"
    else:
        device = "cpu"
        gpu_name = ""
        notes.append("no CUDA device: dataset generation and training will be slow. "
                     "On Modal choose a GPU-backed kernel, or use modal_app.py.")

    data, persistent = resolve_data_dir(repo, platform)
    if not persistent:
        notes.append(f"{data} is container-local: copy results out, or mount a "
                     "Modal Volume and set FNO_DATA_DIR.")

    # Reproducibility knobs that matter for a wave problem: TF32 matmuls silently
    # cost ~3 decimal digits, which is exactly the tolerance the gradient check
    # of §11.2 step 9 is asserting at.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    env = Env(repo=repo, data=data, device=device, platform=platform,
              gpu_name=gpu_name, persistent=persistent, notes=notes)
    if verbose:
        print(env)
    return env
