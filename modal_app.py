"""
Headless launcher for the long jobs.  Optional -- everything here can also be run
by opening the notebooks on any GPU machine.

The notebooks in `notebooks/` are Modal-agnostic by design (see `bootstrap.py`):
no `modal.App`, no `.remote()`, no decorators.  This file is the other half of
that arrangement.  It does two kinds of thing, and the split is deliberate:

  * the *compute* jobs -- solver validation, dataset generation, training --
    call `src/` directly.  They are hours to days long, they need per-unit
    granularity so a single 24 h container can finish one of them, and they
    produce artefacts (HDF5 splits, checkpoints) rather than figures.

  * the *analysis* jobs -- forward evaluation, inversion statistics, transfer
    and the detector -- execute the corresponding notebook with nbconvert and
    leave the executed copy, with all of its figures, on the Volume.  Their
    value *is* the figures, and re-implementing them here would create a second
    version of the analysis that could silently disagree with the first.

Everything reads and writes one Modal Volume, mounted at a path
`bootstrap.resolve_data_dir` already knows about, so the notebooks and these
functions see the same `datasets/`, `checkpoints/`, `figures/` and `results/`.

Usage
-----
    modal run modal_app.py::solver_checks
    modal run modal_app.py::generate_all
    modal run modal_app.py::generate_split --split train
    modal run modal_app.py::train_arm --arm full
    modal run modal_app.py::train_both
    modal run modal_app.py::forward_eval
    modal run modal_app.py::inversion_stats
    modal run modal_app.py::transfer_and_detector

    modal run modal_app.py                      # the whole pipeline, in order

Any notebook can also be run headlessly with its knobs overridden:

    modal run modal_app.py::run_notebook --name 05_inversion --flags QUICK=False
    modal run modal_app.py::run_notebook --name 03_train_fno --flags SMOKE=False

Pull the results back out with the Volume CLI:

    modal volume ls   fno-wave-inverse-data
    modal volume get  fno-wave-inverse-data figures ./figures
    modal volume get  fno-wave-inverse-data results ./results
    modal volume get  fno-wave-inverse-data runs    ./runs

Pick the GPU with an environment variable rather than editing this file:

    FNO_GPU=H100 modal run modal_app.py::train_both
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import modal

# bootstrap.py imports only the standard library, so REQUIRED is readable here
# without torch installed locally.  The image and a local venv are then pinned
# from one list instead of two that drift apart.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap import REQUIRED  # noqa: E402

LOCAL_REPO = pathlib.Path(__file__).resolve().parent
REMOTE_REPO = "/root/fno-wave-inverse"
DATA_MOUNT = "/vol/fno-data"          # one of bootstrap._DATA_CANDIDATES

GPU = os.environ.get("FNO_GPU", "A100")
HOUR = 60 * 60
MAX_TIMEOUT = 24 * HOUR               # Modal's ceiling

VOLUME_NAME = "fno-wave-inverse-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*[spec for _, spec in REQUIRED])
    # Only the analysis functions need these; they are small next to torch.
    .pip_install("jupyter", "nbconvert", "ipykernel", "nbformat")
    .env({
        "FNO_ROOT": REMOTE_REPO,
        "FNO_DATA_DIR": DATA_MOUNT,
        "MPLBACKEND": "Agg",          # nbconvert has no display
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        LOCAL_REPO, remote_path=REMOTE_REPO,
        # The image needs the source and the notebooks and nothing else.  A local
        # .venv is ~2 GB of wheels the image already has, and data/ can be tens
        # of GB of HDF5 -- both would be uploaded on every launch.
        ignore=["**/.venv/**", "**/venv/**", "**/data/**", "**/.git/**",
                "**/__pycache__/**", "**/.ipynb_checkpoints/**",
                "**/*.h5", "**/*.pt", "**/*.pyc", "**/.nbgen/**"],
    )
)

app = modal.App("fno-wave-inverse", image=image)

FN = dict(gpu=GPU, volumes={DATA_MOUNT: vol}, timeout=MAX_TIMEOUT)
# Fan-out coordinators wait on their children, so they need the same ceiling --
# but no GPU: all they do is block on `.starmap`.
DISPATCH = dict(volumes={DATA_MOUNT: vol}, timeout=MAX_TIMEOUT)


# ---------------------------------------------------------------------------
# Shared container-side setup
# ---------------------------------------------------------------------------
def _env():
    """The same `bootstrap.setup()` the notebooks call, with the Volume attached."""
    sys.path.insert(0, REMOTE_REPO)
    import bootstrap

    E = bootstrap.setup(install=False)
    assert str(E.data) == DATA_MOUNT, f"data went to {E.data}, not the Volume"
    assert E.persistent, "the Volume is not mounted -- results would be lost"
    return E


def _write(E, name: str, obj) -> None:
    """Headless records live in results/headless/ so they never clobber a
    notebook's own record for the same step."""
    d = E.results / "headless"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps(obj, indent=1, default=float))
    print("wrote", p)


# ---------------------------------------------------------------------------
# Compute jobs: straight into src/
# ---------------------------------------------------------------------------
@app.function(**{**FN, "timeout": 4 * HOUR})
def solver_checks(nu: float = 1.0 / 3.0):
    """§11.2 steps 1-5.  Nothing downstream is worth running until these pass."""
    E = _env()
    from src.solver import validate

    results = validate.run_all(device=E.device, include_slow=True, nu=nu)
    rec = {r.name: dict(passed=bool(r.passed), value=float(r.value),
                        gate=float(r.gate), units=r.units, detail=r.detail)
           for r in results}
    _write(E, "01_solver_validation.json",
           dict(device=E.device, gpu=E.gpu_name, nu=nu, checks=rec,
                n_pass=sum(r.passed for r in results), n_total=len(results)))
    vol.commit()
    return all(r.passed for r in results)


@app.function(**FN)
def generate_split(split: str, incident_first: bool = True):
    """One HDF5 split.  Skips a split that is already on the Volume, so a
    container that hits the timeout can simply be launched again."""
    E = _env()
    from src import config as cfg
    from src.data import generate as G
    from tqdm.auto import tqdm

    n = {"train": cfg.N_TRAIN, "val": cfg.N_VAL, "test": cfg.N_TEST}[split]
    p = E.datasets / f"{split}.h5"
    if p.exists():
        print(f"{p} exists -- skipping")
        return str(p)

    G.print_projection(n)
    inc = G.run_incident(device=E.device, progress=tqdm) if incident_first else None
    G.generate(str(p), n, split=split, device=E.device, incident=inc, progress=tqdm)
    vol.commit()
    print(f"{p}  {p.stat().st_size / 1e9:.2f} GB")
    return str(p)


@app.function(**FN)
def generate_all():
    """All three splits in one container, sharing one incident-field cache.

    The cache is computed once and passed to each split so the three files carry
    byte-identical incident data -- a case loaded from one split can then be
    compared against a model trained on another without a units mismatch.
    """
    E = _env()
    from src import config as cfg
    from src.data import generate as G
    from tqdm.auto import tqdm

    G.print_projection(cfg.N_TRAIN + cfg.N_VAL + cfg.N_TEST)
    inc = None
    out = {}
    for split, n in (("train", cfg.N_TRAIN), ("val", cfg.N_VAL),
                     ("test", cfg.N_TEST)):
        p = E.datasets / f"{split}.h5"
        if p.exists():
            print(f"{p} exists -- skipping")
            out[split] = str(p)
            continue
        if inc is None:
            inc = G.run_incident(device=E.device, progress=tqdm)
        G.generate(str(p), n, split=split, device=E.device, incident=inc,
                   progress=tqdm)
        vol.commit()
        out[split] = str(p)
        print(f"{split}: {p.stat().st_size / 1e9:.2f} GB")
    _write(E, "02_dataset_generation.json",
           dict(device=E.device, gpu=E.gpu_name, splits=out))
    return out


@app.function(**FN)
def train_arm(arm: str = "full", epochs: int | None = None, workers: int = 4):
    """One arm of the §11.2 step 8 ablation: `full` or `nophys`.

    One arm per container.  The full schedule is 8-16 h on an A100, so two arms
    in one call can exceed the 24 h ceiling -- and a container killed mid-arm
    loses that arm, because the cosine schedule is defined over the whole run
    and cannot be restarted from an epoch checkpoint without changing it.
    """
    E = _env()
    from src import config as cfg
    from src.models import fno2d
    from src import training
    from tqdm.auto import tqdm

    alpha = {"full": cfg.ALPHA_PHYS, "nophys": None}[arm]
    out = E.checkpoints / arm
    model = fno2d.build("primary")
    print(model.summary())
    hist = training.train(
        model, str(E.datasets / "train.h5"), str(E.datasets / "val.h5"),
        out_dir=str(out), device=E.device, epochs=epochs or cfg.EPOCHS,
        num_workers=workers, alpha=alpha, progress=tqdm)
    vol.commit()
    best = min(hist["val"], key=lambda r: r["rel_l2"])
    print(f"arm {arm}: best val rel_l2 {best['rel_l2']:.4f}")
    _write(E, f"03_train_{arm}.json",
           dict(device=E.device, gpu=E.gpu_name, arm=arm, alpha=alpha,
                epochs=epochs or cfg.EPOCHS, best=best))
    return best


@app.function(**DISPATCH)
def train_both(epochs: int | None = None, workers: int = 4):
    """Both ablation arms, as two containers in parallel rather than in series."""
    args = [("nophys", epochs, workers), ("full", epochs, workers)]
    return list(train_arm.starmap(args))


# ---------------------------------------------------------------------------
# Analysis jobs: execute the notebook, keep the figures
# ---------------------------------------------------------------------------
def _flip(nb: dict, flags: dict[str, str]) -> dict:
    """Rewrite top-level knob assignments (`QUICK = True` -> `QUICK = False`).

    Asserts every requested flag was found.  A silently-missed substitution
    would run a quick pass and label the output as a full one, which is worse
    than crashing.
    """
    hit = {k: 0 for k in flags}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        for i, line in enumerate(src):
            for k, v in flags.items():
                if line.startswith(f"{k} =") or line.startswith(f"{k}="):
                    tail = "\n" if line.endswith("\n") else ""
                    src[i] = f"{k} = {v}  # set by modal_app.py{tail}"
                    hit[k] += 1
    missing = [k for k, n in hit.items() if n == 0]
    assert not missing, f"flags not found in the notebook: {missing}"
    print(f"flags applied: {', '.join(f'{k}={v}' for k, v in flags.items())}")
    return nb


def _execute(name: str, flags: str = "", timeout_s: int = -1) -> str:
    """Execute `notebooks/<name>.ipynb` on the Volume and leave it there.

    The copy that runs lives on the Volume, so the executed version -- outputs,
    printed tables, tracebacks and all -- survives the container alongside its
    figures and its JSON record.  It is executed with the *repo* as the working
    directory, because the notebooks locate `bootstrap.py` by walking up from
    the working directory and `/vol` has no repo above it.
    """
    import subprocess

    E = _env()
    nbdir = pathlib.Path(REMOTE_REPO) / "notebooks"
    src = nbdir / f"{name}.ipynb"
    assert src.exists(), f"no such notebook: {src}"

    nb = json.loads(src.read_text(encoding="utf-8"))
    if flags.strip():
        pairs = dict(kv.split("=", 1) for kv in flags.split(",") if kv.strip())
        nb = _flip(nb, {k.strip(): v.strip() for k, v in pairs.items()})

    runs = E.data / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    dst = runs / f"{name}.ipynb"
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
           "--execute", "--inplace", f"--ExecutePreprocessor.timeout={timeout_s}",
           "--ExecutePreprocessor.kernel_name=python3", str(dst)]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(nbdir))
    vol.commit()
    print(f"executed notebook: {dst}  (exit {r.returncode})")
    if r.returncode != 0:
        raise RuntimeError(
            f"{name} failed; the partially-executed notebook is on the Volume at "
            f"runs/{name}.ipynb and holds the traceback")
    return str(dst)


@app.function(**FN)
def run_notebook(name: str, flags: str = "", timeout_s: int = -1):
    """Any notebook, headless.  `flags` is a comma-separated list of `NAME=VALUE`
    knob overrides, e.g. `QUICK=False`."""
    return _execute(name, flags, timeout_s)


@app.function(**FN)
def forward_eval():
    """§11.2 step 8b: the surrogate as a forward operator, and figures 2 and 3."""
    return _execute("04_forward_eval")


@app.function(**FN)
def inversion_stats():
    """§11.2 steps 9-11: the gradient check, the misfit landscape, and the
    success rate over the full case set and the SNR sweep."""
    return _execute("05_inversion", flags="QUICK=False")


@app.function(**FN)
def transfer_and_detector():
    """§11.2 steps 12-13: the CNN baseline, out-of-family transfer, the
    mismatch ROC, and the thesis table."""
    return _execute("06_transfer_and_detector", flags="QUICK=False")


# ---------------------------------------------------------------------------
# The whole thing, in dependency order
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(skip_solver: bool = False, skip_generate: bool = False,
         skip_train: bool = False, epochs: int = 0):
    """
    Run the pipeline in the order §11.2 requires, stopping if the solver fails.

    Each stage is a separate container, so a stage that hits the 24 h ceiling can
    be relaunched on its own without repeating the ones before it -- the dataset
    splits and the arm checkpoints are already on the Volume and are skipped.
    """
    if not skip_solver:
        print("=== 1-5  solver validation")
        if not solver_checks.remote():
            raise SystemExit(
                "solver checks failed -- stopping.  A surrogate trained on these "
                "labels would be an accurate model of the wrong operator.")

    if not skip_generate:
        print("=== 6    dataset generation")
        print(generate_all.remote())

    if not skip_train:
        print("=== 7-8  training, both ablation arms in parallel")
        print(train_both.remote(epochs=epochs or None))

    print("=== 8b   forward evaluation")
    print(forward_eval.remote())
    print("=== 9-11 inversion")
    print(inversion_stats.remote())
    print("=== 12-13 transfer and detector")
    print(transfer_and_detector.remote())

    print(f"\nDone.  Fetch the outputs:\n"
          f"  modal volume get {VOLUME_NAME} figures ./figures\n"
          f"  modal volume get {VOLUME_NAME} results ./results\n"
          f"  modal volume get {VOLUME_NAME} runs    ./runs")
