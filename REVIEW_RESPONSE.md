# Response to the architecture review

What changed in the tree in response to `fno-architecture-review-by-astra.md`, and what
is still open. The review was a static read of a saved snapshot with nothing installed
and nothing run, so every finding was re-checked against the live code before acting —
two were already fixed and one was already correct as written, flagged below.

Status vocabulary, used consistently:

- **fixed + measured** — code changed and a number in the repo backs it
- **fixed, unmeasured** — the code and the gate exist, but `data/datasets/` is empty and
  there is no GPU in this checkout, so no number has been produced
- **open** — not done

`README.md` → `## Known open items` is the live list; this file is the change record.

---

## Blocking

### 1. The spectral-magnitude "envelope" objective — fixed + measured

`|ĝ(ω)|` is invariant under `g(t) → g(t−τ)`, so a magnitude-spectrum objective cannot
see a travel-time shift. Replaced with a time-domain envelope built from a consistently
band-limited analytic signal (`inverse/timedomain.py`, `Reconstruction`). The magnitude
comparison survives under its own name as `spectral_magnitude_misfit`, a **screening
heuristic** with its own capture-rate gate (`GATE_SCREEN_CAPTURE = 0.90`) rather than a
misfit; the guaranteed-coverage language is gone.

The plan/implementation conflict the review found in Stage 2 (plan said envelope, code
used complex waveform) is resolved by declaring the stage table exactly once —
`cfg.STAGE_OBJECTIVE = {1: envelope, 2: envelope, 3: complex}` with `misfit.STAGE_BANDS`
alongside it — so the notebook, the tests and the inversion cannot disagree about which
pipeline is being evaluated.

### 2. Field inputs *enable* shape transfer, they don't establish it — fixed, unmeasured

The claim is now stated as a hypothesis with a named test rather than as a result, in the
scope paragraph at the top of `README.md`. Three gates carry it: `GATE_GRAD_SIGFIGS = 3`
(autodiff vs finite differences **of the surrogate**), `GATE_SENSITIVITY_REL = 0.20` and
`GATE_SENSITIVITY_COSINE = 0.95` (surrogate receiver-field sensitivities vs finite
differences of the **validated solver**, swept over perturbation sizes), and
`GATE_SOLVER_VERIFY_LS = 0.15` (the final recovered geometry re-solved with the reference
FDTD). All three are thresholds with test coverage and no reported number.

### 3. Is the reference problem actually cavity scattering? — fixed + measured

Four new checks answer this against external references instead of internal consistency:
check 6 against the analytic Green's function, check 7 against a traction-free cavity,
check 9 absorber vs an oversized open domain, check 10 the physics residual on solver
labels. The void model is a soft light inclusion, not a cavity, and the residual that
remains is that modelling gap — stated in the README rather than tuned away by widening
the interface. The config/`validate.py`/`sdf.py` paper trail was rewritten with
post-absorber numbers.

---

## Important

### 4. Make the comparison fair — partly open

"Structurally impossible" is softened throughout: on in-distribution circles the direct
regressor is expected to be competitive or better, and `cnn_regressor.py`'s docstring says
so. The regressor is conditioned on source *position* rather than source index, so the
held-out-source test can be run on it at all — a fairness fix the review did not ask for
but which the same argument requires.

Still **open**: the three extra arms. A non-Fourier U-Net field surrogate on the same
`features.pack_inputs` channels and optimiser (dropped by request); a family-aware direct
regressor given acquisition metadata; reference-solver inversion on a small subset.
`README.md` says "one baseline, not three" in as many words.

### 5. Separate shape recovery from model-mismatch detection — fixed

Renamed to a **lack-of-fit indicator** — it no longer claims to detect model mismatch.
The threshold is frozen before testing, it has a false-positive-rate gate
(`GATE_LOF_FPR = 0.10`) with false-positive controls, it reads the **data residual with
the regulariser excluded** (a regularised residual conflates prior weight with misfit),
and the noise normalisation carries a nonzero model-error floor
(`LOF_MODEL_FLOOR = 0.0025`) so a low-noise case cannot be driven to a spurious detection.

### 6. Runtime is a benchmark question — open

Unmeasured: preprocessing, per-iteration online cost, peak memory, break-even query count
against direct FDTD inversion. The sizes that make it a real question are in the README
(six frequencies of input ≈ 302 MB; one float32 width-32 feature tensor ≈ 805 MB), and the
absorber redesign makes every reference solve **1.42×** its former cost, which is now
folded into that entry.

---

## Concrete repository findings

| # | finding | status |
|---|---|---|
| a | `invert.py` always built three parameters | fixed — parameter count follows the family |
| b | spectral magnitude instead of a time envelope | fixed, item 1 |
| c | absorber called a PML | fixed — `ABSORBER_KIND = "graded sponge"`, `N_PML_*` deprecated aliases |
| d | physics-loss and solver discretisations differ | fixed — stated as a deviation, not silently |
| e | void density differs from the plan | fixed — `VOID_DENSITY_SCALE = 1e-2`, residual impedance `1e-3`, with the reason |
| f | "Rayleigh validation" is an internal amplitude-slope fit | documented (deviation 11); an independent analytic reference is **open** |
| g | `Ellipse.sdf` degenerate at `a = b` | already fixed in the live tree |

Smaller corrections: §6.4's claim that propagating wavenumbers band-limit the whole
scattered field is wrong (the evanescent spectrum near the interface is not band-limited,
and masked dense weights still cost storage) — recorded; §5.3–5.4's Fourier sign,
normalisation, sampling and reconstruction conventions are now pinned in code
(`X(ω) = ∫x e^{−iωt}`, so `d/dt ↔ +iω` and outgoing is `H^(2)`), with the two staggered
time origins spelled out; §9's shape-appropriate metrics exist (IoU, permutation-invariant
two-defect matching, axis-ratio and orientation errors in `summarise`). **§8.3 was already
correct in the repo** — the review's own text reverses its directions; the long
sensitivity valley is *transverse* to the ray.

---

## Not in the review: check 2

Fixing item 3 exposed that solver check 2 (P/S arrival times) had been reporting **57.5
steps against a 1.0-step gate** for several revisions. It now reads **0.489 steps** with
`GATE_ARRIVAL_STEPS` untouched at 1.0. Five separate defects, all in the check and none in
the propagator, each measured on its own — the table is deviation 12 in `README.md`:

| cause | worth | fix |
|---|---|---|
| `\|analytic(sqrt(vx²+vy²))\|` is not an envelope | 16–18 steps | `H.vector_envelope` |
| timing P where a vertical point force radiates none | 57.5 → 17.5 steps | `pattern_min = 0.3` |
| source–receiver offsets from grid indices | 0.92 steps P, 1.83 S | `_force_xy` / `_ring_xy` |
| the scheme's own group velocity | 0.83 steps P, 1.36 S | `_dispersive_arrival` |
| integer argmax | ±0.5 steps | parabolic sub-sample refinement |

The gate is now read against the arrival **this discretisation** produces, derived from the
stencil coefficients `(9/8, −1/24)`, `dt`, `dx` and `N_c` with nothing fitted and no
recorded trace read. The continuum `d/c` comparison reads 1.57 steps — a gate on it could
never pass on this grid — and is reported alongside, ungated, in `extras` as
`errs_p_ray` / `errs_s_ray`. Two tempting physical explanations were killed by a-priori
calculation rather than another solve: the near-field group delay (0.079 steps) and the
2-D trailing coda (0.05 steps, two orders too small). Deviation 13 in `README.md` carries
the derivation.

## Files touched

- `src/solver/validate.py` — the three dispersion functions, check 2's physical offsets and
  dual-reference scoring, extended `extras`/`detail`, rewritten docstrings; a comment on
  why check 4's index arithmetic is legitimate where check 2's was not
- `src/inverse/`, `src/losses.py`, `src/config.py`, `src/geometry/sdf.py` — items 1, 5, and
  findings a–e above
- `README.md` — deviations 12 and 13, the gate table, `## Known open items`
- `../.nbgen/nb01.py`, `../.nbgen/master/part2_stageA.py` — check 2's prose, its figure
  (open markers continuum, filled markers this scheme) and its troubleshooting entry;
  notebooks regenerated, **never edited in place**
- 21 `_scratch_*` diagnostic scripts deleted

`pytest -q`: **203 passed, 3 skipped**.

## The check table as it stands

`python -m src.solver.validate` on this CPU, `include_slow=False`, **6/6 passed**:

| check | value | gate |
|---|---|---|
| 1 energy drift, absorber off | 0.0 rel | 5.0e-03 |
| 2 P/S arrival error | 0.489 steps | 1.0 |
| 3 absorber residual energy | 4.66e-06 rel peak | 1.0e-04 |
| 6 Green's function, absolute | 2.56e-02 rel-L2 | 5.0e-02 |
| 7 void vs traction-free cavity | 9.35e-02 rel-L2 | 1.2e-01 |
| 10 physics residual on labels | 5.96e-02 rel residual | 1.5e-01 |

Check 6 also reports the conjugated comparison at **140.7%** — the Fourier and
outgoing-wave conventions are the right way round, which checks 7 and 10 rest on. Check 7
is the cavity-validity number the review asked for: 7.8–9.3% across the production radius
range, worst 17.5% at the top of the band. Checks 4, 5, 8 and 9 did not run — see below.

## Still open

1. Two of the three extra baselines from item 4 (the U-Net arm was dropped by request).
2. The end-to-end runtime benchmark, item 6.
3. An independent analytic Rayleigh-regime coefficient reference, finding (f).
4. Every gate downstream of the surrogate is unmeasured — no dataset and no GPU here. The
   two sensitivity gates are the ones the headline claim rests on.
5. Checks 4, 5, 8 and 9 have never run in one pass on this machine: check 4 alone is
   ~1.27e10 cell-steps, about 4 hours at the ~8.8e5 cell-steps/s this CPU sustains. Two
   figures in `check_interface_width`'s part B (`20.7% → 20.9%`, `84.4% → 84.6%`) and one
   line in `config.py` still quote pre-absorber numbers for that reason.
6. The two upstream design documents still carry the superseded claims — §5.3–5.4, §6.4,
   §8.3 and §9 — and the corrections have not been propagated back into them.
