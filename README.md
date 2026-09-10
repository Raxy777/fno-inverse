# fno-wave-inverse

An FNO surrogate for 2-D plane-strain elastic wave scattering, used as a differentiable
forward model for gradient-based defect localisation. The surrogate predicts *receiver
fields* rather than a shape, so the inverse problem is solved by optimising a geometry
through it, and the same trained network can be handed a shape family it was never
trained on.

**Scope, stated up front.** The claim this repository is built to test is that a
field-predicting surrogate transfers across shape families in a way a direct
shape regressor does not. "Enables" is not "establishes": the transfer claim rests on
two gates that are checked in code and reported as numbers — autodiff agreeing with
finite differences *of the surrogate*, and the surrogate's receiver-field sensitivities
agreeing with finite differences of the *validated FDTD solver* — plus an evaluation of
the final recovered geometry with the reference solver. Where a claim is not yet backed
by a measurement, `Known open items` below says so.

Everything is non-dimensionalised: `rho = c_p = f_c = 1`.

## Layout

    src/config.py            every number, plus `self_check()` which re-derives them
    src/solver/              4th-order staggered velocity-stress elastic FDTD,
                             running DFT, analytic cavity reference, validation checks
    src/geometry/sdf.py      shape families, signed distances, chi and material fields
    src/features.py          input packing (frequency folded into the batch axis)
    src/models/fno2d.py      the surrogate
    src/models/cnn_regressor.py   the direct-regression baseline
    src/losses.py            data + physics residual losses
    src/inverse/             objectives, the four-stage inversion, sensitivity checks
    src/data/                dataset generation and loading
    tests/                   pytest suite (no GPU, no dataset required)
    notebooks/               generated — see below

The notebooks are **generated** from `../.nbgen`; editing an `.ipynb` in place is
reverted on the next build. They are also deliberately Modal-agnostic: no decorators,
no `.remote()`. `bootstrap.py` finds the device and a writable data directory whether
it is run on Modal, on Colab or locally, and `modal_app.py` is an optional headless
launcher that imports the same `src/`.

## Running it

    pip install -e ".[dev]"
    pytest -q                                   # 203 passed, 3 skipped, CPU, 15–35 s
    python -c "from src import config; config.self_check()"
    python -m src.solver.validate                # solver checks; slow ones on CUDA only

Then notebooks 01–06 in order, or `00_master.ipynb`, which is all of them concatenated.
Notebook 01 must pass before generating a dataset: the network learns whatever the
solver does, including its bugs.

## Numeric gates

Every pass/fail threshold lives in `config.py` as a `GATE_*` constant, so a test, a
notebook and this file cannot disagree about what passing means. `src/solver/validate.py`
runs ten checks:

| # | check | gate |
|---|-------|------|
| 1 | energy drift, absorber off | `GATE_ENERGY_DRIFT = 5e-3` |
| 2 | P/S arrival error | `GATE_ARRIVAL_STEPS = 1.0` steps, vs this scheme's own arrival (deviation 13) |
| 3 | absorber residual energy | `GATE_ABSORBER_RESIDUAL = 1e-4` |
| 4 | Rayleigh amplitude slope | internal fit, see deviation 11 |
| 5 | grid convergence, label floor | `GATE_GRID_CONVERGENCE = 0.02` |
| 6 | incident field vs Green's function | `GATE_GREEN_REL_L2 = 0.05` |
| 7 | void vs traction-free cavity | `GATE_CAVITY_REL_L2 = 0.12`, `GATE_CAVITY_PHASE_RAD = 0.20` |
| 8 | interface width vs grid | separates the two refinements 7 conflates |
| 9 | absorber vs open domain | `GATE_ABSORBER_REFLECTION = 2e-2` |
| 10 | physics residual on solver labels | `GATE_PHYS_RESIDUAL_LABEL = 0.15` |

Checks 6–10 exist because the review that prompted this pass found that the forward
model had never been compared against anything external, and that the physics
regulariser had never been measured at all. 4, 5, 8 and 9 sit behind `include_slow`.

Downstream: `GATE_REL_L2 = 0.05` and `GATE_ARRIVAL_PERIODS = 0.05` on the surrogate,
`GATE_GRAD_SIGFIGS = 3` on autodiff vs FD of the surrogate, `GATE_SENSITIVITY_REL = 0.20`
and `GATE_SENSITIVITY_COSINE = 0.95` on surrogate vs solver sensitivities,
`GATE_SOLVER_VERIFY_LS = 0.15` on the recovered geometry re-solved with the FDTD,
`GATE_SOLVER_VERIFY_RESIDUAL_RATIO = 2.0`, `GATE_SOLVER_VERIFY_AXIS_RATIO = 0.15`,
`GATE_SOLVER_VERIFY_ORIENTATION_DEG = 10.0` on the shape criteria returned by the solver
verification (position, residual and shape are reported separately, with `"incomplete"` when
ground truth is absent),
`GATE_POSITION_LS = 0.10` and `GATE_IOU = 0.80` per case, `GATE_SUCCESS_RATE = 0.90`
over cases, `GATE_SCREEN_CAPTURE = 0.90` for the magnitude screen's capture rate, and
`GATE_LOF_FPR = 0.10` for the lack-of-fit indicator at a threshold frozen before testing.

## Documented deviations

Places where the implementation deliberately does something other than what the design
documents (`../fno_wave_inverse_architecture.md`, `../fno_wave_inverse_data_pipeline.md`)
specify. Each one is a decision with a measurement or a derivation behind it, not an
oversight, and each is repeated as a comment at the site.

**1. Time step and `n_t`** (`config.py`). §3.3 states a CFL safety factor of 0.9 and then
quotes `dt = 0.6 dx/c_p`; those disagree. The 4th-order stencil's 2-D von Neumann limit
is `6/(7 sqrt2) = 0.60609`, so the quoted `dt` sits at 99% of it — and the remaining
margin is exactly what averaging moduli across a void interface consumes. The stated
safety factor wins: `CFL_NUMBER = 0.5455`, `NT` 1280 → 1408. `T_END = 24` is unchanged,
so every frequency-domain quantity (which depends on `T_END`, not `n_t`) is unaffected.

**2. The void keeps a small density** (`geometry/sdf.py`, `losses.py`). The documents
retain full density inside the void and floor only the stiffness. That is not a small
deviation: the interior becomes a bag of nearly free masses whose first layer rides on
the interface as an added mass, loading it by `k_s dx = 0.43` at `f_max`. Measured on
the receiver ring against the analytic cavity at the production interface width: **76.7%
relative error with density retained, 9.3% without**, with the fitted amplitude ratio
dropping to 0.71 — a different scatterer, not a slightly wrong cavity. So
`VOID_DENSITY_SCALE = 1e-2` alongside `VOID_STIFFNESS_FLOOR = 1e-4`, giving a residual
impedance of `sqrt(1e-4 · 1e-2) = 1e-3` of the host. `1e-4` was tried and the solver
diverges (step 200 of 1408) because the 4th-order stress stencil reaches two cells and
combines a near-zero density with full-strength stiffness. `losses.py` reads the same
two constants, so `delta_rho` and `delta_C` in the differential residual describe the
medium the solver actually stepped.

**3. The absorber is a graded sponge, not an elastic PML** (`config.py`,
`solver/fdtd_elastic.py`). The documents call for a PML; the implementation is a graded
polynomial sponge, and it is named that way — `ABSORBER_KIND = "graded sponge"`, with
`N_PML_*` and `PML_*` kept only as deprecated aliases. A sponge is not a matched layer,
and the consequence is measurable: WKB round-trip attenuation equals `R_target`
*independently* of thickness and order, so lowering `R_target` only steepens the damping
profile, and it is the gradient of that profile an unmatched layer reflects. `R_target`
therefore has an interior optimum; `ABSORBER_R_TARGET = 3e-2`, `ABSORBER_ORDER = 4`,
`N_ABSORBER_FINE = 60` (1.875 = 1.24 `lambda_p(f_lo)` thick) came out of a sweep against
the open-domain reference of check 9. The cost is real: the padded grid is 376² rather
than 316², so every solve is **1.42×** the pre-redesign cost.

**4. Which objective each stage optimises, and what "envelope" means**
(`inverse/misfit.py`, `inverse/timedomain.py`, `config.STAGE_OBJECTIVE`). v2.0 screened
on `|G|` vs `|d|` — the magnitude spectrum — and defended it as "phase-free by
construction", the property supposed to make an envelope misfit immune to cycle
skipping. That inverts the mathematics. For a trace delayed by `tau`,
`|ĝ_tau(omega)| = |ĝ(omega)|` *exactly*, so a magnitude-spectrum misfit is not
phase-free but travel-time **blind**: it cannot distinguish a defect from the same
defect moved anywhere along a locus of equal scattering amplitude, which is precisely
what a position screen must decide. `timedomain.shift_invariance_demo` proves it
executably. The objective in use is the modulus of a consistently band-limited analytic
signal reconstructed from the phasors, which is shift-*equivariant* and does carry
arrival time. The magnitude screen survives as `spectral_magnitude_misfit`, named as the
heuristic it is, kept because it costs one `abs()` instead of an `[M, 128]`
reconstruction, and no longer allowed to claim an envelope's basin — its capture rate is
measured against the envelope screen's by `invert.screen_capture_rate`
(`GATE_SCREEN_CAPTURE`). Any guaranteed-coverage language is gone.

The pipeline is declared in exactly one place, `cfg.STAGE_OBJECTIVE`, because v2.0 had it
written down in three that disagreed (the documents said envelope refinement in stage 2;
the code ran a complex misfit):

    stage 1  screen, m = 1..6     envelope
    stage 2  Adam,   m = 1..10    envelope
    stage 3  L-BFGS, m = 1..20    complex

**5. FNO spectral weights are initialised zero-mean** (`models/fno2d.py`). The reference
implementation uses `scale * rand`, which is supported on `[0, 1]`, so every coefficient
of every mode starts with mean `scale/2`: the initial operator is a fixed non-random
multiplier added to noise. It trains anyway, but it puts a systematic direction into
every mode at step zero, and this network is later asked for a *gradient* through that
operator. Initialisation is `(rand * 2 - 1) * scale`.

**6. The physics residual's discretisation is stated, not assumed** (`losses.py`). v2.0's
residual left its relationship to the FDTD that produced the labels unexamined, so its
floor was unknown — and a residual that disagrees with the solver penalises the network
for the solver's own truncation error. The two are still not the same operator and are not
claimed to be: the solver is 4th-order *staggered* on the 256² fine grid, the residual is
4th-order *collocated* on the 128² network grid, taken as the divergence of the stress
rather than expanded into constant-coefficient form so that the void's coefficient jump is
differentiated where it actually is. What changed is that the gap is now measured instead
of assumed away. Check 10 evaluates `physics_loss` on labels the solver itself produced,
which is the only way to know the floor (`GATE_PHYS_RESIDUAL_LABEL = 0.15`), and `alpha`
is set by `balance_alpha` from a gradient-norm ratio — a statement about influence rather
than units — because a fixed weight would either be swamped by that floor or spend
capacity fitting discretisation error. Two regions are excluded: three cells at each edge
(`ERODE_CELLS = STENCIL_HALF_WIDTH + 1`, since the coefficient fields are differentiated
too), and a disc of `PHYS_SOURCE_EXCLUDE_CELLS = 3.0` cells around the source, where the
true residual is a delta function.

Related: the default weight `1 - chi` is *zero on the void boundary*, which is the only
place the scattering problem is posed. `interface_weight` restores a band there, built by
dilating `4 chi (1 - chi)` over `STENCIL_HALF_WIDTH = 2` cells rather than using it
directly — at the production interface width `chi` crosses from 0.03 to 0.97 inside one
network cell, so undilated the knob would change the weight sum by nothing at all. It
defaults to 0 because the residual in that band is genuinely large.

**7. Success is two gates, so success rates are not comparable with v2.0's**
(`inverse/invert.py`). A position error alone cannot fail a circle fitted to an ellipse:
place the circle at the ellipse's centre and the position error is *zero*. `success` is
now `position_error_ls < GATE_POSITION_LS` **and** `iou() > GATE_IOU`, with
`success_rate_position_only` carried alongside so the two definitions can be told apart.
Blob matching is permutation-invariant (a two-void truth has no canonical order), IoU
compares each shape through *its own* family's SDF, and cross-family position and size
comparisons go through a named, documented equal-area-circle reduction
(`geometry/sdf.equivalent_circle`) rather than reading one family's `theta` with another
family's column layout.

**8. "Lack-of-fit indicator", not a "model-mismatch detector"** (`inverse/invert.py`). The
old name claimed the statistic could attribute a large residual to model error. It cannot:
a large residual means the model does not fit, and out-of-family geometry, an unmodelled
material, a surrogate error and bad luck with noise all produce one. Four things changed
with the name. The statistic is the **data residual only**, with the regulariser excluded,
because a penalty term is not evidence about the data. Noise normalisation uses a
**nonzero model-error floor** (`LOF_MODEL_FLOOR`), since dividing by measurement noise
alone drives the statistic to infinity as noise falls and would flag every clean
measurement. The threshold is **frozen on in-family calibration cases before any
out-of-family case is scored**, and `GATE_LOF_FPR = 0.10` is a false-positive rate on
*held-out in-family* cases, which is the control that makes a true-positive rate mean
anything. The ROC curve is reported as a diagnostic and is never the operating point.

**9. Fourier sign, outgoing convention, and two time grids** (`solver/harmonic.py`,
`solver/fdtd_elastic.py`, `solver/cavity.py`). Fixed and stated, because the documents
leave them implicit and every one of them is a sign or a half-step somewhere. The
convention is `exp(-i omega t)`, so `d/dt` is `+i omega`, outgoing waves are
`H^{(2)}`, and the 2-D scalar Green's function is `g_alpha(r) = -(i/4) H_0^{(2)}(k_alpha r)`.
Two time grids coexist and must not be merged: velocities live at `(n + 1/2) dt`, so the
recording DFT uses `t_offset = 0.5`, while the point force enters the velocity update
centred on `t = n dt`, so `source_spectrum` uses `t_offset = 0.0`. Getting that wrong
leaves `exp(+i omega dt/2)` on every phasor — 0.072 rad at `f_max`, a 6% error that looks
exactly like a uniform wave-speed error, which is the one quantity the inversion measures.
`tests/test_dft_consistency.py` pins it.

**10. The baseline and the pipeline are scored by one function** (`models/cnn_regressor.py`,
`inverse/invert.py`). v2.0 compared them with two different pieces of code, so
"position error" and "success" each meant two things and the headline comparison was
between two *measurements* rather than two *estimators*. `cnn_regressor.score` now wraps
each prediction in the same `InversionResult` the inversion returns and hands the list to
`invert.summarise`, so both columns of every table in notebook 06 come out of the same
arithmetic. `InversionResult` and `InverseCase` carry a `truth_family`, which is what makes
an exact cross-family IoU possible: the reduction to an equal-area circle is used for
position and size, where it is unavoidable, and *not* for shape, where it would hide the
whole effect. The numbers show why that matters. A circle placed at the exact equal-area
circle of a 2.5:1 ellipse scores position error 0.0000 and radius error 0.0000 — and
**IoU 0.560**, a fail against `GATE_IOU = 0.80`; against two touching voids, IoU 0.310.

**11. "Rayleigh validation" is a scaling-consistency fit, not an amplitude reference**
(`solver/validate.py`, check 4). The check fits the log-log slope of ring-collected
scattered energy against void radius over four radii and gates it into
`slope_window = (3.3, 4.7)` around the theoretical `R^4`, and separately requires S/P
scattered energy above 1% so that mode conversion is demonstrably present. Both numbers
come out of the solver's own output: nothing external enters, so the check can catch a
slope of 2 or 6 — which is what broken interface averaging or a mis-normalised soft
indicator produces — and cannot certify the coefficient. It is also boxed in by its own
premises, needing `kR << 1` and `R >> dx` at once; even on the 4×-refined grid the radii
span `kR` of 0.6 to 1.3, where the next term of the long-wavelength expansion is worth tens
of percent. The gate window is wide for that reason and the docstring says so. The
absolute-amplitude comparison the review asked for is not this check: it is check 6
against the analytic Green's function and check 7 against the traction-free cavity, both at
production radii. A Rayleigh-regime *coefficient* check is still open — see below.

**12. The envelope convention, and the rest of check 2's 57.5 steps**
(`solver/harmonic.py`, `solver/validate.py`). Arrival picking used
`|analytic(sqrt(vx^2 + vy^2))|`. That is not an envelope: rectifying is nonlinear, so it
moves the spectrum to DC and `2 f_c`, and the Hilbert transform of that does not modulate
anything. `vector_envelope` takes the analytic signal of each **signed** component and then
the norm, which is linear and therefore commutes with the rotation from `(vx, vy)` to
radial/transverse — the pick does not depend on the frame the receiver reports in. Measured
on a synthetic two-phase trace with exactly known arrivals (a 2-cycle Hann burst at `d/c_p`
and another at `d/c_s`), rectify-then-analytic misplaces the peaks by **16–18 steps**
against a gate of 1.0, while `vector_envelope` lands within **0.5**.
`tests/test_dft_consistency.py` pins both directions — the correct convention passes the
gate, and the rectified one misses it by more than 10×.

That was the largest of five distinct causes behind the 57.5-step P error this check
reported for several revisions. The wave speeds were never wrong; the check was, in five
places, and each was measured separately rather than absorbed into a widened gate:

| cause | worth | fix |
|---|---|---|
| rectify-then-analytic envelope | 16–18 steps | `vector_envelope` |
| timing P in the P radiation node | 57.5 → 17.5 steps | pre-registered `pattern_min = 0.3` |
| source position from grid indices | 0.92 steps P, 1.83 S | `_force_xy` / `_ring_xy` |
| the scheme's own group velocity | 0.83 steps P, 1.36 S | `_dispersive_arrival`, deviation 13 |
| integer argmax quantisation | ±0.5 steps | parabolic sub-sample refinement |

The radiation-node one is the reason the number was so large and stayed so long. The source
is a **vertical point force**, so its far-field P amplitude goes as `|cos t|`; the eight
receivers on the source's own grid row sit at `t = 90°` exactly and receive no P wave at
all, so the argmax in their P window was timing the S skirt. The three validity conditions
(radiation pattern, P/S separability, far field) are stated from the geometry before the
solve and excluded receivers are reported, not scored. `GATE_ARRIVAL_STEPS` is still 1.0 and
the check now reads **0.489 steps**.

**13. Check 2's reference is the arrival this discretisation produces**
(`solver/validate.py`, `_stencil_wavenumber` / `_numerical_wavenumber` /
`_dispersive_arrival`). Comparing an envelope peak against the continuum `d/c + N_c/(2 f_c)`
on this grid reads **1.57 steps**, and a check that cannot pass measures nothing. So the
gated prediction is obtained by propagating the same burst through this scheme's own
dispersion relation. The 4th-order staggered first difference with `(c1, c2) = (9/8, -1/24)`
has symbol `K dx = 2[c1 sin(t/2) + c2 sin(3t/2)] = t - 0.0046875 t^5 + O(t^7)`, and with
2nd-order leapfrog the 2-D relation is
`(2/dt) sin(w dt/2) = (c/dx) hypot(K(k n_y dx), K(k n_x dx))`, giving
`v_g/c = 1 + nu^2 t^2/8 - 0.046875 t^4`. At this grid's points-per-wavelength the time
term wins below about `1.7 f_c`, so the discrete wave is **fast** and the predicted peak
moves *early* — up to 0.83 steps for P and 1.36 for S over the scored receivers.

Nothing in that is fitted and no recorded trace is read: the stencil coefficients, `dt`,
`dx`, `N_c` and the burst are all fixed before the solver runs, which is what separates a
prediction from a calibration. Both continuum-reference errors are reported in `extras` as
`errs_p_ray` / `errs_s_ray` and in the `detail` string, ungated, because "does it travel at
`c_p`" is a fair question and the answer should not be buried. Two candidate explanations
were **discarded by measurement** first: the near-field group delay
(`t_g = r/c - c/(8 w^2 r)`, 0.079 steps) and the 2-D coda, which propagating the burst with
the exact `H_0^(2)` and exact `k` puts at 0.05 steps — two orders too small. The evidence
that the dispersive model is the right one is the residual's flatness: the signed error sits
at −0.26 to −0.40 steps across every distance and every angle, where before the
source-position fix the same column tracked both. What remains is a leftover convention
constant plus the fact that the relation is the scalar one applied per wave type, exact
along the grid axes and only approximate off them.

## Known open items

Things that are wired, gated and named but **not yet backed by a measurement**. Listed here
rather than left for a reader to discover, because the difference between "the gate exists"
and "the gate passed" is the whole difference between a claim and a result.

**No trained model in this checkout.** `data/datasets/` is empty and there is no GPU here,
so every gate downstream of the surrogate — `GATE_REL_L2`, `GATE_ARRIVAL_PERIODS`,
`GATE_GRAD_SIGFIGS`, `GATE_SENSITIVITY_REL`, `GATE_SENSITIVITY_COSINE`,
`GATE_SOLVER_VERIFY_LS`, `GATE_POSITION_LS`, `GATE_IOU`, `GATE_SUCCESS_RATE`,
`GATE_SCREEN_CAPTURE`, `GATE_LOF_FPR` — is a threshold with test coverage and no reported
number. The two sensitivity gates are the ones the headline claim rests on; until they are
run, the transfer claim is a hypothesis with a stated test, which is what the scope
paragraph at the top says.

**One baseline, not three.** The comparison is currently FNO-plus-inversion against a direct
ring regressor. Three arms the review asked for are still missing, and each removes a
different confound: a **non-Fourier U-Net** field surrogate on the same
`features.pack_inputs` channels with the same optimiser (does the transfer come from
predicting fields, or from the Fourier layers?); a **family-aware direct regressor** given
acquisition metadata (is the baseline losing on architecture, or on information?); and
**reference-solver inversion** on a small subset (how much of the error is the surrogate's).
Until those exist, "field inputs transfer where shape regression does not" is supported by
one contrast.

**No end-to-end runtime benchmark.** Preprocessing, per-iteration online cost, peak memory
and the break-even query count against direct FDTD inversion are unmeasured. The sizes that
make this a real question are known — six frequencies of input are ~302 MB, one float32
width-32 feature tensor is ~805 MB, and the absorber redesign makes every reference solve
**1.42×** its former cost — which is exactly why the benchmark should be a number rather
than an argument.

**No Rayleigh-regime coefficient reference.** Check 4 fits a slope (deviation 11); checks 6
and 7 compare amplitude and phase against analytic references at production radii. Nothing
yet compares a scattered *amplitude* against an analytic small-void coefficient in the
regime where the expansion is valid.

**The full ten-check table has not been produced in a single CPU run.** Checks 4, 5, 8 and 9
sit behind `include_slow` because they refine the grid or run an oversized open domain.
Check 4 is the one that is genuinely out of reach here: at `refine = 4` and
`l_domain = 3.0` its padded grid is 864² and it steps 3400 times for a batch of 5, which is
1.27e10 cell-steps — about **4 hours** at the ~8.8e5 cell-steps/s this machine sustains,
against 66 s for check 1. It is a CUDA check in practice. `python -m src.solver.validate`
reports whatever it ran, and a check that did not run is not a check that passed.

**The upstream design documents still carry superseded claims.**
`../fno_wave_inverse_architecture.md` and `../fno_wave_inverse_data_pipeline.md` are the
source of the "documented deviations" above, and the corrections have not been propagated
back into them: §5.3–5.4 leave the Fourier sign, normalisation and sampling implicit, §6.4
argues that propagating wavenumbers band-limit the whole scattered field (they do not — the
evanescent spectrum near the interface is not band-limited, and masked dense weights still
cost storage), §8.3 states the sensitivity anisotropy with its directions reversed (the long
valley is **transverse** to the ray, not along it), and §9 still specifies
circle-shaped metrics for non-circular truths. Where the documents and this file disagree,
this file and `config.py` are what the code does.
