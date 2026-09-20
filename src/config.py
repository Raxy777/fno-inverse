"""
Single source of truth for every number in the architecture document.

Nothing else in the repository is allowed to hard-code a physical or numerical
constant.  If a number appears twice, it appears here once and is imported.

Run this file directly to print the full sizing table and execute every
self-consistency assertion:

    python -m src.config

That is the "verification script" §3.5 implies: it recomputes the mode budget,
the CFL limit, the ppw table, the wrap-around bound and the parameter counts
from first principles, so the document's arithmetic is checked by code rather
than by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Non-dimensionalisation (§3.5)
#
# rho = c_p = f_c = 1  =>  lambda_p = 1 and T_p = 1.
# Every length below is in P-wavelengths, every time in carrier periods.
# ---------------------------------------------------------------------------
RHO0: float = 1.0
CP: float = 1.0
FC: float = 1.0

LAMBDA_P: float = CP / FC          # = 1
T_P: float = 1.0 / FC              # = 1
OMEGA_C: float = 2.0 * math.pi * FC

# ---------------------------------------------------------------------------
# Material range (§3.5, §5.5)
# ---------------------------------------------------------------------------
NU_LIST: tuple[float, ...] = (0.25, 0.29, 0.33, 0.37)


def cs_over_cp(nu: float) -> float:
    """c_s/c_p = sqrt((1-2nu) / (2(1-nu))).  Bijective on nu in (-1, 1/2)."""
    return math.sqrt((1.0 - 2.0 * nu) / (2.0 * (1.0 - nu)))


def lame_from_nu(nu: float) -> tuple[float, float]:
    """
    (lambda_0, mu_0) for rho=1, c_p=1.

    c_p^2 = (lambda + 2 mu) / rho = 1  =>  lambda + 2 mu = 1
    c_s^2 = mu / rho                   =>  mu = c_s^2
    """
    mu = cs_over_cp(nu) ** 2
    lam = 1.0 - 2.0 * mu
    return lam, mu


# The worst case for every resolution requirement is the *smallest* c_s/c_p,
# i.e. the largest nu.  §3.5: sizing on lambda_p instead is the standard way to
# silently destroy a wave FNO.
NU_WORST: float = max(NU_LIST)
CS_MIN: float = cs_over_cp(NU_WORST)
LAMBDA_S_MIN: float = CS_MIN / FC       # shortest wavelength in the problem

# ---------------------------------------------------------------------------
# Domain and grids (§3.5)
# ---------------------------------------------------------------------------
DOMAIN_IN_LAMBDA_P: float = 8.0
L_DOMAIN: float = DOMAIN_IN_LAMBDA_P * LAMBDA_P     # = 8

N_NET: int = 128                                    # network grid
N_FINE: int = 256                                   # FDTD physical grid
DX_NET: float = L_DOMAIN / N_NET                    # = lambda_p / 16
DX_FINE: float = L_DOMAIN / N_FINE                  # = lambda_p / 32
DOWNSAMPLE: int = N_FINE // N_NET                   # = 2

# ---------------------------------------------------------------------------
# Absorbing layer (§3.6)
#
# The document says "15 cells".  That is 15 cells at the *network* resolution;
# at the solver resolution (dx twice as fine) the same physical thickness is 30
# cells, which is what the solver needs.  Expressed as a thickness this is
# ~1 lambda_p, which is the honest way to state it.
#
# WHAT THIS LAYER IS.  It is a graded sponge: a polynomial-profile damping term added
# to the velocity update inside the layer (`solver/fdtd_elastic.py:absorber_profile`).
# It is NOT a perfectly matched layer.  A split-field or CFS PML stretches the
# coordinate in the complex plane and is reflectionless for all angles and all
# frequencies at the continuous level; a sponge is not matched at all, and its
# reflection coefficient depends on the grading profile, the layer thickness in
# wavelengths, and the angle of incidence.  R_TARGET below is the value the *design
# formula* for a graded damping profile aims at -- it is a design input, not a
# measurement, and `solver/validate.py:check_absorber_reflection` measures the artefact
# the layer actually leaves and compares it against GATE_ABSORBER_REFLECTION instead of
# assuming the formula.  It does that by differencing the production acquisition against
# the same acquisition in a padded open domain, rather than by fitting a normal-incidence
# reflection coefficient: sources sit RING_INSET_NET cells from the layer, so it is
# struck at every angle out to grazing, and grazing is where a sponge is worst.
# (`check_absorber` is a different and weaker test: it measures how much energy is left
# in the domain at t_end, which a layer could pass while still reflecting, so both are
# run.)
#
# The names ABSORBER_* are primary.  The PML_* spellings are retained as aliases
# because they appear throughout the solver and the notebooks, but nothing in this
# project implements a PML and the docstrings no longer claim one.
#
# Consequence for scope: an absorbing outer boundary models an *open domain* -- an
# unbounded medium containing one defect.  It does not model the reflecting edges of a
# finite specimen, and any inversion result here is a result about the open-domain
# problem.  Stated once, here, so §1.4's scope limits are complete.
#
# THE NUMBERS BELOW ARE MEASURED, NOT DESIGNED.  v2.0 used 30 fine cells, p = 2,
# R_target = 1e-4, on the reasoning that a smaller R_target is a better layer.  Against
# an open domain (the production acquisition with the physical region padded by 208 fine
# cells, converged in the padding) that layer leaves **22.3%** on the incident ring field
# and 6.3% on the scattered field -- 11x over GATE_ABSORBER_REFLECTION, and at the time
# the largest single term in check 7's cavity error budget.  Its per-frequency signature
# falls 31.0% -> 15.3% across the band, which is what a layer too thin at the *low* end
# looks like: 30 cells is 0.62 lambda_p at FREQS[0].
#
# 42 designs were measured on that reference (`validate.check_absorber_reflection` is
# the same measurement, kept as check 9).  Incident-field artefact, rel-L2:
#
#     thickness, at R_target = 1e-2:
#        cells  lam_p(f_lo)  cost    p=4    p=5    p=6
#           36        0.74   1.08x  8.47%  8.07%  8.11%
#           44        0.91   1.19x  4.77%  4.41%  4.46%
#           52        1.07   1.30x  2.47%      -      -
#           60        1.24   1.42x  1.20%  1.05%      -
#
#     60 cells, the two knobs jointly:
#              R=3e-1  1e-1   3e-2   1e-2   1e-3   1e-4
#        p=3        -     -  1.57%  2.00%  3.01%  3.92%
#        p=4    8.30% 2.21%  1.02%  1.20%  1.84%  2.42%
#        p=5        -     -  0.97%  1.05%  1.51%  1.93%
#
# Three findings, none of them guessable from the design formula.  (1) Thickness in
# wavelengths dominates everything and saturates near 1.2 lambda_p(f_lo); below ~1
# lambda_p no choice of p or R_target reaches the gate, because 44 cells bottoms out at
# 3.7% and 36 at 8.1%.  (2) Order buys about a factor of two at fixed thickness and then
# saturates too (p = 6 is no better than p = 5).  (3) R_target has an interior *minimum*,
# and the production value was on the wrong side of it.  d_0 ~ -ln(R_target), so a
# smaller target steepens the profile, and for an unmatched layer it is the gradient of
# d that reflects; the WKB round-trip attenuation is R_target independently of thickness
# and order, so lowering it buys transmission the layer does not need while paying in
# reflection.  The two mechanisms cross near 3e-2 here, and the crossing moves *up* as
# the layer thins (44 cells is best at 1e-1), exactly as that reading predicts.
#
# Chosen: 60 fine cells, p = 4, R_target = 3e-2 -> 1.02% incident, per-frequency 2.0% ->
# 0.6%, i.e. within a factor of two of the gate at every frequency instead of failing at
# all of them.  Both knobs are on the flat bottom of the table above (p = 5 is 0.05
# points better, which is inside the reference's own convergence), so the choice is not
# perched on a cliff.  The cost is (256 + 120)^2 / 316^2 = 1.42x on every solve in the
# project, which is charged to dataset generation, physics-loss evaluation and every
# validation check; it is paid because a 22% boundary artefact is not a small
# perturbation of a scattering label, and because `data/datasets/` was empty when this
# changed, so no existing label had to be regenerated.  N_ABSORBER_FINE, ABSORBER_ORDER
# and ABSORBER_R_TARGET are snapshotted into every dataset (`data/generate.py`).
# ---------------------------------------------------------------------------
N_ABSORBER_NET: int = 30
N_ABSORBER_FINE: int = N_ABSORBER_NET * DOWNSAMPLE   # = 60
ABSORBER_ORDER: float = 4.0
ABSORBER_R_TARGET: float = 3.0e-2                    # design target, not a measurement
N_FINE_TOTAL: int = N_FINE + 2 * N_ABSORBER_FINE     # = 376

# What the layer *is*, as a string, so that any figure caption, table or paper
# sentence generated from the config says "sponge" and cannot say "PML" by
# accident.  Checked by tests/test_config.py.
ABSORBER_KIND: str = "graded sponge"

# Deprecated aliases -- same objects, honest primary names above.
N_PML_NET: int = N_ABSORBER_NET
N_PML_FINE: int = N_ABSORBER_FINE
PML_ORDER: float = ABSORBER_ORDER
PML_R_TARGET: float = ABSORBER_R_TARGET

# ---------------------------------------------------------------------------
# Time stepping (§3.3)
#
# The CFL limit is derived, not quoted.  For the 4th-order staggered operator
# the symbol max is |D~|_max = (2/dx)(9/8 + 1/24) = (7/3)/dx, attained at
# Nyquist, so in 2D von Neumann stability requires
#
#     c dt sqrt(2) (7/3) / dx <= 2   =>   c dt / dx <= 6/(7 sqrt(2)) = 0.6061
#
# equivalently dt <= C dx / (c sqrt(2)) with C = 6/7 = 0.857, which is the
# document's "C ~ 0.86".
#
# NOTE (deliberate deviation from the document, with reason).  §3.3 states a
# safety factor of 0.9 and then quotes dt = 0.6 dx/c_p.  Those disagree:
# 0.6/0.6061 = 0.99, so dt = 0.6 dx/c_p sits at 99% of the stability limit with
# essentially no margin -- and the margin is exactly what the void-interface
# moduli averaging eats into.  We honour the stated safety factor rather than
# the quoted number.  T_end is unchanged, so every frequency-domain quantity
# (which depends on T_end, not on n_t) is unaffected; only n_t moves,
# 1280 -> 1408.
# ---------------------------------------------------------------------------
CFL_LIMIT_4TH: float = 6.0 / (7.0 * math.sqrt(2.0))     # 0.606092
CFL_SAFETY: float = 0.9
CFL_NUMBER_TARGET: float = CFL_SAFETY * CFL_LIMIT_4TH   # 0.545483

T_END: float = 24.0 * T_P
NT: int = int(math.ceil(T_END / (CFL_NUMBER_TARGET * DX_FINE / CP)))   # 1408
DT: float = T_END / NT                                  # exact division
CFL_NUMBER: float = CP * DT / DX_FINE

N_SAVED_FRAMES: int = 64
SAVE_EVERY: int = NT // N_SAVED_FRAMES                  # 22

# ---------------------------------------------------------------------------
# Tone burst (§2.3)
# ---------------------------------------------------------------------------
N_CYCLES: int = 5
BURST_DURATION: float = N_CYCLES / FC                   # = 5

# ---------------------------------------------------------------------------
# Frequency band (§5.4)
# ---------------------------------------------------------------------------
M_FREQ: int = 20
F_START: float = 0.66 * FC
DF: float = 0.0358 * FC
FREQS: tuple[float, ...] = tuple(F_START + DF * m for m in range(M_FREQ))
OMEGAS: tuple[float, ...] = tuple(2.0 * math.pi * f for f in FREQS)

# Continuation bands used by the inversion stages (§8.5).
BAND_STAGE1 = slice(0, 6)      # m = 1..6   -> f = 0.660 .. 0.839 f_c
BAND_STAGE2 = slice(0, 10)     # m = 1..10  -> f = 0.660 .. 0.982 f_c
BAND_STAGE3 = slice(0, M_FREQ)  # full band

# ---------------------------------------------------------------------------
# Mode budget (§3.5) -- the calculation that decides whether the FNO works
# ---------------------------------------------------------------------------
BURST_SPREAD: float = 1.4                                # main-lobe half-width factor
K_CARRIER_WORST: float = L_DOMAIN / LAMBDA_S_MIN         # 17.61
K_REQUIRED: float = K_CARRIER_WORST * BURST_SPREAD       # 24.65
KMAX: int = 28
K_NYQUIST: int = N_NET // 2                              # 64

# ---------------------------------------------------------------------------
# Defect geometry (§2.4, §3.7)
# ---------------------------------------------------------------------------
R_MIN_LS: float = 0.4          # radius range in *shear* wavelengths
R_MAX_LS: float = 1.2
SDF_CLIP_CELLS: float = 16.0           # |phi/dx| clip
BOUNDARY_KEEPOUT_LS: float = 1.5       # keep centre >= 1.5 lambda_s from edges/source

# The interface is a physical length, not a cell count, and it is quoted both ways
# here because quoting only the cell count is what let "1.5 cells" read as "narrow".
#
# chi = sigmoid(-phi/eps) has 10-90% transition width 2 ln(9) eps = 4.394 eps, so the
# primary constant is eps as a fraction of the *fine* cell -- the grid the material
# actually lives on -- and the network-cell count is derived.  v2.0 had this the
# other way round at eps = 1.5 network cells: a 10-90% width of 6.59 network cells,
# 0.91 lambda_s, wider than twice the smallest specified radius.  That is a graded
# soft inclusion, not a slightly smoothed cavity, and it was never measured.
#
# MEASURED (solver/validate.py:check_cavity_scattering and check_interface_width;
# receiver-ring scattered field against the analytic traction-free cavity, R = 1.2
# lambda_s, nu = 1/3).  The two errors are independent and neither fix works alone:
#
#                            eps = 1.5 net cells   eps = 0.375 fine cells
#     rho_void = RHO0                87.4%                  76.7%
#     rho_void = 1e-2 RHO0           86.3%                   9.3%
#
# and the width dependence at rho_void = 1e-2 is a U with a broad, flat minimum --
# 15.7% at 2.64 fine cells of 10-90% width, 11.0% at 1.98, 9.3% at 1.65, 8.9% at 1.32,
# 11.4% at 0.99 -- whose left branch is the physical width and whose right branch is a
# sigmoid thinner than the cell that samples it, and the fitted amplitude ratio crossing
# 1.0 between the last two (0.9885 -> 1.0002 -> 1.0130) is what identifies the branches:
# a sub-cell sigmoid is a staircase, and a staircase over-scatters.  Tripling dx at
# fixed physical eps moves the error by 0.2 points, so this is a property of the
# interface model and not of the discretisation.
#
# The true minimum is one step narrower than the value below (8.9% at 1.32 fine cells
# against 9.3% at 1.65) and it is deliberately not taken.  0.4 points is 4% of a
# residual whose 9% is dominated by the void model, it is bought by moving *towards* the
# staircase branch where the amplitude ratio has already crossed 1, and it would narrow
# the interface the network grid has to represent from 0.82 to 0.66 net cells.
# Everything else in the project wants eps large: dchi/dtheta has support eps, and that
# is the inversion's shape gradient.  So the value sits at the wide end of the flat
# minimum, and the 0.4 points is recorded rather than chased.
EPS_INTERFACE_FINE_CELLS: float = 0.375                  # eps, in fine solver cells
EPS_TRANSITION_FACTOR: float = 2.0 * math.log(9.0)       # = 4.3944, the 10-90% width
EPS_INTERFACE_PHYS: float = EPS_INTERFACE_FINE_CELLS * DX_FINE          # 0.01171875
EPS_INTERFACE_CELLS: float = EPS_INTERFACE_PHYS / DX_NET                # 0.1875
#
# EPS_INTERFACE_PHYS is what any grid-refinement comparison must hold fixed: refining
# dx while keeping eps at a fixed *cell* count refines the interface at the same time
# and the two effects cannot then be separated (§3 of the review).

# Interface width annealing during inversion (§8.2, §8.5).
#
# OFF by default.  Training used eps = EPS_INTERFACE_CELLS and nothing else, so a
# schedule that brackets it asks the network three questions it was never trained on
# and calls the answers gradients.  Bracketing the training value is not validation
# of the endpoints.  Turn it on only after
# `inverse/sensitivity.py:eps_transfer_report` shows the surrogate's receiver
# sensitivities still agree with the solver's at the schedule's endpoints.  The
# endpoints are 4/3 and 2/3 of the trained width, as they were at eps = 1.5 cells.
EPS_INVERT_ANNEAL: bool = False
EPS_INVERT_START: float = 0.25
EPS_INVERT_END: float = 0.125

# Void model.  A void has no mass, and v2.0 kept RHO0 inside it while nulling the
# stiffness -- a soft *heavy* inclusion.  Measured against the analytic cavity, that
# is the single largest error in the labels once the interface is narrow (76.7%
# against 9.3%), and the mechanism is not subtle: the interior is a bag of
# effectively free masses (the coupling length sqrt(VOID_STIFFNESS_FLOOR) lambda_s is
# a hundredth of a cell, so nothing propagates inside), and the first layer of them
# rides on the interface as an added mass.  The load per unit area is rho dx omega^2 u
# against a solid traction scale rho c_s omega u, i.e. a ratio of k_s dx = 0.43 at
# f_max -- a 43% perturbation of the boundary condition, which is what the numbers
# above show.
#
# 1e-2 rather than the 1e-4 of the stiffness floor, and this *is* a stability limit,
# though not the one the docstring used to claim.  Cell-centred c = sqrt(solid/rho)
# never exceeds CP for any chi when both floors are hit together, so there is no 0/0
# problem at centres.  The failure is at the *faces*: rho_vy = avg_plus(rho, -2)
# averages a one-cell density jump arithmetically while the 4th-order stress stencil
# reaches two cells into full-stiffness material, so a face inside a sharp void sees
# a small stiffness over a much smaller density.  At rho_void = 1e-4 with a sharp
# interface that diverges (measured: step 200 of 1408, Courant 0.5455); at 1e-2 with
# the interface width above, stiffness and density fall together across the same two
# cells and it is stable.
#
# The remaining impedance ratio is sqrt(VOID_STIFFNESS_FLOOR * VOID_DENSITY_SCALE)
# = 1e-3 of the host, not zero.  Its accuracy is a measured quantity
# (`solver/cavity.py:cavity_accuracy_report`), not a consequence of the run being
# stable.  losses.py reads these same constants so that delta_rho in the
# differential scattered-field residual matches the solver exactly.
VOID_DENSITY_SCALE: float = 1.0e-2     # relative density retained inside the void
VOID_STIFFNESS_FLOOR: float = 1.0e-4   # relative floor, keeps the update finite

# Harmonic-average guard for mu at corners (`solver/fdtd_elastic.py`).
# Absolute, tiny, and batch-independent: it exists only to avoid 0/0 if a zero
# modulus is ever passed in.  `material_fields` already guarantees
# mu >= min(mu0)*VOID_STIFFNESS_FLOOR ~ 2e-05, so 1e-12 is ~7 orders below any
# real value and never alters physics.  It must NOT be derived from
# `mu.max()` over the batch: that made the same physical sample solve
# differently depending on its batch mates (up to 0.36 rel-L2 on the scattered
# field, median 0 / bimodal), which is what §11.2 step 6 caught.
MU_HARMONIC_FLOOR: float = 1.0e-12

# ---------------------------------------------------------------------------
# Source / receiver ring (§3.7)
# ---------------------------------------------------------------------------
N_RECV: int = 32               # 8 per side
N_RECV_PER_SIDE: int = 8
N_SRC: int = 8                 # 2 per side
N_SRC_PER_SIDE: int = 2
RING_INSET_NET: int = 3        # cells inside the cropped edge
N_RECV_SUBSET: int = 8         # random 8-of-32 per training step (§7.3)

# Source-cell exclusion radius for the physics residual (see README).
PHYS_SOURCE_EXCLUDE_CELLS: float = 3.0

# ---------------------------------------------------------------------------
# Dataset (§3.7)
# ---------------------------------------------------------------------------
N_TRAIN: int = 2000
N_VAL: int = 400
N_TEST: int = 400
SEED: int = 20260829

# Source indices withheld from training entirely (§11.2 step 11: "including a
# held-out source").  Indices into SOURCES_NET, whose order is bottom, right, top,
# left.  3 and 6 are the upper-right and upper-left positions: two sources on
# different sides, each with a same-side sibling that *is* trained on, so the
# held-out test is about the specific illumination and not about a whole unseen
# edge of the domain.
SRC_HELDOUT: tuple[int, ...] = (3, 6)
SRC_TRAIN: tuple[int, ...] = tuple(i for i in range(N_SRC) if i not in SRC_HELDOUT)

# Samples per forward-solver batch during dataset generation.  Each sample costs
# ~8 MB of material and field arrays on the 376^2 padded grid plus ~5 MB of phasor
# accumulator, so 16 fits comfortably in 16 GB while keeping the GPU busy.
GEN_BATCH: int = 16

# Subsampling interval for the running DFT.  1 = accumulate every step, which is
# exact and is the default.
#
# Larger values are tempting (the accumulator is ~40 planes of 128^2 complex64, so
# touching it every step roughly doubles the cost of the solve) and the sampling
# theorem says a band-limited signal can be transformed exactly from samples below
# its Nyquist rate.  The catch is what "band-limited" means here: a point source is
# a spatial delta, so it excites every wavenumber the grid supports, including
# content near the grid Nyquist at c/(2 dx) = 16 f_c.  At DFT_EVERY = 2 the
# sampling Nyquist is 14.7 f_c and that grid noise folds to 13.3 f_c -- harmlessly
# outside the 0.66-1.34 f_c band.  At DFT_EVERY = 4 the Nyquist is 7.3 f_c and the
# same noise folds to 1.34 f_c, landing exactly on the top of the operating band.
# So 2 is safe and 4 is not, which is not a distinction worth risking a whole
# dataset on for a 17% saving.
DFT_EVERY: int = 1

# Frames are ~64x the size of the phasors, so they are stored for a small subset
# only -- enough for the wavefield montage of §11.3 figure 1 and nothing more.
N_VIS_SAMPLES: int = 8

# Measurement noise levels used in evaluation (§9.3).  30 dB is the headline
# number the >90% success-rate gate is quoted at.
SNR_DB_DEFAULT: float = 30.0
SNR_DB_SWEEP: tuple[float, ...] = (60.0, 40.0, 30.0, 20.0)

# ---------------------------------------------------------------------------
# Numeric gates (§11.2).  Every one of these is a number, not a feeling.
# Kept here so that a test, a notebook and the README cannot disagree about
# what "passing" means.
# ---------------------------------------------------------------------------
GATE_ENERGY_DRIFT: float = 5.0e-3        # step 1: |dE/E| over the full run
GATE_ARRIVAL_STEPS: float = 1.0          # step 2: P/S arrival, in time steps
GATE_ABSORBER_RESIDUAL: float = 1.0e-4   # step 3: residual/peak energy
GATE_GRID_CONVERGENCE: float = 0.02      # step 5: 256^2 vs 512^2 rel-L2
GATE_REL_L2: float = 0.05                # step 7: surrogate rel-L2
GATE_ARRIVAL_PERIODS: float = 0.05       # step 7: arrival-time error
GATE_POSITION_LS: float = 0.10           # steps 10/11: position error, in lam_s
GATE_SUCCESS_RATE: float = 0.90          # step 11: success rate at 30 dB
GATE_GRAD_SIGFIGS: int = 3               # step 9a: autodiff vs FD of the surrogate

# The absorber is a graded sponge, not a split-field or CFS PML.  The old name
# GATE_PML_RESIDUAL claimed otherwise; it is kept as an alias so nothing breaks, but
# the artefact the sponge *leaves on the receiver ring* is the number that matters and
# it gets its own gate rather than being folded into the energy-decay check.  It is not
# a normal-incidence |R|: check_absorber_reflection differences the production
# acquisition against the same acquisition in a padded open domain, so it reads the
# layer at every angle out to grazing, which is where a sponge is worst.  Measured 1.08%
# on the incident field and 1.81% on the scattered field; 2% leaves a factor of ~2 of
# headroom, which is the least a gate can have and still be a gate.
GATE_PML_RESIDUAL: float = GATE_ABSORBER_RESIDUAL           # deprecated alias
GATE_ABSORBER_REFLECTION: float = 2.0e-2   # step 3b: ring artefact vs an open domain

# --- gates the review requires and v2.0 did not have ----------------------
#
# 9b is the one that distinguishes implementation correctness from physical accuracy:
# 9a differentiates the surrogate twice by two methods and can pass while the
# surrogate's sensitivities are physically wrong.  9b compares the surrogate's
# receiver sensitivity to a finite difference of the *validated solver*, over several
# perturbation sizes, and is quoted as a relative error rather than significant
# figures because two different models are never going to agree to 3 s.f.
GATE_SENSITIVITY_REL: float = 0.20         # step 9b: surrogate vs solver dJ/dtheta
GATE_SENSITIVITY_COSINE: float = 0.95      # step 9b: direction agreement
GATE_SCREEN_CAPTURE: float = 0.90          # step 10a: fraction of screens whose
#                                            survivor set contains the true basin
# Step 1a: the *incident* field against the analytic Green's tensor, absolutely.
# Tighter than the cavity gate because it has strictly fewer error sources -- no
# interface, no void model, no scattered field to drag through the sponge -- and
# because it is the calibration the cavity comparison then rests on.  Measured 2.6%
# with the source at the domain centre and 1.9% with the production source; the floor
# with no absorber at all is 1.0%, rising like f^2, which is numerical dispersion at
# 14.5 points per shear wavelength.  (It read 3.6% and 22.8% before the absorber was
# redesigned; the second number is why check 9 exists.)
GATE_GREEN_REL_L2: float = 0.05            # step 1a: incident field vs Green's tensor

# Step 1b: the solver's void against the analytic traction-free cavity, on the
# receiver ring, absolutely (no fitted scale or phase).  0.12 is an error budget, not
# a round number, and it is quoted here because a gate that a measurement was tuned
# to meet is worth nothing.  Measured over the whole specified radius range: 8.6% at
# R = 0.4 lambda_s, 7.8% at 0.8, 9.3% at 1.2, with amplitude ratio 0.989 and phase
# +0.025 rad at the worst radius.  What is accounted for:
#
#   ~1.9%  the incident field itself, i.e. the illumination the void scatters, which is
#          what check_green_incident reads with the production source
#   ~1.8%  the absorber acting on the scattered field on its way to the ring, which is
#          the second number check_absorber_reflection reports
#   ~2%    the circle's area on a fine grid (chi.sum() matches pi R^2 / dx^2 to 0.4%,
#          but the boundary is sampled, not conformal)
#   ~1%    the residual of the A-scan sampling model, which is exact only for a plane
#          wave at normal incidence
#
# ~3.4% in quadrature against 9.3% measured, and the gap is the point.  Before the
# absorber was redesigned these four terms were ~3.6/6.3/2/1 = 7.6% against 10.4%, so
# fixing a 22% boundary artefact moved the cavity error by one point.  What is left is
# not a discretisation and check 8 is the evidence: refining dx threefold at fixed
# physical interface width moves it by 0.2 points, and the width itself is at the
# bottom of its own U.  It is the *void model*.  A cavity has zero impedance; this one
# has sqrt(VOID_STIFFNESS_FLOOR * VOID_DENSITY_SCALE) = 1e-3 of the solid's, over a
# transition 1.65 fine cells wide, and both floors exist because the scheme diverges
# without them (see geometry/sdf.py:material_fields).  So roughly 9% is the price of
# representing a traction-free boundary as a soft light inclusion on a fixed grid, and
# it is a *model* error that no amount of mesh refinement removes.  Quoting it as such
# is the honest version of §2.4.
#
# The gate stays at 0.12 rather than being pulled down to 0.10: check 8 measures 12.1%
# at half the production interface width, so 0.12 is one width step above the
# measurement and fails if the interface, the void floors or the absorber regress,
# while passing across the whole radius range.  Tighten it by making the void a real
# cavity -- an explicit traction-free boundary condition on a cut cell -- not by
# widening the interface.
GATE_CAVITY_REL_L2: float = 0.12           # step 1b: solver void vs Pao-Mow
GATE_CAVITY_PHASE_RAD: float = 0.20        # step 1b: receiver-phase error

# Step 4b: what losses.physics_loss reads on a field the solver produced.  Review finding
# (d) -- the residual and the solver do not share a discretisation, so this floor is
# nonzero and nobody had measured it.  Measured 0.033 homogeneous, 0.060 at the largest
# radius, 0.075 with interface_weight restoring the band the default weight discards; see
# ALPHA_PHYS for what those numbers mean for the loss balance.  0.15 rather than the 0.30
# this started at, which was a guess made before the measurement existed: 0.15 is ~2.5x
# the worst production reading, and the headroom is sized for perimeter rather than
# invented -- the interface contributes +0.027 over the floor, and a two-void or eccentric
# family at the same area carries up to ~sqrt(2) more boundary, so ~0.09 is reachable
# without anything being wrong.  Below that a regression in the interface, the stencil or
# the erosion shows up here rather than silently inflating the regulariser.
GATE_PHYS_RESIDUAL_LABEL: float = 0.15     # step 4b: collocated physics residual
#                                            measured *on reference labels*, relative
GATE_SOLVER_VERIFY_LS: float = 0.15        # step 10b: position error of the recovered
#                                            geometry when re-scored by the solver
GATE_SOLVER_VERIFY_RESIDUAL_RATIO: float = 2.0  # step 10b: solver J(answer)/J(truth)
GATE_SOLVER_VERIFY_AXIS_RATIO: float = 0.15     # step 10b: ellipse axis-ratio error
GATE_SOLVER_VERIFY_ORIENTATION_DEG: float = 10.0 # step 10b: ellipse orientation error
GATE_LOF_FPR: float = 0.10                 # step 12: lack-of-fit false-positive rate
#                                            at the frozen threshold

# Shape-appropriate success (§9).  GATE_POSITION_LS scores a *centre*, which is the
# whole story only for a circle whose radius is also right, and is not size-normalised:
# 0.10 lambda_s is a 25% miss on the smallest void in the box and an 8% miss on the
# largest.  It also cannot express an error in an ellipse's axes or orientation, or a
# two-void recovery that found one void twice.  So the primary score for the transfer
# families is the soft-indicator IoU, measured at the trained interface width on the
# network grid, with permutation-invariant blob matching for multi-void families.
#
# The threshold is calibrated against the position gate rather than invented: a circle
# displaced by exactly GATE_POSITION_LS with a perfect radius gives IoU 0.829 at
# R = 0.4 lambda_s, 0.855 at 0.6 and 0.908 at 1.2.  0.80 is therefore a hair more
# permissive than the position gate on the hardest (smallest) void and materially
# stricter on the easiest, which is the intended correction -- and unlike the position
# gate it also fails a run that got the centre right and the size wrong.
GATE_IOU: float = 0.80                     # steps 10/11: soft-indicator IoU

# Diagnostic (not a gate): when is an estimate "pressed against the wall" of the
# feasible box?  `ShapeFamily.to_unconstrained` clamps the normalised coordinate to
# [1e-4, 1 - 1e-4], so the largest |z| any *theta* can map back to is log(9999) =
# 9.2103 -- which is why a threshold of 12 could never fire, however saturated the
# optimiser's internal z became.  9.0 is the largest round number below the clamp:
# sigmoid(9) = 0.999877, i.e. theta within 1.2e-4 of the box width from the edge,
# 8.2e-4 in length units, 0.013 network cells.  A run reporting an answer that close to
# the wall has almost certainly stopped because the reparameterisation ran out of room,
# not because it found a minimum -- and in float32 sigmoid' underflows to exactly zero
# by |z| = 16.75, so the optimiser's own coordinate can be somewhere the gradient no
# longer exists while theta still reads as merely "at the boundary".
WALL_Z_CLAMP: float = math.log((1.0 - 1e-4) / 1e-4)     # 9.2103, the entry clamp
WALL_Z_WARN: float = 9.0                                # summarise() counts z above this

# Model-error floor for the lack-of-fit statistic (§9.3).  The statistic normalises the
# converged *data* residual by what noise alone would explain, and at high SNR -- or in
# the noiseless synthetic case -- that denominator goes to zero and the statistic
# diverges for reasons that have nothing to do with fit.  What actually floors the
# residual is surrogate error: the network's own error at the receivers, which no amount
# of optimisation can go below.  So the denominator is (noise term + LOF_MODEL_FLOOR).
#
# Units: the same as the misfit, i.e. *squared* relative error, hence GATE_REL_L2^2
# rather than GATE_REL_L2.  This is a placeholder with a known bias -- GATE_REL_L2 is a
# whole-domain field error and the receiver ring is harder than the domain average --
# so `inverse.invert.calibrate_lack_of_fit` compares it against the observed in-family
# residuals and warns when the floor is the smaller of the two.  Substitute the trained
# model's measured receiver misfit before freezing a threshold.
LOF_MODEL_FLOOR: float = GATE_REL_L2 ** 2

# ---------------------------------------------------------------------------
# Network (§6.2 - §6.5)
# ---------------------------------------------------------------------------
C_IN: int = 12
C_OUT: int = 4
D_V: int = 32
N_BLOCKS: int = 4
LIFT_HIDDEN: int = 64
PROJ_HIDDEN: int = 128

LR: float = 1.0e-3
WEIGHT_DECAY: float = 1.0e-4
LR_FINAL: float = 1.0e-5
BATCH_SIZE: int = 16
EPOCHS: int = 300
GRAD_CLIP: float = 1.0

# The network is time-harmonic: frequency enters as a conditioning channel, so one
# stored sample (geometry, source, nu) yields M_FREQ = 20 distinct training
# examples.  Using all 20 with BATCH_SIZE = 16 would put 320 fields through the
# network at once -- at d_v = 32 and 128^2 that is ~0.67 GB per stored activation
# and several GB once the complex spectral intermediates are counted.  Drawing a
# random subset per sample per step keeps the effective batch at
# BATCH_SIZE * N_FREQ_PER_SAMPLE = 64 while still seeing every frequency many
# times per epoch, and the randomness acts as augmentation across the band.
N_FREQ_PER_SAMPLE: int = 4

GAMMA_H1: float = 0.1          # §7.1
BETA_MEAS: float = 1.0         # §7.4

# The physics term has a *measured* floor and it is not zero.  check 10
# (validate.check_physics_residual_on_labels) evaluates losses.physics_loss on fields the
# solver produced, i.e. on the very labels the network is trained to reproduce, and reads
# 0.033 on a homogeneous medium and 0.044/0.047/0.060 at R = 0.4/0.8/1.2 lambda_s.  The
# homogeneous number is pure discretisation mismatch -- the residual is a nested pair of
# 4th-order *centred* differences of the time-harmonic Navier operator on the 128^2
# network grid, while the label came from a staggered velocity-stress scheme on the 376^2
# fine grid, and nothing makes the first vanish on a solution of the second.  The excess
# above it, +0.027 at the largest radius, is the interface.
#
# So a *perfect* prediction is still charged about 0.06 of this term, and that is the
# number ALPHA_PHYS has to be read against: at 3e-2 the irreducible contribution is
# ~2e-3, three orders below a data term of order 1, which is the intended regime.  Raise
# it and the optimiser starts buying that floor down by making the field smooth on the
# network grid, which is not the same thing as making it right.  Gradient-norm balancing
# (§7.4) can defeat this on its own, since it will happily scale up a term whose gradient
# is dominated by its own floor; cap it rather than trusting it.
ALPHA_PHYS: float = 3.0e-2     # §7.4 -- overwritten by gradient-norm balancing
STENCIL_HALF_WIDTH: int = 2     # 4th order
ERODE_CELLS: int = STENCIL_HALF_WIDTH + 1   # §7.2: stencil half-width + 1

# Capacity sweep variants (§6.4)
VARIANTS = {
    "primary": dict(d_v=32, kmax=28),
    "small": dict(d_v=24, kmax=24),
    "tiny": dict(d_v=16, kmax=16),
}

# ---------------------------------------------------------------------------
# Inversion (§8.5)
# ---------------------------------------------------------------------------
SCREEN_GRID: int = 16                  # 16 x 16 = 256 candidates
N_SURVIVORS: int = 16
ADAM_STEPS_STAGE2: int = 200
ADAM_LR_STAGE2: float = 5.0e-2
LBFGS_STEPS_STAGE3: int = 60
TIKHONOV_MU: float = 1.0e-3

# Which objective each stage actually optimises.  This table exists because v2.0's
# prose and the code disagreed: the prose said "envelope refinement" in stage 2 and
# the code ran a complex waveform misfit.  Rather than leave the reader to diff them,
# the intended pipeline is declared here, every stage reads it, and
# `inverse.misfit.OBJECTIVES` is keyed by exactly these names.
#
#   spectral_magnitude   |G| vs |d| per frequency.  A *heuristic* screen: provably
#                        blind to travel time (see inverse/timedomain.py identity A),
#                        so it is retained only because it is 20x cheaper than a
#                        reconstruction and its capture rate is measured, not assumed.
#   envelope             |z(t)| of the band-limited analytic signal.  Sensitive to
#                        arrival time, blind to carrier phase.
#   traveltime           soft-argmax cross-correlation lag, squared and weighted.
#   correlation          1 - peak normalised correlation; the complement.
#   complex              full complex residual; the only one with carrier resolution.
STAGE_OBJECTIVE: dict[int, str] = {
    1: "envelope",       # was "spectral_magnitude"; that is now the fallback screen
    2: "envelope",       # was "complex"; §8.4 always said envelope here
    3: "complex",
}
# The cheap screen is still available, and Stage 1 can be switched to it to reproduce
# the v2.0 numbers or to measure the difference.  `SCREEN_OBJECTIVE` is what
# `inverse.invert.screen` uses; `screen_capture_rate` compares both.
SCREEN_OBJECTIVE: str = "envelope"
SCREEN_FALLBACK_OBJECTIVE: str = "spectral_magnitude"

# Time grid for the band-limited analytic reconstruction (inverse/timedomain.py).
# 128 samples: definition R's highest frequency is 1.34 f_c and 128 * DF = 4.58 f_c,
# so R is exactly representable on this grid and the envelope is not aliased.
RECON_N_T: int = 128
RECON_TAPER: str = "hann"

# Screen batching.  The chunk is 16 candidates, not 64, and the reason is arithmetic
# rather than taste (§6 of the review).  One chunk's *input* tensor is
#     chunk * F * C_IN * N_NET^2 * 4 B
# and the frequency axis is folded into the batch by features.pack_inputs, so the
# stage-1 band (6 frequencies) multiplies it by six.  At chunk = 64 that is 302 MB of
# input alone, and the first lifted feature tensor -- chunk * F * D_V * N_NET^2 * 4 B
# -- is 805 MB, before the complex spectral intermediates inside each block.  16
# keeps the input at 75 MB and the feature tensor at 201 MB; `screen_memory_mb()`
# below reports both so the number in the notebook is computed, not quoted.
SCREEN_CHUNK: int = 16

# ---------------------------------------------------------------------------
# Physical units, for reporting only (§3.5)
# ---------------------------------------------------------------------------
CP_ALUMINIUM: float = 6300.0           # m/s
FC_PHYSICAL: float = 250.0e3           # Hz


def to_mm(length_nondim: float, cp: float = CP_ALUMINIUM, fc: float = FC_PHYSICAL) -> float:
    """Convert a non-dimensional length (in lambda_p) to millimetres."""
    lambda_p_mm = 1e3 * cp / fc
    return length_nondim * lambda_p_mm


def interface_report(eps_cells: float = EPS_INTERFACE_CELLS,
                     dx: float = DX_NET) -> dict:
    """
    What the sigmoid interface actually measures, in every unit that matters.

    The review's §3 arithmetic, made a function so it cannot drift out of date:
    quoting eps as a cell count understates the transition by a factor of 4.39, and
    the honest comparison is against the *smallest* radius at the *worst-case* shear
    wavelength, not against the domain.

    `ratio_to_r_min` above 1 means the transition is wider than the smallest defect
    is across, which is where v2.0's eps = 1.5 network cells sat (2.27).  At the
    present width it is 0.28, and `solver/validate.py:check_cavity_scattering`
    measures what that costs against the analytic traction-free cavity.
    """
    w = EPS_TRANSITION_FACTOR * eps_cells * dx         # 10-90% width, non-dim
    r_min = R_MIN_LS * LAMBDA_S_MIN
    return dict(
        eps_cells=eps_cells,
        eps_fine_cells=eps_cells * dx / DX_FINE,
        eps_phys=eps_cells * dx,
        transition_cells=EPS_TRANSITION_FACTOR * eps_cells,
        transition_fine_cells=w / DX_FINE,
        transition_phys=w,
        transition_mm=to_mm(w),
        transition_over_lambda_s_worst=w / LAMBDA_S_MIN,
        r_min_phys=r_min,
        r_min_over_lambda_s=R_MIN_LS,
        ratio_to_r_min=w / r_min,
        impedance_ratio=math.sqrt(VOID_STIFFNESS_FLOOR * VOID_DENSITY_SCALE),
    )


def screen_memory_mb(chunk: int = SCREEN_CHUNK, band: slice = BAND_STAGE1, *,
                     d_v: int = D_V, bytes_per: int = 4) -> dict:
    """
    Peak-ish memory of one screening chunk, counting the frequency axis.

    v2.0 quoted 50 MB.  That was `chunk * C_IN * N_NET^2 * 4` -- the input tensor for
    one frequency.  `features.pack_inputs` folds the band into the batch axis, so the
    real row count is chunk * F, and the lifted feature tensor at width d_v is larger
    than the input whenever d_v > C_IN, which it is (32 > 12).  Both are reported;
    neither is the true peak, because each spectral block also allocates a complex
    [rows, d_v, N_NET, N_NET/2+1] intermediate, so `spectral_mb` is included too.
    """
    f = len(range(*band.indices(M_FREQ)))
    rows = chunk * f
    cell = N_NET * N_NET * bytes_per
    return dict(
        chunk=chunk, n_freq=f, rows=rows,
        input_mb=rows * C_IN * cell / 1e6,
        feature_mb=rows * d_v * cell / 1e6,
        spectral_mb=rows * d_v * N_NET * (N_NET // 2 + 1) * 2 * bytes_per / 1e6,
        chunks=-(-SCREEN_GRID ** 2 // chunk),
    )


# ---------------------------------------------------------------------------
# Derived per-material table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Material:
    nu: float
    cs_cp: float
    lam: float
    mu: float
    lambda_s: float
    ppw_s_fine: float
    ppw_s_net: float
    domain_in_lambda_s: float
    k_carrier: float

    @property
    def cs(self) -> float:
        return self.cs_cp * CP


def material(nu: float) -> Material:
    r = cs_over_cp(nu)
    lam, mu = lame_from_nu(nu)
    lam_s = r / FC
    return Material(
        nu=nu,
        cs_cp=r,
        lam=lam,
        mu=mu,
        lambda_s=lam_s,
        ppw_s_fine=lam_s / DX_FINE,
        ppw_s_net=lam_s / DX_NET,
        domain_in_lambda_s=L_DOMAIN / lam_s,
        k_carrier=L_DOMAIN / lam_s,
    )


MATERIALS: tuple[Material, ...] = tuple(material(nu) for nu in NU_LIST)


# ---------------------------------------------------------------------------
# Ring geometry
# ---------------------------------------------------------------------------
def ring_positions(n_per_side: int, n_grid: int = N_NET, inset: int = RING_INSET_NET
                   ) -> list[tuple[int, int]]:
    """
    Evenly spaced positions on a square ring `inset` cells inside the domain.

    Returns (iy, ix) index pairs on an `n_grid` x `n_grid` grid, ordered
    bottom / right / top / left.  Corners are avoided by construction because
    the along-side coordinates are strictly interior fractions.
    """
    lo, hi = inset, n_grid - 1 - inset
    span = hi - lo
    fracs = [(j + 0.5) / n_per_side for j in range(n_per_side)]
    coords = [lo + int(round(f * span)) for f in fracs]
    pos: list[tuple[int, int]] = []
    pos += [(lo, c) for c in coords]              # bottom edge, y = lo
    pos += [(c, hi) for c in coords]              # right edge,  x = hi
    pos += [(hi, c) for c in reversed(coords)]    # top edge,    y = hi
    pos += [(c, lo) for c in reversed(coords)]    # left edge,   x = lo
    return pos


RECEIVERS_NET: list[tuple[int, int]] = ring_positions(N_RECV_PER_SIDE)
SOURCES_NET: list[tuple[int, int]] = ring_positions(N_SRC_PER_SIDE)


def net_to_fine(iy: int, ix: int) -> tuple[int, int]:
    """Network-grid index -> index in the *padded* fine solver grid."""
    return (iy * DOWNSAMPLE + N_PML_FINE, ix * DOWNSAMPLE + N_PML_FINE)


def receiver_position(i: int) -> tuple[float, float]:
    """
    Physical (x, y) of receiver i.

    Exact: a receiver reads the 2x2-averaged network cell, and the centroid of a
    2x2 block of fine cells is precisely the network cell centre.
    """
    iy, ix = RECEIVERS_NET[i]
    return ((ix + 0.5) * DX_NET, (iy + 0.5) * DX_NET)


def source_position(i: int) -> tuple[float, float]:
    """
    Physical (x, y) of source i -- the centre of the fine cell the force is
    actually injected into, which is *not* the nominal network cell centre.

    With DOWNSAMPLE = 2 there is no fine cell centred on a network cell centre:
    the network centre sits on the corner shared by fine cells 2i+P and 2i+1+P,
    so `net_to_fine` picks the lower-left of the four and the true source sits
    dx_fine/2 = dx_net/4 below and left of nominal.  That is a real 0.0156
    lambda_p offset, and the reason to expose it here rather than let each caller
    recompute (ix+0.5)*dx_net is that the inversion's forward model and the
    physics loss both need the source location: a quarter-cell inconsistency
    between where the solver put the force and where the loss thinks it is
    appears as a small fixed phase error, i.e. as a bias in the recovered
    position, which is exactly the quantity being measured.

    Not worth removing by moving to DOWNSAMPLE = 3 -- an odd factor would align
    exactly, but it would cost 2.25x the solver work for an offset that is
    identical in every simulation and therefore cancels out of every difference.
    Worth *knowing about*, which is the point of writing it down.
    """
    iy, ix = SOURCES_NET[i]
    fy, fx = net_to_fine(iy, ix)
    return ((fx + 0.5 - N_PML_FINE) * DX_FINE, (fy + 0.5 - N_PML_FINE) * DX_FINE)


SOURCE_XY: tuple[tuple[float, float], ...] = tuple(
    source_position(i) for i in range(N_SRC))


def source_force_position(i: int) -> tuple[float, float]:
    """
    Physical (x, y) of the *point force* of source i: half a fine cell above
    `source_position(i)`.

    The force is added to `vy`, and on a staggered grid `vy[j, i]` does not live at
    the centre of cell (j, i) -- it lives on the y-face at (x_i, y_j + dx_fine/2)
    (the same offset `absorber_profile` is evaluated at with `shift_y=0.5`).  So the
    delta the solver actually applies sits dx_fine/2 = 0.0156 lambda_p above the
    cell centre.

    That is 0.26 rad of shear phase at the top of the band -- larger than
    GATE_CAVITY_PHASE_RAD -- so an analytic reference evaluated at
    `source_position` disagrees with the solver by more than the gate allows for
    reasons that have nothing to do with the solver being wrong.  This is the
    position `solver/cavity.py` must be given; `source_position` remains the cell
    centre because that is what the figures, the acquisition metadata and the
    regressor's source features mean, and because the offset is identical in every
    simulation and cancels from every difference of two solves.

    The remaining discrepancy is a form factor, not a position: the discrete delta
    is 1/dx^2 spread over one fine cell rather than a true point, which multiplies
    the radiated field by sinc(k_x dx/2) sinc(k_y dx/2) ~ 1 - (k dx)^2/24, i.e.
    0.3% at f_max.  Below every gate here, and not removable by moving a point.
    """
    x, y = source_position(i)
    return (x, y + 0.5 * DX_FINE)


SOURCE_FORCE_XY: tuple[tuple[float, float], ...] = tuple(
    source_force_position(i) for i in range(N_SRC))
RECEIVER_XY: tuple[tuple[float, float], ...] = tuple(
    receiver_position(i) for i in range(N_RECV))


# ---------------------------------------------------------------------------
# Parameter counting (§6.4)
# ---------------------------------------------------------------------------
def spectral_params(d_v: int, kmax: int, radial: bool) -> int:
    """Real parameter count of one spectral convolution."""
    if not radial:
        kept = 2 * kmax * kmax                       # two half-spectrum quadrants
    else:
        kept = 0
        for iy in range(kmax):
            for ix in range(kmax):
                if math.hypot(iy, ix) <= kmax:       # w1: ky = +iy
                    kept += 1
                if math.hypot(kmax - iy, ix) <= kmax:  # w2: ky = -(kmax - iy)
                    kept += 1
    return 2 * d_v * d_v * kept                      # factor 2 = (Re, Im)


def block_params(d_v: int, kmax: int, radial: bool = True) -> int:
    return spectral_params(d_v, kmax, radial) + d_v * d_v + d_v   # + pointwise W


def total_params(d_v: int = D_V, kmax: int = KMAX, n_blocks: int = N_BLOCKS,
                 radial: bool = True) -> int:
    spectral = n_blocks * block_params(d_v, kmax, radial)
    lift = C_IN * LIFT_HIDDEN + LIFT_HIDDEN + LIFT_HIDDEN * d_v + d_v
    proj = d_v * PROJ_HIDDEN + PROJ_HIDDEN + PROJ_HIDDEN * C_OUT + C_OUT
    return spectral + lift + proj


# ---------------------------------------------------------------------------
# Numerical dispersion (§3.4)
# ---------------------------------------------------------------------------
def dispersion_error(ppw: float, order: int) -> float:
    """Relative phase-velocity error of the staggered operator."""
    k_dx = 2.0 * math.pi / ppw
    if order == 2:
        return k_dx ** 2 / 24.0
    if order == 4:
        return 3.0 * k_dx ** 4 / 640.0
    raise ValueError(f"order must be 2 or 4, got {order}")


def accumulated_phase(ppw: float, order: int, n_wavelengths: float) -> float:
    """Accumulated phase error in radians after propagating n_wavelengths."""
    return 2.0 * math.pi * n_wavelengths * dispersion_error(ppw, order)


# ---------------------------------------------------------------------------
# Sponge damping profile (§3.6)
# ---------------------------------------------------------------------------
def absorber_d0(thickness_cells: int = N_ABSORBER_FINE, dx: float = DX_FINE,
                c: float = CP, p: float = ABSORBER_ORDER,
                r_target: float = ABSORBER_R_TARGET) -> float:
    """
    d_0 = -(p+1) c ln(R_target) / (2 L), the standard graded-profile design formula.

    The formula is borrowed from PML design, where R_target is the reflection of the
    *matched* layer under normal incidence.  Applied to a sponge it is only a way of
    picking a damping magnitude with the right scaling in thickness and wave speed;
    the layer it produces is not matched, so the R_target that goes in is not the
    reflection that comes out.  Measure the latter --
    `solver.validate.check_absorber_reflection` -- and gate on
    GATE_ABSORBER_REFLECTION.
    """
    l_abs = thickness_cells * dx
    return -(p + 1.0) * c * math.log(r_target) / (2.0 * l_abs)


pml_d0 = absorber_d0        # deprecated alias


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def self_check(verbose: bool = True) -> None:
    """Recompute every claim in the document and assert it."""
    lines: list[str] = []
    ok = lines.append

    # -- grids -------------------------------------------------------------
    assert abs(DX_NET - LAMBDA_P / 16) < 1e-12, "network dx must be lambda_p/16"
    assert abs(DX_FINE - LAMBDA_P / 32) < 1e-12, "fine dx must be lambda_p/32"
    assert N_FINE == 2 * N_NET
    ok(f"grids            : net {N_NET}^2 (dx=lp/{LAMBDA_P/DX_NET:.0f}), "
       f"fine {N_FINE}^2 (dx=lp/{LAMBDA_P/DX_FINE:.0f}), "
       f"padded {N_FINE_TOTAL}^2")

    # -- CFL ---------------------------------------------------------------
    assert CFL_NUMBER <= CFL_LIMIT_4TH, (
        f"CFL violated: {CFL_NUMBER:.4f} > {CFL_LIMIT_4TH:.4f}")
    margin = CFL_NUMBER / CFL_LIMIT_4TH
    assert margin <= 0.95, f"CFL margin too thin: at {margin:.1%} of the limit"
    ok(f"CFL              : dt={DT:.6f} T_p, c dt/dx={CFL_NUMBER:.4f}, "
       f"limit={CFL_LIMIT_4TH:.4f} ({margin:.1%} of limit), nt={NT}")
    assert NT % N_SAVED_FRAMES == 0, "saved frames must divide nt evenly"

    # -- frequency band ----------------------------------------------------
    assert DF <= 1.0 / T_END + 1e-12, (
        f"time-domain wrap-around: df={DF:.4f} > 1/T_end={1.0/T_END:.4f}")
    assert FREQS[0] > (1.0 - 2.0 / N_CYCLES) * FC, "band edge hits the lower null"
    assert FREQS[-1] < (1.0 + 2.0 / N_CYCLES) * FC, "band edge hits the upper null"
    ok(f"band             : M={M_FREQ}, f in [{FREQS[0]:.3f}, {FREQS[-1]:.3f}] f_c, "
       f"df={DF:.4f} <= 1/T_end={1.0/T_END:.4f}; nulls at "
       f"{1-2/N_CYCLES:.2f} and {1+2/N_CYCLES:.2f} f_c")

    # -- material table ----------------------------------------------------
    ok("")
    ok("  nu     cs/cp   lam_s   L/lam_s  ppw_s(fine)  ppw_s(net)  k_carrier  x1.4")
    for m in MATERIALS:
        ok(f"  {m.nu:.2f}   {m.cs_cp:.3f}   {m.lambda_s:.3f}   "
           f"{m.domain_in_lambda_s:6.1f}   {m.ppw_s_fine:9.1f}   "
           f"{m.ppw_s_net:9.1f}   {m.k_carrier:7.1f}   {m.k_carrier*1.4:5.1f}")
        # every material must be resolved on the network grid
        assert m.ppw_s_net >= 6.0, f"nu={m.nu}: shear wave under-resolved on net grid"
        lam, mu = m.lam, m.mu
        assert lam + mu > 0, "positive-definiteness requires lambda + mu > 0"
        assert abs((lam + 2 * mu) - 1.0) < 1e-12, "c_p must be 1"
        assert abs(mu - m.cs_cp ** 2) < 1e-12, "mu must be c_s^2"
        # round-trip nu
        nu_back = lam / (2.0 * (lam + mu))
        assert abs(nu_back - m.nu) < 1e-9, f"Lame round-trip failed for nu={m.nu}"
    ok("")

    # -- mode budget -------------------------------------------------------
    assert KMAX >= K_REQUIRED, (
        f"mode budget: kmax={KMAX} < required {K_REQUIRED:.1f}")
    assert KMAX < K_NYQUIST, f"kmax={KMAX} exceeds Nyquist {K_NYQUIST}"
    ok(f"mode budget      : worst carrier k={K_CARRIER_WORST:.1f} (nu={NU_WORST}), "
       f"x{BURST_SPREAD} -> {K_REQUIRED:.1f}, kmax={KMAX}, Nyquist={K_NYQUIST}")

    # -- dispersion (§3.4 table) -------------------------------------------
    ok("")
    ok("  ppw     eps_2      dpsi_2 (16 lam)    eps_4      dpsi_4 (16 lam)")
    for ppw in (6.4, 8.0, 10.75, 12.8, 16.0):
        e2 = dispersion_error(ppw, 2)
        e4 = dispersion_error(ppw, 4)
        ok(f"  {ppw:5.2f}  {100*e2:7.3f} %  {accumulated_phase(ppw,2,16):10.3f} rad  "
           f"{100*e4:7.4f} %  {accumulated_phase(ppw,4,16):10.4f} rad")
    worst = MATERIALS[-1]
    e4_worst = dispersion_error(worst.ppw_s_net, 4)
    psi4_worst = accumulated_phase(worst.ppw_s_net, 4, worst.domain_in_lambda_s)
    psi2_worst = accumulated_phase(worst.ppw_s_net, 2, worst.domain_in_lambda_s)
    ok("")
    ok(f"worst case nu={worst.nu}: ppw_s(net)={worst.ppw_s_net:.1f}, "
       f"eps_4={100*e4_worst:.2f} %/lam, accumulated over "
       f"{worst.domain_in_lambda_s:.1f} lam_s = {psi4_worst:.2f} rad "
       f"(2nd order would give {psi2_worst:.2f} rad)")
    assert psi4_worst < math.pi / 4, (
        "4th-order dispersion is not comfortably below the cycle-skip threshold")
    assert psi2_worst > math.pi, (
        "the 2nd-order argument of §3.4 does not hold at this resolution")

    # -- parameter counts (§6.4 table) -------------------------------------
    ok("")
    ok("  variant     d_v  kmax   square/layer   radial/layer   kept    total(4)")
    for name, v in VARIANTS.items():
        d_v, kmax = v["d_v"], v["kmax"]
        sq = spectral_params(d_v, kmax, radial=False)
        ra = spectral_params(d_v, kmax, radial=True)
        ok(f"  {name:10s} {d_v:4d} {kmax:5d}   {sq/1e6:9.2f} M   {ra/1e6:9.2f} M   "
           f"{ra/sq:5.1%}   {total_params(d_v, kmax)/1e6:6.2f} M")
    frac = spectral_params(D_V, KMAX, True) / spectral_params(D_V, KMAX, False)
    assert 0.72 <= frac <= 0.82, f"radial mask keeps {frac:.1%}, expected 75-79%"

    # -- training batch arithmetic ----------------------------------------
    assert 1 <= N_FREQ_PER_SAMPLE <= M_FREQ, "frequency subset out of range"
    eff = BATCH_SIZE * N_FREQ_PER_SAMPLE
    act_mb = eff * D_V * N_NET * N_NET * 4 / 1e6
    ok(f"batch            : {BATCH_SIZE} samples x {N_FREQ_PER_SAMPLE} freqs "
       f"= {eff} fields; {act_mb:.0f} MB per d_v={D_V} activation; "
       f"{M_FREQ/N_FREQ_PER_SAMPLE:.0f} epochs to see every (sample, freq) pair once")

    # -- rings -------------------------------------------------------------
    assert len(RECEIVERS_NET) == N_RECV, f"{len(RECEIVERS_NET)} receivers, want {N_RECV}"
    assert len(SOURCES_NET) == N_SRC, f"{len(SOURCES_NET)} sources, want {N_SRC}"
    assert len(set(RECEIVERS_NET)) == N_RECV, "duplicate receiver positions"
    assert len(set(SOURCES_NET)) == N_SRC, "duplicate source positions"
    for iy, ix in RECEIVERS_NET + SOURCES_NET:
        assert RING_INSET_NET <= iy < N_NET - RING_INSET_NET
        assert RING_INSET_NET <= ix < N_NET - RING_INSET_NET
    ok(f"ring             : {N_RECV} receivers, {N_SRC} sources, "
       f"{RING_INSET_NET} cells inside the cropped edge")

    # -- inversion basins (§8.3, §8.4) -------------------------------------
    #
    # The waveform basin is a half-wavelength resolution estimate and stands.  The
    # "envelope basin = N_c lam_s / 2" of v2.0 does NOT stand and is not asserted
    # here any more: it was derived by treating a magnitude-spectrum misfit as an
    # envelope misfit, and a magnitude spectrum is exactly invariant to the time
    # shifts the basin was supposed to cover (inverse/timedomain.py, identity A).
    # The genuine envelope objective does have a wider basin, but its width is an
    # empirical property of elastic multipath, so it is *measured* --
    # `inverse.misfit.envelope_basin_width` -- and the screen's coverage is measured
    # too, by `inverse.invert.screen_capture_rate` against GATE_SCREEN_CAPTURE.
    #
    # What remains asserted is only the part that is a definition: the screen must be
    # finer than the wide objective's basin and coarser than the narrow one's, or the
    # four-stage structure has no reason to exist.
    lam_s_ref = material(1.0 / 3.0).lambda_s          # the nu = 1/3 reference
    basin_waveform = lam_s_ref / 4.0
    basin_envelope_nominal = N_CYCLES * lam_s_ref / 2.0   # v2.0's number, kept for
    #                                                      comparison only
    # interior span / (SCREEN_GRID - 1), because `inverse.invert.screen_candidates`
    # builds the lattice with an endpoint-inclusive linspace: 16 nodes, 15 intervals.
    # L_DOMAIN / SCREEN_GRID would overstate the interior reach and understate the
    # spacing at the same time, in opposite directions.
    screen_spacing = ((L_DOMAIN - 2.0 * BOUNDARY_KEEPOUT_LS * LAMBDA_S_MIN)
                      / (SCREEN_GRID - 1))
    assert screen_spacing > basin_waveform, (
        f"screen spacing {screen_spacing:.3f} is finer than the waveform basin "
        f"{basin_waveform:.3f}: stages 1-2 are then redundant")
    ok(f"inversion        : waveform basin lam_s/4 = {basin_waveform:.3f} "
       f"({to_mm(basin_waveform):.1f} mm), screen spacing {screen_spacing:.3f} "
       f"({to_mm(screen_spacing):.0f} mm) over "
       f"{SCREEN_GRID}x{SCREEN_GRID}={SCREEN_GRID**2} candidates; "
       f"envelope basin MEASURED, not assumed "
       f"(v2.0 asserted {basin_envelope_nominal:.2f})")
    ok(f"objectives       : stage 1 {STAGE_OBJECTIVE[1]}, "
       f"stage 2 {STAGE_OBJECTIVE[2]}, stage 3 {STAGE_OBJECTIVE[3]}; "
       f"screen {SCREEN_OBJECTIVE} "
       f"(fallback {SCREEN_FALLBACK_OBJECTIVE}), "
       f"capture-rate gate {GATE_SCREEN_CAPTURE:.2f}")

    # -- interface width, in physical units (§2.4, §3.6) -------------------
    ir = interface_report()
    assert ir["ratio_to_r_min"] > 0.0
    ok(f"interface        : eps = {ir['eps_fine_cells']:.3f} fine cells = "
       f"{ir['eps_phys']:.5f}, 10-90% width "
       f"{ir['transition_fine_cells']:.2f} fine cells = "
       f"{ir['transition_mm']:.2f} mm = "
       f"{ir['transition_over_lambda_s_worst']:.3f} lam_s(worst); "
       f"r_min = {ir['r_min_over_lambda_s']:.2f} lam_s, ratio "
       f"{ir['ratio_to_r_min']:.2f}, impedance ratio "
       f"{ir['impedance_ratio']:.1e}")
    if ir["ratio_to_r_min"] > 1.0:
        ok(f"                   WARNING: the transition is wider than the smallest "
           f"radius.  This is a graded soft inclusion, not a smoothed cavity; "
           f"solver/cavity.py measures the error against Pao-Mow "
           f"(gate {GATE_CAVITY_REL_L2:.2f} rel-L2, "
           f"{GATE_CAVITY_PHASE_RAD:.2f} rad phase)")
    else:
        ok(f"                   width and void density are both set from the "
           f"measured cavity error; validate.check_cavity_scattering re-measures "
           f"it (gate {GATE_CAVITY_REL_L2:.2f} rel-L2, "
           f"{GATE_CAVITY_PHASE_RAD:.2f} rad phase)")
    ok(f"eps anneal       : {'ON' if EPS_INVERT_ANNEAL else 'OFF'} "
       f"({EPS_INVERT_START:.3f} -> {EPS_INVERT_END:.3f} net cells, training used "
       f"{EPS_INTERFACE_CELLS:.4f}); off until sensitivity transfer is shown")

    # screen memory footprint, §8.4(d) -- counting the frequency axis this time
    sm = screen_memory_mb()
    ok(f"screen memory    : {sm['chunk']} candidates x {sm['n_freq']} freqs = "
       f"{sm['rows']} rows -> input {sm['input_mb']:.0f} MB, "
       f"features {sm['feature_mb']:.0f} MB, spectral {sm['spectral_mb']:.0f} MB, "
       f"{sm['chunks']} chunks "
       f"(v2.0 quoted 50 MB, which was one frequency's input only)")

    # -- physical units ----------------------------------------------------
    ok("")
    ok(f"aluminium        : c_p={CP_ALUMINIUM:.0f} m/s, f_c={FC_PHYSICAL/1e3:.0f} kHz "
       f"-> lam_p={to_mm(1.0):.1f} mm, lam_s={to_mm(lam_s_ref):.1f} mm, "
       f"L={to_mm(L_DOMAIN):.0f} mm, "
       f"R in [{to_mm(R_MIN_LS*lam_s_ref):.0f}, {to_mm(R_MAX_LS*lam_s_ref):.0f}] mm")

    # -- SDF clip consistency (§2.4) ---------------------------------------
    clip_in_ls = SDF_CLIP_CELLS * DX_NET / LAMBDA_S_MIN
    assert clip_in_ls >= 1.7, f"SDF clip is only {clip_in_ls:.2f} lambda_s at worst nu"
    ok(f"sdf clip         : {SDF_CLIP_CELLS:.0f} cells = "
       f"{clip_in_ls:.2f} lam_s at nu={NU_WORST}, "
       f"{SDF_CLIP_CELLS*DX_NET/material(1/3).lambda_s:.1f} lam_s at nu=1/3")

    # -- absorber ----------------------------------------------------------
    abs_lam_p_lo = N_ABSORBER_FINE * DX_FINE / (CP / FREQS[0])
    assert abs_lam_p_lo > 1.0, (
        f"absorber is {abs_lam_p_lo:.2f} lambda_p at the lowest frequency; measured, "
        "no order or R_target reaches GATE_ABSORBER_REFLECTION below about 1")
    ok(f"absorber         : graded sponge (NOT a PML), {N_ABSORBER_FINE} fine "
       f"cells = {N_ABSORBER_FINE*DX_FINE:.3f} = {abs_lam_p_lo:.2f} "
       f"lambda_p(f_lo), p={ABSORBER_ORDER:.0f}, design R={ABSORBER_R_TARGET:g}, "
       f"d0={absorber_d0():.2f}; measured artefact 1.02% vs open domain, gated at "
       f"{GATE_ABSORBER_REFLECTION:g}, open-domain scope")

    if verbose:
        print("\n".join(lines))
        print("\nall self-checks passed")


if __name__ == "__main__":
    self_check()
