"""
The analytic reference the reference problem is supposed to be: a traction-free
circular cavity in an unbounded elastic plane (§2.4, §3.6-3.7, §7.2).

Everything else in this project takes for granted that the labels are cavity
scattering.  Nothing so far has checked it.  The production geometry is not a
cavity: `geometry/sdf.material_fields` builds a sigmoid-graded soft inclusion whose
stiffness bottoms out at `VOID_STIFFNESS_FLOOR` and whose density is left at the
solid value, and `config.interface_report()` shows the 10-90% transition is
`2 ln(9) eps = 6.59` network cells, about 0.97 of the shortest shear wavelength and
*wider than the smallest defect radius*.  A graded soft inclusion that wide is a
different scattering problem from a hole, and the difference has to be a measured
number, not an assumption -- if the labels are not cavity scattering then every
inversion result is an inversion for the wrong physics, and no amount of network
accuracy repairs that.

This module supplies the reference, and only the reference: it is pure numpy plus
`scipy.special`, imports no torch and never touches the solver.  The solves and the
comparison live in `solver/validate.py` (`check_green_incident`,
`check_cavity_scattering`, `check_interface_width`), which hands the measured
receiver phasors to `cavity_accuracy_report` here.  Keeping the analytic side free
of the solver is what makes it usable as an independent check at all: a reference
that shared code with the thing under test would agree with it for the wrong
reasons.

Conventions, all forced by choices made elsewhere and none of them free:

* `fdtd_elastic.dft_at_freqs` transforms with `exp(-i omega t)`, so `d/dt` maps to
  `+i omega`, phasors carry `exp(+i omega t)`, and an *outgoing* cylindrical wave is
  `H^(2)`, not `H^(1)`.  Get this backwards and every scattered phase flips sign
  while every magnitude stays right, which is the most expensive kind of error to
  find late.  `cavity_accuracy_report` therefore reports the residual against the
  conjugate reference as well, so a convention slip shows up as one number instead
  of a mystery.
* `g_alpha(r) = -(i/4) H_0^(2)(k_alpha r)` solves `(lap + k^2) g = -delta`.
* the solver injects `b_vy * s(t) / dx^2 / rho`, a body-force density integrating to
  a total force `s(t)` in `+y`, and `harmonic.transfer_factor` divides out
  `i omega s_hat`.  So `harmonic.displacement_from_ascans` of an incident run *is*
  the Green's tensor column for a unit `+y` point force -- `(G_xy, G_yy)`, with no
  free amplitude to fit.  That is why `check_green_incident` can be a direct
  comparison and not a correlation.

What is checked and how, since an analytic reference nobody has verified is just a
second implementation of the same guess (`self_test`, and `tests/test_cavity.py`):

* `green_tensor` is finite-differenced and substituted into the Navier operator; the
  residual falls as `h^2`.  The `(g_p - g_s)` ordering, which is the sign error this
  formula invites, does not converge at all, and the test asserts that too.
* `incident_potential_coeffs` (Graf's addition theorem, derivative moved onto the
  source coordinate) is compared against a direct `green_tensor` evaluation.  They
  are two unrelated computations of the same field and they agree to ~1e-15.
* `_traction_matrix` is compared against numerical differentiation of the modal
  displacement field followed by Hooke's law; the error falls as `h^2`.
* the traction of the matched total field on `r = radius` is machine zero.

Together those pin the solution down completely: it satisfies the PDE, it satisfies
the traction-free boundary condition, and it is outgoing, so by uniqueness it is the
cavity field.  No comparison against published tables is needed or offered.

Scope: this is an *open-domain* solution.  The solver's graded sponge only
approximates an open domain, so any comparison against it inherits the sponge's
residual reflection -- measured separately by
`validate.check_absorber_reflection`.  Both errors land in the same residual and the
two checks are what separate them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import hankel2 as _h2, jv as _jv

from .. import config as cfg

__all__ = [
    "scalar_green", "green_tensor", "green_displacement",
    "incident_potential_coeffs", "cavity_field", "CavityField",
    "traction_residual", "n_modes_for", "sample_like_solver",
    "product_rule_symbol", "collocated_symbol", "staggered_symbol",
    "operator_defect_report", "cavity_accuracy_report", "self_test",
]


# ---------------------------------------------------------------------------
# Elastic constants
# ---------------------------------------------------------------------------
def _speeds(nu: float, *, cp: float = cfg.CP, rho: float = cfg.RHO0):
    """(cp, cs, lam, mu, rho) from Poisson's ratio, via config's own conversion.

    Reads `cfg.lame_from_nu` rather than re-deriving the moduli: the reference has
    to be built from the *same* constants the solver runs with, or the comparison
    measures a materials mismatch and calls it a discretisation error.
    """
    lam, mu = (float(v) for v in cfg.lame_from_nu(nu))
    cs = cp * float(cfg.cs_over_cp(nu))
    return cp, cs, lam * rho * cp ** 2, mu * rho * cp ** 2, rho


# ---------------------------------------------------------------------------
# Scalar Helmholtz Green's function
# ---------------------------------------------------------------------------
def scalar_green(r, k):
    """
    `g = -(i/4) H_0^(2)(k r)` and `dg/dr`, the outgoing solution of
    `(lap + k^2) g = -delta`.

    `dg/dr = +(i/4) k H_1^(2)(k r)` because `d/dz H_0 = -H_1`.  The second
    derivative is never needed as `H_2`: the Helmholtz equation in polar
    coordinates gives `g'' = -g'/r - k^2 g` away from the origin, which is both
    cheaper and one fewer special-function call to get wrong.
    """
    r = np.asarray(r, dtype=float)
    z = k * r
    return -0.25j * _h2(0, z), 0.25j * k * _h2(1, z)


# ---------------------------------------------------------------------------
# Elastodynamic Green's tensor
# ---------------------------------------------------------------------------
def green_tensor(dx, dy, omega, *, cs: float, cp: float = cfg.CP,
                 rho: float = cfg.RHO0, ordering: int = +1) -> np.ndarray:
    """
    2D plane-strain Green's tensor at offset `(dx, dy)` from a point force.

        rho w^2 G_ij = d_i d_j (g_s - g_p) + delta_ij k_s^2 g_s

    `dx, dy, omega` broadcast against each other; the result is that shape with a
    trailing `(2, 2)`.  Index order is `(x, y)` -- the same order the A-scan
    components come out in, so `G[..., :, 1]` is directly the column
    `harmonic.displacement_from_ascans` returns for the solver's `+y` force.

    The second derivatives avoid `H_2` by way of the Helmholtz ODE:

        d_i d_j g = -k^2 g n_i n_j + (g'/r) (delta_ij - 2 n_i n_j),   n = x/r

    `ordering = -1` selects the `(g_p - g_s)` form.  It is here on purpose: that
    ordering is the natural-looking guess, it is wrong, and `self_test` asserts it
    fails the Navier residual rather than leaving the sign as something a future
    reader has to re-derive.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    om = np.asarray(omega, dtype=float)
    r = np.hypot(dx, dy)
    if np.any(r == 0.0):
        raise ValueError("green_tensor is singular at the source point")
    kp, ks = om / cp, om / cs
    gp, dgp = scalar_green(r, kp)
    gs, dgs = scalar_green(r, ks)
    nvec = np.stack(np.broadcast_arrays(dx / r, dy / r), axis=-1)
    nn = nvec[..., :, None] * nvec[..., None, :]
    eye = np.eye(2)

    def hess(g, dg, k):
        """d_i d_j g, with the k^2 and 1/r factors folded in before adding axes.

        The order matters for broadcasting: `dx, dy` may carry a receiver axis and
        `omega` a frequency axis, so every scalar coefficient has to be formed at
        the natural broadcast shape and only then given its trailing (2, 2).
        """
        a = (-(k ** 2) * g)[..., None, None]
        b = (dg / r)[..., None, None]
        return a * nn + b * (eye - 2 * nn)

    diff = ordering * (hess(gs, dgs, ks) - hess(gp, dgp, kp))
    iso = eye * (ks ** 2 * gs)[..., None, None]
    return (diff + iso) / (rho * (om ** 2)[..., None, None])


def green_displacement(src_xy, recv_xy, freqs, *, nu: float,
                       force=(0.0, 1.0), cp: float = cfg.CP,
                       rho: float = cfg.RHO0) -> np.ndarray:
    """
    Incident displacement `[R, 2, M]` at `recv_xy` from a point force at `src_xy`.

    Positions are physical `(x, y)` in the same frame `cfg.receiver_position` and
    `cfg.source_position` return, `freqs` is in units of `f_c` like `cfg.FREQS`, and
    the default force is the solver's unit `+y`.  The layout `[R, 2, M]` is the one
    `harmonic.displacement_from_ascans` produces after its batch axis, so the two
    can be subtracted without a reshape and without an opportunity to transpose
    components against frequencies.
    """
    _, cs, _, _, rho = _speeds(nu, cp=cp, rho=rho)
    recv = np.asarray(recv_xy, dtype=float).reshape(-1, 2)
    src = np.asarray(src_xy, dtype=float).reshape(2)
    om = 2.0 * np.pi * np.asarray(freqs, dtype=float).reshape(-1)
    d = recv - src
    G = green_tensor(d[:, 0:1], d[:, 1:2], om[None, :], cs=cs, cp=cp, rho=rho)
    u = G @ np.asarray(force, dtype=float)
    return np.ascontiguousarray(np.moveaxis(u, -1, 1))


# ---------------------------------------------------------------------------
# The solver's sampling operator, applied to an analytic field
# ---------------------------------------------------------------------------
def _sampling_stencil(downsample: int) -> tuple[dict, dict]:
    """
    Offsets (in fine cells, from the network cell centre) and weights for the two
    velocity components, reproducing `ElasticFDTD2D.run`'s recording path.

    That path is `avg_minus` onto fine cell centres followed by `avg_pool2d` over
    the D x D block, so for vx each fine cell contributes the mean of its two
    x-faces and for vy the mean of its two y-faces.  Built by accumulation rather
    than written out, because the collapse of coincident points (fine cell q's
    upper face is cell q+1's lower face, and gets weight 2) is exactly the sort of
    thing that is wrong when written out by hand.
    """
    wx: dict[tuple[float, float], float] = {}
    wy: dict[tuple[float, float], float] = {}
    w_cell = 1.0 / (downsample * downsample)
    for p in range(downsample):
        oy = p + 0.5 - 0.5 * downsample            # fine centre, y
        for q in range(downsample):
            ox = q + 0.5 - 0.5 * downsample        # fine centre, x
            for s in (-0.5, 0.5):
                kx = (ox + s, oy)
                wx[kx] = wx.get(kx, 0.0) + 0.5 * w_cell
                ky = (ox, oy + s)
                wy[ky] = wy.get(ky, 0.0) + 0.5 * w_cell
    return wx, wy


def sample_like_solver(field_fn, recv_xy, *, dx_fine: float = cfg.DX_FINE,
                       downsample: int = cfg.DOWNSAMPLE) -> np.ndarray:
    """
    Pass an analytic field through the operator the solver's A-scans come out of.

    An A-scan is not the field at the receiver.  Velocities live on faces, so
    `run` averages each component onto fine cell centres and then block-averages
    2x2 fine cells into one network cell; the number recorded is therefore the
    field convolved with a [1,2,1]/4 stencil at spacing dx_fine along its own
    component direction and [1,1]/2 across it.  For a shear wave at 45 degrees at
    the top of the band that is a 5% amplitude reduction, half of
    GATE_CAVITY_REL_L2 -- so a check that compares a point-evaluated analytic
    field against an A-scan is measuring this smoothing, not the solver.

    `field_fn(points) -> [P, 2, ...]` is evaluated once, at every distinct offset
    of the stencil for every receiver (12 offsets when downsample = 2).  The
    smoothing is applied to the reference rather than deconvolved out of the
    measurement because the reference is exact there and the measurement is not.
    """
    recv = np.asarray(recv_xy, dtype=float).reshape(-1, 2)
    wx, wy = _sampling_stencil(int(downsample))
    offs = sorted(set(wx) | set(wy))
    pts = np.concatenate([recv + np.asarray(o) * dx_fine for o in offs], axis=0)
    u = np.asarray(field_fn(pts))
    if u.shape[0] != len(offs) * recv.shape[0] or u.shape[1] != 2:
        raise ValueError(f"field_fn returned {u.shape}, want "
                         f"[{len(offs) * recv.shape[0]}, 2, ...]")
    u = u.reshape((len(offs), recv.shape[0]) + u.shape[1:])
    out = np.zeros(u.shape[1:], dtype=complex)
    for k, o in enumerate(offs):
        out[:, 0] += wx.get(o, 0.0) * u[k, :, 0]
        out[:, 1] += wy.get(o, 0.0) * u[k, :, 1]
    return out


# ---------------------------------------------------------------------------
# Cylindrical modes about the cavity
# ---------------------------------------------------------------------------
def _dh2(n, z):
    """d/dz H_n^(2)(z), by the recurrence, so only orders n-1 and n are evaluated."""
    return _h2(n - 1, z) - (n / z) * _h2(n, z)


def _dj(n, z):
    return _jv(n - 1, z) - (n / z) * _jv(n, z)


def n_modes_for(ks: float, radius: float) -> int:
    """
    Truncation order for the cylindrical series.

    `ks * radius + 4 (ks * radius)^(1/3) + 10` is the usual rule for Bessel series
    on a circle: the coefficients decay factorially once `n` clears the argument,
    and the cube-root term covers the transition region.  A floor of 12 keeps the
    small-cavity case honest, and `cavity_field` reports the size of the last
    retained term so truncation is a measured quantity rather than a hope.
    """
    z = max(float(ks) * float(radius), 1e-12)
    return max(12, int(math.ceil(z + 4.0 * z ** (1.0 / 3.0) + 10.0)))


def incident_potential_coeffs(ns, r_s: float, th_s: float, force, *,
                              kp: float, ks: float, omega: float,
                              rho: float = cfg.RHO0):
    """
    Cylindrical-harmonic coefficients `(a_n, b_n)` of the incident point-force field
    about the cavity centre, valid for `r < r_s`:

        phi_inc = sum_n a_n J_n(k_p r) e^{i n th}
        psi_inc = sum_n b_n J_n(k_s r) e^{i n th}
        u_inc   = grad phi + curl(psi z)

    The potentials come from the Green's tensor itself.  Writing
    `u_i = (1/rho w^2)[d_i d_j (g_s - g_p) f_j + k_s^2 g_s f_i]`, the first term is
    already a gradient and the rest is divergence-free by the Helmholtz equation, so

        phi = -(1/rho w^2) (f . grad) g_p
        psi =  (1/rho w^2) (f_y d_x - f_x d_y) g_s

    Then `grad_x g(|x - x_s|) = -grad_{x_s} g`, which moves both derivatives onto the
    source coordinate.  Under Graf's addition theorem the source coordinate appears
    only in `H_n^(2)(k r_s) e^{-i n th_s}`, so the derivative acts on a single factor
    and the expansion stays modal.  Doing it the other way round -- differentiating
    `J_n(k r) e^{i n th}` -- mixes orders and loses the whole point of the basis.

    Returned in the `(f_r, f_th)` combinations of the force, which is what makes the
    two coefficients each a single term rather than a sum over Cartesian components.
    """
    ns = np.asarray(ns)
    fx, fy = float(force[0]), float(force[1])
    f_r = fx * math.cos(th_s) + fy * math.sin(th_s)
    f_th = -fx * math.sin(th_s) + fy * math.cos(th_s)
    pre = 1.0 / (rho * omega ** 2)
    phase = np.exp(-1j * ns * th_s)
    a = (-0.25j * pre) * (kp * _dh2(ns, kp * r_s) * f_r
                          - 1j * ns / r_s * _h2(ns, kp * r_s) * f_th) * phase
    b = (+0.25j * pre) * (ks * _dh2(ns, ks * r_s) * f_th
                          + 1j * ns / r_s * _h2(ns, ks * r_s) * f_r) * phase
    return a, b


def _traction_matrix(n, r: float, kp: float, ks: float, *, outgoing: bool):
    """
    `(r^2 / 2mu) (sigma_rr, sigma_rth)` as a 2x2 map from `(A_n, B_n)`.

    Scaled by `r^2 / 2mu` so the entries are dimensionless combinations of the
    radial functions and `n` only -- no elastic constants survive except through
    `k_s r`, which is why the same matrix serves every Poisson ratio.  The Bessel ODE
    has been used to eliminate the second derivatives, so only `Z_n` and `Z_n'`
    appear.  `outgoing` selects `H_n^(2)` (scattered) over `J_n` (incident).
    """
    x, y = kp * r, ks * r
    if outgoing:
        zp, dzp, zs, dzs = _h2(n, x), _dh2(n, x), _h2(n, y), _dh2(n, y)
    else:
        zp, dzp, zs, dzs = _jv(n, x), _dj(n, x), _jv(n, y), _dj(n, y)
    nn = n ** 2
    half = 0.5 * y ** 2
    return np.array([[(nn - half) * zp - x * dzp, 1j * n * (y * dzs - zs)],
                     [1j * n * (x * dzp - zp), -((nn - half) * zs - y * dzs)]])


@dataclass
class CavityField:
    """Receiver displacements `[R, 2, M]` for the traction-free-cavity problem."""
    incident: np.ndarray
    scattered: np.ndarray
    total: np.ndarray
    n_modes: int
    tail: float = 0.0          # largest retained mode / largest mode, worst frequency
    extras: dict = field(default_factory=dict)


def cavity_field(src_xy, recv_xy, freqs, *, radius: float, nu: float,
                 centre=(0.0, 0.0), force=(0.0, 1.0), cp: float = cfg.CP,
                 rho: float = cfg.RHO0, n_modes: int | None = None) -> CavityField:
    """
    Exact displacement at `recv_xy` for a traction-free circular cavity of radius
    `radius` centred at `centre`, excited by a point force at `src_xy`.

    The incident field at the receivers is evaluated *directly* from
    `green_tensor`, not from the modal series.  This is not an optimisation: the
    receiver ring in this geometry sits at `r ~ 5.0` while the sources sit at
    `r ~ 4.3`, so the receivers are outside the source radius and the `r < r_s`
    expansion does not converge there.  The series is used only where it belongs --
    on `r = radius`, where `radius / r_s ~ 0.1` and it converges in a handful of
    terms.  The scattered field is outgoing everywhere outside the cavity, so its
    modal sum is valid at any receiver.
    """
    _, cs, _, _, rho = _speeds(nu, cp=cp, rho=rho)
    centre = np.asarray(centre, dtype=float).reshape(2)
    recv = np.asarray(recv_xy, dtype=float).reshape(-1, 2) - centre
    src = np.asarray(src_xy, dtype=float).reshape(2) - centre
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    r_s = float(np.hypot(*src))
    th_s = float(math.atan2(src[1], src[0]))
    r_r = np.hypot(recv[:, 0], recv[:, 1])
    th_r = np.arctan2(recv[:, 1], recv[:, 0])
    if r_s <= radius or np.any(r_r <= radius):
        raise ValueError("source and receivers must lie outside the cavity")

    om = 2.0 * np.pi * freqs
    ks_max = float(om.max()) / cs
    nmax = n_modes_for(ks_max, radius) if n_modes is None else int(n_modes)
    ns = np.arange(-nmax, nmax + 1)

    scattered = np.zeros((recv.shape[0], 2, om.size), dtype=complex)
    tails = []
    for m, omega in enumerate(om):
        kp, ks = omega / cp, omega / cs
        a, b = incident_potential_coeffs(ns, r_s, th_s, force,
                                         kp=kp, ks=ks, omega=omega, rho=rho)
        u_r = np.zeros(recv.shape[0], dtype=complex)
        u_t = np.zeros(recv.shape[0], dtype=complex)
        mags = np.zeros(ns.size)
        for i, n in enumerate(ns):
            rhs = -_traction_matrix(n, radius, kp, ks, outgoing=False) @ np.array([a[i], b[i]])
            mat = _traction_matrix(n, radius, kp, ks, outgoing=True)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                sol = np.linalg.solve(mat, rhs)
            if not np.all(np.isfinite(sol)):
                continue        # H_n overflowed: the true term is far below roundoff
            an, bn = sol
            # u = grad phi + curl(psi z), outgoing radial functions
            zp, dzp = _h2(n, kp * r_r), _dh2(n, kp * r_r)
            zs, dzs = _h2(n, ks * r_r), _dh2(n, ks * r_r)
            e = np.exp(1j * n * th_r)
            du_r = (an * kp * dzp + bn * (1j * n / r_r) * zs) * e
            du_t = (an * (1j * n / r_r) * zp - bn * ks * dzs) * e
            u_r += du_r
            u_t += du_t
            mags[i] = max(np.abs(du_r).max(), np.abs(du_t).max())
        scattered[:, 0, m] = u_r * np.cos(th_r) - u_t * np.sin(th_r)
        scattered[:, 1, m] = u_r * np.sin(th_r) + u_t * np.cos(th_r)
        peak = mags.max()
        tails.append(max(mags[0], mags[-1]) / peak if peak > 0 else 0.0)

    incident = green_displacement(src_xy, recv_xy, freqs, nu=nu, force=force,
                                  cp=cp, rho=rho)
    return CavityField(incident=incident, scattered=scattered,
                       total=incident + scattered, n_modes=nmax,
                       tail=float(max(tails)),
                       extras={"r_source": r_s, "r_recv_min": float(r_r.min()),
                               "ks_radius": ks_max * radius})


def traction_residual(*, radius: float, r_s: float, th_s: float, freq: float,
                      nu: float, force=(0.0, 1.0), cp: float = cfg.CP,
                      rho: float = cfg.RHO0, n_theta: int = 64) -> float:
    """
    `max |t_inc + t_sc| / max |t_inc|` on `r = radius` after mode matching.

    The number that says the boundary condition is actually satisfied.  It is
    machine zero by construction -- the match is done mode by mode -- so what this
    really tests is that the *same* traction operator was used on both sides and
    that the linear solve was not silently singular.  The independent test of the
    operator itself is `self_test`'s finite-difference comparison.
    """
    _, cs, _, _, rho = _speeds(nu, cp=cp, rho=rho)
    omega = 2.0 * np.pi * float(freq)
    kp, ks = omega / cp, omega / cs
    nmax = n_modes_for(ks, radius)
    ns = np.arange(-nmax, nmax + 1)
    a, b = incident_potential_coeffs(ns, r_s, th_s, force,
                                     kp=kp, ks=ks, omega=omega, rho=rho)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    tot = np.zeros((n_theta, 2), dtype=complex)
    inc = np.zeros((n_theta, 2), dtype=complex)
    for i, n in enumerate(ns):
        e_in = _traction_matrix(n, radius, kp, ks, outgoing=False) @ np.array([a[i], b[i]])
        mat = _traction_matrix(n, radius, kp, ks, outgoing=True)
        sol = np.linalg.solve(mat, -e_in)
        if not np.all(np.isfinite(sol)):
            continue
        e_sc = mat @ sol
        phase = np.exp(1j * n * thetas)[:, None]
        inc += e_in[None, :] * phase
        tot += (e_in + e_sc)[None, :] * phase
    return float(np.abs(tot).max() / np.abs(inc).max())


# ---------------------------------------------------------------------------
# Discrete-operator symbols (repository finding (d))
# ---------------------------------------------------------------------------
_COLLOCATED = {-2: 1.0 / 12.0, -1: -8.0 / 12.0, 1: 8.0 / 12.0, 2: -1.0 / 12.0}
_STAGGERED = {-1.5: 1.0 / 24.0, -0.5: -9.0 / 8.0, 0.5: 9.0 / 8.0, 1.5: -1.0 / 24.0}


def _symbol(weights: dict, theta):
    """(sum_o w_o e^{i o theta}) / (i theta): the operator's symbol over the exact one."""
    theta = np.asarray(theta, dtype=float)
    s = sum(w * np.exp(1j * o * theta) for o, w in weights.items())
    return s / (1j * theta)


def collocated_symbol(theta):
    """`losses.d1`'s 4th-order 5-point derivative, relative to exact: `-> 1`."""
    return _symbol(_COLLOCATED, theta).real


def staggered_symbol(theta):
    """`fdtd_elastic.d_plus/d_minus`'s 4th-order staggered derivative, same scaling."""
    return _symbol(_STAGGERED, theta).real


def product_rule_symbol(theta):
    """
    `D(fg) / (f Dg + g Df)` at `f = g = exp(i k x)`, `theta = k dx`, for `losses.d1`.

    A discrete derivative does not obey the product rule, and this is the cleanest
    scalar statement of by how much: put both factors at the same wavenumber and take
    the ratio.  For the 5-point collocated operator it closes to

        cos t (4 - cos 2t) / (4 - cos t)  =  1 - O(t^4)

    which is the fourth-order accuracy showing up as the second-order term
    cancelling.  It matters because `losses.navier_residual` builds the stress as a
    product of a material field and a derivative of the displacement and then
    differentiates *that*, whereas the solver never forms such a product: it
    differentiates stress components that live on their own staggered nodes.  The
    two operators therefore disagree by an amount this function measures, and
    `operator_defect_report` evaluates it at the shortest propagating shear
    wavelength on the network grid, where it is not small.
    """
    theta = np.asarray(theta, dtype=float)
    return np.cos(theta) * (4.0 - np.cos(2.0 * theta)) / (4.0 - np.cos(theta))


def operator_defect_report(nu: float = 1.0 / 3.0, *, dx: float = cfg.DX_NET,
                           f_max: float | None = None) -> dict:
    """
    The three numbers behind repository finding (d), at the worst grid point count.

    `theta = 2 pi dx f_max / c_s` is the phase advance per network cell of the
    shortest propagating shear wave -- the network grid, because that is the grid the
    physics loss is evaluated on, and the shortest wavelength, because that is where
    a difference operator is least accurate.  All three quantities are exact symbols
    of the actual stencils, and `self_test` re-measures them from the stencil
    coefficients so a formula can never drift away from the code it describes.
    """
    f_max = float(cfg.FREQS[-1]) if f_max is None else float(f_max)
    _, cs, _, _, _ = _speeds(nu)
    lam_s = cs / f_max
    theta = 2.0 * np.pi * dx / lam_s
    return {
        "nu": nu, "theta": theta, "points_per_wavelength": lam_s / dx,
        "lambda_s_min": lam_s,
        "collocated": float(collocated_symbol(theta)),
        "staggered": float(staggered_symbol(theta)),
        "derivative_mismatch": float(collocated_symbol(theta) / staggered_symbol(theta)),
        "product_rule": float(product_rule_symbol(theta)),
    }


# ---------------------------------------------------------------------------
# Comparison against measured phasors
# ---------------------------------------------------------------------------
def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / den) if den > 0.0 else float("inf")


def cavity_accuracy_report(measured: np.ndarray, reference: np.ndarray, *,
                           freqs=None) -> dict:
    """
    How far a measured receiver field is from the analytic one.

    `measured` and `reference` are `[R, 2, M]` complex.  Four numbers, because one
    would not separate the things that go wrong:

    `rel_l2`     uncalibrated relative L2 over all receivers, components and
                 frequencies.  The headline, gated by `cfg.GATE_CAVITY_REL_L2`.
    `phase_rad`  `|arg c|` for the least-squares complex scale
                 `c = <ref, meas> / <ref, ref>`.  A single systematic phase --
                 wrong wave speed, wrong quadrature offset, wrong outgoing
                 convention -- lands here and nowhere else, and it is the number
                 `cfg.GATE_CAVITY_PHASE_RAD` gates.
    `amp_ratio`  `|c|`.  Separates an amplitude calibration error (source scaling,
                 a missing `1/dx^2`) from a shape error.
    `rel_l2_cal` residual after dividing out `c`: what is left once a global
                 amplitude and phase are forgiven.  If `rel_l2` is large and
                 `rel_l2_cal` is small the disagreement is a calibration, not
                 physics.

    `rel_l2_conj` compares against the conjugated reference.  It exists because the
    `exp(-i omega t)` convention is asserted in three files and verified in none; if
    that number is the small one, the convention is backwards somewhere.
    """
    measured = np.asarray(measured, dtype=complex)
    reference = np.asarray(reference, dtype=complex)
    if measured.shape != reference.shape:
        raise ValueError(f"shape mismatch: {measured.shape} vs {reference.shape}")
    m, r = measured.reshape(-1), reference.reshape(-1)
    denom = float(np.vdot(r, r).real)
    c = complex(np.vdot(r, m) / denom) if denom > 0.0 else complex("nan")
    per_freq = [_rel_l2(measured[..., i], reference[..., i])
                for i in range(measured.shape[-1])]
    per_recv = [_rel_l2(measured[i], reference[i]) for i in range(measured.shape[0])]
    out = {
        "rel_l2": _rel_l2(m, r),
        "rel_l2_conj": _rel_l2(m, np.conj(r)),
        "rel_l2_cal": _rel_l2(m / c, r) if np.isfinite(c) and c != 0 else float("inf"),
        "amp_ratio": abs(c),
        "phase_rad": abs(math.atan2(c.imag, c.real)),
        "rel_l2_per_freq": per_freq,
        "rel_l2_per_recv": per_recv,
        "rel_l2_worst_freq": max(per_freq),
        "rel_l2_worst_recv": max(per_recv),
    }
    if freqs is not None:
        out["freqs"] = [float(f) for f in np.asarray(freqs).reshape(-1)]
    return out


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------
def _modal_u(ns, a, b, x: float, y: float, kp: float, ks: float, *,
             outgoing: bool) -> np.ndarray:
    """Cartesian `(u_x, u_y)` of a potential expansion at one point."""
    r, th = math.hypot(x, y), math.atan2(y, x)
    if outgoing:
        zp, dzp, zs, dzs = _h2(ns, kp * r), _dh2(ns, kp * r), _h2(ns, ks * r), _dh2(ns, ks * r)
    else:
        zp, dzp, zs, dzs = _jv(ns, kp * r), _dj(ns, kp * r), _jv(ns, ks * r), _dj(ns, ks * r)
    e = np.exp(1j * ns * th)
    u_r = np.sum((a * kp * dzp + b * (1j * ns / r) * zs) * e)
    u_t = np.sum((a * (1j * ns / r) * zp - b * ks * dzs) * e)
    return np.array([u_r * math.cos(th) - u_t * math.sin(th),
                     u_r * math.sin(th) + u_t * math.cos(th)])


def _navier_residual(x: float, y: float, omega: float, *, cs: float, lam: float,
                     mu: float, rho: float, h: float, ordering: int = +1,
                     force=(0.0, 1.0)) -> float:
    """`|mu lap u + (lam+mu) grad div u + rho w^2 u| / |u|`, second-order FD."""
    f = np.asarray(force, dtype=float)

    def u(px, py):
        return green_tensor(px, py, omega, cs=cs, rho=rho, ordering=ordering) @ f

    u0 = u(x, y)
    uxx = (u(x + h, y) - 2 * u0 + u(x - h, y)) / h ** 2
    uyy = (u(x, y + h) - 2 * u0 + u(x, y - h)) / h ** 2
    uxy = (u(x + h, y + h) - u(x + h, y - h) - u(x - h, y + h) + u(x - h, y - h)) / (4 * h ** 2)
    gdiv = np.array([uxx[0] + uxy[1], uxy[0] + uyy[1]])
    res = mu * (uxx + uyy) + (lam + mu) * gdiv + rho * omega ** 2 * u0
    return float(np.abs(res).max() / np.abs(u0).max())


def _traction_fd_error(ns, a, b, radius: float, kp: float, ks: float, *,
                       lam: float, mu: float, outgoing: bool, h: float) -> float:
    """
    Relative error of `_traction_matrix` against differentiating the field itself.

    The only independent check of the traction algebra there is: build the modal
    displacement field, finite-difference it into a strain, apply Hooke's law, and
    project onto the normal.  Everything else in this module would agree with a wrong
    traction matrix, because everything else uses it.
    """
    worst = scale = 0.0
    for th in (0.0, 0.9, 2.4, -1.7):
        px, py = radius * math.cos(th), radius * math.sin(th)
        grad = np.empty((2, 2), dtype=complex)
        for j, (ex, ey) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            up = _modal_u(ns, a, b, px + h * ex, py + h * ey, kp, ks, outgoing=outgoing)
            um = _modal_u(ns, a, b, px - h * ex, py - h * ey, kp, ks, outgoing=outgoing)
            grad[:, j] = (up - um) / (2.0 * h)
        eps = 0.5 * (grad + grad.T)
        sig = lam * np.trace(eps) * np.eye(2) + 2.0 * mu * eps
        nh = np.array([math.cos(th), math.sin(th)])
        tg = np.array([-math.sin(th), math.cos(th)])
        num = np.array([nh @ sig @ nh, tg @ sig @ nh])
        ana = np.zeros(2, dtype=complex)
        for i, n in enumerate(ns):
            ana += (_traction_matrix(n, radius, kp, ks, outgoing=outgoing)
                    @ np.array([a[i], b[i]])) * np.exp(1j * n * th)
        ana *= 2.0 * mu / radius ** 2
        worst = max(worst, float(np.abs(num - ana).max()))
        scale = max(scale, float(np.abs(ana).max()))
    return worst / scale


def self_test(nu: float = 1.0 / 3.0, freq: float | None = None) -> dict:
    """
    Verify the analytic reference against nothing but itself and calculus.

    Returns measured numbers; `tests/test_cavity.py` holds the thresholds, and
    `validate.run_all` prints the summary.  The point of collecting them in one
    function is that the reference is used to judge the solver, so it has to be
    judged first, by arguments that do not involve the solver at all.
    """
    freq = float(cfg.FC) if freq is None else float(freq)
    cp, cs, lam, mu, rho = _speeds(nu)
    omega = 2.0 * np.pi * freq
    kp, ks = omega / cp, omega / cs

    navier = [_navier_residual(0.7, -0.4, omega, cs=cs, lam=lam, mu=mu, rho=rho, h=h)
              for h in (0.01, 0.005, 0.0025)]
    navier_flipped = [_navier_residual(0.7, -0.4, omega, cs=cs, lam=lam, mu=mu,
                                       rho=rho, h=h, ordering=-1)
                      for h in (0.01, 0.005)]

    g = green_tensor(0.7, -0.4, omega, cs=cs, rho=rho)
    symmetry = float(abs(g[0, 1] - g[1, 0]) / abs(g).max())

    far = 400.0
    th = 0.7
    gf = green_tensor(far * math.cos(th), far * math.sin(th), omega, cs=cs, rho=rho)
    nvec = np.array([math.cos(th), math.sin(th)])
    proj = np.outer(nvec, nvec)
    gp_far, _ = scalar_green(far, kp)
    gs_far, _ = scalar_green(far, ks)
    ideal = proj * gp_far / (lam + 2.0 * mu) + (np.eye(2) - proj) * gs_far / mu
    far_field = float(np.abs(gf - ideal).max() / np.abs(ideal).max())

    # Graf coefficients against a direct evaluation, at r well inside r_s
    r_s, th_s = 4.26, 0.37
    xs, ys = r_s * math.cos(th_s), r_s * math.sin(th_s)
    ns = np.arange(-40, 41)
    a, b = incident_potential_coeffs(ns, r_s, th_s, (0.0, 1.0),
                                     kp=kp, ks=ks, omega=omega, rho=rho)
    graf = 0.0
    for (r, t) in ((0.5, 0.0), (1.2, 2.0), (2.0, -1.1)):
        px, py = r * math.cos(t), r * math.sin(t)
        modal = _modal_u(ns, a, b, px, py, kp, ks, outgoing=False)
        direct = green_tensor(px - xs, py - ys, omega, cs=cs, rho=rho) @ np.array([0.0, 1.0])
        graf = max(graf, float(np.abs(modal - direct).max() / np.abs(direct).max()))

    rng = np.random.default_rng(0)
    ns6 = np.arange(-6, 7)
    ra = rng.normal(size=ns6.size) + 1j * rng.normal(size=ns6.size)
    rb = rng.normal(size=ns6.size) + 1j * rng.normal(size=ns6.size)
    traction_fd = {
        f"{kind}_h{h:g}": _traction_fd_error(ns6, ra, rb, 0.6, kp, ks, lam=lam, mu=mu,
                                             outgoing=(kind == "outgoing"), h=h)
        for kind in ("regular", "outgoing") for h in (1e-3, 2.5e-4)
    }
    bc = traction_residual(radius=0.45, r_s=r_s, th_s=th_s, freq=freq, nu=nu)

    small = [cavity_field((xs, ys), [(1.3, 0.9)], [freq], radius=rr, nu=nu)
             for rr in (0.45, 0.1, 0.02)]
    shrink = [float(np.abs(f.scattered).max() / np.abs(f.incident).max()) for f in small]

    theta = np.pi / 4.0
    symbols = {
        "collocated_formula": float(collocated_symbol(theta)),
        "collocated_stencil": float(_symbol(_COLLOCATED, theta).real),
        "staggered_formula": float(staggered_symbol(theta)),
        "staggered_stencil": float(_symbol(_STAGGERED, theta).real),
        "product_formula": float(product_rule_symbol(theta)),
        "product_stencil": float((
            sum(w * np.exp(2j * theta * o) for o, w in _COLLOCATED.items())
            / (2.0 * sum(w * np.exp(1j * theta * o) for o, w in _COLLOCATED.items()))).real),
    }
    return {
        "navier_residual": navier,
        "navier_order": math.log2(navier[0] / navier[-1]) / 2.0,
        "navier_flipped": navier_flipped,
        "symmetry": symmetry,
        "far_field": far_field,
        "graf_vs_direct": graf,
        "traction_fd": traction_fd,
        "traction_bc_residual": bc,
        "scattered_over_incident": shrink,
        "symbols": symbols,
        "operator_defect": operator_defect_report(nu),
    }
