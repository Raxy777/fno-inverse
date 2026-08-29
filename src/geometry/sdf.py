"""
Differentiable defect geometry (§2.4, §8.2).

Everything here is a function of a parameter tensor `theta` of shape [B, P] and
returns fields of shape [B, ny, nx].  The whole point is that the map
theta -> field is smooth, because §8.2 shows the inversion gradient

    dJ/dtheta = (residual) . dG/da . da/dtheta

dies on the last factor if the geometry enters as a hard mask: the derivative of
1[phi < 0] is a Dirac delta on the interface, which on a discrete grid evaluates
to exactly zero at every grid point not lying precisely on the boundary.  The
sigmoid replaces that delta with a bump of finite width, and the width is a real
trade-off -- too narrow and the gradient support falls between grid points, too
wide and the void boundary is physically smeared.

Three shape families are provided.  Only `Circle` is ever used to generate
training data; `Ellipse` and `TwoCircle` exist so that the inversion can be run
against shapes the network never saw, which is the headline transfer experiment
of §8.5.  Nothing outside this file needs to change to add a family -- that is
the claim being tested.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .. import config as cfg


# ---------------------------------------------------------------------------
# Coordinate grids
# ---------------------------------------------------------------------------
def grid_coords(n: int, dx: float, offset: float = 0.0, *,
                device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """
    Cell-centre coordinates of an n x n grid, as (yy, xx), each [n, n].

    Index i maps to (i + 0.5 - offset) * dx.  `offset` lets the padded solver
    grid share an origin with the cropped physical grid: passing
    offset = N_PML_FINE puts x = 0 at the first physical cell's left edge, so
    that a 2x2 average-pool of the fine grid lands exactly on the network grid's
    cell centres.
    """
    idx = (torch.arange(n, device=device, dtype=dtype) + 0.5 - offset) * dx
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    return yy, xx


def net_coords(device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    return grid_coords(cfg.N_NET, cfg.DX_NET, device=device, dtype=dtype)


def fine_coords(device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    return grid_coords(cfg.N_FINE_TOTAL, cfg.DX_FINE, offset=cfg.N_PML_FINE,
                       device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Smooth reductions
# ---------------------------------------------------------------------------
def smooth_min(a: Tensor, b: Tensor, k: float) -> Tensor:
    """
    -k log(e^{-a/k} + e^{-b/k}).  Approaches min(a, b) as k -> 0 but has a
    continuous derivative everywhere, which matters for the two-void family:
    a hard min puts a gradient kink along the locus where the two SDFs cross,
    and L-BFGS handles kinks badly.
    """
    m = torch.minimum(a, b)
    return m - k * torch.log(torch.exp((m - a) / k) + torch.exp((m - b) / k))


# ---------------------------------------------------------------------------
# Shape families
# ---------------------------------------------------------------------------
class ShapeFamily(ABC):
    """A differentiable parametric family of signed distance functions."""

    name: str = "abstract"
    n_params: int = 0
    param_names: tuple[str, ...] = ()

    @abstractmethod
    def sdf(self, theta: Tensor, yy: Tensor, xx: Tensor) -> Tensor:
        """theta [B, P], yy/xx [ny, nx] -> phi [B, ny, nx]."""

    @abstractmethod
    def bounds(self, lambda_s: float) -> tuple[Tensor, Tensor]:
        """Physical (lo, hi) box for theta, each [P]."""

    # -- shared helpers ----------------------------------------------------
    def to_unconstrained(self, theta: Tensor, lambda_s: float) -> Tensor:
        """
        Invert the sigmoid reparameterisation of §8.1.

        Bounds are enforced by reparameterisation, not clipping, so that the
        gradient stays defined at the boundary of the feasible set.
        """
        lo, hi = self.bounds(lambda_s)
        lo, hi = lo.to(theta), hi.to(theta)
        u = ((theta - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        return torch.log(u / (1 - u))

    def to_physical(self, z: Tensor, lambda_s: float) -> Tensor:
        lo, hi = self.bounds(lambda_s)
        lo, hi = lo.to(z), hi.to(z)
        return lo + (hi - lo) * torch.sigmoid(z)

    def area(self, theta: Tensor) -> Tensor:
        """Void area, used by the Rayleigh-limit check and for reporting."""
        raise NotImplementedError


class Circle(ShapeFamily):
    """theta = (xc, yc, R).  The only family used for training data."""

    name = "circle"
    n_params = 3
    param_names = ("xc", "yc", "R")

    def sdf(self, theta: Tensor, yy: Tensor, xx: Tensor) -> Tensor:
        xc = theta[:, 0].view(-1, 1, 1)
        yc = theta[:, 1].view(-1, 1, 1)
        r = theta[:, 2].view(-1, 1, 1)
        d = torch.sqrt((xx - xc) ** 2 + (yy - yc) ** 2 + 1e-30)
        return d - r

    def bounds(self, lambda_s: float) -> tuple[Tensor, Tensor]:
        pad = cfg.BOUNDARY_KEEPOUT_LS * lambda_s
        lo = torch.tensor([pad, pad, cfg.R_MIN_LS * lambda_s])
        hi = torch.tensor([cfg.L_DOMAIN - pad, cfg.L_DOMAIN - pad,
                           cfg.R_MAX_LS * lambda_s])
        return lo, hi

    def area(self, theta: Tensor) -> Tensor:
        return math.pi * theta[:, 2] ** 2

    # -- analytic derivatives, for the gradient check of §11.2 step 9 -------
    def dchi_dtheta_analytic(self, theta: Tensor, yy: Tensor, xx: Tensor,
                             eps_len: float) -> Tensor:
        """
        The expressions derived in §8.2, returned as [B, 3, ny, nx].

        d chi / d xc = sigma'(-phi/eps) * (1/eps) * cos(alpha)
        d chi / d yc = sigma'(-phi/eps) * (1/eps) * sin(alpha)
        d chi / d R  = sigma'(-phi/eps) * (1/eps)

        Note the geometry these encode: sigma' is sharply peaked on phi = 0, so
        all sensitivity lives in an annulus of width ~eps around the boundary.
        The R-derivative is uniform over that annulus (a monopole, breathing
        mode); the position derivatives carry cos/sin (dipole modes).  That is
        why all three parameters are separately identifiable from the same data.
        """
        xc = theta[:, 0].view(-1, 1, 1)
        yc = theta[:, 1].view(-1, 1, 1)
        d = torch.sqrt((xx - xc) ** 2 + (yy - yc) ** 2 + 1e-30)
        phi = d - theta[:, 2].view(-1, 1, 1)
        s = torch.sigmoid(-phi / eps_len)
        bump = s * (1.0 - s) / eps_len          # sigma'(-phi/eps) / eps
        cos_a = (xx - xc) / d
        sin_a = (yy - yc) / d
        return torch.stack([bump * cos_a, bump * sin_a, bump], dim=1)


class Ellipse(ShapeFamily):
    """
    theta = (xc, yc, a, b, alpha).  Never trained on -- see §8.5.

    An ellipse has no closed-form signed distance, so we use the first-order
    normalisation phi ~ g/|grad g| of the implicit function
    g = (x'/a)^2 + (y'/b)^2 - 1.  This is exact on the boundary (which is the
    only place the soft indicator has support) and smooth and monotone away from
    it, which is all the network input needs.
    """

    name = "ellipse"
    n_params = 5
    param_names = ("xc", "yc", "a", "b", "alpha")

    def sdf(self, theta: Tensor, yy: Tensor, xx: Tensor) -> Tensor:
        xc, yc, a, b, al = (theta[:, i].view(-1, 1, 1) for i in range(5))
        ca, sa = torch.cos(al), torch.sin(al)
        dx_, dy_ = xx - xc, yy - yc
        xp = ca * dx_ + sa * dy_
        yp = -sa * dx_ + ca * dy_
        g = (xp / a) ** 2 + (yp / b) ** 2 - 1.0
        grad = 2.0 * torch.sqrt((xp / a ** 2) ** 2 + (yp / b ** 2) ** 2 + 1e-20)
        return g / (grad + 1e-12)

    def bounds(self, lambda_s: float) -> tuple[Tensor, Tensor]:
        pad = cfg.BOUNDARY_KEEPOUT_LS * lambda_s
        lo = torch.tensor([pad, pad, cfg.R_MIN_LS * lambda_s,
                           cfg.R_MIN_LS * lambda_s, -math.pi / 2])
        hi = torch.tensor([cfg.L_DOMAIN - pad, cfg.L_DOMAIN - pad,
                           cfg.R_MAX_LS * lambda_s * 1.6,
                           cfg.R_MAX_LS * lambda_s * 1.6, math.pi / 2])
        return lo, hi

    def area(self, theta: Tensor) -> Tensor:
        return math.pi * theta[:, 2] * theta[:, 3]


class TwoCircle(ShapeFamily):
    """theta = (xc1, yc1, R1, xc2, yc2, R2).  Never trained on -- see §8.5."""

    name = "two_circle"
    n_params = 6
    param_names = ("xc1", "yc1", "R1", "xc2", "yc2", "R2")

    def __init__(self, blend_cells: float = 0.5):
        self.blend = blend_cells * cfg.DX_NET

    def sdf(self, theta: Tensor, yy: Tensor, xx: Tensor) -> Tensor:
        one = Circle()
        p1 = one.sdf(theta[:, 0:3], yy, xx)
        p2 = one.sdf(theta[:, 3:6], yy, xx)
        return smooth_min(p1, p2, self.blend)

    def bounds(self, lambda_s: float) -> tuple[Tensor, Tensor]:
        lo1, hi1 = Circle().bounds(lambda_s)
        return torch.cat([lo1, lo1]), torch.cat([hi1, hi1])

    def area(self, theta: Tensor) -> Tensor:
        # Ignores overlap; adequate for reporting only.
        return math.pi * (theta[:, 2] ** 2 + theta[:, 5] ** 2)


FAMILIES: dict[str, ShapeFamily] = {
    "circle": Circle(),
    "ellipse": Ellipse(),
    "two_circle": TwoCircle(),
}


# ---------------------------------------------------------------------------
# Fields derived from an SDF
# ---------------------------------------------------------------------------
def soft_indicator(phi: Tensor, eps_len: float) -> Tensor:
    """chi = sigmoid(-phi/eps).  1 inside the void, 0 in the solid."""
    return torch.sigmoid(-phi / eps_len)


def phi_tilde(phi: Tensor, dx: float, clip: float = cfg.SDF_CLIP_CELLS) -> Tensor:
    """
    SDF in grid-cell units, clipped (§2.4).

    Grid units rather than wavelengths because lambda_s moves with nu; the
    wavelengths are supplied separately as conditioning channels (§6.2), so this
    representation stays fixed while the physics scales.
    """
    return (phi / dx).clamp(-clip, clip)


def geometry_channels(theta: Tensor, family: ShapeFamily, *,
                      dx: float = cfg.DX_NET,
                      eps_cells: float = cfg.EPS_INTERFACE_CELLS,
                      yy: Tensor | None = None,
                      xx: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    Channels 1 and 2 of the network input: (phi_tilde, chi), each [B, ny, nx].
    """
    if yy is None or xx is None:
        n = cfg.N_NET if dx == cfg.DX_NET else cfg.N_FINE_TOTAL
        if n == cfg.N_NET:
            yy, xx = net_coords(device=theta.device, dtype=theta.dtype)
        else:
            yy, xx = fine_coords(device=theta.device, dtype=theta.dtype)
    phi = family.sdf(theta, yy, xx)
    return phi_tilde(phi, dx), soft_indicator(phi, eps_cells * dx)


def material_fields(chi: Tensor, lam0: float, mu0: float, rho0: float = cfg.RHO0
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """
    (lambda, mu, rho) fields for the solver.

    Stiffness is scaled by (1 - chi) down to a small floor.  Density is scaled by
    VOID_DENSITY_SCALE, which defaults to 1.0 -- i.e. *unchanged* inside the
    void.  See README "Documented deviations": §7.2 writes delta_rho = -rho0 chi,
    but nulling density alongside stiffness makes c = sqrt(stiffness/rho) a 0/0
    limit whose numerical value can exceed c_p and break the CFL condition.
    Keeping rho fixed and nulling stiffness gives c_void -> 0, which is
    unconditionally stable, and an impedance rho*c -> 0, which is the physical
    content of a traction-free void.  losses.py reads the same constant so the
    Lippmann-Schwinger residual matches the solver rather than the document.
    """
    solid = (1.0 - chi).clamp_min(cfg.VOID_STIFFNESS_FLOOR)
    lam = lam0 * solid
    mu = mu0 * solid
    rho = rho0 * (1.0 - (1.0 - cfg.VOID_DENSITY_SCALE) * chi)
    return lam, mu, rho
