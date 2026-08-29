"""
The channel vocabulary shared by the dataset, the network, the losses and the
inversion (§6.2).

There is exactly one definition of "what the 12 input channels are" and one
definition of "what the 4 output channels are", and it lives here.  The reason for
a separate module rather than putting this in the model: the inversion builds
network inputs itself, from geometry parameters it is differentiating with respect
to, and if it packs the channels in a different order from the training loader the
symptom is a network that works on the test set and produces garbage gradients --
which looks like an optimisation failure, not a bookkeeping one.

Input channels
--------------
 0  phi_tilde          clipped signed distance in network cells (§2.4)
 1  chi                soft void indicator, sigmoid(-phi/eps)
 2  Re u_inc_x         incident displacement phasor, x component
 3  Im u_inc_x
 4  Re u_inc_y
 5  Im u_inc_y
 6  f / f_c            frequency, in [0.66, 1.34]
 7  k_p dx             P-wave phase advance per cell, in radians
 8  k_s dx             S-wave phase advance per cell, in radians
 9  nu_centred         Poisson ratio mapped to [-1, 1]
10  x_norm             2x/L - 1
11  y_norm             2y/L - 1

Channels 6-8 are algebraically redundant at fixed dx and c_p = 1 (k_p dx = 2 pi f
dx).  They are kept separate deliberately: the discretisation-invariance claim of
§4.3 is that the *same weights* work at a different dx, and at a different dx the
frequency and the per-cell phase advance are no longer proportional.  Feeding both
is what lets the network condition on the physical frequency and on the grid
resolution independently.  Channel 9 is not redundant with 7-8 in the way it looks
either: nu fixes the P-to-S conversion coefficients at the void boundary, which is
amplitude information, not phase information.

Channels 10-11 exist because the operator is not translation invariant: the
absorbing boundary and the receiver ring break homogeneity, so the network needs to
know where in the domain it is.  Without them a defect near the edge and one in the
centre present identically.

Output channels
---------------
 0  Re u_s_x           *scattered* displacement phasor, x component
 1  Im u_s_x
 2  Re u_s_y
 3  Im u_s_y

Scattered rather than total, per §6.1: the incident field is known analytically-ish
(it is one cached solve) and carries most of the energy, so predicting the total
field would spend the network's capacity reproducing something already known and
would put the error metric's denominator on the large incident amplitude, hiding
exactly the scattered signal the inversion depends on.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from . import config as cfg

INPUT_CHANNELS: tuple[str, ...] = (
    "phi_tilde", "chi",
    "re_uinc_x", "im_uinc_x", "re_uinc_y", "im_uinc_y",
    "f_over_fc", "kp_dx", "ks_dx", "nu_centred", "x_norm", "y_norm",
)
OUTPUT_CHANNELS: tuple[str, ...] = ("re_usx", "im_usx", "re_usy", "im_usy")

assert len(INPUT_CHANNELS) == cfg.C_IN, "INPUT_CHANNELS disagrees with config.C_IN"
assert len(OUTPUT_CHANNELS) == cfg.C_OUT, "OUTPUT_CHANNELS disagrees with config.C_OUT"

# nu -> [-1, 1] over the training range
NU_MID: float = 0.5 * (min(cfg.NU_LIST) + max(cfg.NU_LIST))
NU_HALF: float = 0.5 * (max(cfg.NU_LIST) - min(cfg.NU_LIST))


def nu_centred(nu: Tensor | float) -> Tensor | float:
    return (nu - NU_MID) / NU_HALF


# ---------------------------------------------------------------------------
# Complex <-> real channel packing
# ---------------------------------------------------------------------------
def complex_to_channels(z: Tensor) -> Tensor:
    """
    [..., 2, ny, nx] complex (x, y components) -> [..., 4, ny, nx] real.

    Interleaved (Re_x, Im_x, Re_y, Im_y) rather than blocked (Re_x, Re_y, Im_x,
    Im_y) so that the two channels belonging to one physical component are
    adjacent; the projection MLP is pointwise so it cannot exploit adjacency, but
    every plotting and slicing operation downstream can.
    """
    assert z.shape[-3] == 2, f"expected 2 components, got shape {tuple(z.shape)}"
    assert z.is_complex(), f"expected a complex tensor, got {z.dtype}"
    return torch.stack([z[..., 0, :, :].real, z[..., 0, :, :].imag,
                        z[..., 1, :, :].real, z[..., 1, :, :].imag], dim=-3)


def channels_to_complex(x: Tensor) -> Tensor:
    """[..., 4, ny, nx] real -> [..., 2, ny, nx] complex.  Inverse of the above."""
    assert x.shape[-3] == 4, f"expected 4 channels, got shape {tuple(x.shape)}"
    ux = torch.complex(x[..., 0, :, :], x[..., 1, :, :])
    uy = torch.complex(x[..., 2, :, :], x[..., 3, :, :])
    return torch.stack([ux, uy], dim=-3)


# ---------------------------------------------------------------------------
# Static channels
# ---------------------------------------------------------------------------
def coordinate_channels(n: int = cfg.N_NET, l_domain: float = cfg.L_DOMAIN, *,
                        device=None, dtype=torch.float32) -> Tensor:
    """[2, n, n] with (x_norm, y_norm) in [-1, 1]."""
    from .geometry.sdf import grid_coords
    yy, xx = grid_coords(n, l_domain / n, device=device, dtype=dtype)
    return torch.stack([2.0 * xx / l_domain - 1.0, 2.0 * yy / l_domain - 1.0])


def wavenumber_channels(freq: Tensor, nu: Tensor, dx: float) -> Tensor:
    """
    (f/f_c, k_p dx, k_s dx) for a batch of (freq, nu) pairs -> [B, 3].

    freq [B] in units of f_c, nu [B].  c_p = 1, so k_p = 2 pi f and k_s = k_p/c_s.
    """
    cs = torch.sqrt((1.0 - 2.0 * nu) / (2.0 * (1.0 - nu)))
    kp = 2.0 * math.pi * freq / cfg.CP
    ks = kp / cs
    return torch.stack([freq / cfg.FC, kp * dx, ks * dx], dim=-1)


# ---------------------------------------------------------------------------
# Assembling a network input
# ---------------------------------------------------------------------------
def pack_inputs(phi_t: Tensor, chi: Tensor, u_inc: Tensor, freqs: Tensor,
                nu: Tensor, *, dx: float = cfg.DX_NET,
                coords: Tensor | None = None,
                dtype: torch.dtype | None = None) -> Tensor:
    """
    Build the [B*F, C_IN, ny, nx] network input.

    phi_t   [B, ny, nx]        clipped SDF, shared across frequencies
    chi     [B, ny, nx]        soft indicator, shared across frequencies
    u_inc   [B, F, 2, ny, nx]  complex incident displacement phasors
    freqs   [B, F]             frequency of each column, in units of f_c
    nu      [B]                Poisson ratio per sample

    The frequency axis is folded into the batch axis, F-major within each sample,
    so a target tensor of shape [B, F, 2, ny, nx] flattened with
    `flatten_freq(complex_to_channels(target))` lines up row for row.  Getting this
    wrong is silent: the loss still decreases, it just decreases towards the wrong
    operator.  `tests/test_features.py` asserts the round trip.

    The output dtype defaults to the real dtype matching `u_inc` -- float32 for
    complex64, float64 for complex128 -- rather than being pinned to float32.  It has
    to be inferrable, because the gradient check of §11.2 step 9 runs the whole chain
    in double precision and a float32 cast anywhere in it would cap the agreement at
    about seven digits of the *field*, which after two nested differences is fewer
    digits than the check asserts.
    """
    B, F = u_inc.shape[0], u_inc.shape[1]
    ny, nx = u_inc.shape[-2], u_inc.shape[-1]
    dev = u_inc.device
    if dtype is None:
        dtype = torch.float64 if u_inc.dtype == torch.complex128 else torch.float32
    dt_ = dtype

    assert phi_t.shape == (B, ny, nx), f"phi_t {tuple(phi_t.shape)}"
    assert chi.shape == (B, ny, nx), f"chi {tuple(chi.shape)}"
    assert freqs.shape == (B, F), f"freqs {tuple(freqs.shape)}"
    assert nu.shape == (B,), f"nu {tuple(nu.shape)}"

    geom = torch.stack([phi_t, chi], dim=1)                     # [B, 2, ny, nx]
    geom = geom.unsqueeze(1).expand(B, F, 2, ny, nx)

    inc = complex_to_channels(u_inc)                            # [B, F, 4, ny, nx]

    scal = wavenumber_channels(freqs.reshape(-1),
                               nu.repeat_interleave(F), dx)     # [B*F, 3]
    nuc = nu_centred(nu).repeat_interleave(F).unsqueeze(-1)     # [B*F, 1]
    scal = torch.cat([scal, nuc], dim=-1)                       # [B*F, 4]
    scal = scal.view(B, F, 4, 1, 1).expand(B, F, 4, ny, nx)

    if coords is None:
        coords = coordinate_channels(ny, device=dev, dtype=dt_)
    xy = coords.view(1, 1, 2, ny, nx).expand(B, F, 2, ny, nx)

    x = torch.cat([geom.to(dt_), inc.to(dt_), scal.to(dt_), xy.to(dt_)], dim=2)
    assert x.shape[2] == cfg.C_IN, f"packed {x.shape[2]} channels, want {cfg.C_IN}"
    return x.reshape(B * F, cfg.C_IN, ny, nx)


def flatten_freq(t: Tensor) -> Tensor:
    """[B, F, C, ny, nx] -> [B*F, C, ny, nx], matching pack_inputs' ordering."""
    B, F = t.shape[0], t.shape[1]
    return t.reshape(B * F, *t.shape[2:])


def unflatten_freq(t: Tensor, n_freq: int) -> Tensor:
    """Inverse of flatten_freq."""
    return t.reshape(-1, n_freq, *t.shape[1:])


def channel_index(name: str) -> int:
    """Index of a named input channel.  Use this instead of a literal."""
    return INPUT_CHANNELS.index(name)


__all__ = [
    "INPUT_CHANNELS",
    "OUTPUT_CHANNELS",
    "channel_index",
    "channels_to_complex",
    "complex_to_channels",
    "coordinate_channels",
    "flatten_freq",
    "nu_centred",
    "pack_inputs",
    "unflatten_freq",
    "wavenumber_channels",
]
