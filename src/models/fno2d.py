"""
The Fourier neural operator (§6.2 - §6.5).

Architecture, in one line: lift 12 -> 32 pointwise, four Fourier blocks, project
32 -> 4 pointwise.  No normalisation layers anywhere.

Three decisions in here are worth being able to defend out loud.

**Radial mode truncation, applied in `forward`.**  The retained set is
{k : |k| <= KMAX} on the integer lattice, not the square {|k_x|, |k_y| <= KMAX}
that the reference FNO implementation uses.  The square keeps corner modes out to
|k| = sqrt(2) KMAX along the diagonals, so a square truncation is anisotropic: it
resolves diagonal wavenumbers 41% better than axial ones.  For an operator whose
kernel is, in the homogeneous limit, a function of |k| alone (§4.4 -- the Christoffel
matrix depends on the direction of k only through the outer product k k^T, and its
eigenvalues depend only on |k|), that anisotropy is a modelling error the network has
to spend capacity undoing.  It also breaks the rotational symmetry of the physics in
a way that shows up as a preferred scattering direction, which is precisely the kind
of artefact that would discredit a defect-localisation result.

The mask is applied inside `forward`, multiplying the weights, rather than being
baked in once at construction.  If it were only applied at init, the discarded
coefficients would be ordinary live parameters again from the first optimiser step
and the truncation would quietly stop being a truncation.  Masking in the forward
pass makes the retained set a property of the operator instead of a property of the
initialisation.  The masked entries are also zeroed at init so that
`effective_params()` and the parameter file agree.

**No normalisation layers.**  BatchNorm would make the output depend on the other
samples in the batch, which is fatal here: the inversion (§8) differentiates the
network with respect to geometry for a *single* sample, and a batch-dependent output
means a batch-dependent gradient.  LayerNorm/InstanceNorm would divide by the field's
own amplitude, destroying the radius information the same way per-sample target
normalisation would (§7.1).  Stability instead comes from the block being close to
the identity at initialisation: the spectral weights start at scale 1/(d_in d_out),
so `v + act(W v + K v)` is a small perturbation of `v` in every block.

**norm='ortho' on both transforms.**  Composed over a forward and an inverse, the
overall scaling is 1/N^2 either way, so this is the same operator as the default
convention -- but the intermediate spectral coefficients stay O(field) instead of
O(N^2 field).  At N = 128 that is a factor of 16384, about four of float32's seven
significant digits, spent for nothing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch import Tensor

from .. import config as cfg


# ---------------------------------------------------------------------------
# Spectral convolution
# ---------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    """
    Learned Fourier multiplier: R(k) in C^{d_out x d_in} for each retained k.

    Index layout follows `config.spectral_params` exactly, so the counted and the
    allocated parameter numbers cannot drift apart.  With an rfft2 of a [ny, nx]
    real field the output is [ny, nx//2 + 1]; the retained block is

        w1 -> rows [0 : kmax]     i.e. k_y = 0 .. kmax-1
        w2 -> rows [-kmax : ]     i.e. k_y = -kmax .. -1
        both  cols [0 : kmax]     i.e. k_x = 0 .. kmax-1

    Only k_x >= 0 is stored because the field is real and the negative-k_x half is
    the conjugate mirror; `irfft2` reconstructs it.  That is also why the learned
    multiplier is unconstrained complex rather than Hermitian: Hermitian symmetry is
    imposed by the transform, not by the weights.
    """

    def __init__(self, d_in: int, d_out: int, kmax: int = cfg.KMAX, *,
                 radial: bool = True):
        super().__init__()
        self.d_in, self.d_out, self.kmax, self.radial = d_in, d_out, kmax, radial

        scale = 1.0 / (d_in * d_out)
        self.w1 = nn.Parameter(self._init(d_in, d_out, kmax, scale))
        self.w2 = nn.Parameter(self._init(d_in, d_out, kmax, scale))

        m1, m2 = self._masks(kmax, radial)
        # non-persistent: derived from kmax, and a checkpoint should not be able to
        # restore a mask that disagrees with the kmax it was loaded with
        self.register_buffer("m1", m1, persistent=False)
        self.register_buffer("m2", m2, persistent=False)
        with torch.no_grad():                      # keep the count honest
            self.w1 *= m1
            self.w2 *= m2

    @staticmethod
    def _init(d_in: int, d_out: int, kmax: int, scale: float) -> Tensor:
        """
        Zero-mean uniform, unlike the reference implementation's `scale * rand`.

        `rand` is supported on [0, 1], so every coefficient of every mode starts
        with mean scale/2: the initial operator is a fixed non-random multiplier
        added to the noise.  It trains anyway, but it puts a systematic direction
        into every mode at step zero, and this network is later asked for a
        *gradient* through that operator.  Recorded in README as a deliberate
        deviation.
        """
        re = (torch.rand(d_in, d_out, kmax, kmax) * 2.0 - 1.0) * scale
        im = (torch.rand(d_in, d_out, kmax, kmax) * 2.0 - 1.0) * scale
        return torch.complex(re, im)

    @staticmethod
    def _masks(kmax: int, radial: bool) -> tuple[Tensor, Tensor]:
        if not radial:
            one = torch.ones(1, 1, kmax, kmax)
            return one, one.clone()
        iy = torch.arange(kmax).view(-1, 1).float()
        ix = torch.arange(kmax).view(1, -1).float()
        m1 = (torch.hypot(iy, ix) <= kmax).float().view(1, 1, kmax, kmax)
        # row iy of the [-kmax:] slice is k_y = -(kmax - iy)
        m2 = (torch.hypot(kmax - iy, ix) <= kmax).float().view(1, 1, kmax, kmax)
        return m1, m2

    def effective_params(self) -> int:
        """Real parameters that can actually move.  Matches config.spectral_params."""
        kept = int(self.m1.sum().item() + self.m2.sum().item())
        return 2 * self.d_in * self.d_out * kept

    def forward(self, x: Tensor) -> Tensor:
        B, C, ny, nx = x.shape
        assert C == self.d_in, f"expected {self.d_in} channels, got {C}"
        k = self.kmax
        assert 2 * k <= ny and k <= nx // 2 + 1, (
            f"kmax={k} does not fit a {ny}x{nx} grid; the retained band must be "
            "strictly inside the resolved band or the two weight blocks overlap")

        xh = torch.fft.rfft2(x, norm="ortho")                    # [B, C, ny, nx//2+1]
        out = torch.zeros(B, self.d_out, ny, nx // 2 + 1,
                          dtype=xh.dtype, device=x.device)
        out[:, :, :k, :k] = torch.einsum(
            "bixy,ioxy->boxy", xh[:, :, :k, :k], self.w1 * self.m1)
        out[:, :, -k:, :k] = torch.einsum(
            "bixy,ioxy->boxy", xh[:, :, -k:, :k], self.w2 * self.m2)
        return torch.fft.irfft2(out, s=(ny, nx), norm="ortho")

    def extra_repr(self) -> str:
        return (f"d_in={self.d_in}, d_out={self.d_out}, kmax={self.kmax}, "
                f"radial={self.radial}, kept={self.effective_params()}")


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class FourierBlock(nn.Module):
    """
    v <- v + act(K v + W v).

    The residual is what removes the need for a norm layer: at initialisation K is
    O(1/d^2) and W is a small random 1x1 convolution, so the block is a perturbation
    of the identity and a four-block stack cannot blow up before the first update.
    """

    def __init__(self, d_v: int, kmax: int = cfg.KMAX, *, act: bool = True,
                 radial: bool = True):
        super().__init__()
        self.spectral = SpectralConv2d(d_v, d_v, kmax, radial=radial)
        self.pointwise = nn.Conv2d(d_v, d_v, kernel_size=1)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        h = self.spectral(x) + self.pointwise(x)
        return x + (Fn.gelu(h) if self.act else h)


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------
class FNO2d(nn.Module):
    """
    12 input channels -> 4 output channels on a 128^2 grid (or any other; §4.3).

    The final block is linear.  A GELU on the output of the last block would make
    the network's response to its input one-sided at small amplitude, and the target
    here is a *signed* phasor field whose two halves are physically symmetric: the
    scattered field from a void has no preferred sign.  Ending on a nonlinearity
    would force the projection MLP to undo that asymmetry.
    """

    def __init__(self, *, c_in: int = cfg.C_IN, c_out: int = cfg.C_OUT,
                 d_v: int = cfg.D_V, n_blocks: int = cfg.N_BLOCKS,
                 kmax: int = cfg.KMAX, lift_hidden: int = cfg.LIFT_HIDDEN,
                 proj_hidden: int = cfg.PROJ_HIDDEN, radial: bool = True):
        super().__init__()
        self.c_in, self.c_out, self.d_v, self.kmax = c_in, c_out, d_v, kmax
        self.lift = nn.Sequential(
            nn.Conv2d(c_in, lift_hidden, 1), nn.GELU(),
            nn.Conv2d(lift_hidden, d_v, 1))
        self.blocks = nn.ModuleList([
            FourierBlock(d_v, kmax, act=(i < n_blocks - 1), radial=radial)
            for i in range(n_blocks)])
        self.project = nn.Sequential(
            nn.Conv2d(d_v, proj_hidden, 1), nn.GELU(),
            nn.Conv2d(proj_hidden, c_out, 1))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.c_in, (
            f"got {x.shape[1]} input channels, expected {self.c_in}; "
            "build inputs with features.pack_inputs, not by hand")
        v = self.lift(x)
        for b in self.blocks:
            v = b(v)
        return self.project(v)

    # -- reporting --------------------------------------------------------
    def effective_params(self) -> int:
        n = sum(p.numel() for name, p in self.named_parameters()
                if not name.endswith((".w1", ".w2")))
        n += sum(m.effective_params() for m in self.modules()
                 if isinstance(m, SpectralConv2d))
        return n

    def allocated_params(self) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1)
                   for p in self.parameters())

    def summary(self) -> str:
        eff, alloc = self.effective_params(), self.allocated_params()
        want = cfg.total_params(self.d_v, self.kmax, len(self.blocks), radial=True)
        lines = [
            f"FNO2d  d_v={self.d_v}  blocks={len(self.blocks)}  kmax={self.kmax}",
            f"  effective parameters : {eff:,}",
            f"  allocated (incl. masked-off) : {alloc:,}",
            f"  config.total_params()        : {want:,}",
        ]
        if eff != want:
            lines.append("  MISMATCH -- the model and the sizing table disagree; "
                         "one of them is wrong and the document quotes the table")
        return "\n".join(lines)


def build(variant: str = "primary", **kw) -> FNO2d:
    """Instantiate one of the §6.4 capacity variants."""
    assert variant in cfg.VARIANTS, f"unknown variant {variant!r}"
    return FNO2d(**{**cfg.VARIANTS[variant], **kw})


def to_double(model: nn.Module) -> nn.Module:
    """
    Convert a model to full double precision, complex parameters included.

    `nn.Module.double()` is not enough.  Its implementation converts a parameter only
    when `t.is_floating_point()` is true, and a complex tensor is not floating point
    by that predicate, so the spectral weights stay complex64 while everything around
    them becomes float64.  The result runs -- PyTorch promotes complex64 * float64 to
    complex128 -- and is silently accurate to about seven digits, which is fewer than
    the three significant figures the gradient check of §11.2 step 9 asserts *after*
    two nested finite differences have eaten most of them.  A check that fails for
    precision reasons is worse than no check, because it gets disabled.

    Mutates in place and returns the model, so it composes: `to_double(build())`.
    """
    for m in model.modules():
        for name, p in list(m._parameters.items()):
            if p is None:
                continue
            target = torch.complex128 if p.is_complex() else torch.float64
            m._parameters[name] = nn.Parameter(p.detach().to(target),
                                               requires_grad=p.requires_grad)
        for name, b in list(m._buffers.items()):
            if b is None or not (b.is_floating_point() or b.is_complex()):
                continue
            m._buffers[name] = b.to(torch.complex128 if b.is_complex()
                                    else torch.float64)
    return model


def band_in_modes(nu: float = min(cfg.NU_LIST),
                  f_over_fc: float | None = None) -> float:
    """
    The highest *physical* mode index the operating band needs, as a check on KMAX.

    A mode index is a physical wavenumber here: index k means 2 pi k / L, and L is
    fixed, so truncating by index is truncating by wavenumber and survives a change
    of grid resolution unchanged.  That equivalence is the whole of the
    discretisation-invariance claim in §4.3, so it is worth one function.
    """
    if f_over_fc is None:
        f_over_fc = max(cfg.FREQS) / cfg.FC
    k_s = 2.0 * math.pi * f_over_fc * cfg.FC / (cfg.cs_over_cp(nu) * cfg.CP)
    return k_s * cfg.L_DOMAIN / (2.0 * math.pi)


__all__ = ["FNO2d", "FourierBlock", "SpectralConv2d", "band_in_modes", "build",
           "to_double"]


if __name__ == "__main__":
    net = build()
    print(net.summary())
    kneed = band_in_modes()
    print(f"\nband top needs mode index {kneed:.1f} (nu={min(cfg.NU_LIST)}, "
          f"f={max(cfg.FREQS)/cfg.FC:.2f} f_c); KMAX = {cfg.KMAX} gives "
          f"{cfg.KMAX/kneed:.2f}x headroom for near-field content")
    x = torch.randn(2, cfg.C_IN, cfg.N_NET, cfg.N_NET)
    print("forward:", tuple(net(x).shape))
    y = net(torch.randn(2, cfg.C_IN, 2 * cfg.N_NET, 2 * cfg.N_NET))
    print("forward at 2x resolution:", tuple(y.shape), "(same weights)")
