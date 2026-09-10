"""
The channel vocabulary.

`features.pack_inputs` names this file in its docstring as the place the F-major
round trip is asserted, and that is the one test here that really matters.  The
frequency axis is folded into the batch axis in two different places -- the loader's
`batch_to_model` and the inversion's `SurrogateForward` -- and if the two disagree
about whether the flattening is F-major or B-major, training still converges.  It
converges towards an operator that has learned to average over frequency, and the
only visible symptom is that the inversion's misfit landscape is mysteriously flat.
So the ordering is asserted explicitly, with distinguishable values in every slot,
rather than being checked by shape agreement (which would pass either way).
"""

from __future__ import annotations

import math

import pytest
import torch

from src import config as cfg
from src import features as feat
from src import training


# ---------------------------------------------------------------------------
# Complex <-> channel packing
# ---------------------------------------------------------------------------
def test_complex_channel_round_trip():
    z = torch.randn(3, 5, 2, 7, 7, dtype=torch.complex64)
    x = feat.complex_to_channels(z)
    assert x.shape == (3, 5, 4, 7, 7)
    assert torch.equal(feat.channels_to_complex(x), z)


def test_per_sample_relative_errors_reduce_one_batch_correctly():
    """Notebook 04's helper must preserve one value per sample, not reduce a missing axis."""
    pred = torch.arange(2 * 3 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 3, 2, 4, 4)
    target = pred + 1.0
    recv = torch.tensor([[0, 0], [1, 2], [3, 3]])
    field, ring = training.per_sample_relative_errors(pred, target, recv)
    expected_field = ((pred - target).pow(2).sum((1, 2, 3)).sqrt()
                      / target.pow(2).sum((1, 2, 3)).sqrt())
    pr, tr = pred[..., recv[:, 0], recv[:, 1]], target[..., recv[:, 0], recv[:, 1]]
    expected_ring = ((pr - tr).pow(2).sum((1, 2, 3)).sqrt()
                     / tr.pow(2).sum((1, 2, 3)).sqrt())
    assert field.shape == ring.shape == (2,)
    assert torch.allclose(field, expected_field)
    assert torch.allclose(ring, expected_ring)




def test_channel_packing_is_interleaved_not_blocked():
    """(Re_x, Im_x, Re_y, Im_y), not (Re_x, Re_y, Im_x, Im_y)."""
    z = torch.zeros(1, 2, 4, 4, dtype=torch.complex64)
    z[0, 0] = 1.0 + 2.0j          # x component
    z[0, 1] = 3.0 + 4.0j          # y component
    x = feat.complex_to_channels(z)[0]
    assert x[0].unique().tolist() == [1.0]
    assert x[1].unique().tolist() == [2.0]
    assert x[2].unique().tolist() == [3.0]
    assert x[3].unique().tolist() == [4.0]


def test_complex_to_channels_rejects_real_input():
    with pytest.raises(AssertionError):
        feat.complex_to_channels(torch.zeros(1, 2, 4, 4))


# ---------------------------------------------------------------------------
# The channel table
# ---------------------------------------------------------------------------
def test_channel_table_matches_config():
    assert len(feat.INPUT_CHANNELS) == cfg.C_IN == 12
    assert len(feat.OUTPUT_CHANNELS) == cfg.C_OUT == 4
    assert len(set(feat.INPUT_CHANNELS)) == cfg.C_IN
    assert feat.channel_index("phi_tilde") == 0
    assert feat.channel_index("chi") == 1
    assert feat.channel_index("re_uinc_x") == 2
    assert feat.channel_index("im_uinc_y") == 5
    assert feat.channel_index("f_over_fc") == 6
    assert feat.channel_index("ks_dx") == 8
    assert feat.channel_index("nu_centred") == 9
    assert feat.channel_index("x_norm") == 10
    assert feat.channel_index("y_norm") == 11


def test_nu_centred_spans_minus_one_to_one():
    assert feat.nu_centred(min(cfg.NU_LIST)) == pytest.approx(-1.0)
    assert feat.nu_centred(max(cfg.NU_LIST)) == pytest.approx(1.0)
    assert feat.nu_centred(feat.NU_MID) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Static channels
# ---------------------------------------------------------------------------
def test_coordinate_channels_orientation_and_range():
    n = 16
    c = feat.coordinate_channels(n, cfg.L_DOMAIN)
    assert c.shape == (2, n, n)
    x_norm, y_norm = c[0], c[1]
    # x varies along the last axis, y along the second-to-last.  A transpose here
    # would be invisible on a square grid except through this test.
    assert torch.allclose(x_norm[0], x_norm[-1])
    assert torch.allclose(y_norm[:, 0], y_norm[:, -1])
    assert x_norm[0, 0] == pytest.approx(2.0 * 0.5 / n - 1.0)
    assert x_norm[0, -1] == pytest.approx(2.0 * (n - 0.5) / n - 1.0)
    # Cell centres, so the range is open: never exactly +-1.
    assert c.abs().max() < 1.0
    assert c.abs().max() > 1.0 - 2.0 / n


def test_coordinate_channels_are_resolution_independent():
    """
    The same physical point gets the same coordinate value at any n.  Cell centres
    of an n-grid coincide with every third cell centre of a 3n-grid, which is the
    same odd-refinement alignment the solver's grid-convergence check relies on.
    """
    n = 8
    c1 = feat.coordinate_channels(n, cfg.L_DOMAIN)
    c3 = feat.coordinate_channels(3 * n, cfg.L_DOMAIN)
    assert torch.allclose(c1, c3[:, 1::3, 1::3], atol=1e-6)


def test_wavenumber_channels_against_analytic():
    freqs = torch.tensor([0.66, 1.0, 1.34])
    nus = torch.tensor([0.25, 0.33, 0.37])
    w = feat.wavenumber_channels(freqs, nus, cfg.DX_NET)
    assert w.shape == (3, 3)
    for i, (f, nu) in enumerate(zip(freqs.tolist(), nus.tolist())):
        kp = 2.0 * math.pi * f / cfg.CP
        ks = kp / cfg.cs_over_cp(nu)
        assert w[i, 0].item() == pytest.approx(f / cfg.FC, rel=1e-6)
        assert w[i, 1].item() == pytest.approx(kp * cfg.DX_NET, rel=1e-6)
        assert w[i, 2].item() == pytest.approx(ks * cfg.DX_NET, rel=1e-6)


def test_per_cell_phase_advance_is_under_a_quarter_turn():
    """
    k_s dx at the worst material and the top of the band, on the network grid.  If
    this approached pi the network's own grid would be aliasing the field it is
    asked to represent, whatever the mode truncation did.
    """
    f = max(cfg.FREQS)
    ks_dx = 2.0 * math.pi * f / cfg.CS_MIN * cfg.DX_NET
    assert ks_dx < math.pi / 2.0
    assert ks_dx == pytest.approx(1.159, abs=2e-3)


# ---------------------------------------------------------------------------
# pack_inputs: the F-major ordering contract
# ---------------------------------------------------------------------------
def _fake_batch(B=3, F=4, n=8, *, cdtype=torch.complex64):
    """
    Every slot filled with a value that identifies its (b, f) origin, so a
    misordering cannot hide behind a plausible-looking shape.
    """
    u = torch.zeros(B, F, 2, n, n, dtype=cdtype)
    for b in range(B):
        for f in range(F):
            tag = 10.0 * b + f
            u[b, f, 0] = complex(tag, tag + 0.25)
            u[b, f, 1] = complex(-tag, -tag - 0.25)
    phi_t = torch.arange(B, dtype=torch.float32).view(B, 1, 1).expand(B, n, n) * 1.0
    chi = -phi_t
    freqs = torch.tensor([[0.7 + 0.1 * f + 0.01 * b for f in range(F)]
                          for b in range(B)])
    nu = torch.tensor([cfg.NU_LIST[b % len(cfg.NU_LIST)] for b in range(B)])
    return dict(phi_t=phi_t.contiguous(), chi=chi.contiguous(), u_inc=u,
                freqs=freqs, nu=nu, B=B, F=F, n=n)


def test_pack_inputs_is_f_major():
    d = _fake_batch()
    B, F, n = d["B"], d["F"], d["n"]
    x = feat.pack_inputs(d["phi_t"], d["chi"], d["u_inc"], d["freqs"], d["nu"])
    assert x.shape == (B * F, cfg.C_IN, n, n)

    ci = feat.channel_index
    for b in range(B):
        for f in range(F):
            row = x[b * F + f]                      # <- F-major: b*F + f
            tag = 10.0 * b + f
            assert row[ci("phi_tilde")].unique().tolist() == [float(b)]
            assert row[ci("chi")].unique().tolist() == [float(-b)]
            assert row[ci("re_uinc_x")].unique().tolist() == [tag]
            assert row[ci("im_uinc_x")].unique().tolist() == [tag + 0.25]
            assert row[ci("re_uinc_y")].unique().tolist() == [-tag]
            assert row[ci("im_uinc_y")].unique().tolist() == [-tag - 0.25]
            assert row[ci("f_over_fc")].unique().item() == pytest.approx(
                d["freqs"][b, f].item(), rel=1e-6)
            assert row[ci("nu_centred")].unique().item() == pytest.approx(
                float(feat.nu_centred(d["nu"][b])), rel=1e-6)


def test_pack_inputs_target_row_alignment():
    """
    The contract named in `pack_inputs`' docstring: inputs built by pack_inputs and
    targets built by flatten_freq(complex_to_channels(.)) line up row for row.
    """
    d = _fake_batch()
    B, F = d["B"], d["F"]
    u_s = d["u_inc"] * (2.0 + 0.0j)                 # any per-(b,f) distinguishable field
    x = feat.pack_inputs(d["phi_t"], d["chi"], d["u_inc"], d["freqs"], d["nu"])
    y = feat.flatten_freq(feat.complex_to_channels(u_s))
    assert x.shape[0] == y.shape[0] == B * F

    ci = feat.channel_index
    # The target's Re_x must be exactly twice the input's Re u_inc_x on every row,
    # which can only hold if the rows describe the same (b, f).
    assert torch.allclose(y[:, 0], 2.0 * x[:, ci("re_uinc_x")], atol=1e-5)
    assert torch.allclose(y[:, 3], 2.0 * x[:, ci("im_uinc_y")], atol=1e-5)


def test_flatten_unflatten_are_inverse():
    t = torch.randn(3, 4, 5, 6, 6)
    flat = feat.flatten_freq(t)
    assert flat.shape == (12, 5, 6, 6)
    assert torch.equal(feat.unflatten_freq(flat, 4), t)


def test_pack_inputs_dtype_is_inferred_from_the_incident_field():
    """
    Required by the gradient check: the whole chain has to be able to run in double
    precision, and a hardcoded float32 anywhere in the packing would cap agreement
    at about seven digits of the field -- fewer than three significant figures after
    two nested finite differences.
    """
    d32 = _fake_batch(B=1, F=2, n=8, cdtype=torch.complex64)
    x32 = feat.pack_inputs(d32["phi_t"], d32["chi"], d32["u_inc"],
                           d32["freqs"], d32["nu"])
    assert x32.dtype == torch.float32

    d64 = _fake_batch(B=1, F=2, n=8, cdtype=torch.complex128)
    x64 = feat.pack_inputs(d64["phi_t"].double(), d64["chi"].double(),
                           d64["u_inc"], d64["freqs"].double(), d64["nu"].double())
    assert x64.dtype == torch.float64

    # And an explicit override wins over inference.
    x_forced = feat.pack_inputs(d64["phi_t"].double(), d64["chi"].double(),
                                d64["u_inc"], d64["freqs"].double(),
                                d64["nu"].double(), dtype=torch.float32)
    assert x_forced.dtype == torch.float32


def test_pack_inputs_accepts_precomputed_coords():
    d = _fake_batch(B=2, F=2, n=8)
    coords = feat.coordinate_channels(8, cfg.L_DOMAIN)
    a = feat.pack_inputs(d["phi_t"], d["chi"], d["u_inc"], d["freqs"], d["nu"])
    b = feat.pack_inputs(d["phi_t"], d["chi"], d["u_inc"], d["freqs"], d["nu"],
                         coords=coords)
    assert torch.allclose(a, b)


@pytest.mark.parametrize("bad", ["phi_t", "chi", "freqs", "nu"])
def test_pack_inputs_rejects_mismatched_shapes(bad):
    """
    Each of these asserts guards a real confusion: a phi_t built on the wrong grid,
    a freqs vector that was already flattened, a scalar nu for a batch.  All of them
    would otherwise broadcast into something plausible.
    """
    d = _fake_batch(B=2, F=3, n=8)
    kw = dict(phi_t=d["phi_t"], chi=d["chi"], u_inc=d["u_inc"],
              freqs=d["freqs"], nu=d["nu"])
    if bad == "phi_t":
        kw["phi_t"] = d["phi_t"][:, :4]
    elif bad == "chi":
        kw["chi"] = d["chi"][:1]
    elif bad == "freqs":
        kw["freqs"] = d["freqs"].reshape(-1)
    else:
        kw["nu"] = d["nu"][:1]
    with pytest.raises(AssertionError):
        feat.pack_inputs(**kw)
