"""
Batch-independence of the solver (§11.2 step 6 regression).

Reproduces the 20-sample re-solve FAIL (median 0.000e+00, max 3.660e-01,
gate 1e-05): the same physical sample solved differently depending on its
batch mates, because the harmonic-average guard was
`VOID_STIFFNESS_FLOOR * mu.max()` over the whole batch.  In a mixed-nu batch
the floor (~3.3e-05) exceeded the void mu of low-mu samples (~2.1e-05) and
raised it by ~60%, while same-floor batches reproduced bitwise -- hence the
bimodal 0-vs-0.3 signature.

The fix (`cfg.MU_HARMONIC_FLOOR`, absolute 1e-12) must keep this test green:
same theta+nu in different batch contexts gives bitwise-identical mu_xy and
identical short-run fields.
"""

from __future__ import annotations

import torch

from src import config as cfg


def _batch(nu_list, theta=None):
    from src.data.generate import _materials, fine_chi

    if theta is None:
        theta = torch.tensor([[4.0, 4.0, 0.35]], dtype=torch.float32).repeat(len(nu_list), 1)
    nu_vals = torch.tensor(nu_list)
    chi = fine_chi(theta)
    return _materials(chi, nu_vals)


def test_harmonic_floor_is_batch_independent():
    assert cfg.MU_HARMONIC_FLOOR == 1e-12
    # Far below any physical mu: min void mu is min(mu0)*1e-4 ~ 2e-05.
    min_void_mu = min((1.0 - 2.0 * nu) / (2.0 * (1.0 - nu)) for nu in cfg.NU_LIST) * 1e-4
    assert cfg.MU_HARMONIC_FLOOR < min_void_mu / 1e3


def test_same_sample_same_mu_xy_regardless_of_batch_mates():
    from src.solver.fdtd_elastic import ElasticFDTD2D

    theta0 = torch.tensor([[4.0, 4.0, 0.35]], dtype=torch.float32)
    # Same physical sample (nu=0.37) paired once with a stiff mate, once with itself.
    lam_a, mu_a, rho_a = _batch([0.37, 0.25], theta0.repeat(2, 1))
    lam_b, mu_b, rho_b = _batch([0.37, 0.37], theta0.repeat(2, 1))
    sim_a = ElasticFDTD2D(lam_a, mu_a, rho_a)
    sim_b = ElasticFDTD2D(lam_b, mu_b, rho_b)
    assert torch.equal(sim_a.mu_xy[0], sim_b.mu_xy[0]), (
        f"mu_xy differs by {float((sim_a.mu_xy[0] - sim_b.mu_xy[0]).abs().max()):.3e} "
        "for the same sample in different batches"
    )


def test_short_run_is_batch_independent():
    from src.solver.fdtd_elastic import ElasticFDTD2D
    from src.solver import harmonic as H

    theta0 = torch.tensor([[4.0, 4.0, 0.35]], dtype=torch.float32)
    om = torch.tensor([2.0 * 3.141592653589793 * 1.0], dtype=torch.float64)
    outs = []
    for nus in ([0.37, 0.25], [0.37, 0.37]):
        lam, mu, rho = _batch(nus, theta0.repeat(2, 1))
        # Tiny grid for speed: n_total=24, absorber 4 -> core 16, downsample 2 -> net 8.
        sim = ElasticFDTD2D(lam, mu, rho, dx=0.25, dt=0.05, n_pml=4, downsample=2)
        src = [(8, 8), (8, 8)]
        res = sim.run(src, nt=20, recv_yx=[(2, 2)], omegas=om)
        u = H.displacement_from_field(res.phasors, omegas=om)
        outs.append(u[0].clone())
    assert torch.equal(outs[0], outs[1]), (
        f"short-run field differs by "
        f"{float((outs[0] - outs[1]).abs().max()):.3e} for the same sample"
    )


def test_snapshot_covers_harmonic_floor():
    from src.data.generate import _SNAPSHOT_KEYS, config_snapshot

    assert "MU_HARMONIC_FLOOR" in _SNAPSHOT_KEYS
    assert config_snapshot()["MU_HARMONIC_FLOOR"] == cfg.MU_HARMONIC_FLOOR
