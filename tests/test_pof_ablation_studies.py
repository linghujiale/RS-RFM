# 中文注释：检验 PoF 消融实验设置和结果结构。
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from lrfm.nonlinear.vorticity_streamfunction import VorticityStreamfunctionSolution2D


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pof_ablation_studies.py"
SPEC = importlib.util.spec_from_file_location("run_pof_ablation_studies", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def args(**updates):
    values = {
        "case": "two_rate_physics_range",
        "temporal_mode": "relaxation_quotient",
        "rho_static": 0.25,
        "tau_min": 0.08,
        "tau_max": 3.0,
        "sampling_mode": "log_stratified",
        "tau0": 1.0,
        "seed": 10,
        "P": 384,
        "output": Path("unused"),
    }
    values.update(updates)
    return Namespace(**values)


def test_full_and_naive_share_all_random_draws():
    full_args = args()
    candidate, pde = MODULE.build_problem(full_args)
    full, full_audit = MODULE.make_paired_basis(full_args, candidate, pde)
    naive_args = args(temporal_mode="naive_exponential")
    naive_candidate, naive_pde = MODULE.build_problem(naive_args)
    naive, naive_audit = MODULE.make_paired_basis(naive_args, naive_candidate, naive_pde)
    np.testing.assert_array_equal(full.W_space, naive.W_space)
    np.testing.assert_array_equal(full.b_space, naive.b_space)
    np.testing.assert_array_equal(full.tau, naive.tau)
    np.testing.assert_array_equal(full.is_static, naive.is_static)
    assert full_audit["spatial_draw_sha256"] == naive_audit["spatial_draw_sha256"]
    assert full_audit["tau_draw_sha256"] == naive_audit["tau_draw_sha256"]


def test_single_rate_changes_only_dynamic_tau_values():
    full_args = args()
    candidate, pde = MODULE.build_problem(full_args)
    full, _ = MODULE.make_paired_basis(full_args, candidate, pde)
    single_args = args(temporal_mode="single_rate")
    single_candidate, single_pde = MODULE.build_problem(single_args)
    single, _ = MODULE.make_paired_basis(single_args, single_candidate, single_pde)
    np.testing.assert_array_equal(full.W_space, single.W_space)
    np.testing.assert_array_equal(full.b_space, single.b_space)
    np.testing.assert_array_equal(full.is_static, single.is_static)
    np.testing.assert_allclose(single.tau[~single.is_static], single_pde.T)


def test_anchor_counts_are_deterministic_and_sum_to_P():
    expected = {0.0: 0, 0.125: 48, 0.25: 96, 0.5: 192}
    hashes = {}
    for rho, n_static in expected.items():
        run_args = args(rho_static=rho)
        candidate, pde = MODULE.build_problem(run_args)
        basis, audit = MODULE.make_paired_basis(run_args, candidate, pde)
        assert audit["static_features"] == n_static
        assert audit["dynamic_features"] + audit["static_features"] == run_args.P
        hashes[rho] = audit["spatial_draw_sha256"]
        repeat, repeat_audit = MODULE.make_paired_basis(run_args, candidate, pde)
        np.testing.assert_array_equal(basis.tau, repeat.tau)
        assert audit == repeat_audit
    assert len(set(hashes.values())) == 1


def test_hard_lift_matches_initial_condition_for_every_temporal_mode():
    points = np.asarray([[0.2, 0.4], [1.1, 2.3]])
    for mode in ("relaxation_quotient", "naive_exponential", "single_rate"):
        run_args = args(temporal_mode=mode)
        candidate, pde = MODULE.build_problem(run_args)
        basis, _ = MODULE.make_paired_basis(run_args, candidate, pde)
        coefficients = np.linspace(-0.3, 0.7, 2 * basis.n_features)
        solution = VorticityStreamfunctionSolution2D(
            pde=pde,
            basis=basis,
            coefficients=coefficients,
            hard_initial=True,
        )
        actual = solution(points, np.zeros(points.shape[0]))
        expected = np.vstack([pde.initial_condition(point) for point in points])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)
