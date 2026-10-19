# 中文注释：检验二维流动适用性扫描的算例与指标。
import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pof_suitability_scan.py"
SPEC = importlib.util.spec_from_file_location("run_pof_suitability_scan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_ic_compatible_temporal_factor_matches_exponential_hard_lift():
    tau = np.asarray([0.125, 2.0])
    static = np.asarray([False, False])
    for time in (0.0, 1.0e-9, 0.03, 0.7):
        value, derivative = MODULE.ICCompatibleRelaxationBasis2D._temporal_terms(
            tau, static, time
        )
        assert np.all(np.isfinite(value))
        assert np.all(np.isfinite(derivative))
        if time > 0.0:
            np.testing.assert_allclose(
                time * value,
                np.exp(-time / tau) - 1.0,
                rtol=2.0e-10,
                atol=2.0e-12,
            )


def test_static_anchor_is_linear_under_hard_lift():
    tau = np.asarray([np.inf])
    static = np.asarray([True])
    value, derivative = MODULE.ICCompatibleRelaxationBasis2D._temporal_terms(
        tau, static, 0.4
    )
    np.testing.assert_allclose(value, np.ones(1))
    np.testing.assert_allclose(derivative, np.zeros(1))


def test_naive_exponential_temporal_factor_and_derivative():
    tau = np.asarray([0.125, 2.0])
    static = np.asarray([False, False])
    for time in (0.0, 0.03, 0.7):
        value, derivative = MODULE.ICCompatibleRelaxationBasis2D._temporal_terms(
            tau, static, time, "naive_exponential"
        )
        expected = np.exp(-time / tau)
        np.testing.assert_allclose(value, expected, rtol=1.0e-13, atol=1.0e-14)
        np.testing.assert_allclose(derivative, -expected / tau, rtol=1.0e-13, atol=1.0e-14)
        np.testing.assert_allclose(time * value, time * expected, rtol=1.0e-13, atol=1.0e-14)


def test_temporal_derivatives_match_finite_differences():
    tau = np.asarray([0.125, 2.0])
    static = np.asarray([False, False])
    time = 0.37
    step = 1.0e-6
    for mode in ("relaxation_quotient", "naive_exponential"):
        value, derivative = MODULE.ICCompatibleRelaxationBasis2D._temporal_terms(
            tau, static, time, mode
        )
        plus = MODULE.ICCompatibleRelaxationBasis2D._temporal_terms(
            tau, static, time + step, mode
        )[0]
        minus = MODULE.ICCompatibleRelaxationBasis2D._temporal_terms(
            tau, static, time - step, mode
        )[0]
        finite_difference = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-7, atol=2.0e-9)


def test_candidate_matrix_has_predeclared_original_and_aligned_cases():
    matrix = MODULE.candidate_matrix()
    assert set(matrix) == {
        "taylor_green",
        "two_rate_vortex",
        "interacting_vortex",
        "linear_interacting_stokes",
        "double_shear",
        "multiscale_shear",
        "multiscale_shear_k12",
    }
    assert len(matrix["taylor_green"]) == 2
    assert len(matrix["two_rate_vortex"]) == 2
    assert len(matrix["interacting_vortex"]) == 1
    assert len(matrix["linear_interacting_stokes"]) == 1
    assert len(matrix["double_shear"]) == 2
    assert len(matrix["multiscale_shear"]) == 1
    assert len(matrix["multiscale_shear_k12"]) == 1
    assert matrix["multiscale_shear_k12"][0].fast_wavenumber == 2
    assert matrix["taylor_green"][1].viscosity > matrix["taylor_green"][0].viscosity
    assert matrix["double_shear"][1].viscosity > matrix["double_shear"][0].viscosity
