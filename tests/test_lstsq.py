# 中文注释：检验正则化最小二乘求解器。
import numpy as np

from lrfm.lstsq.solver import solve_lstsq


def test_lstsq_recovers_constructed_solution():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(30, 5))
    coeffs = rng.normal(size=5)
    b = A @ coeffs

    result = solve_lstsq(A, b)
    np.testing.assert_allclose(result.coefficients, coeffs, rtol=1.0e-11, atol=1.0e-11)
    assert result.rank == 5
    assert result.residual_norm < 1.0e-10
