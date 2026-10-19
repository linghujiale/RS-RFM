# 中文注释：检验 Picard 非线性迭代和耦合求解。
import numpy as np

from lrfm.assembly.matrix import LRFMBasis1D, sample_collocation_1d
from lrfm.assembly.weighting import default_row_weights
from lrfm.ltc.init import make_reservoirs_from_pou_1d
from lrfm.nonlinear.picard import solve_picard_1d
from lrfm.pdes.allen_cahn import AllenCahn1D, solve_allen_cahn_reference_1d
from lrfm.pdes.burgers import Burgers1D, cole_hopf_burgers_reference_1d
from lrfm.pdes.kdv import KdV1D, solve_kdv_reference_1d
from lrfm.pou.decomposition import PartitionOfUnity1D


def test_allen_cahn_picard_feature_residual_matches_manual_formula():
    pde = AllenCahn1D(epsilon=0.01)
    value = np.asarray([1.0, 2.0])
    dt = np.asarray([0.2, 0.3])
    dxx = np.asarray([4.0, 5.0])
    row = pde.picard_feature_residual(value, dt, np.zeros(2), dxx, 0.1, 0.2, u_prev=0.5)
    expected = dt - 0.01 * dxx + (0.5**2 - 1.0) * value
    np.testing.assert_allclose(row, expected)


def test_burgers_cole_hopf_recovers_initial_condition():
    pde = Burgers1D()
    x = np.linspace(-1.0, 1.0, 32, endpoint=False)
    got = cole_hopf_burgers_reference_1d(pde, x=x, t=np.zeros_like(x))
    np.testing.assert_allclose(got, pde.initial_condition(x), atol=1.0e-12)


def test_burgers_cole_hopf_recovers_scaled_initial_condition():
    pde = Burgers1D(amplitude=1.7, mode=2)
    x = np.linspace(-1.0, 1.0, 32, endpoint=False)
    got = cole_hopf_burgers_reference_1d(pde, x=x, t=np.zeros_like(x))
    np.testing.assert_allclose(got, pde.initial_condition(x), atol=1.0e-12)


def test_kdv_picard_feature_residual_uses_third_derivative():
    pde = KdV1D()
    dt = np.asarray([0.2, 0.3])
    dx = np.asarray([1.0, -2.0])
    dxx = np.asarray([4.0, 5.0])
    dxxx = np.asarray([-0.5, 0.75])
    row = pde.picard_feature_residual(np.zeros(2), dt, dx, dxx, dxxx, 0.1, 0.2, u_prev=0.25)
    expected = dt + 6.0 * 0.25 * dx + dxxx
    np.testing.assert_allclose(row, expected)


def test_allen_cahn_reference_solver_shapes():
    pde = AllenCahn1D(epsilon=1.0e-3, T=0.02)
    reference = solve_allen_cahn_reference_1d(pde, nx=32, nt=5, rtol=1.0e-5, atol=1.0e-7)
    assert reference.u.shape == (5, 32)
    assert np.all(np.isfinite(reference.u))


def test_kdv_reference_solver_shapes():
    pde = KdV1D(T=0.05)
    reference = solve_kdv_reference_1d(pde, nx=32, nt=5)
    assert reference.u.shape == (5, 32)
    assert np.all(np.isfinite(reference.u))
    np.testing.assert_allclose(reference.u[0], pde.initial_condition(reference.x), atol=1.0e-14)


def test_tiny_picard_solve_runs_for_allen_cahn():
    pde = AllenCahn1D(epsilon=1.0e-3, T=0.02)
    pou = PartitionOfUnity1D.uniform(pde.domain, 2)
    rng = np.random.default_rng(21)
    reservoirs = make_reservoirs_from_pou_1d(
        rng,
        centers=pou.centers,
        radius=pou.radius,
        m_features=6,
        tau_min=0.01 * pde.T,
        tau_max=10.0 * pde.T,
    )
    basis = LRFMBasis1D(pou, reservoirs)
    collocation = sample_collocation_1d(pde, n_int=30, n_ic=16, n_bc=8, seed=22)
    result = solve_picard_1d(
        pde,
        basis,
        collocation,
        weights=default_row_weights(collocation.n_int, collocation.n_ic, collocation.n_bc),
        max_iters=2,
        monitor_nx=16,
        monitor_nt=5,
    )
    assert result.iterations == 2
    assert np.all(np.isfinite(result.coefficients))
