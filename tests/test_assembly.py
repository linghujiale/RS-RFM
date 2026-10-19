# 中文注释：检验配点采样和残差矩阵装配。
import numpy as np

from lrfm.assembly.matrix import assemble_system, sample_collocation_1d
from lrfm.assembly.matrix import LRFMBasis1D
from lrfm.assembly.weighting import default_row_weights
from lrfm.lstsq.solver import solve_lstsq
from lrfm.ltc.init import make_reservoirs_from_pou_1d
from lrfm.pdes.heat import HeatEquation1D
from lrfm.pou.decomposition import PartitionOfUnity1D
from lrfm.solution import LRFMSolution1D


def test_assembly_shapes_for_heat_1d():
    pde = HeatEquation1D(T=0.1)
    pou = PartitionOfUnity1D.uniform(pde.domain, 2)
    rng = np.random.default_rng(0)
    reservoirs = make_reservoirs_from_pou_1d(
        rng,
        centers=pou.centers,
        radius=pou.radius,
        m_features=3,
        tau_min=0.1,
        tau_max=1.0,
    )
    collocation = sample_collocation_1d(pde, n_int=7, n_ic=5, n_bc=4, seed=1)
    A, b = assemble_system(
        pde,
        pou,
        reservoirs,
        collocation,
        weights=default_row_weights(collocation.n_int, collocation.n_ic, collocation.n_bc),
    )

    assert A.shape == (7 + 5 + 2 * 4, 2 * 3)
    assert b.shape == (A.shape[0],)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(b))


def test_lrfm_global_basis_has_third_spatial_derivative():
    pde = HeatEquation1D(T=0.1)
    pou = PartitionOfUnity1D.uniform(pde.domain, 3)
    rng = np.random.default_rng(4)
    reservoirs = make_reservoirs_from_pou_1d(
        rng,
        centers=pou.centers,
        radius=pou.radius,
        m_features=4,
        tau_min=0.05,
        tau_max=1.0,
    )
    basis = LRFMBasis1D(pou, reservoirs)
    x = 0.2
    t = 0.03
    eps = 2.0e-5
    terms = basis.terms(x, t)
    fd_dxxx = (basis.terms(x + eps, t)["dxx"] - basis.terms(x - eps, t)["dxx"]) / (2.0 * eps)

    assert terms["dxxx"].shape == terms["value"].shape
    np.testing.assert_allclose(terms["dxxx"], fd_dxxx, rtol=2.0e-5, atol=5.0e-6)


def test_tiny_heat_solve_reduces_training_residual():
    pde = HeatEquation1D(T=0.05)
    pou = PartitionOfUnity1D.uniform(pde.domain, 2)
    rng = np.random.default_rng(5)
    reservoirs = make_reservoirs_from_pou_1d(
        rng,
        centers=pou.centers,
        radius=pou.radius,
        m_features=8,
        tau_min=0.05,
        tau_max=1.0,
    )
    collocation = sample_collocation_1d(pde, n_int=50, n_ic=20, n_bc=10, seed=6)
    A, b = assemble_system(
        pde,
        pou,
        reservoirs,
        collocation,
        weights=default_row_weights(collocation.n_int, collocation.n_ic, collocation.n_bc),
    )
    result = solve_lstsq(A, b)
    solution = LRFMSolution1D(pde, pou, reservoirs, result.coefficients)
    pred = solution(np.asarray([0.1, -0.4]), np.asarray([0.02, 0.04]))

    assert result.residual_norm < np.linalg.norm(b)
    assert np.all(np.isfinite(pred))
