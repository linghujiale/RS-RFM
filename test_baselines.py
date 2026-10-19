# 中文注释：检验 ST-RFM、PIELM 等对比基函数。
import numpy as np

from lrfm.assembly.matrix import sample_collocation_1d
from lrfm.assembly.weighting import default_row_weights
from lrfm.baselines import (
    assemble_feature_system_1d,
    make_separable_rfm_global_basis_1d,
    make_pielm_basis_1d,
    make_piln_fixed_liquid_basis_1d,
    make_strfm_sov_basis_1d,
    make_strfm_stc_basis_1d,
)
from lrfm.lstsq.solver import solve_lstsq
from lrfm.pdes.heat import HeatEquation1D
from lrfm.pou.decomposition import PartitionOfUnity1D


def _assert_basis_derivatives_match_finite_differences(basis):
    x = 0.13
    t = 0.07
    hx = 1.0e-5
    ht = 1.0e-5
    terms = basis.terms(x, t)
    value = terms["value"]
    x_plus = basis.terms(x + hx, t)["value"]
    x_minus = basis.terms(x - hx, t)["value"]
    dxx_plus = basis.terms(x + hx, t)["dxx"]
    dxx_minus = basis.terms(x - hx, t)["dxx"]
    t_plus = basis.terms(x, t + ht)["value"]
    t_minus = basis.terms(x, t - ht)["value"]
    fd_dx = (x_plus - x_minus) / (2.0 * hx)
    fd_dxx = (x_plus - 2.0 * value + x_minus) / hx**2
    fd_dxxx = (dxx_plus - dxx_minus) / (2.0 * hx)
    fd_dt = (t_plus - t_minus) / (2.0 * ht)

    np.testing.assert_allclose(terms["dx"], fd_dx, rtol=2.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(terms["dxx"], fd_dxx, rtol=2.0e-4, atol=5.0e-5)
    np.testing.assert_allclose(terms["dxxx"], fd_dxxx, rtol=5.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(terms["dt"], fd_dt, rtol=2.0e-5, atol=1.0e-6)


def test_baseline_derivatives_match_finite_differences():
    rng = np.random.default_rng(11)
    pde = HeatEquation1D(T=0.2)
    pou = PartitionOfUnity1D.uniform(pde.domain, 3)
    bases = [
        make_separable_rfm_global_basis_1d(
            rng, n_features=12, domain=pde.domain, T=pde.T
        ),
        make_strfm_sov_basis_1d(rng, pou=pou, m_features=4, T=pde.T),
        make_strfm_stc_basis_1d(rng, pou=pou, m_features=4, T=pde.T),
        make_pielm_basis_1d(rng, n_features=12, domain=pde.domain, T=pde.T),
        make_piln_fixed_liquid_basis_1d(rng, n_features=12, domain=pde.domain, T=pde.T),
    ]
    for basis in bases:
        _assert_basis_derivatives_match_finite_differences(basis)


def test_baseline_assembly_shapes_for_heat_1d():
    rng = np.random.default_rng(12)
    pde = HeatEquation1D(T=0.1)
    basis = make_pielm_basis_1d(rng, n_features=9, domain=pde.domain, T=pde.T)
    collocation = sample_collocation_1d(pde, n_int=7, n_ic=5, n_bc=4, seed=13)
    A, b = assemble_feature_system_1d(
        pde,
        basis,
        collocation,
        weights=default_row_weights(collocation.n_int, collocation.n_ic, collocation.n_bc),
    )
    assert A.shape == (7 + 5 + 2 * 4, 9)
    assert b.shape == (A.shape[0],)
    assert np.all(np.isfinite(A))


def test_baseline_tiny_heat_solve_reduces_residual():
    rng = np.random.default_rng(14)
    pde = HeatEquation1D(T=0.05)
    basis = make_pielm_basis_1d(rng, n_features=16, domain=pde.domain, T=pde.T)
    collocation = sample_collocation_1d(pde, n_int=50, n_ic=20, n_bc=10, seed=15)
    A, b = assemble_feature_system_1d(
        pde,
        basis,
        collocation,
        weights=default_row_weights(collocation.n_int, collocation.n_ic, collocation.n_bc),
    )
    result = solve_lstsq(A, b)
    assert result.residual_norm < np.linalg.norm(b)
