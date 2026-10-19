# 中文注释：检验单位分解权重及其归一化性质。
import numpy as np

from lrfm.pou.decomposition import PartitionOfUnity1D


def test_pou_sums_to_one_on_fine_grid():
    pou = PartitionOfUnity1D.uniform((-1.0, 1.0), 5)
    for x in np.linspace(-1.0, 1.0, 101):
        vals = pou.evaluate(float(x))
        assert np.isclose(vals.psi.sum(), 1.0, atol=1.0e-12)
        assert np.isclose(vals.dpsi_dx.sum(), 0.0, atol=1.0e-11)
        assert np.isclose(vals.d2psi_dx2.sum(), 0.0, atol=1.0e-9)
        assert np.isclose(vals.d3psi_dx3.sum(), 0.0, atol=1.0e-7)


def test_single_patch_pou_is_constant():
    pou = PartitionOfUnity1D.uniform((-1.0, 1.0), 1)
    vals = pou.evaluate(0.37)
    np.testing.assert_allclose(vals.psi, np.asarray([1.0]))
    np.testing.assert_allclose(vals.dpsi_dx, np.asarray([0.0]))
    np.testing.assert_allclose(vals.d2psi_dx2, np.asarray([0.0]))
    np.testing.assert_allclose(vals.d3psi_dx3, np.asarray([0.0]))
