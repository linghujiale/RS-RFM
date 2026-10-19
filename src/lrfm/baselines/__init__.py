# 中文注释：导出论文对比方法的公共接口。
"""Baseline methods for comparison experiments."""

from lrfm.baselines.common import (
    CompositeFeatureBasis2D,
    CompositeFeatureBasis3D,
    FittedBasisSolution1D,
    FittedBasisSolution2D,
    FittedBasisSolution3D,
    assemble_feature_system_1d,
)
from lrfm.baselines.exp_rfm import ExpRFMBasis3D, make_exp_rfm_basis_3d
from lrfm.baselines.moving_interface import (
    MovingWavyInterfaceProfileBasis3D,
    make_moving_wavy_interface_profile_basis_3d,
)
from lrfm.baselines.pielm import (
    PIELMBasis1D,
    PIELMBasis2D,
    PIELMBasis3D,
    make_pielm_basis_1d,
    make_pielm_basis_2d,
    make_pielm_basis_3d,
)
# PILN is an optional JAX/Equinox baseline.  Keep the NumPy-only baselines
# importable on lightweight CPU nodes where those optional packages are absent.
_PILN_EXPORTS: list[str] = []
try:
    from lrfm.baselines.piln import (
        PILNFixedLiquidBasis1D,
        PILNFixedLiquidBasis2D,
        PILNFixedLiquidBasis3D,
        make_piln_fixed_liquid_basis_1d,
        make_piln_fixed_liquid_basis_2d,
        make_piln_fixed_liquid_basis_3d,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"equinox", "jax", "jaxlib"}:
        raise
else:
    _PILN_EXPORTS = [
        "PILNFixedLiquidBasis1D",
        "PILNFixedLiquidBasis2D",
        "PILNFixedLiquidBasis3D",
        "make_piln_fixed_liquid_basis_1d",
        "make_piln_fixed_liquid_basis_2d",
        "make_piln_fixed_liquid_basis_3d",
    ]
from lrfm.baselines.radial import (
    GrayScottSpotGaussianBasis3D,
    ShrinkingCircleRadialBasis2D,
    make_grayscott_spot_gaussian_basis_3d,
    make_shrinking_circle_radial_basis_2d,
)
from lrfm.baselines.strfm import (
    SeparableRFMGlobalBasis1D,
    SeparableRFMGlobalBasis2D,
    SeparableRFMGlobalBasis3D,
    STRFMSoVBasis1D,
    STRFMSoVBasis2D,
    STRFMSoVBasis3D,
    STRFMSTCBasis1D,
    STRFMSTCBasis2D,
    STRFMSTCBasis3D,
    make_separable_rfm_global_basis_1d,
    make_separable_rfm_global_basis_2d,
    make_separable_rfm_global_basis_3d,
    make_strfm_sov_basis_1d,
    make_strfm_sov_basis_2d,
    make_strfm_sov_basis_3d,
    make_strfm_stc_basis_1d,
    make_strfm_stc_basis_2d,
    make_strfm_stc_basis_3d,
)

__all__ = [
    "FittedBasisSolution1D",
    "FittedBasisSolution2D",
    "FittedBasisSolution3D",
    "CompositeFeatureBasis2D",
    "CompositeFeatureBasis3D",
    "ExpRFMBasis3D",
    "MovingWavyInterfaceProfileBasis3D",
    "PIELMBasis1D",
    "PIELMBasis2D",
    "PIELMBasis3D",
    "GrayScottSpotGaussianBasis3D",
    "ShrinkingCircleRadialBasis2D",
    "SeparableRFMGlobalBasis1D",
    "SeparableRFMGlobalBasis2D",
    "SeparableRFMGlobalBasis3D",
    "STRFMSoVBasis1D",
    "STRFMSoVBasis2D",
    "STRFMSoVBasis3D",
    "STRFMSTCBasis1D",
    "STRFMSTCBasis2D",
    "STRFMSTCBasis3D",
    "assemble_feature_system_1d",
    "make_exp_rfm_basis_3d",
    "make_moving_wavy_interface_profile_basis_3d",
    "make_pielm_basis_1d",
    "make_pielm_basis_2d",
    "make_pielm_basis_3d",
    "make_grayscott_spot_gaussian_basis_3d",
    "make_shrinking_circle_radial_basis_2d",
    "make_separable_rfm_global_basis_1d",
    "make_separable_rfm_global_basis_2d",
    "make_separable_rfm_global_basis_3d",
    "make_strfm_sov_basis_1d",
    "make_strfm_sov_basis_2d",
    "make_strfm_sov_basis_3d",
    "make_strfm_stc_basis_1d",
    "make_strfm_stc_basis_2d",
    "make_strfm_stc_basis_3d",
] + _PILN_EXPORTS
