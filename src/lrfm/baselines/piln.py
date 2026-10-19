# 中文注释：实现固定动力学的 PILN 风格基线。
"""PILN baseline implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lrfm.ltc.init import make_reservoir_1d, make_reservoir_2d, make_reservoir_3d
from lrfm.ltc.reservoir import LiquidReservoir


@dataclass(frozen=True)
class PILNFixedLiquidBasis1D:
    """P2 heat-sanity PILN-style liquid basis.

    PILN trains an LTC network. For the P2 heat sanity gate we keep the same
    liquid temporal primitive but use a frozen global reservoir with an LS
    readout. This keeps the comparison pipeline and derivative checks in place;
    hidden-state training can be added when the nonlinear PILN benchmark is
    needed.
    """

    reservoir: LiquidReservoir
    name: str = "piln_fixed_liquid"

    @property
    def n_features(self) -> int:
        return self.reservoir.n_features

    def terms(self, x: float, t: float) -> dict[str, np.ndarray]:
        local = self.reservoir.features_with_derivs(np.asarray([x]), t, order=3)
        return {
            "value": local["phi"],
            "dt": local["dphi_dt"],
            "dx": local["grad_x"][:, 0],
            "dxx": local["lap_x"],
            "dxxx": local["d3phi_dx3"],
        }


@dataclass(frozen=True)
class PILNFixedLiquidBasis2D:
    """2D fixed-liquid baseline with a global frozen reservoir and LS readout."""

    reservoir: LiquidReservoir
    name: str = "piln_fixed_liquid"

    @property
    def n_features(self) -> int:
        return self.reservoir.n_features

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        local = self.reservoir.features_with_derivs(np.asarray(x, dtype=float), t, order=2)
        return {
            "value": local["phi"],
            "dt": local["dphi_dt"],
            "grad": local["grad_x"],
            "lap": local["lap_x"],
        }


@dataclass(frozen=True)
class PILNFixedLiquidBasis3D:
    """3D fixed-liquid baseline with a global frozen reservoir and LS readout."""

    reservoir: LiquidReservoir
    name: str = "piln_fixed_liquid"

    @property
    def n_features(self) -> int:
        return self.reservoir.n_features

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        local = self.reservoir.features_with_derivs(np.asarray(x, dtype=float), t, order=2)
        return {
            "value": local["phi"],
            "dt": local["dphi_dt"],
            "grad": local["grad_x"],
            "lap": local["lap_x"],
        }


def make_piln_fixed_liquid_basis_1d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[float, float],
    T: float,
    tau_min: float | None = None,
    tau_max: float | None = None,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> PILNFixedLiquidBasis1D:
    """Sample a global liquid basis used for P2 PILN sanity checks."""
    lo, hi = map(float, domain)
    reservoir = make_reservoir_1d(
        rng,
        m_features=n_features,
        x_center=0.5 * (lo + hi),
        radius=0.5 * (hi - lo),
        tau_min=0.01 * T if tau_min is None else tau_min,
        tau_max=10.0 * T if tau_max is None else tau_max,
        r_w=r_w,
        r_b=r_b,
        basis_mode="piln_fixed_liquid",
    )
    return PILNFixedLiquidBasis1D(reservoir=reservoir)


def make_piln_fixed_liquid_basis_2d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[tuple[float, float], tuple[float, float]],
    T: float,
    tau_min: float | None = None,
    tau_max: float | None = None,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> PILNFixedLiquidBasis2D:
    """Sample a global 2D liquid basis used for PILN-style comparisons."""
    bounds = np.asarray(domain, dtype=float)
    if bounds.shape != (2, 2):
        raise ValueError("2D domain must have shape (2, 2)")
    center = 0.5 * (bounds[:, 0] + bounds[:, 1])
    radius = 0.5 * float(np.max(bounds[:, 1] - bounds[:, 0]))
    reservoir = make_reservoir_2d(
        rng,
        m_features=n_features,
        x_center=center,
        radius=radius,
        tau_min=0.01 * T if tau_min is None else tau_min,
        tau_max=10.0 * T if tau_max is None else tau_max,
        r_w=r_w,
        r_b=r_b,
        basis_mode="piln_fixed_liquid",
    )
    return PILNFixedLiquidBasis2D(reservoir=reservoir)


def make_piln_fixed_liquid_basis_3d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    T: float,
    tau_min: float | None = None,
    tau_max: float | None = None,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> PILNFixedLiquidBasis3D:
    """Sample a global 3D liquid basis used for PILN-style comparisons."""
    bounds = np.asarray(domain, dtype=float)
    if bounds.shape != (3, 2):
        raise ValueError("3D domain must have shape (3, 2)")
    center = 0.5 * (bounds[:, 0] + bounds[:, 1])
    radius = 0.5 * float(np.max(bounds[:, 1] - bounds[:, 0]))
    reservoir = make_reservoir_3d(
        rng,
        m_features=n_features,
        x_center=center,
        radius=radius,
        tau_min=0.01 * T if tau_min is None else tau_min,
        tau_max=10.0 * T if tau_max is None else tau_max,
        r_w=r_w,
        r_b=r_b,
        basis_mode="piln_fixed_liquid",
    )
    return PILNFixedLiquidBasis3D(reservoir=reservoir)
