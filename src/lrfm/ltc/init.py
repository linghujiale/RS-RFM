# 中文注释：按全局或单位分解支撑初始化多松弛率随机特征库。
"""Random parameter initializers for frozen relaxation-response reservoirs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def sample_log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int | tuple[int, ...],
) -> np.ndarray:
    """Sample positive values uniformly in log-space."""
    if low <= 0.0 or high <= 0.0:
        raise ValueError("log-uniform bounds must be positive")
    if high < low:
        raise ValueError("high must be greater than or equal to low")
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def sample_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int | tuple[int, ...],
) -> np.ndarray:
    """Sample uniformly from a closed interval."""
    return rng.uniform(low, high, size=size)


def make_reservoir_1d(
    rng: np.random.Generator,
    *,
    m_features: int,
    x_center: float,
    radius: float,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
    r_w: float = 1.0,
    r_b: float = 1.0,
    basis_mode: str = "l_rfm_full",
    constant_tau: float | None = None,
):
    """Sample a 1D :class:`~lrfm.ltc.reservoir.LiquidReservoir`.

    The import is local to avoid a module cycle: ``reservoir`` imports the
    scalar feature routines from ``neuron``.
    """
    from lrfm.ltc.reservoir import LiquidReservoir

    if m_features <= 0:
        raise ValueError("m_features must be positive")
    if constant_tau is not None:
        tau = np.full(m_features, float(constant_tau))
    elif basis_mode == "l_rfm_constant_tau":
        tau = np.full(m_features, np.sqrt(float(tau_min) * float(tau_max)))
    else:
        tau = sample_log_uniform(rng, tau_min, tau_max, size=m_features)

    return LiquidReservoir(
        tau=tau,
        W_in=sample_uniform(rng, -r_w, r_w, size=(m_features, 1)),
        b_in=sample_uniform(rng, -r_b, r_b, size=m_features),
        A=sample_uniform(rng, -1.0, 1.0, size=m_features),
        W_h0=sample_uniform(rng, -r_w, r_w, size=(m_features, 1)),
        b_h0=sample_uniform(rng, -r_b, r_b, size=m_features),
        x_center=np.asarray([x_center], dtype=float),
        radius=float(radius),
        basis_mode=basis_mode,
    )


def make_reservoirs_from_pou_1d(
    rng: np.random.Generator,
    *,
    centers: ArrayLike,
    radius: float,
    m_features: int,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
    r_w: float = 1.0,
    r_b: float = 1.0,
    basis_mode: str = "l_rfm_full",
    constant_tau: float | None = None,
) -> list:
    """Sample one reservoir for each 1D PoU center."""
    return [
        make_reservoir_1d(
            rng,
            m_features=m_features,
            x_center=float(center),
            radius=radius,
            tau_min=tau_min,
            tau_max=tau_max,
            r_w=r_w,
            r_b=r_b,
            basis_mode=basis_mode,
            constant_tau=constant_tau,
        )
        for center in np.asarray(centers, dtype=float)
    ]


def make_reservoir_2d(
    rng: np.random.Generator,
    *,
    m_features: int,
    x_center: ArrayLike,
    radius: float,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
    r_w: float = 1.0,
    r_b: float = 1.0,
    basis_mode: str = "l_rfm_full",
    constant_tau: float | None = None,
):
    """Sample a 2D frozen relaxation-response reservoir."""
    from lrfm.ltc.reservoir import LiquidReservoir

    if m_features <= 0:
        raise ValueError("m_features must be positive")
    center = np.asarray(x_center, dtype=float)
    if center.shape != (2,):
        raise ValueError("x_center must have shape (2,)")
    if constant_tau is not None:
        tau = np.full(m_features, float(constant_tau))
    elif basis_mode == "l_rfm_constant_tau":
        tau = np.full(m_features, np.sqrt(float(tau_min) * float(tau_max)))
    else:
        tau = sample_log_uniform(rng, tau_min, tau_max, size=m_features)
    return LiquidReservoir(
        tau=tau,
        W_in=sample_uniform(rng, -r_w, r_w, size=(m_features, 2)),
        b_in=sample_uniform(rng, -r_b, r_b, size=m_features),
        A=sample_uniform(rng, -1.0, 1.0, size=m_features),
        W_h0=sample_uniform(rng, -r_w, r_w, size=(m_features, 2)),
        b_h0=sample_uniform(rng, -r_b, r_b, size=m_features),
        x_center=center,
        radius=float(radius),
        basis_mode=basis_mode,
    )


def make_reservoir_3d(
    rng: np.random.Generator,
    *,
    m_features: int,
    x_center: ArrayLike,
    radius: float,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
    r_w: float = 1.0,
    r_b: float = 1.0,
    basis_mode: str = "l_rfm_full",
):
    """Sample a 3D frozen LTC reservoir."""
    from lrfm.ltc.reservoir import LiquidReservoir

    if m_features <= 0:
        raise ValueError("m_features must be positive")
    center = np.asarray(x_center, dtype=float)
    if center.shape != (3,):
        raise ValueError("x_center must have shape (3,)")
    return LiquidReservoir(
        tau=sample_log_uniform(rng, tau_min, tau_max, size=m_features),
        W_in=sample_uniform(rng, -r_w, r_w, size=(m_features, 3)),
        b_in=sample_uniform(rng, -r_b, r_b, size=m_features),
        A=sample_uniform(rng, -1.0, 1.0, size=m_features),
        W_h0=sample_uniform(rng, -r_w, r_w, size=(m_features, 3)),
        b_h0=sample_uniform(rng, -r_b, r_b, size=m_features),
        x_center=center,
        radius=float(radius),
        basis_mode=basis_mode,
    )


def make_reservoirs_from_pou_2d(
    rng: np.random.Generator,
    *,
    centers: ArrayLike,
    radius: float,
    m_features: int,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
    r_w: float = 1.0,
    r_b: float = 1.0,
    basis_mode: str = "l_rfm_full",
    constant_tau: float | None = None,
) -> list:
    """Sample one 2D reservoir for each 2D PoU center."""
    center_arr = np.asarray(centers, dtype=float)
    if center_arr.ndim != 2 or center_arr.shape[1] != 2:
        raise ValueError("centers must have shape (K, 2)")
    return [
        make_reservoir_2d(
            rng,
            m_features=m_features,
            x_center=center,
            radius=radius,
            tau_min=tau_min,
            tau_max=tau_max,
            r_w=r_w,
            r_b=r_b,
            basis_mode=basis_mode,
            constant_tau=constant_tau,
        )
        for center in center_arr
    ]


def make_reservoirs_from_pou_3d(
    rng: np.random.Generator,
    *,
    centers: ArrayLike,
    radius: float,
    m_features: int,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
    r_w: float = 1.0,
    r_b: float = 1.0,
    basis_mode: str = "l_rfm_full",
) -> list:
    """Sample one 3D reservoir for each 3D PoU center."""
    center_arr = np.asarray(centers, dtype=float)
    if center_arr.ndim != 2 or center_arr.shape[1] != 3:
        raise ValueError("centers must have shape (K, 3)")
    return [
        make_reservoir_3d(
            rng,
            m_features=m_features,
            x_center=center,
            radius=radius,
            tau_min=tau_min,
            tau_max=tau_max,
            r_w=r_w,
            r_b=r_b,
            basis_mode=basis_mode,
        )
        for center in center_arr
    ]
