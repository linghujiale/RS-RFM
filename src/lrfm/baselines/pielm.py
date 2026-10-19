# 中文注释：实现 PIELM 基线及其二维、三维随机基构造。
"""PIELM baseline implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lrfm.baselines.common import scale_space_1d, scale_space_2d, scale_space_nd, scale_time, tanh_terms, tanh_terms_2d


@dataclass(frozen=True)
class PIELMBasis1D:
    """Global physics-informed ELM basis ``tanh(w_x x + w_t t + b)``."""

    W_x: np.ndarray
    W_t: np.ndarray
    b: np.ndarray
    domain: tuple[float, float]
    T: float
    name: str = "pielm"

    @property
    def n_features(self) -> int:
        return int(self.b.shape[0])

    def terms(self, x: float, t: float) -> dict[str, np.ndarray]:
        x_scaled, dx_scaled = scale_space_1d(self.domain, x)
        t_scaled, dt_scaled = scale_time(self.T, t)
        s = self.W_x * x_scaled + self.W_t * t_scaled + self.b
        return tanh_terms(s, self.W_x * dx_scaled, self.W_t * dt_scaled)


@dataclass(frozen=True)
class PIELMBasis2D:
    """Global 2D PIELM basis ``tanh(w_x*x + w_y*y + w_t*t + b)``."""

    W_space: np.ndarray
    W_t: np.ndarray
    b: np.ndarray
    domain: tuple[tuple[float, float], tuple[float, float]]
    T: float
    name: str = "pielm"

    @property
    def n_features(self) -> int:
        return int(self.b.shape[0])

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        x_scaled, dx_scaled = scale_space_2d(self.domain, x)
        t_scaled, dt_scaled = scale_time(self.T, t)
        s = self.W_space @ x_scaled + self.W_t * t_scaled + self.b
        return tanh_terms_2d(s, self.W_space * dx_scaled[None, :], self.W_t * dt_scaled)


@dataclass(frozen=True)
class PIELMBasis3D:
    """Global 3D PIELM basis ``tanh(w_x*x + w_y*y + w_z*z + w_t*t + b)``."""

    W_space: np.ndarray
    W_t: np.ndarray
    b: np.ndarray
    domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    T: float
    name: str = "pielm"

    @property
    def n_features(self) -> int:
        return int(self.b.shape[0])

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        x_scaled, dx_scaled = scale_space_nd(self.domain, x)
        t_scaled, dt_scaled = scale_time(self.T, t)
        s = self.W_space @ x_scaled + self.W_t * t_scaled + self.b
        return tanh_terms_2d(s, self.W_space * dx_scaled[None, :], self.W_t * dt_scaled)


def make_pielm_basis_1d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[float, float],
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> PIELMBasis1D:
    """Sample a global PIELM basis."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    return PIELMBasis1D(
        W_x=rng.uniform(-r_w, r_w, size=n_features),
        W_t=rng.uniform(-r_w, r_w, size=n_features),
        b=rng.uniform(-r_b, r_b, size=n_features),
        domain=(float(domain[0]), float(domain[1])),
        T=float(T),
    )


def make_pielm_basis_2d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[tuple[float, float], tuple[float, float]],
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> PIELMBasis2D:
    """Sample a global 2D PIELM basis."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    return PIELMBasis2D(
        W_space=rng.uniform(-r_w, r_w, size=(n_features, 2)),
        W_t=rng.uniform(-r_w, r_w, size=n_features),
        b=rng.uniform(-r_b, r_b, size=n_features),
        domain=(
            (float(domain[0][0]), float(domain[0][1])),
            (float(domain[1][0]), float(domain[1][1])),
        ),
        T=float(T),
    )


def make_pielm_basis_3d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
    r_w_space: float | None = None,
    r_w_time: float | None = None,
) -> PIELMBasis3D:
    """Sample a global 3D PIELM basis."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    space_range = float(r_w if r_w_space is None else r_w_space)
    time_range = float(r_w if r_w_time is None else r_w_time)
    return PIELMBasis3D(
        W_space=rng.uniform(-space_range, space_range, size=(n_features, 3)),
        W_t=rng.uniform(-time_range, time_range, size=n_features),
        b=rng.uniform(-r_b, r_b, size=n_features),
        domain=(
            (float(domain[0][0]), float(domain[0][1])),
            (float(domain[1][0]), float(domain[1][1])),
            (float(domain[2][0]), float(domain[2][1])),
        ),
        T=float(T),
    )
