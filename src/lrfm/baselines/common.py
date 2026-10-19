# 中文注释：提供各随机特征对比方法共用的数据结构和辅助函数。
"""Common feature-basis assembly for baseline methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lrfm.assembly.matrix import CollocationPoints1D
from lrfm.assembly.weighting import normalize_weight_keys


class FeatureBasis1D(Protocol):
    """Protocol for 1D feature bases used by LS baselines."""

    name: str

    @property
    def n_features(self) -> int:
        """Number of scalar basis functions."""

    def terms(self, x: float, t: float) -> dict[str, np.ndarray]:
        """Return value, dt, dx, dxx, and optional dxxx arrays at one point."""


class FeatureBasis2D(Protocol):
    """Protocol for 2D feature bases used by LS and Picard baselines."""

    name: str

    @property
    def n_features(self) -> int:
        """Number of scalar basis functions."""

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        """Return value, dt, grad, and lap arrays at one 2D point."""


class FeatureBasis3D(Protocol):
    """Protocol for 3D feature bases used by LS and Picard baselines."""

    name: str

    @property
    def n_features(self) -> int:
        """Number of scalar basis functions."""

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        """Return value, dt, grad, and lap arrays at one 3D point."""


@dataclass(frozen=True)
class FittedBasisSolution1D:
    """Fitted solution backed by any 1D feature basis."""

    pde: object
    basis: FeatureBasis1D
    coefficients: np.ndarray

    def __call__(self, x: float | np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Evaluate the fitted baseline at broadcastable ``x`` and ``t``."""
        x_arr, t_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
        out = np.empty_like(x_arr, dtype=float)
        for idx in np.ndindex(x_arr.shape):
            out[idx] = self.basis.terms(float(x_arr[idx]), float(t_arr[idx]))["value"] @ self.coefficients
        return out

    def derivatives(self, x: float, t: float) -> dict[str, float]:
        """Evaluate value and derivatives at one point."""
        terms = self.basis.terms(float(x), float(t))
        return {
            "value": float(terms["value"] @ self.coefficients),
            "dt": float(terms["dt"] @ self.coefficients),
            "dx": float(terms["dx"] @ self.coefficients),
            "dxx": float(terms["dxx"] @ self.coefficients),
            "dxxx": float(terms.get("dxxx", np.zeros_like(terms["value"])) @ self.coefficients),
        }


@dataclass(frozen=True)
class FittedBasisSolution2D:
    """Fitted scalar solution backed by any 2D feature basis."""

    pde: object
    basis: FeatureBasis2D
    coefficients: np.ndarray

    def __call__(self, x: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Evaluate at points whose last axis is ``(x, y)``."""
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape[-1:] != (2,):
            raise ValueError("x must have last dimension 2")
        t_arr = np.broadcast_to(np.asarray(t, dtype=float), x_arr.shape[:-1])
        out = np.empty(x_arr.shape[:-1], dtype=float)
        flat_x = x_arr.reshape((-1, 2))
        flat_t = t_arr.reshape(-1)
        flat_out = out.reshape(-1)
        for idx, (point, time) in enumerate(zip(flat_x, flat_t, strict=True)):
            flat_out[idx] = self.basis.terms(point, float(time))["value"] @ self.coefficients
        return out

    def derivatives(self, x: np.ndarray, t: float) -> dict[str, float | np.ndarray]:
        """Evaluate value and derivatives at one 2D point."""
        terms = self.basis.terms(np.asarray(x, dtype=float), float(t))
        return {
            "value": float(terms["value"] @ self.coefficients),
            "dt": float(terms["dt"] @ self.coefficients),
            "grad": terms["grad"].T @ self.coefficients,
            "lap": float(terms["lap"] @ self.coefficients),
        }


@dataclass(frozen=True)
class FittedBasisSolution3D:
    """Fitted scalar solution backed by any 3D feature basis."""

    pde: object
    basis: FeatureBasis3D
    coefficients: np.ndarray

    def __call__(self, x: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Evaluate at points whose last axis is ``(x, y, z)``."""
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape[-1:] != (3,):
            raise ValueError("x must have last dimension 3")
        t_arr = np.broadcast_to(np.asarray(t, dtype=float), x_arr.shape[:-1])
        out = np.empty(x_arr.shape[:-1], dtype=float)
        flat_x = x_arr.reshape((-1, 3))
        flat_t = t_arr.reshape(-1)
        flat_out = out.reshape(-1)
        for idx, (point, time) in enumerate(zip(flat_x, flat_t, strict=True)):
            flat_out[idx] = self.basis.terms(point, float(time))["value"] @ self.coefficients
        return out

    def derivatives(self, x: np.ndarray, t: float) -> dict[str, float | np.ndarray]:
        """Evaluate value and derivatives at one 3D point."""
        terms = self.basis.terms(np.asarray(x, dtype=float), float(t))
        return {
            "value": float(terms["value"] @ self.coefficients),
            "dt": float(terms["dt"] @ self.coefficients),
            "grad": terms["grad"].T @ self.coefficients,
            "lap": float(terms["lap"] @ self.coefficients),
        }


@dataclass(frozen=True)
class CompositeFeatureBasis2D:
    """Concatenate several scalar 2D feature bases into one basis."""

    bases: tuple[FeatureBasis2D, ...]
    name: str = "composite_2d"

    @property
    def n_features(self) -> int:
        return int(sum(basis.n_features for basis in self.bases))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        pieces = [basis.terms(x, t) for basis in self.bases]
        return {
            "value": np.concatenate([piece["value"] for piece in pieces]),
            "dt": np.concatenate([piece["dt"] for piece in pieces]),
            "grad": np.vstack([piece["grad"] for piece in pieces]),
            "lap": np.concatenate([piece["lap"] for piece in pieces]),
        }


@dataclass(frozen=True)
class CompositeFeatureBasis3D:
    """Concatenate several scalar 3D feature bases into one basis."""

    bases: tuple[FeatureBasis3D, ...]
    name: str = "composite_3d"

    @property
    def n_features(self) -> int:
        return int(sum(basis.n_features for basis in self.bases))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        pieces = [basis.terms(x, t) for basis in self.bases]
        return {
            "value": np.concatenate([piece["value"] for piece in pieces]),
            "dt": np.concatenate([piece["dt"] for piece in pieces]),
            "grad": np.vstack([piece["grad"] for piece in pieces]),
            "lap": np.concatenate([piece["lap"] for piece in pieces]),
        }

    def terms_value_dt_lap(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        pieces = [getattr(basis, "terms_value_dt_lap", basis.terms)(x, t) for basis in self.bases]
        return {
            "value": np.concatenate([piece["value"] for piece in pieces]),
            "dt": np.concatenate([piece["dt"] for piece in pieces]),
            "lap": np.concatenate([piece["lap"] for piece in pieces]),
        }


def assemble_feature_system_1d(
    pde,
    basis: FeatureBasis1D,
    collocation: CollocationPoints1D,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble a linear LS system for any 1D feature basis."""
    if not pde.is_linear():
        raise NotImplementedError("P2 baseline assembly handles linear PDE sanity checks")
    row_weights = normalize_weight_keys(weights)
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for x, t in zip(collocation.x_int, collocation.t_int, strict=True):
        terms = basis.terms(float(x), float(t))
        if int(getattr(pde, "spatial_derivative_order", 2)) >= 3:
            row = pde.feature_residual(
                terms["value"],
                terms["dt"],
                terms["dx"],
                terms["dxx"],
                terms["dxxx"],
                float(x),
                float(t),
            )
        else:
            row = pde.feature_residual(
                terms["value"],
                terms["dt"],
                terms["dx"],
                terms["dxx"],
                float(x),
                float(t),
            )
        rows.append(row_weights["PDE"] * row)
        rhs.append(row_weights["PDE"] * float(pde.source_term(float(x), float(t))))

    for x in collocation.x_ic:
        row = basis.terms(float(x), 0.0)["value"]
        rows.append(row_weights["IC"] * row)
        rhs.append(row_weights["IC"] * float(pde.initial_condition(float(x))))

    lo, hi = map(float, pde.domain)
    if getattr(pde, "boundary", None) == "periodic":
        for t in collocation.t_bc:
            left = basis.terms(lo, float(t))
            right = basis.terms(hi, float(t))
            rows.append(row_weights["BC"] * (left["value"] - right["value"]))
            rhs.append(0.0)
            rows.append(row_weights["BC"] * (left["dx"] - right["dx"]))
            rhs.append(0.0)
            if int(getattr(pde, "periodic_bc_order", 1)) >= 2:
                rows.append(row_weights["BC"] * (left["dxx"] - right["dxx"]))
                rhs.append(0.0)
    else:
        for t in collocation.t_bc:
            left = basis.terms(lo, float(t))["value"]
            right = basis.terms(hi, float(t))["value"]
            rows.append(row_weights["BC"] * left)
            rhs.append(row_weights["BC"] * float(pde.boundary_condition(lo, float(t))))
            rows.append(row_weights["BC"] * right)
            rhs.append(row_weights["BC"] * float(pde.boundary_condition(hi, float(t))))

    return np.vstack(rows), np.asarray(rhs, dtype=float)


def tanh_terms(s: np.ndarray, ds_dx: np.ndarray, ds_dt: np.ndarray) -> dict[str, np.ndarray]:
    """Return tanh feature value and derivatives through third order in x."""
    value = np.tanh(s)
    sech2 = 1.0 - value**2
    return {
        "value": value,
        "dt": sech2 * ds_dt,
        "dx": sech2 * ds_dx,
        "dxx": -2.0 * value * sech2 * ds_dx**2,
        "dxxx": -2.0 * sech2 * (1.0 - 3.0 * value**2) * ds_dx**3,
    }


def tanh_terms_2d(s: np.ndarray, grad_s: np.ndarray, ds_dt: np.ndarray) -> dict[str, np.ndarray]:
    """Return tanh feature value, time derivative, gradient, and Laplacian."""
    value = np.tanh(s)
    sech2 = 1.0 - value**2
    grad = sech2[:, None] * grad_s
    lap = -2.0 * value * sech2 * np.sum(grad_s**2, axis=1)
    return {
        "value": value,
        "dt": sech2 * ds_dt,
        "grad": grad,
        "lap": lap,
    }


def tanh_terms_nd(s: np.ndarray, grad_s: np.ndarray, ds_dt: np.ndarray) -> dict[str, np.ndarray]:
    """Return tanh feature terms for an arbitrary spatial dimension."""
    return tanh_terms_2d(s, grad_s, ds_dt)


def scale_space_1d(domain: tuple[float, float], x: float) -> tuple[float, float]:
    """Return scaled coordinate in [-1,1] and its derivative."""
    lo, hi = map(float, domain)
    half_width = 0.5 * (hi - lo)
    midpoint = 0.5 * (hi + lo)
    return (float(x) - midpoint) / half_width, 1.0 / half_width


def scale_space_2d(
    domain: tuple[tuple[float, float], tuple[float, float]],
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled 2D coordinate in ``[-1,1]^2`` and coordinate scales."""
    point = np.asarray(x, dtype=float).reshape(-1)
    if point.shape != (2,):
        raise ValueError("x must have shape (2,)")
    bounds = np.asarray(domain, dtype=float)
    if bounds.shape != (2, 2):
        raise ValueError("2D domain must have shape (2, 2)")
    half_width = 0.5 * (bounds[:, 1] - bounds[:, 0])
    midpoint = 0.5 * (bounds[:, 1] + bounds[:, 0])
    return (point - midpoint) / half_width, 1.0 / half_width


def scale_space_nd(domain, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled ND coordinate in ``[-1,1]^d`` and coordinate scales."""
    point = np.asarray(x, dtype=float).reshape(-1)
    bounds = np.asarray(domain, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] != point.shape[0]:
        raise ValueError("domain must have shape (d, 2) matching x")
    half_width = 0.5 * (bounds[:, 1] - bounds[:, 0])
    midpoint = 0.5 * (bounds[:, 1] + bounds[:, 0])
    return (point - midpoint) / half_width, 1.0 / half_width


def scale_time(T: float, t: float) -> tuple[float, float]:
    """Return scaled time and its derivative."""
    if T <= 0.0:
        raise ValueError("T must be positive")
    return float(t) / float(T), 1.0 / float(T)
