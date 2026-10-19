# 中文注释：实现一至三维随机特征基、配点采样和 PDE 残差矩阵装配。
"""Matrix assembly for PDE, initial, and boundary residual rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lrfm.assembly.weighting import normalize_weight_keys


@dataclass(frozen=True)
class CollocationPoints1D:
    """Interior, initial, and boundary time samples for 1D IBVP assembly."""

    x_int: np.ndarray
    t_int: np.ndarray
    x_ic: np.ndarray
    t_bc: np.ndarray

    @property
    def n_int(self) -> int:
        return int(self.x_int.shape[0])

    @property
    def n_ic(self) -> int:
        return int(self.x_ic.shape[0])

    @property
    def n_bc(self) -> int:
        return int(self.t_bc.shape[0])


@dataclass(frozen=True)
class LRFMBasis1D:
    """Feature-basis wrapper for a 1D PoU relaxation-response collection."""

    pou: object
    reservoirs: Sequence
    name: str = "l_rfm_full"

    @property
    def n_features(self) -> int:
        return int(sum(reservoir.n_features for reservoir in self.reservoirs))

    def terms(self, x: float, t: float) -> dict[str, np.ndarray]:
        return _global_basis_terms_1d(self.pou, self.reservoirs, float(x), float(t))


@dataclass(frozen=True)
class CollocationPoints2D:
    """Interior, initial, and boundary samples for 2D IBVP assembly."""

    x_int: np.ndarray
    t_int: np.ndarray
    x_ic: np.ndarray
    x_bc: np.ndarray
    t_bc: np.ndarray

    @property
    def n_int(self) -> int:
        return int(self.x_int.shape[0])

    @property
    def n_ic(self) -> int:
        return int(self.x_ic.shape[0])

    @property
    def n_bc(self) -> int:
        return int(self.x_bc.shape[0])


@dataclass(frozen=True)
class LRFMBasis2D:
    """Feature-basis wrapper for a 2D PoU relaxation-response collection."""

    pou: object
    reservoirs: Sequence
    name: str = "l_rfm_full"

    @property
    def n_features(self) -> int:
        return int(sum(reservoir.n_features for reservoir in self.reservoirs))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        return _global_basis_terms_2d(self.pou, self.reservoirs, np.asarray(x, dtype=float), float(t))


@dataclass(frozen=True)
class CollocationPoints3D:
    """Interior, initial, and boundary samples for 3D IBVP assembly."""

    x_int: np.ndarray
    t_int: np.ndarray
    x_ic: np.ndarray
    x_bc: np.ndarray
    t_bc: np.ndarray

    @property
    def n_int(self) -> int:
        return int(self.x_int.shape[0])

    @property
    def n_ic(self) -> int:
        return int(self.x_ic.shape[0])

    @property
    def n_bc(self) -> int:
        return int(self.x_bc.shape[0])


@dataclass(frozen=True)
class LRFMBasis3D:
    """Feature-basis wrapper for a 3D PoU relaxation-response collection."""

    pou: object
    reservoirs: Sequence
    name: str = "l_rfm_full"

    @property
    def n_features(self) -> int:
        return int(sum(reservoir.n_features for reservoir in self.reservoirs))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        return _global_basis_terms_3d(self.pou, self.reservoirs, np.asarray(x, dtype=float), float(t))

    def terms_value_dt_lap(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        return _global_basis_terms_3d(
            self.pou,
            self.reservoirs,
            np.asarray(x, dtype=float),
            float(t),
            include_grad=False,
        )


def sample_collocation_1d(
    pde,
    *,
    n_int: int,
    n_ic: int,
    n_bc: int,
    seed: int = 42,
    include_boundary_endpoints: bool = False,
) -> CollocationPoints1D:
    """Sample uniform random 1D collocation points."""
    if n_int <= 0 or n_ic <= 0 or n_bc <= 0:
        raise ValueError("collocation counts must be positive")
    rng = np.random.default_rng(seed)
    lo, hi = map(float, pde.domain)
    x_int = rng.uniform(lo, hi, size=n_int)
    t_int = rng.uniform(0.0, float(pde.T), size=n_int)
    if include_boundary_endpoints:
        x_ic = np.linspace(lo, hi, n_ic)
    else:
        x_ic = rng.uniform(lo, hi, size=n_ic)
    t_bc = rng.uniform(0.0, float(pde.T), size=n_bc)
    return CollocationPoints1D(x_int=x_int, t_int=t_int, x_ic=x_ic, t_bc=t_bc)


def sample_collocation_2d(
    pde,
    *,
    n_int: int,
    n_ic: int,
    n_bc: int,
    seed: int = 42,
) -> CollocationPoints2D:
    """Sample uniform random 2D collocation points.

    ``n_bc`` is interpreted as the number of samples per side, so the returned
    boundary arrays contain ``4*n_bc`` rows.
    """
    if n_int <= 0 or n_ic <= 0 or n_bc <= 0:
        raise ValueError("collocation counts must be positive")
    rng = np.random.default_rng(seed)
    (xlo, xhi), (ylo, yhi) = pde.domain
    x_int = np.column_stack([rng.uniform(xlo, xhi, size=n_int), rng.uniform(ylo, yhi, size=n_int)])
    t_int = rng.uniform(0.0, float(pde.T), size=n_int)
    x_ic = np.column_stack([rng.uniform(xlo, xhi, size=n_ic), rng.uniform(ylo, yhi, size=n_ic)])

    y_pair = rng.uniform(0.0, 1.0, size=n_bc)
    x_pair = rng.uniform(0.0, 1.0, size=n_bc)
    x_bc = np.vstack(
        [
            np.column_stack([np.full(n_bc, xlo), ylo + (yhi - ylo) * y_pair]),
            np.column_stack([np.full(n_bc, xhi), ylo + (yhi - ylo) * y_pair]),
            np.column_stack([xlo + (xhi - xlo) * x_pair, np.full(n_bc, ylo)]),
            np.column_stack([xlo + (xhi - xlo) * x_pair, np.full(n_bc, yhi)]),
        ]
    )
    t_x_pair = rng.uniform(0.0, float(pde.T), size=n_bc)
    t_y_pair = rng.uniform(0.0, float(pde.T), size=n_bc)
    t_bc = np.concatenate([t_x_pair, t_x_pair, t_y_pair, t_y_pair])
    return CollocationPoints2D(x_int=x_int, t_int=t_int, x_ic=x_ic, x_bc=x_bc, t_bc=t_bc)


def sample_collocation_3d(
    pde,
    *,
    n_int: int,
    n_ic: int,
    n_bc: int,
    seed: int = 42,
) -> CollocationPoints3D:
    """Sample uniform random 3D collocation points.

    ``n_bc`` is interpreted as the number of paired samples per coordinate
    direction; the returned boundary arrays contain ``6*n_bc`` rows ordered as
    x-low, x-high, y-low, y-high, z-low, z-high.
    """
    if n_int <= 0 or n_ic <= 0 or n_bc <= 0:
        raise ValueError("collocation counts must be positive")
    rng = np.random.default_rng(seed)
    bounds = np.asarray(pde.domain, dtype=float)
    if bounds.shape != (3, 2):
        raise ValueError("3D domain must have shape (3, 2)")
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    x_int = rng.uniform(lo, hi, size=(n_int, 3))
    t_int = rng.uniform(0.0, float(pde.T), size=n_int)
    x_ic = rng.uniform(lo, hi, size=(n_ic, 3))

    yz = rng.uniform(lo[[1, 2]], hi[[1, 2]], size=(n_bc, 2))
    xz = rng.uniform(lo[[0, 2]], hi[[0, 2]], size=(n_bc, 2))
    xy = rng.uniform(lo[[0, 1]], hi[[0, 1]], size=(n_bc, 2))
    x_bc = np.vstack(
        [
            np.column_stack([np.full(n_bc, lo[0]), yz]),
            np.column_stack([np.full(n_bc, hi[0]), yz]),
            np.column_stack([xz[:, 0], np.full(n_bc, lo[1]), xz[:, 1]]),
            np.column_stack([xz[:, 0], np.full(n_bc, hi[1]), xz[:, 1]]),
            np.column_stack([xy, np.full(n_bc, lo[2])]),
            np.column_stack([xy, np.full(n_bc, hi[2])]),
        ]
    )
    t_x = rng.uniform(0.0, float(pde.T), size=n_bc)
    t_y = rng.uniform(0.0, float(pde.T), size=n_bc)
    t_z = rng.uniform(0.0, float(pde.T), size=n_bc)
    t_bc = np.concatenate([t_x, t_x, t_y, t_y, t_z, t_z])
    return CollocationPoints3D(x_int=x_int, t_int=t_int, x_ic=x_ic, x_bc=x_bc, t_bc=t_bc)


def _global_basis_terms_1d(pou, reservoirs: Sequence, x: float, t: float) -> dict[str, np.ndarray]:
    """Evaluate global basis values and derivatives at one point."""
    weights = pou.evaluate(float(x))
    values: list[np.ndarray] = []
    dt_values: list[np.ndarray] = []
    dx_values: list[np.ndarray] = []
    dxx_values: list[np.ndarray] = []
    dxxx_values: list[np.ndarray] = []

    for patch_idx, reservoir in enumerate(reservoirs):
        local = reservoir.features_with_derivs(np.asarray([x]), t, order=3)
        phi = local["phi"]
        phi_t = local["dphi_dt"]
        phi_x = local["grad_x"][:, 0]
        phi_xx = local["lap_x"]
        phi_xxx = local["d3phi_dx3"]

        psi = weights.psi[patch_idx]
        psi_x = weights.dpsi_dx[patch_idx]
        psi_xx = weights.d2psi_dx2[patch_idx]
        psi_xxx = weights.d3psi_dx3[patch_idx]
        values.append(psi * phi)
        dt_values.append(psi * phi_t)
        dx_values.append(psi_x * phi + psi * phi_x)
        dxx_values.append(psi_xx * phi + 2.0 * psi_x * phi_x + psi * phi_xx)
        dxxx_values.append(
            psi_xxx * phi
            + 3.0 * psi_xx * phi_x
            + 3.0 * psi_x * phi_xx
            + psi * phi_xxx
        )

    return {
        "value": np.concatenate(values),
        "dt": np.concatenate(dt_values),
        "dx": np.concatenate(dx_values),
        "dxx": np.concatenate(dxx_values),
        "dxxx": np.concatenate(dxxx_values),
    }


def evaluate_basis_1d(pou, reservoirs: Sequence, x: float, t: float) -> np.ndarray:
    """Evaluate all global basis functions at one 1D space-time point."""
    return _global_basis_terms_1d(pou, reservoirs, x, t)["value"]


def evaluate_basis_derivative_1d(pou, reservoirs: Sequence, x: float, t: float) -> np.ndarray:
    """Evaluate all global basis-function first derivatives in x."""
    return _global_basis_terms_1d(pou, reservoirs, x, t)["dx"]


def _global_basis_terms_2d(pou, reservoirs: Sequence, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
    """Evaluate global 2D basis values and derivatives at one point."""
    point = np.asarray(x, dtype=float).reshape(-1)
    if point.shape != (2,):
        raise ValueError("x must have shape (2,)")
    weights = pou.evaluate(point)
    values: list[np.ndarray] = []
    dt_values: list[np.ndarray] = []
    grad_values: list[np.ndarray] = []
    lap_values: list[np.ndarray] = []

    for patch_idx, reservoir in enumerate(reservoirs):
        psi = weights.psi[patch_idx]
        grad_psi = weights.grad_psi[patch_idx]
        lap_psi = weights.lap_psi[patch_idx]
        if psi == 0.0 and lap_psi == 0.0 and not np.any(grad_psi):
            n_features = reservoir.n_features
            values.append(np.zeros(n_features, dtype=float))
            dt_values.append(np.zeros(n_features, dtype=float))
            grad_values.append(np.zeros((n_features, 2), dtype=float))
            lap_values.append(np.zeros(n_features, dtype=float))
            continue

        local = reservoir.features_with_derivs(point, t, order=2)
        phi = local["phi"]
        phi_t = local["dphi_dt"]
        phi_grad = local["grad_x"]
        phi_lap = local["lap_x"]

        values.append(psi * phi)
        dt_values.append(psi * phi_t)
        grad_values.append(grad_psi[None, :] * phi[:, None] + psi * phi_grad)
        lap_values.append(lap_psi * phi + 2.0 * (phi_grad @ grad_psi) + psi * phi_lap)

    return {
        "value": np.concatenate(values),
        "dt": np.concatenate(dt_values),
        "grad": np.vstack(grad_values),
        "lap": np.concatenate(lap_values),
    }


def evaluate_basis_2d(pou, reservoirs: Sequence, x: np.ndarray, t: float) -> np.ndarray:
    """Evaluate all global 2D basis functions at one space-time point."""
    return _global_basis_terms_2d(pou, reservoirs, x, t)["value"]


def _global_basis_terms_3d(
    pou,
    reservoirs: Sequence,
    x: np.ndarray,
    t: float,
    *,
    include_grad: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate global 3D basis values and derivatives at one point."""
    point = np.asarray(x, dtype=float).reshape(-1)
    if point.shape != (3,):
        raise ValueError("x must have shape (3,)")
    weights = pou.evaluate(point)
    values: list[np.ndarray] = []
    dt_values: list[np.ndarray] = []
    grad_values: list[np.ndarray] = []
    lap_values: list[np.ndarray] = []

    for patch_idx, reservoir in enumerate(reservoirs):
        psi = weights.psi[patch_idx]
        grad_psi = weights.grad_psi[patch_idx]
        lap_psi = weights.lap_psi[patch_idx]
        if psi == 0.0 and lap_psi == 0.0 and not np.any(grad_psi):
            n_features = reservoir.n_features
            values.append(np.zeros(n_features, dtype=float))
            dt_values.append(np.zeros(n_features, dtype=float))
            if include_grad:
                grad_values.append(np.zeros((n_features, 3), dtype=float))
            lap_values.append(np.zeros(n_features, dtype=float))
            continue

        local = reservoir.features_with_derivs(point, t, order=2)
        phi = local["phi"]
        phi_t = local["dphi_dt"]
        phi_grad = local["grad_x"]
        phi_lap = local["lap_x"]

        values.append(psi * phi)
        dt_values.append(psi * phi_t)
        if include_grad:
            grad_values.append(grad_psi[None, :] * phi[:, None] + psi * phi_grad)
        lap_values.append(lap_psi * phi + 2.0 * (phi_grad @ grad_psi) + psi * phi_lap)

    out = {
        "value": np.concatenate(values),
        "dt": np.concatenate(dt_values),
        "lap": np.concatenate(lap_values),
    }
    if include_grad:
        out["grad"] = np.vstack(grad_values)
    return out


def evaluate_basis_3d(pou, reservoirs: Sequence, x: np.ndarray, t: float) -> np.ndarray:
    """Evaluate all global 3D basis functions at one space-time point."""
    return _global_basis_terms_3d(pou, reservoirs, x, t)["value"]


def assemble_system(
    pde,
    pou,
    reservoirs: Sequence,
    collocation: CollocationPoints1D,
    weights: dict[str, float] | None = None,
    u_prev=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``A, b`` for one linear 1D L-RFM least-squares solve."""
    del u_prev
    if not pde.is_linear():
        raise NotImplementedError("P1 assembly handles linear PDEs; Picard starts in P3")
    row_weights = normalize_weight_keys(weights)
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for x, t in zip(collocation.x_int, collocation.t_int, strict=True):
        terms = _global_basis_terms_1d(pou, reservoirs, float(x), float(t))
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
        row = evaluate_basis_1d(pou, reservoirs, float(x), 0.0)
        rows.append(row_weights["IC"] * row)
        rhs.append(row_weights["IC"] * float(pde.initial_condition(float(x))))

    lo, hi = map(float, pde.domain)
    if getattr(pde, "boundary", None) == "periodic":
        for t in collocation.t_bc:
            left = _global_basis_terms_1d(pou, reservoirs, lo, float(t))
            right = _global_basis_terms_1d(pou, reservoirs, hi, float(t))
            rows.append(row_weights["BC"] * (left["value"] - right["value"]))
            rhs.append(0.0)
            rows.append(row_weights["BC"] * (left["dx"] - right["dx"]))
            rhs.append(0.0)
            if int(getattr(pde, "periodic_bc_order", 1)) >= 2:
                rows.append(row_weights["BC"] * (left["dxx"] - right["dxx"]))
                rhs.append(0.0)
    else:
        for t in collocation.t_bc:
            left_row = evaluate_basis_1d(pou, reservoirs, lo, float(t))
            right_row = evaluate_basis_1d(pou, reservoirs, hi, float(t))
            rows.append(row_weights["BC"] * left_row)
            rhs.append(row_weights["BC"] * float(pde.boundary_condition(lo, float(t))))
            rows.append(row_weights["BC"] * right_row)
            rhs.append(row_weights["BC"] * float(pde.boundary_condition(hi, float(t))))

    return np.vstack(rows), np.asarray(rhs, dtype=float)


def assemble_system_2d(
    pde,
    pou,
    reservoirs: Sequence,
    collocation: CollocationPoints2D,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``A, b`` for one linear 2D L-RFM least-squares solve."""
    if not pde.is_linear():
        raise NotImplementedError("2D assembly currently handles linear PDEs")
    row_weights = normalize_weight_keys(weights)
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for x, t in zip(collocation.x_int, collocation.t_int, strict=True):
        terms = _global_basis_terms_2d(pou, reservoirs, x, float(t))
        row = pde.feature_residual(
            terms["value"],
            terms["dt"],
            terms["grad"],
            terms["lap"],
            x,
            float(t),
        )
        rows.append(row_weights["PDE"] * row)
        rhs.append(row_weights["PDE"] * float(pde.source_term(x, float(t))))

    for x in collocation.x_ic:
        row = evaluate_basis_2d(pou, reservoirs, x, 0.0)
        rows.append(row_weights["IC"] * row)
        rhs.append(row_weights["IC"] * float(pde.initial_condition(x)))

    for x, t in zip(collocation.x_bc, collocation.t_bc, strict=True):
        row = evaluate_basis_2d(pou, reservoirs, x, float(t))
        rows.append(row_weights["BC"] * row)
        rhs.append(row_weights["BC"] * float(pde.boundary_condition(x, float(t))))

    return np.vstack(rows), np.asarray(rhs, dtype=float)
