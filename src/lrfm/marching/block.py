# 中文注释：实现长时间问题的分块时间推进与块间状态传递。
"""Block time-marching implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lrfm.assembly.matrix import sample_collocation_1d
from lrfm.assembly.weighting import default_row_weights
from lrfm.nonlinear.picard import PicardResult1D, solve_picard_1d


@dataclass(frozen=True)
class BlockPDE1D:
    """Local-in-time PDE wrapper used by block marching."""

    base_pde: object
    T: float
    t_start: float
    initial_condition_fn: Callable

    @property
    def domain(self):
        return self.base_pde.domain

    @property
    def boundary(self):
        return getattr(self.base_pde, "boundary", None)

    @property
    def periodic_bc_order(self):
        return getattr(self.base_pde, "periodic_bc_order", 1)

    @property
    def spatial_derivative_order(self):
        return getattr(self.base_pde, "spatial_derivative_order", 2)

    def initial_condition(self, x):
        return self.initial_condition_fn(x)

    def boundary_condition(self, x, t):
        return self.base_pde.boundary_condition(x, self.t_start + np.asarray(t))

    def residual(self, u, du_dt, du_dx, d2u_dx2, x, t):
        return self.base_pde.residual(u, du_dt, du_dx, d2u_dx2, x, self.t_start + np.asarray(t))

    def feature_residual(self, value, du_dt, du_dx, d2u_dx2, *args):
        if self.spatial_derivative_order >= 3:
            d3u_dx3, x, t = args
            return self.base_pde.feature_residual(
                value,
                du_dt,
                du_dx,
                d2u_dx2,
                d3u_dx3,
                x,
                self.t_start + float(t),
            )
        x, t = args
        return self.base_pde.feature_residual(value, du_dt, du_dx, d2u_dx2, x, self.t_start + float(t))

    def picard_feature_residual(self, value, du_dt, du_dx, d2u_dx2, *args):
        if self.spatial_derivative_order >= 3:
            d3u_dx3, x, t, u_prev = args
            return self.base_pde.picard_feature_residual(
                value,
                du_dt,
                du_dx,
                d2u_dx2,
                d3u_dx3,
                x,
                self.t_start + float(t),
                u_prev,
            )
        x, t, u_prev = args
        return self.base_pde.picard_feature_residual(
            value,
            du_dt,
            du_dx,
            d2u_dx2,
            x,
            self.t_start + float(t),
            u_prev,
        )

    def source_term(self, x, t):
        return self.base_pde.source_term(x, self.t_start + float(t))

    def is_linear(self) -> bool:
        return self.base_pde.is_linear()


@dataclass(frozen=True)
class BlockSolveRecord1D:
    """Diagnostics for one solved time block."""

    index: int
    t_start: float
    t_end: float
    picard: PicardResult1D


@dataclass(frozen=True)
class BlockTimeSolution1D:
    """Piecewise-in-time solution assembled from local block solves."""

    base_pde: object
    block_starts: np.ndarray
    block_ends: np.ndarray
    local_solutions: tuple

    def __call__(self, x: float | np.ndarray, t: float | np.ndarray) -> np.ndarray:
        x_arr, t_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
        out = np.empty_like(x_arr, dtype=float)
        for idx in np.ndindex(x_arr.shape):
            time = float(np.clip(t_arr[idx], self.block_starts[0], self.block_ends[-1]))
            block_idx = int(np.searchsorted(self.block_ends, time, side="left"))
            block_idx = min(block_idx, len(self.local_solutions) - 1)
            local_t = time - float(self.block_starts[block_idx])
            out[idx] = self.local_solutions[block_idx](float(x_arr[idx]), local_t)
        return out


@dataclass(frozen=True)
class BlockPicardResult1D:
    """Full block-marching result and diagnostics."""

    solution: BlockTimeSolution1D
    records: tuple[BlockSolveRecord1D, ...]

    @property
    def iterations(self) -> int:
        return int(sum(record.picard.iterations for record in self.records))

    @property
    def converged(self) -> bool:
        return all(record.picard.converged for record in self.records)

    @property
    def max_last_update(self) -> float:
        return float(max(record.picard.relative_updates[-1] for record in self.records))


def solve_block_picard_1d(
    base_pde,
    *,
    basis_factory: Callable,
    n_blocks: int,
    block_edges: np.ndarray | None = None,
    n_int: int,
    n_ic: int,
    n_bc: int,
    seed: int,
    max_iters: int = 6,
    tol: float = 1.0e-7,
    monitor_nx: int = 64,
    monitor_nt: int = 12,
    cond_threshold: float = 1.0e-12,
    ridge_lambda: float = 0.0,
    weight_multipliers: dict[str, float] | None = None,
) -> BlockPicardResult1D:
    """Solve a nonlinear 1D PDE by Picard iteration on consecutive time blocks."""
    if block_edges is None:
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        block_edges = np.linspace(0.0, float(base_pde.T), n_blocks + 1)
    else:
        block_edges = np.asarray(block_edges, dtype=float)
        if block_edges.ndim != 1 or block_edges.size < 2:
            raise ValueError("block_edges must be a one-dimensional array with at least two entries")
        if not np.isclose(block_edges[0], 0.0):
            raise ValueError("block_edges must start at 0")
        if not np.isclose(block_edges[-1], float(base_pde.T)):
            raise ValueError("block_edges must end at base_pde.T")
        if np.any(np.diff(block_edges) <= 0.0):
            raise ValueError("block_edges must be strictly increasing")
        n_blocks = block_edges.size - 1
    local_solutions = []
    records: list[BlockSolveRecord1D] = []

    def current_ic(x):
        return base_pde.initial_condition(x)

    for block_idx in range(n_blocks):
        t_start = float(block_edges[block_idx])
        t_end = float(block_edges[block_idx + 1])
        local_pde = BlockPDE1D(
            base_pde=base_pde,
            T=t_end - t_start,
            t_start=t_start,
            initial_condition_fn=current_ic,
        )
        rng = np.random.default_rng(seed + 7919 * (block_idx + 1))
        basis = basis_factory(local_pde, block_idx, rng)
        collocation = sample_collocation_1d(
            local_pde,
            n_int=n_int,
            n_ic=n_ic,
            n_bc=n_bc,
            seed=seed + 104729 * (block_idx + 1),
            include_boundary_endpoints=True,
        )
        weights = default_row_weights(collocation.n_int, collocation.n_ic, collocation.n_bc)
        if weight_multipliers is not None:
            weights = {
                key: float(value) * float(weight_multipliers.get(key, weight_multipliers.get(key.lower(), 1.0)))
                for key, value in weights.items()
            }
        picard = solve_picard_1d(
            local_pde,
            basis,
            collocation,
            weights=weights,
            max_iters=max_iters,
            tol=tol,
            monitor_nx=monitor_nx,
            monitor_nt=monitor_nt,
            cond_threshold=cond_threshold,
            ridge_lambda=ridge_lambda,
        )
        local_solution = picard.solution
        local_solutions.append(local_solution)
        records.append(
            BlockSolveRecord1D(
                index=block_idx,
                t_start=t_start,
                t_end=t_end,
                picard=picard,
            )
        )

        def current_ic(x, local_solution=local_solution, local_T=local_pde.T):
            return local_solution(x, local_T)

    return BlockPicardResult1D(
        solution=BlockTimeSolution1D(
            base_pde=base_pde,
            block_starts=block_edges[:-1],
            block_ends=block_edges[1:],
            local_solutions=tuple(local_solutions),
        ),
        records=tuple(records),
    )
