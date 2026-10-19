# 中文注释：实现复值 PDE 的实块 Picard 迭代。
"""Picard iteration for complex-valued 1D PDEs with real feature bases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lrfm.assembly.matrix import CollocationPoints1D
from lrfm.assembly.weighting import normalize_weight_keys
from lrfm.baselines.common import FeatureBasis1D
from lrfm.lstsq.solver import LeastSquaresResult, solve_lstsq


@dataclass(frozen=True)
class FittedComplexBasisSolution1D:
    """Complex-valued fitted solution backed by one real 1D feature basis."""

    pde: object
    basis: FeatureBasis1D
    coefficients: np.ndarray

    @property
    def n_real_features(self) -> int:
        return int(self.coefficients.shape[0] // 2)

    def _split(self) -> tuple[np.ndarray, np.ndarray]:
        n_features = self.n_real_features
        return self.coefficients[:n_features], self.coefficients[n_features:]

    def __call__(self, x: float | np.ndarray, t: float | np.ndarray) -> np.ndarray:
        x_arr, t_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
        out = np.empty_like(x_arr, dtype=np.complex128)
        coeff_re, coeff_im = self._split()
        for idx in np.ndindex(x_arr.shape):
            value = self.basis.terms(float(x_arr[idx]), float(t_arr[idx]))["value"]
            out[idx] = value @ coeff_re + 1j * (value @ coeff_im)
        return out

    def derivatives(self, x: float, t: float) -> dict[str, complex]:
        terms = self.basis.terms(float(x), float(t))
        coeff_re, coeff_im = self._split()
        return {
            "value": complex(terms["value"] @ coeff_re, terms["value"] @ coeff_im),
            "dt": complex(terms["dt"] @ coeff_re, terms["dt"] @ coeff_im),
            "dx": complex(terms["dx"] @ coeff_re, terms["dx"] @ coeff_im),
            "dxx": complex(terms["dxx"] @ coeff_re, terms["dxx"] @ coeff_im),
        }


@dataclass(frozen=True)
class ComplexPicardResult1D:
    """Result of a Picard-linearized complex LS solve."""

    solution: FittedComplexBasisSolution1D
    coefficients: np.ndarray
    iterations: int
    converged: bool
    relative_updates: list[float]
    residual_norms: list[float]
    last_lstsq: LeastSquaresResult


def _initial_complex_guess(pde):
    def guess(x: float | np.ndarray, t: float | np.ndarray):
        del t
        return pde.initial_condition(x)

    return guess


def assemble_complex_picard_system_1d(
    pde,
    basis: FeatureBasis1D,
    collocation: CollocationPoints1D,
    *,
    psi_prev,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble one real-valued LS system for a complex Picard step."""
    row_weights = normalize_weight_keys(weights)
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for x, t in zip(collocation.x_int, collocation.t_int, strict=True):
        terms = basis.terms(float(x), float(t))
        row_re, row_im = pde.picard_complex_feature_rows(
            terms["value"],
            terms["dt"],
            terms["dx"],
            terms["dxx"],
            float(x),
            float(t),
            complex(psi_prev(float(x), float(t))),
        )
        rows.append(row_weights["PDE"] * row_re)
        rhs.append(0.0)
        rows.append(row_weights["PDE"] * row_im)
        rhs.append(0.0)

    n_features = basis.n_features
    zeros = np.zeros(n_features)
    for x in collocation.x_ic:
        value = basis.terms(float(x), 0.0)["value"]
        initial_value = complex(pde.initial_condition(float(x)))
        rows.append(row_weights["IC"] * np.concatenate([value, zeros]))
        rhs.append(row_weights["IC"] * initial_value.real)
        rows.append(row_weights["IC"] * np.concatenate([zeros, value]))
        rhs.append(row_weights["IC"] * initial_value.imag)

    lo, hi = map(float, pde.domain)
    if getattr(pde, "boundary", None) == "periodic":
        for t in collocation.t_bc:
            left = basis.terms(lo, float(t))
            right = basis.terms(hi, float(t))
            for key in ("value", "dx"):
                jump = left[key] - right[key]
                rows.append(row_weights["BC"] * np.concatenate([jump, zeros]))
                rhs.append(0.0)
                rows.append(row_weights["BC"] * np.concatenate([zeros, jump]))
                rhs.append(0.0)
    else:
        for t in collocation.t_bc:
            for boundary_x in (lo, hi):
                value = basis.terms(boundary_x, float(t))["value"]
                target = complex(pde.boundary_condition(boundary_x, float(t)))
                rows.append(row_weights["BC"] * np.concatenate([value, zeros]))
                rhs.append(row_weights["BC"] * target.real)
                rows.append(row_weights["BC"] * np.concatenate([zeros, value]))
                rhs.append(row_weights["BC"] * target.imag)

    return np.vstack(rows), np.asarray(rhs, dtype=float)


def _relative_complex_update(new_solution, old_solution, pde, *, nx: int, nt: int) -> float:
    x = np.linspace(pde.domain[0], pde.domain[1], nx)
    t = np.linspace(0.0, pde.T, nt)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    new = new_solution(xx, tt)
    old = old_solution(xx, tt)
    return float(np.linalg.norm((new - old).ravel()) / (np.linalg.norm(new.ravel()) + 1.0e-14))


def solve_complex_picard_1d(
    pde,
    basis: FeatureBasis1D,
    collocation: CollocationPoints1D,
    *,
    weights: dict[str, float] | None = None,
    max_iters: int = 5,
    tol: float = 1.0e-8,
    monitor_nx: int = 64,
    monitor_nt: int = 16,
    driver: str = "gelsd",
    cond_threshold: float = 1.0e-12,
    ridge_lambda: float = 0.0,
) -> ComplexPicardResult1D:
    """Run Picard iteration for a complex-valued 1D PDE."""
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    psi_prev = _initial_complex_guess(pde)
    relative_updates: list[float] = []
    residual_norms: list[float] = []
    solution = None
    last_lstsq = None
    converged = False

    for iteration in range(1, max_iters + 1):
        A, b = assemble_complex_picard_system_1d(
            pde,
            basis,
            collocation,
            psi_prev=psi_prev,
            weights=weights,
        )
        last_lstsq = solve_lstsq(
            A,
            b,
            driver=driver,
            cond_threshold=cond_threshold,
            ridge_lambda=ridge_lambda,
        )
        solution = FittedComplexBasisSolution1D(pde, basis, last_lstsq.coefficients)
        residual_norms.append(last_lstsq.residual_norm)
        update = _relative_complex_update(solution, psi_prev, pde, nx=monitor_nx, nt=monitor_nt)
        relative_updates.append(update)
        if update < tol:
            converged = True
            break
        psi_prev = solution

    if solution is None or last_lstsq is None:
        raise RuntimeError("Picard iteration did not run")
    return ComplexPicardResult1D(
        solution=solution,
        coefficients=last_lstsq.coefficients,
        iterations=iteration,
        converged=converged,
        relative_updates=relative_updates,
        residual_norms=residual_norms,
        last_lstsq=last_lstsq,
    )
