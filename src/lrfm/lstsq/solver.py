# 中文注释：实现带正则化和列缩放的线性最小二乘求解器。
"""QR/SVD least-squares wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class LeastSquaresResult:
    """Least-squares solution and diagnostics."""

    coefficients: np.ndarray
    residual_norm: float
    rank: int
    singular_values: np.ndarray
    condition_number: float
    driver: str


def solve_lstsq(
    A: ArrayLike,
    b: ArrayLike,
    *,
    driver: str = "gelsd",
    ridge_lambda: float = 0.0,
    cond_threshold: float = 1.0e-12,
) -> LeastSquaresResult:
    """Solve ``min ||A c - b||_2`` with SciPy LAPACK wrappers."""
    matrix = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("A must be a matrix")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("b must have shape (A.shape[0],)")
    if ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be nonnegative")
    if ridge_lambda > 0.0:
        reg = np.sqrt(ridge_lambda) * np.eye(matrix.shape[1])
        matrix = np.vstack([matrix, reg])
        rhs = np.concatenate([rhs, np.zeros(matrix.shape[1])])

    coeffs, _residues, rank, singular_values = scipy.linalg.lstsq(
        matrix,
        rhs,
        lapack_driver=driver,
        cond=cond_threshold,
    )
    residual_norm = float(np.linalg.norm(matrix @ coeffs - rhs))
    if singular_values is None or singular_values.size == 0:
        condition_number = float("inf")
        singular_values = np.asarray([], dtype=float)
    elif singular_values[-1] == 0.0:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])
    return LeastSquaresResult(
        coefficients=np.asarray(coeffs, dtype=float),
        residual_norm=residual_norm,
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
        condition_number=condition_number,
        driver=driver,
    )
