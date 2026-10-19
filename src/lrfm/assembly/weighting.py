# 中文注释：计算不同残差块的缩放权重，平衡方程、边界和初值约束。
"""Residual-row weighting helpers."""

from __future__ import annotations


def default_row_weights(n_int: int, n_ic: int, n_bc: int) -> dict[str, float]:
    """Return RFM-style row weights with IC/BC scaled to interior rows."""
    if n_int <= 0 or n_ic <= 0 or n_bc <= 0:
        raise ValueError("row counts must be positive")
    return {
        "PDE": 1.0,
        "IC": float(n_int) / float(n_ic),
        "BC": float(n_int) / float(n_bc),
    }


def normalize_weight_keys(weights: dict[str, float] | None) -> dict[str, float]:
    """Fill missing residual weights with one."""
    if weights is None:
        return {"PDE": 1.0, "IC": 1.0, "BC": 1.0}
    return {
        "PDE": float(weights.get("PDE", weights.get("pde", 1.0))),
        "IC": float(weights.get("IC", weights.get("ic", 1.0))),
        "BC": float(weights.get("BC", weights.get("bc", 1.0))),
    }
