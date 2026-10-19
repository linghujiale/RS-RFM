# 中文注释：组合松弛特征并计算时间、空间及混合导数。
"""Vectorized frozen relaxation-response reservoirs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from lrfm.ltc.neuron import ltc_feature_closed_form_1d, ltc_feature_closed_form_nd_order2


class LiquidReservoir:
    """Frozen reservoir of random relaxation-response features.

    The legacy class name is retained for compatibility with archived
    experiment configurations.
    """

    tau: np.ndarray
    W_in: np.ndarray
    b_in: np.ndarray
    A: np.ndarray
    W_h0: np.ndarray
    b_h0: np.ndarray
    x_center: np.ndarray
    radius: float
    basis_mode: str

    def __init__(
        self,
        *,
        tau: ArrayLike,
        W_in: ArrayLike,
        b_in: ArrayLike,
        A: ArrayLike,
        W_h0: ArrayLike,
        b_h0: ArrayLike,
        x_center: ArrayLike,
        radius: float,
        basis_mode: str = "l_rfm_full",
    ) -> None:
        self.tau = np.asarray(tau, dtype=float)
        self.W_in = np.asarray(W_in, dtype=float)
        self.b_in = np.asarray(b_in, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.W_h0 = np.asarray(W_h0, dtype=float)
        self.b_h0 = np.asarray(b_h0, dtype=float)
        self.x_center = np.asarray(x_center, dtype=float)
        self.radius = float(radius)
        self.basis_mode = basis_mode
        self._validate()

    @property
    def n_features(self) -> int:
        """Number of frozen LTC neurons."""
        return int(self.tau.shape[0])

    @property
    def dim(self) -> int:
        """Spatial dimension of this reservoir."""
        return int(self.W_in.shape[1])

    def _validate(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.W_in.ndim != 2:
            raise ValueError("W_in must have shape (M, d)")
        if self.W_h0.shape != self.W_in.shape:
            raise ValueError("W_h0 must match W_in shape")
        m_features = self.W_in.shape[0]
        for name, arr in {
            "tau": self.tau,
            "b_in": self.b_in,
            "A": self.A,
            "b_h0": self.b_h0,
        }.items():
            if arr.shape != (m_features,):
                raise ValueError(f"{name} must have shape ({m_features},)")
        if self.x_center.shape != (self.W_in.shape[1],):
            raise ValueError("x_center must have shape (d,)")
        if np.any(self.tau <= 0.0):
            raise ValueError("all tau values must be positive")

    def features(self, x: ArrayLike, t: float) -> np.ndarray:
        """Return ``phi_i(x,t)`` with shape ``(M,)`` at one point."""
        return self.features_with_derivs(x, t, order=0)["phi"]

    def features_with_derivs(self, x: ArrayLike, t: float, order: int = 2) -> dict[str, np.ndarray]:
        """Return feature values and derivatives through ``order``.

        The 1D hierarchy is available through third order. The ND hierarchy is
        available through second order, which covers Laplacian-based PDEs.
        """
        if self.dim == 1 and (order < 0 or order > 3):
            raise NotImplementedError("1D reservoirs implement derivative orders 0 through 3")
        if self.dim >= 2 and (order < 0 or order > 2):
            raise NotImplementedError("ND reservoirs implement derivative orders 0 through 2")

        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if self.dim >= 2:
            if x_arr.shape != (self.dim,):
                raise ValueError(f"x must have shape ({self.dim},) for ND reservoirs")
            if self.basis_mode == "l_rfm_static":
                base = ltc_feature_closed_form_nd_order2(
                    x_arr,
                    0.0,
                    tau=self.tau,
                    w_in=self.W_in,
                    b_in=self.b_in,
                    asymptote=self.A,
                    w_h0=self.W_h0,
                    b_h0=self.b_h0,
                    x_center=self.x_center,
                    radius=self.radius,
                )
                base["dphi_dt"] = np.zeros_like(base["phi"])
                return base
            return ltc_feature_closed_form_nd_order2(
                x_arr,
                float(t),
                tau=self.tau,
                w_in=self.W_in,
                b_in=self.b_in,
                asymptote=self.A,
                w_h0=self.W_h0,
                b_h0=self.b_h0,
                x_center=self.x_center,
                radius=self.radius,
            )

        if x_arr.shape != (1,):
            raise ValueError("x must be a scalar or shape-(1,) array for 1D reservoirs")

        if self.basis_mode == "l_rfm_static":
            base = ltc_feature_closed_form_1d(
                float(x_arr[0]),
                0.0,
                tau=self.tau,
                w_in=self.W_in[:, 0],
                b_in=self.b_in,
                asymptote=self.A,
                w_h0=self.W_h0[:, 0],
                b_h0=self.b_h0,
                x_center=float(self.x_center[0]),
                radius=self.radius,
            )
            base["dphi_dt"] = np.zeros_like(base["phi"])
            return base

        return ltc_feature_closed_form_1d(
            float(x_arr[0]),
            float(t),
            tau=self.tau,
            w_in=self.W_in[:, 0],
            b_in=self.b_in,
            asymptote=self.A,
            w_h0=self.W_h0[:, 0],
            b_h0=self.b_h0,
            x_center=float(self.x_center[0]),
            radius=self.radius,
        )
