# 中文注释：实现指数时间特征随机特征方法。
"""Non-liquid exponential-time random feature baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lrfm.baselines.common import scale_space_nd


def _tanh_profile_terms(W: np.ndarray, b: np.ndarray, x_scaled: np.ndarray, dx_scaled: np.ndarray):
    s = W @ x_scaled + b
    value = np.tanh(s)
    sech2 = 1.0 - value**2
    grad_s = W * dx_scaled[None, :]
    grad = sech2[:, None] * grad_s
    lap = -2.0 * value * sech2 * np.sum(grad_s**2, axis=1)
    return value, grad, lap


@dataclass(frozen=True)
class ExpRFMBasis3D:
    """Global 3D Exp-RFM basis with separable exponential time response.

    Each feature has the form ``tanh(w_i . z(x) + b_i) exp(-t/tau_i)``.
    This gives exponential temporal atoms without the liquid spatial coupling
    used by L-RFM.
    """

    W: np.ndarray
    b: np.ndarray
    tau: np.ndarray
    domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    name: str = "exp_rfm"

    @property
    def n_features(self) -> int:
        return int(self.tau.shape[0])

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        x_scaled, dx_scaled = scale_space_nd(self.domain, x)
        space_value, space_grad, space_lap = _tanh_profile_terms(self.W, self.b, x_scaled, dx_scaled)
        decay = np.exp(-float(t) / self.tau)
        return {
            "value": space_value * decay,
            "dt": -(space_value / self.tau) * decay,
            "grad": space_grad * decay[:, None],
            "lap": space_lap * decay,
        }


def make_exp_rfm_basis_3d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    tau_min: float = 0.003,
    tau_max: float = 0.60,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> ExpRFMBasis3D:
    """Sample a global 3D Exp-RFM basis."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if tau_min <= 0.0 or tau_max <= tau_min:
        raise ValueError("tau_min and tau_max must satisfy 0 < tau_min < tau_max")
    bounds = np.asarray(domain, dtype=float)
    if bounds.shape != (3, 2):
        raise ValueError("3D domain must have shape (3, 2)")
    tau = np.exp(rng.uniform(np.log(tau_min), np.log(tau_max), size=n_features))
    return ExpRFMBasis3D(
        W=rng.uniform(-r_w, r_w, size=(n_features, 3)),
        b=rng.uniform(-r_b, r_b, size=n_features),
        tau=tau,
        domain=(
            (float(domain[0][0]), float(domain[0][1])),
            (float(domain[1][0]), float(domain[1][1])),
            (float(domain[2][0]), float(domain[2][1])),
        ),
    )
