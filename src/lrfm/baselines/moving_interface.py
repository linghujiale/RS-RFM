# 中文注释：实现移动界面问题使用的对比基函数和装配逻辑。
"""Interface-aligned profile features for manufactured moving-front tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MovingWavyInterfaceProfileBasis3D:
    """Profile features aligned with the moving wavy-interface level set."""

    pde: object
    alpha: np.ndarray
    beta: np.ndarray
    name: str = "moving_wavy_interface_profile"

    @property
    def n_features(self) -> int:
        return int(self.alpha.shape[0])

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        data = self.pde.level_data(np.asarray(x, dtype=float), float(t))
        scale = self.alpha / (np.sqrt(2.0) * float(self.pde.epsilon))
        q = scale * float(data["D"]) + self.beta
        value = np.tanh(q)
        sech2 = 1.0 - value**2
        q_t = scale * float(data["D_t"])
        q_grad = scale[:, None] * np.asarray(data["grad_D"], dtype=float)[None, :]
        q_lap = scale * float(data["lap_D"])
        q_grad_norm2 = np.sum(q_grad**2, axis=1)
        return {
            "value": value,
            "dt": sech2 * q_t,
            "grad": sech2[:, None] * q_grad,
            "lap": sech2 * q_lap - 2.0 * value * sech2 * q_grad_norm2,
        }


def make_moving_wavy_interface_profile_basis_3d(
    rng: np.random.Generator,
    *,
    pde: object,
    n_features: int,
    alpha_min: float = 0.75,
    alpha_max: float = 1.35,
    beta_range: float = 0.75,
    include_exact: bool = True,
) -> MovingWavyInterfaceProfileBasis3D:
    """Create moving-interface profile atoms, including the exact profile by default."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if alpha_min <= 0.0 or alpha_max < alpha_min:
        raise ValueError("alpha range must satisfy 0 < alpha_min <= alpha_max")
    alpha = rng.uniform(float(alpha_min), float(alpha_max), size=n_features)
    beta = rng.uniform(-float(beta_range), float(beta_range), size=n_features)
    if include_exact:
        alpha[0] = 1.0
        beta[0] = 0.0
    return MovingWavyInterfaceProfileBasis3D(pde=pde, alpha=alpha, beta=beta)
