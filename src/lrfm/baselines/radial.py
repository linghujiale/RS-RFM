# 中文注释：实现径向空间随机特征基线。
"""Geometry-aware radial feature bases for circular phase-field tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShrinkingCircleRadialBasis2D:
    """Random radial phase-field atoms following the shrinking-circle radius law.

    Each atom has the form

    ``q((r - (R(t) + delta_i)) / (epsilon * s_i))``,

    where ``q(z)=0.5*(1-tanh(z/2))``.  The shifts ``delta_i`` and scales
    ``s_i`` are sampled once and then kept fixed, like the other random feature
    families.  This basis is deliberately problem-specific and is used only as
    an Exp13 geometry-aware enrichment.
    """

    pde: object
    offsets: np.ndarray
    width_scales: np.ndarray
    name: str = "shrinking_circle_radial"

    @property
    def n_features(self) -> int:
        return int(self.offsets.shape[0])

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.shape != (2,):
            raise ValueError("x must have shape (2,)")
        cx, cy = self.pde.center
        dx = float(point[0] - cx)
        dy = float(point[1] - cy)
        r = float(np.sqrt(dx**2 + dy**2))
        r_safe = max(r, float(getattr(self.pde, "radius_floor", 1.0e-8)))
        radius = float(self.pde.radius_exact(float(t)))
        widths = self.width_scales * float(self.pde.epsilon)
        z = (r - (radius + self.offsets)) / widths
        value = 0.5 * (1.0 - np.tanh(0.5 * z))
        q_prime = -value * (1.0 - value)
        q_second = value * (1.0 - value) * (1.0 - 2.0 * value)
        radial_dir = np.asarray([dx / r_safe, dy / r_safe], dtype=float)
        return {
            "value": value,
            "dt": q_prime / (radius * widths),
            "grad": (q_prime / widths)[:, None] * radial_dir[None, :],
            "lap": q_second / widths**2 + q_prime / (widths * r_safe),
        }


def make_shrinking_circle_radial_basis_2d(
    rng: np.random.Generator,
    *,
    pde,
    n_features: int,
    offset_scale: float = 2.0,
    width_min: float = 0.5,
    width_max: float = 2.0,
) -> ShrinkingCircleRadialBasis2D:
    """Sample random radial atoms for the shrinking-circle phase field."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if width_min <= 0.0 or width_max < width_min:
        raise ValueError("width_min and width_max must be positive with width_max >= width_min")
    offsets = rng.normal(loc=0.0, scale=float(offset_scale) * float(pde.epsilon), size=n_features)
    width_scales = np.exp(rng.uniform(np.log(width_min), np.log(width_max), size=n_features))
    return ShrinkingCircleRadialBasis2D(pde=pde, offsets=offsets, width_scales=width_scales)


@dataclass(frozen=True)
class GrayScottSpotGaussianBasis3D:
    """Deterministic Gaussian atoms for the localized 3D Gray--Scott spot.

    The atoms are centered at the spherical seed and a few small axis offsets,
    with several widths and exponential time rates.  The first atom is an exact
    constant, so the near-steady background field does not have to be assembled
    from random features.
    """

    centers: np.ndarray
    widths: np.ndarray
    rates: np.ndarray
    include_constant: bool = True
    name: str = "grayscott_spot_gaussian"

    @property
    def n_features(self) -> int:
        return int(self.widths.shape[0] + (1 if self.include_constant else 0))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.shape != (3,):
            raise ValueError("x must have shape (3,)")
        delta = point[None, :] - self.centers
        r2 = np.sum(delta * delta, axis=1)
        inv_width2 = 1.0 / (self.widths * self.widths)
        value = np.exp(-0.5 * r2 * inv_width2) * np.exp(self.rates * float(t))
        dt = self.rates * value
        grad = -(delta * inv_width2[:, None]) * value[:, None]
        lap = (r2 * inv_width2 * inv_width2 - 3.0 * inv_width2) * value
        if self.include_constant:
            value = np.concatenate([[1.0], value])
            dt = np.concatenate([[0.0], dt])
            grad = np.vstack([np.zeros(3), grad])
            lap = np.concatenate([[0.0], lap])
        return {"value": value, "dt": dt, "grad": grad, "lap": lap}


def make_grayscott_spot_gaussian_basis_3d(
    *,
    pde,
    n_features: int,
    offset_scale: float = 0.35,
    width_scales: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00, 1.20, 1.45, 1.75),
    rates: tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20),
) -> GrayScottSpotGaussianBasis3D:
    """Build deterministic spot atoms for the 3D Gray--Scott benchmark."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    center = np.asarray(pde.spot_center, dtype=float)
    directions = [np.zeros(3)]
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        directions.extend([unit, -unit])

    n_gaussian = n_features - 1
    if n_gaussian <= 0:
        return GrayScottSpotGaussianBasis3D(
            centers=np.empty((0, 3), dtype=float),
            widths=np.empty(0, dtype=float),
            rates=np.empty(0, dtype=float),
            include_constant=True,
        )

    width_scale_values = np.asarray(width_scales, dtype=float)
    rate_values = np.asarray(rates, dtype=float)
    if np.any(width_scale_values <= 0.0):
        raise ValueError("width scales must be positive")

    centers: list[np.ndarray] = []
    widths: list[float] = []
    sampled_rates: list[float] = []
    for direction in directions:
        atom_center = center + float(offset_scale) * float(pde.spot_sigma) * direction
        for width_scale in width_scale_values:
            for rate in rate_values:
                centers.append(atom_center)
                widths.append(float(pde.spot_sigma) * float(width_scale))
                sampled_rates.append(float(rate))
                if len(widths) >= n_gaussian:
                    break
            if len(widths) >= n_gaussian:
                break
        if len(widths) >= n_gaussian:
            break

    return GrayScottSpotGaussianBasis3D(
        centers=np.asarray(centers, dtype=float),
        widths=np.asarray(widths, dtype=float),
        rates=np.asarray(sampled_rates, dtype=float),
        include_constant=True,
    )
