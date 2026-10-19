# 中文注释：实现联合时空及变量分离形式的 ST-RFM 对比方法。
"""ST-RFM baseline implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lrfm.baselines.common import (
    scale_space_1d,
    scale_space_2d,
    scale_space_nd,
    scale_time,
    tanh_terms,
    tanh_terms_2d,
)
from lrfm.pou.decomposition import PartitionOfUnity1D, PartitionOfUnity2D, PartitionOfUnity3D


def _localize(pou: PartitionOfUnity1D, local_terms: list[dict[str, np.ndarray]], x: float) -> dict[str, np.ndarray]:
    weights = pou.evaluate(float(x))
    values: list[np.ndarray] = []
    dts: list[np.ndarray] = []
    dxs: list[np.ndarray] = []
    dxxs: list[np.ndarray] = []
    dxxxs: list[np.ndarray] = []

    for patch_idx, terms in enumerate(local_terms):
        psi = weights.psi[patch_idx]
        psi_x = weights.dpsi_dx[patch_idx]
        psi_xx = weights.d2psi_dx2[patch_idx]
        psi_xxx = weights.d3psi_dx3[patch_idx]
        value = terms["value"]
        dx = terms["dx"]
        dxx = terms["dxx"]
        values.append(psi * value)
        dts.append(psi * terms["dt"])
        dxs.append(psi_x * value + psi * dx)
        dxxs.append(psi_xx * value + 2.0 * psi_x * dx + psi * dxx)
        dxxxs.append(psi_xxx * value + 3.0 * psi_xx * dx + 3.0 * psi_x * dxx + psi * terms["dxxx"])

    return {
        "value": np.concatenate(values),
        "dt": np.concatenate(dts),
        "dx": np.concatenate(dxs),
        "dxx": np.concatenate(dxxs),
        "dxxx": np.concatenate(dxxxs),
    }


def _localize_2d(pou: PartitionOfUnity2D, local_terms: list[dict[str, np.ndarray]], x: np.ndarray) -> dict[str, np.ndarray]:
    weights = pou.evaluate(np.asarray(x, dtype=float))
    values: list[np.ndarray] = []
    dts: list[np.ndarray] = []
    grads: list[np.ndarray] = []
    laps: list[np.ndarray] = []

    for patch_idx, terms in enumerate(local_terms):
        psi = weights.psi[patch_idx]
        grad_psi = weights.grad_psi[patch_idx]
        lap_psi = weights.lap_psi[patch_idx]
        value = terms["value"]
        grad = terms["grad"]
        values.append(psi * value)
        dts.append(psi * terms["dt"])
        grads.append(grad_psi[None, :] * value[:, None] + psi * grad)
        laps.append(lap_psi * value + 2.0 * (grad @ grad_psi) + psi * terms["lap"])

    return {
        "value": np.concatenate(values),
        "dt": np.concatenate(dts),
        "grad": np.vstack(grads),
        "lap": np.concatenate(laps),
    }


def _localize_3d(pou: PartitionOfUnity3D, local_terms: list[dict[str, np.ndarray]], x: np.ndarray) -> dict[str, np.ndarray]:
    weights = pou.evaluate(np.asarray(x, dtype=float))
    values: list[np.ndarray] = []
    dts: list[np.ndarray] = []
    grads: list[np.ndarray] = []
    laps: list[np.ndarray] = []

    for patch_idx, terms in enumerate(local_terms):
        psi = weights.psi[patch_idx]
        grad_psi = weights.grad_psi[patch_idx]
        lap_psi = weights.lap_psi[patch_idx]
        value = terms["value"]
        grad = terms["grad"]
        values.append(psi * value)
        dts.append(psi * terms["dt"])
        grads.append(grad_psi[None, :] * value[:, None] + psi * grad)
        laps.append(lap_psi * value + 2.0 * (grad @ grad_psi) + psi * terms["lap"])

    return {
        "value": np.concatenate(values),
        "dt": np.concatenate(dts),
        "grad": np.vstack(grads),
        "lap": np.concatenate(laps),
    }


@dataclass(frozen=True)
class SeparableRFMGlobalBasis1D:
    """Global separated-variable RFM basis ``tanh(w_x x+b_x)tanh(w_t t+b_t)``."""

    W_x: np.ndarray
    b_x: np.ndarray
    W_t: np.ndarray
    b_t: np.ndarray
    domain: tuple[float, float]
    T: float
    name: str = "sep_rfm_global"

    @property
    def n_features(self) -> int:
        return int(self.W_x.size)

    def terms(self, x: float, t: float) -> dict[str, np.ndarray]:
        x_scaled, dx_scaled = scale_space_1d(self.domain, x)
        t_scaled, dt_scaled = scale_time(self.T, t)
        spatial = np.tanh(self.W_x * x_scaled + self.b_x)
        temporal = np.tanh(self.W_t * t_scaled + self.b_t)
        spatial_sech2 = 1.0 - spatial**2
        temporal_sech2 = 1.0 - temporal**2
        ds_dx = self.W_x * dx_scaled
        return {
            "value": spatial * temporal,
            "dt": spatial * temporal_sech2 * self.W_t * dt_scaled,
            "dx": spatial_sech2 * ds_dx * temporal,
            "dxx": -2.0 * spatial * spatial_sech2 * ds_dx**2 * temporal,
            "dxxx": -2.0 * spatial_sech2 * (1.0 - 3.0 * spatial**2) * ds_dx**3 * temporal,
        }


@dataclass(frozen=True)
class SeparableRFMGlobalBasis2D:
    """Global two-dimensional separated-variable RFM basis."""

    W_space: np.ndarray
    b_space: np.ndarray
    W_t: np.ndarray
    b_t: np.ndarray
    domain: tuple[tuple[float, float], tuple[float, float]]
    T: float
    name: str = "sep_rfm_global"

    @property
    def n_features(self) -> int:
        return int(self.W_t.size)

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        x_scaled, dx_scaled = scale_space_2d(self.domain, x)
        t_scaled, dt_scaled = scale_time(self.T, t)
        spatial = np.tanh(self.W_space @ x_scaled + self.b_space)
        temporal = np.tanh(self.W_t * t_scaled + self.b_t)
        spatial_sech2 = 1.0 - spatial**2
        temporal_sech2 = 1.0 - temporal**2
        grad_s = self.W_space * dx_scaled[None, :]
        return {
            "value": spatial * temporal,
            "dt": spatial * temporal_sech2 * self.W_t * dt_scaled,
            "grad": spatial_sech2[:, None] * grad_s * temporal[:, None],
            "lap": -2.0 * spatial * spatial_sech2 * np.sum(grad_s**2, axis=1) * temporal,
        }


@dataclass(frozen=True)
class SeparableRFMGlobalBasis3D:
    """Global three-dimensional separated-variable RFM basis."""

    W_space: np.ndarray
    b_space: np.ndarray
    W_t: np.ndarray
    b_t: np.ndarray
    domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    T: float
    name: str = "sep_rfm_global"

    @property
    def n_features(self) -> int:
        return int(self.W_t.size)

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        x_scaled, dx_scaled = scale_space_nd(self.domain, x)
        if x_scaled.shape != (3,):
            raise ValueError("x must have shape (3,)")
        t_scaled, dt_scaled = scale_time(self.T, t)
        spatial = np.tanh(self.W_space @ x_scaled + self.b_space)
        temporal = np.tanh(self.W_t * t_scaled + self.b_t)
        spatial_sech2 = 1.0 - spatial**2
        temporal_sech2 = 1.0 - temporal**2
        grad_s = self.W_space * dx_scaled[None, :]
        return {
            "value": spatial * temporal,
            "dt": spatial * temporal_sech2 * self.W_t * dt_scaled,
            "grad": spatial_sech2[:, None] * grad_s * temporal[:, None],
            "lap": -2.0 * spatial * spatial_sech2 * np.sum(grad_s**2, axis=1) * temporal,
        }


@dataclass(frozen=True)
class STRFMSoVBasis1D:
    """Partitioned ST-RFM separated-variable basis.

    Local features are products ``tanh(w_x z + b_x) * tanh(w_t t/T + b_t)``.
    """

    W_x: np.ndarray
    b_x: np.ndarray
    W_t: np.ndarray
    b_t: np.ndarray
    pou: PartitionOfUnity1D
    T: float
    name: str = "strfm_sov"

    @property
    def n_features(self) -> int:
        return int(self.W_x.size)

    def terms(self, x: float, t: float) -> dict[str, np.ndarray]:
        t_scaled, dt_scaled = scale_time(self.T, t)
        locals_: list[dict[str, np.ndarray]] = []
        for patch_idx, center in enumerate(self.pou.centers):
            z = (float(x) - float(center)) / self.pou.radius
            sx = self.W_x[patch_idx] * z + self.b_x[patch_idx]
            st = self.W_t[patch_idx] * t_scaled + self.b_t[patch_idx]
            x_value = np.tanh(sx)
            t_value = np.tanh(st)
            x_sech2 = 1.0 - x_value**2
            t_sech2 = 1.0 - t_value**2
            ds_dx = self.W_x[patch_idx] / self.pou.radius
            x_dx = x_sech2 * ds_dx
            x_dxx = -2.0 * x_value * x_sech2 * ds_dx**2
            x_dxxx = -2.0 * x_sech2 * (1.0 - 3.0 * x_value**2) * ds_dx**3
            t_dt = t_sech2 * self.W_t[patch_idx] * dt_scaled
            locals_.append(
                {
                    "value": x_value * t_value,
                    "dt": x_value * t_dt,
                    "dx": x_dx * t_value,
                    "dxx": x_dxx * t_value,
                    "dxxx": x_dxxx * t_value,
                }
            )
        return _localize(self.pou, locals_, x)


@dataclass(frozen=True)
class STRFMSoVBasis2D:
    """Partitioned 2D ST-RFM separated-variable basis."""

    W_space: np.ndarray
    b_space: np.ndarray
    W_t: np.ndarray
    b_t: np.ndarray
    pou: PartitionOfUnity2D
    T: float
    name: str = "strfm_sov"

    @property
    def n_features(self) -> int:
        return int(np.prod(self.W_t.shape))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.shape != (2,):
            raise ValueError("x must have shape (2,)")
        t_scaled, dt_scaled = scale_time(self.T, t)
        locals_: list[dict[str, np.ndarray]] = []
        for patch_idx, center in enumerate(self.pou.centers):
            z = (point - center) / self.pou.radius
            s_space = self.W_space[patch_idx] @ z + self.b_space[patch_idx]
            s_time = self.W_t[patch_idx] * t_scaled + self.b_t[patch_idx]
            space_value = np.tanh(s_space)
            time_value = np.tanh(s_time)
            space_sech2 = 1.0 - space_value**2
            time_sech2 = 1.0 - time_value**2
            grad_s = self.W_space[patch_idx] / self.pou.radius
            space_grad = space_sech2[:, None] * grad_s
            space_lap = -2.0 * space_value * space_sech2 * np.sum(grad_s**2, axis=1)
            time_dt = time_sech2 * self.W_t[patch_idx] * dt_scaled
            locals_.append(
                {
                    "value": space_value * time_value,
                    "dt": space_value * time_dt,
                    "grad": space_grad * time_value[:, None],
                    "lap": space_lap * time_value,
                }
            )
        return _localize_2d(self.pou, locals_, point)


@dataclass(frozen=True)
class STRFMSoVBasis3D:
    """Partitioned 3D ST-RFM separated-variable basis."""

    W_space: np.ndarray
    b_space: np.ndarray
    W_t: np.ndarray
    b_t: np.ndarray
    pou: PartitionOfUnity3D
    T: float
    name: str = "strfm_sov"

    @property
    def n_features(self) -> int:
        return int(np.prod(self.W_t.shape))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.shape != (3,):
            raise ValueError("x must have shape (3,)")
        t_scaled, dt_scaled = scale_time(self.T, t)
        locals_: list[dict[str, np.ndarray]] = []
        for patch_idx, center in enumerate(self.pou.centers):
            z = (point - center) / self.pou.radius
            s_space = self.W_space[patch_idx] @ z + self.b_space[patch_idx]
            s_time = self.W_t[patch_idx] * t_scaled + self.b_t[patch_idx]
            space_value = np.tanh(s_space)
            time_value = np.tanh(s_time)
            space_sech2 = 1.0 - space_value**2
            time_sech2 = 1.0 - time_value**2
            grad_s = self.W_space[patch_idx] / self.pou.radius
            space_grad = space_sech2[:, None] * grad_s
            space_lap = -2.0 * space_value * space_sech2 * np.sum(grad_s**2, axis=1)
            time_dt = time_sech2 * self.W_t[patch_idx] * dt_scaled
            locals_.append(
                {
                    "value": space_value * time_value,
                    "dt": space_value * time_dt,
                    "grad": space_grad * time_value[:, None],
                    "lap": space_lap * time_value,
                }
            )
        return _localize_3d(self.pou, locals_, point)


@dataclass(frozen=True)
class STRFMSTCBasis1D:
    """Partitioned ST-RFM space-time coupled basis."""

    W_x: np.ndarray
    W_t: np.ndarray
    b: np.ndarray
    pou: PartitionOfUnity1D
    T: float
    name: str = "strfm_stc"

    @property
    def n_features(self) -> int:
        return int(self.W_x.size)

    def terms(self, x: float, t: float) -> dict[str, np.ndarray]:
        t_scaled, dt_scaled = scale_time(self.T, t)
        locals_: list[dict[str, np.ndarray]] = []
        for patch_idx, center in enumerate(self.pou.centers):
            z = (float(x) - float(center)) / self.pou.radius
            s = self.W_x[patch_idx] * z + self.W_t[patch_idx] * t_scaled + self.b[patch_idx]
            locals_.append(
                tanh_terms(
                    s,
                    self.W_x[patch_idx] / self.pou.radius,
                    self.W_t[patch_idx] * dt_scaled,
                )
            )
        return _localize(self.pou, locals_, x)


@dataclass(frozen=True)
class STRFMSTCBasis2D:
    """Partitioned 2D ST-RFM space-time coupled basis."""

    W_space: np.ndarray
    W_t: np.ndarray
    b: np.ndarray
    pou: PartitionOfUnity2D
    T: float
    name: str = "strfm_stc"

    @property
    def n_features(self) -> int:
        return int(np.prod(self.W_t.shape))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.shape != (2,):
            raise ValueError("x must have shape (2,)")
        t_scaled, dt_scaled = scale_time(self.T, t)
        locals_: list[dict[str, np.ndarray]] = []
        for patch_idx, center in enumerate(self.pou.centers):
            z = (point - center) / self.pou.radius
            s = self.W_space[patch_idx] @ z + self.W_t[patch_idx] * t_scaled + self.b[patch_idx]
            locals_.append(
                tanh_terms_2d(
                    s,
                    self.W_space[patch_idx] / self.pou.radius,
                    self.W_t[patch_idx] * dt_scaled,
                )
            )
        return _localize_2d(self.pou, locals_, point)


@dataclass(frozen=True)
class STRFMSTCBasis3D:
    """Partitioned 3D ST-RFM space-time coupled basis."""

    W_space: np.ndarray
    W_t: np.ndarray
    b: np.ndarray
    pou: PartitionOfUnity3D
    T: float
    name: str = "strfm_stc"

    @property
    def n_features(self) -> int:
        return int(np.prod(self.W_t.shape))

    def terms(self, x: np.ndarray, t: float) -> dict[str, np.ndarray]:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.shape != (3,):
            raise ValueError("x must have shape (3,)")
        t_scaled, dt_scaled = scale_time(self.T, t)
        locals_: list[dict[str, np.ndarray]] = []
        for patch_idx, center in enumerate(self.pou.centers):
            z = (point - center) / self.pou.radius
            s = self.W_space[patch_idx] @ z + self.W_t[patch_idx] * t_scaled + self.b[patch_idx]
            locals_.append(
                tanh_terms_2d(
                    s,
                    self.W_space[patch_idx] / self.pou.radius,
                    self.W_t[patch_idx] * dt_scaled,
                )
            )
        return _localize_3d(self.pou, locals_, point)


def make_separable_rfm_global_basis_1d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[float, float],
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> SeparableRFMGlobalBasis1D:
    """Sample a global separated-variable RFM basis."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    return SeparableRFMGlobalBasis1D(
        W_x=rng.uniform(-r_w, r_w, size=n_features),
        b_x=rng.uniform(-r_b, r_b, size=n_features),
        W_t=rng.uniform(-r_w, r_w, size=n_features),
        b_t=rng.uniform(-r_b, r_b, size=n_features),
        domain=(float(domain[0]), float(domain[1])),
        T=float(T),
    )


def make_separable_rfm_global_basis_2d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[tuple[float, float], tuple[float, float]],
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> SeparableRFMGlobalBasis2D:
    """Sample a global two-dimensional separated-variable RFM basis."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    normalized_domain = tuple((float(lo), float(hi)) for lo, hi in domain)
    return SeparableRFMGlobalBasis2D(
        W_space=rng.uniform(-r_w, r_w, size=(n_features, 2)),
        b_space=rng.uniform(-r_b, r_b, size=n_features),
        W_t=rng.uniform(-r_w, r_w, size=n_features),
        b_t=rng.uniform(-r_b, r_b, size=n_features),
        domain=normalized_domain,
        T=float(T),
    )


def make_separable_rfm_global_basis_3d(
    rng: np.random.Generator,
    *,
    n_features: int,
    domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
    r_w_space: float | None = None,
    r_w_time: float | None = None,
) -> SeparableRFMGlobalBasis3D:
    """Sample a global three-dimensional separated-variable RFM basis."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    space_range = float(r_w if r_w_space is None else r_w_space)
    time_range = float(r_w if r_w_time is None else r_w_time)
    normalized_domain = tuple((float(lo), float(hi)) for lo, hi in domain)
    return SeparableRFMGlobalBasis3D(
        W_space=rng.uniform(-space_range, space_range, size=(n_features, 3)),
        b_space=rng.uniform(-r_b, r_b, size=n_features),
        W_t=rng.uniform(-time_range, time_range, size=n_features),
        b_t=rng.uniform(-r_b, r_b, size=n_features),
        domain=normalized_domain,
        T=float(T),
    )


def make_strfm_sov_basis_1d(
    rng: np.random.Generator,
    *,
    pou: PartitionOfUnity1D,
    m_features: int,
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> STRFMSoVBasis1D:
    """Sample a partitioned ST-RFM SoV basis."""
    if m_features <= 0:
        raise ValueError("m_features must be positive")
    shape = (pou.n_subdomains, m_features)
    return STRFMSoVBasis1D(
        W_x=rng.uniform(-r_w, r_w, size=shape),
        b_x=rng.uniform(-r_b, r_b, size=shape),
        W_t=rng.uniform(-r_w, r_w, size=shape),
        b_t=rng.uniform(-r_b, r_b, size=shape),
        pou=pou,
        T=float(T),
    )


def make_strfm_sov_basis_2d(
    rng: np.random.Generator,
    *,
    pou: PartitionOfUnity2D,
    m_features: int,
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> STRFMSoVBasis2D:
    """Sample a partitioned 2D ST-RFM SoV basis."""
    if m_features <= 0:
        raise ValueError("m_features must be positive")
    shape = (pou.n_subdomains, m_features)
    return STRFMSoVBasis2D(
        W_space=rng.uniform(-r_w, r_w, size=shape + (2,)),
        b_space=rng.uniform(-r_b, r_b, size=shape),
        W_t=rng.uniform(-r_w, r_w, size=shape),
        b_t=rng.uniform(-r_b, r_b, size=shape),
        pou=pou,
        T=float(T),
    )


def make_strfm_sov_basis_3d(
    rng: np.random.Generator,
    *,
    pou: PartitionOfUnity3D,
    m_features: int,
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
    r_w_space: float | None = None,
    r_w_time: float | None = None,
) -> STRFMSoVBasis3D:
    """Sample a partitioned 3D ST-RFM SoV basis."""
    if m_features <= 0:
        raise ValueError("m_features must be positive")
    space_range = float(r_w if r_w_space is None else r_w_space)
    time_range = float(r_w if r_w_time is None else r_w_time)
    shape = (pou.n_subdomains, m_features)
    return STRFMSoVBasis3D(
        W_space=rng.uniform(-space_range, space_range, size=shape + (3,)),
        b_space=rng.uniform(-r_b, r_b, size=shape),
        W_t=rng.uniform(-time_range, time_range, size=shape),
        b_t=rng.uniform(-r_b, r_b, size=shape),
        pou=pou,
        T=float(T),
    )


def make_strfm_stc_basis_1d(
    rng: np.random.Generator,
    *,
    pou: PartitionOfUnity1D,
    m_features: int,
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> STRFMSTCBasis1D:
    """Sample a partitioned ST-RFM STC basis."""
    if m_features <= 0:
        raise ValueError("m_features must be positive")
    shape = (pou.n_subdomains, m_features)
    return STRFMSTCBasis1D(
        W_x=rng.uniform(-r_w, r_w, size=shape),
        W_t=rng.uniform(-r_w, r_w, size=shape),
        b=rng.uniform(-r_b, r_b, size=shape),
        pou=pou,
        T=float(T),
    )


def make_strfm_stc_basis_2d(
    rng: np.random.Generator,
    *,
    pou: PartitionOfUnity2D,
    m_features: int,
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
) -> STRFMSTCBasis2D:
    """Sample a partitioned 2D ST-RFM STC basis."""
    if m_features <= 0:
        raise ValueError("m_features must be positive")
    shape = (pou.n_subdomains, m_features)
    return STRFMSTCBasis2D(
        W_space=rng.uniform(-r_w, r_w, size=shape + (2,)),
        W_t=rng.uniform(-r_w, r_w, size=shape),
        b=rng.uniform(-r_b, r_b, size=shape),
        pou=pou,
        T=float(T),
    )


def make_strfm_stc_basis_3d(
    rng: np.random.Generator,
    *,
    pou: PartitionOfUnity3D,
    m_features: int,
    T: float,
    r_w: float = 1.0,
    r_b: float = 1.0,
    r_w_space: float | None = None,
    r_w_time: float | None = None,
) -> STRFMSTCBasis3D:
    """Sample a partitioned 3D ST-RFM STC basis."""
    if m_features <= 0:
        raise ValueError("m_features must be positive")
    space_range = float(r_w if r_w_space is None else r_w_space)
    time_range = float(r_w if r_w_time is None else r_w_time)
    shape = (pou.n_subdomains, m_features)
    return STRFMSTCBasis3D(
        W_space=rng.uniform(-space_range, space_range, size=shape + (3,)),
        W_t=rng.uniform(-time_range, time_range, size=shape),
        b=rng.uniform(-r_b, r_b, size=shape),
        pou=pou,
        T=float(T),
    )
