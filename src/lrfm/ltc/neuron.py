# 中文注释：定义满足初值相容性的单个松弛响应随机特征。
"""Single-neuron LTC feature equations.

The scalar L-RFM feature in HANDOFF Section 2.2 has time-independent spatial
coupling ``g(x)``. That makes the augmented sensitivity hierarchy a linear ODE
with constant coefficients at each fixed ``x``. The public helpers below expose
both the ODE right-hand side and its closed-form solution, so tests can check
the ODE structure directly while assembly can stay fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class LocalDerivatives1D:
    """Value and first three spatial derivatives of a scalar local function."""

    value: np.ndarray
    dx: np.ndarray
    dxx: np.ndarray
    dxxx: np.ndarray


@dataclass(frozen=True)
class LocalDerivativesND:
    """Value, gradient, Hessian, and Laplacian of a scalar local function."""

    value: np.ndarray
    grad: np.ndarray
    hessian: np.ndarray
    lap: np.ndarray


def tanh_with_derivatives_1d(s: ArrayLike, ds_dx: ArrayLike) -> LocalDerivatives1D:
    """Return ``tanh(s)`` and its first three 1D spatial derivatives.

    The local coordinates are affine in ``x``, so ``d2s/dx2 = 0``.
    """
    s_arr = np.asarray(s, dtype=float)
    ds = np.asarray(ds_dx, dtype=float)
    value = np.tanh(s_arr)
    one_minus_sq = 1.0 - value**2
    dx = one_minus_sq * ds
    dxx = -2.0 * value * one_minus_sq * ds**2
    dxxx = -2.0 * one_minus_sq * (1.0 - 3.0 * value**2) * ds**3
    return LocalDerivatives1D(value=value, dx=dx, dxx=dxx, dxxx=dxxx)


def local_tanh_1d(
    x: float,
    w: ArrayLike,
    b: ArrayLike,
    x_center: float,
    radius: float,
) -> LocalDerivatives1D:
    """Evaluate ``tanh(w * (x-c)/r + b)`` and 1D derivatives."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    w_arr = np.asarray(w, dtype=float)
    z = (float(x) - float(x_center)) / float(radius)
    s = w_arr * z + np.asarray(b, dtype=float)
    return tanh_with_derivatives_1d(s, w_arr / float(radius))


def local_tanh_nd(
    x: ArrayLike,
    w: ArrayLike,
    b: ArrayLike,
    x_center: ArrayLike,
    radius: float,
) -> LocalDerivativesND:
    """Evaluate ``tanh(w dot ((x-c)/r) + b)`` and ND derivatives."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    center = np.asarray(x_center, dtype=float).reshape(-1)
    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim != 2:
        raise ValueError("w must have shape (M, d)")
    if center.shape != (w_arr.shape[1],) or x_arr.shape != center.shape:
        raise ValueError("x, x_center, and w dimension must agree")
    z = (x_arr - center) / float(radius)
    ds = w_arr / float(radius)
    s = w_arr @ z + np.asarray(b, dtype=float)
    value = np.tanh(s)
    one_minus_sq = 1.0 - value**2
    grad = one_minus_sq[:, None] * ds
    hessian = -2.0 * value[:, None, None] * one_minus_sq[:, None, None] * (
        ds[:, :, None] * ds[:, None, :]
    )
    lap = np.trace(hessian, axis1=1, axis2=2)
    return LocalDerivativesND(value=value, grad=grad, hessian=hessian, lap=lap)


def augmented_rhs_1d(
    state: ArrayLike,
    *,
    tau: ArrayLike,
    g: ArrayLike,
    dg_dx: ArrayLike,
    d2g_dx2: ArrayLike,
    d3g_dx3: ArrayLike | None = None,
    asymptote: ArrayLike,
) -> np.ndarray:
    """Right-hand side for the 1D spatial sensitivity hierarchy.

    This is the hierarchy obtained by differentiating the scalar LTC feature
    ODE in space. ``state`` may contain derivatives through order two or three.
    """
    y = np.asarray(state, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    g_arr = np.asarray(g, dtype=float)
    gx = np.asarray(dg_dx, dtype=float)
    gxx = np.asarray(d2g_dx2, dtype=float)
    asym = np.asarray(asymptote, dtype=float)

    phi, phi_x, phi_xx = y[:3]
    lam = 1.0 / tau_arr + g_arr
    dphi_dt = -lam * phi + g_arr * asym
    dphi_x_dt = -lam * phi_x + gx * (asym - phi)
    dphi_xx_dt = -lam * phi_xx - 2.0 * gx * phi_x + gxx * (asym - phi)
    if y.shape[0] == 3:
        return np.stack([dphi_dt, dphi_x_dt, dphi_xx_dt], axis=0)
    if y.shape[0] != 4:
        raise ValueError("state must contain derivatives through order two or three")
    if d3g_dx3 is None:
        raise ValueError("d3g_dx3 is required for order-three augmented states")
    gxxx = np.asarray(d3g_dx3, dtype=float)
    phi_xxx = y[3]
    dphi_xxx_dt = -lam * phi_xxx - 3.0 * gx * phi_xx - 3.0 * gxx * phi_x + gxxx * (asym - phi)
    return np.stack([dphi_dt, dphi_x_dt, dphi_xx_dt, dphi_xxx_dt], axis=0)


def ltc_feature_closed_form_1d(
    x: float,
    t: float,
    *,
    tau: ArrayLike,
    w_in: ArrayLike,
    b_in: ArrayLike,
    asymptote: ArrayLike,
    w_h0: ArrayLike,
    b_h0: ArrayLike,
    x_center: float,
    radius: float,
) -> dict[str, np.ndarray]:
    """Evaluate 1D LTC features and derivatives through order three.

    The formulas are the exact solution of the augmented linear ODE at fixed
    spatial coordinate. They avoid differentiating through a numerical ODE
    solver during LS assembly.
    """
    tau_arr = np.asarray(tau, dtype=float)
    asym = np.asarray(asymptote, dtype=float)
    time = float(t)

    g = local_tanh_1d(x, w_in, b_in, x_center, radius)
    h0 = local_tanh_1d(x, w_h0, b_h0, x_center, radius)

    q = 1.0 / tau_arr
    lam = q + g.value
    if np.any(np.abs(lam) < 1.0e-10):
        raise FloatingPointError("encountered near-zero LTC decay rate")

    beta = g.value * asym
    exp_term = np.exp(-lam * time)
    steady = beta / lam
    steady_x = asym * q * g.dx / lam**2
    steady_xx = asym * q * (g.dxx / lam**2 - 2.0 * g.dx**2 / lam**3)
    steady_xxx = asym * q * (
        g.dxxx / lam**2
        - 6.0 * g.dx * g.dxx / lam**3
        + 6.0 * g.dx**3 / lam**4
    )

    transient = h0.value - steady
    transient_x = h0.dx - steady_x
    transient_xx = h0.dxx - steady_xx
    transient_xxx = h0.dxxx - steady_xxx

    exp_x = -time * g.dx * exp_term
    exp_xx = (time**2 * g.dx**2 - time * g.dxx) * exp_term
    exp_xxx = (-time**3 * g.dx**3 + 3.0 * time**2 * g.dx * g.dxx - time * g.dxxx) * exp_term

    phi = steady + transient * exp_term
    phi_x = steady_x + transient_x * exp_term + transient * exp_x
    phi_xx = (
        steady_xx
        + transient_xx * exp_term
        + 2.0 * transient_x * exp_x
        + transient * exp_xx
    )
    phi_xxx = (
        steady_xxx
        + transient_xxx * exp_term
        + 3.0 * transient_xx * exp_x
        + 3.0 * transient_x * exp_xx
        + transient * exp_xxx
    )
    dphi_dt = -lam * phi + beta

    return {
        "phi": phi,
        "dphi_dt": dphi_dt,
        "grad_x": phi_x[:, None],
        "lap_x": phi_xx,
        "d2phi_dx2": phi_xx,
        "d3phi_dx3": phi_xxx,
        "g": g.value,
        "dg_dx": g.dx,
        "d2g_dx2": g.dxx,
        "d3g_dx3": g.dxxx,
        "h0": h0.value,
        "dh0_dx": h0.dx,
        "d2h0_dx2": h0.dxx,
        "d3h0_dx3": h0.dxxx,
    }


def ltc_feature_closed_form_nd_order2(
    x: ArrayLike,
    t: float,
    *,
    tau: ArrayLike,
    w_in: ArrayLike,
    b_in: ArrayLike,
    asymptote: ArrayLike,
    w_h0: ArrayLike,
    b_h0: ArrayLike,
    x_center: ArrayLike,
    radius: float,
) -> dict[str, np.ndarray]:
    """Evaluate ND LTC features and derivatives through second order.

    This is the 2D/ND analogue of the augmented sensitivity hierarchy in the
    handoff. The local coupling and initial state are affine-tanh functions of
    space, so the value, gradient, and Hessian have closed forms and can be
    propagated through the scalar LTC solution without differentiating through
    a numerical ODE solver.
    """
    tau_arr = np.asarray(tau, dtype=float)
    asym = np.asarray(asymptote, dtype=float)
    time = float(t)

    g = local_tanh_nd(x, w_in, b_in, x_center, radius)
    h0 = local_tanh_nd(x, w_h0, b_h0, x_center, radius)

    q = 1.0 / tau_arr
    lam = q + g.value
    if np.any(np.abs(lam) < 1.0e-10):
        raise FloatingPointError("encountered near-zero LTC decay rate")

    beta = g.value * asym
    exp_term = np.exp(-lam * time)
    g_outer = g.grad[:, :, None] * g.grad[:, None, :]

    steady = beta / lam
    steady_grad = asym[:, None] * q[:, None] * g.grad / lam[:, None] ** 2
    steady_hessian = asym[:, None, None] * q[:, None, None] * (
        g.hessian / lam[:, None, None] ** 2
        - 2.0 * g_outer / lam[:, None, None] ** 3
    )

    transient = h0.value - steady
    transient_grad = h0.grad - steady_grad
    transient_hessian = h0.hessian - steady_hessian

    exp_grad = -time * g.grad * exp_term[:, None]
    exp_hessian = (time**2 * g_outer - time * g.hessian) * exp_term[:, None, None]

    phi = steady + transient * exp_term
    phi_grad = steady_grad + transient_grad * exp_term[:, None] + transient[:, None] * exp_grad
    phi_hessian = (
        steady_hessian
        + transient_hessian * exp_term[:, None, None]
        + transient_grad[:, :, None] * exp_grad[:, None, :]
        + exp_grad[:, :, None] * transient_grad[:, None, :]
        + transient[:, None, None] * exp_hessian
    )
    phi_lap = np.trace(phi_hessian, axis1=1, axis2=2)
    dphi_dt = -lam * phi + beta

    return {
        "phi": phi,
        "dphi_dt": dphi_dt,
        "grad_x": phi_grad,
        "hessian_x": phi_hessian,
        "lap_x": phi_lap,
        "g": g.value,
        "grad_g": g.grad,
        "hessian_g": g.hessian,
        "h0": h0.value,
        "grad_h0": h0.grad,
        "hessian_h0": h0.hessian,
    }
