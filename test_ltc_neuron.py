# 中文注释：检验松弛响应特征及其解析导数。
import numpy as np

from lrfm.ltc.init import make_reservoir_1d
from lrfm.ltc.neuron import local_tanh_1d
from lrfm.ltc.reservoir import LiquidReservoir
from lrfm.ode_solver.augmented import solve_augmented_ltc_1d_diffrax


def test_ltc_recovers_exponential_decay_when_g_is_zero():
    tau = np.asarray([0.7])
    b_h0 = np.asarray([0.4])
    reservoir = LiquidReservoir(
        tau=tau,
        W_in=np.asarray([[0.0]]),
        b_in=np.asarray([0.0]),
        A=np.asarray([0.3]),
        W_h0=np.asarray([[0.0]]),
        b_h0=b_h0,
        x_center=np.asarray([0.0]),
        radius=1.0,
    )

    t = 0.9
    got = reservoir.features(np.asarray([0.2]), t)[0]
    expected = np.tanh(b_h0[0]) * np.exp(-t / tau[0])
    assert np.isclose(got, expected, rtol=1.0e-12, atol=1.0e-12)


def test_augmented_derivatives_match_finite_differences():
    rng = np.random.default_rng(123)
    reservoir = make_reservoir_1d(
        rng,
        m_features=4,
        x_center=0.0,
        radius=1.5,
        tau_min=0.2,
        tau_max=2.0,
    )
    x = 0.17
    t = 0.35
    eps = 1.0e-5

    derivs = reservoir.features_with_derivs(np.asarray([x]), t, order=2)
    f0 = reservoir.features(np.asarray([x]), t)
    f_plus = reservoir.features(np.asarray([x + eps]), t)
    f_minus = reservoir.features(np.asarray([x - eps]), t)

    fd_x = (f_plus - f_minus) / (2.0 * eps)
    fd_xx = (f_plus - 2.0 * f0 + f_minus) / eps**2
    np.testing.assert_allclose(derivs["grad_x"][:, 0], fd_x, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(derivs["lap_x"], fd_xx, rtol=1.0e-4, atol=2.0e-5)


def test_order_three_augmented_derivatives_match_finite_differences():
    rng = np.random.default_rng(321)
    reservoir = make_reservoir_1d(
        rng,
        m_features=4,
        x_center=0.0,
        radius=1.7,
        tau_min=0.2,
        tau_max=2.0,
    )
    x = -0.11
    t = 0.28
    eps = 2.0e-5

    derivs = reservoir.features_with_derivs(np.asarray([x]), t, order=3)
    dxx_plus = reservoir.features_with_derivs(np.asarray([x + eps]), t, order=2)["lap_x"]
    dxx_minus = reservoir.features_with_derivs(np.asarray([x - eps]), t, order=2)["lap_x"]
    fd_dxxx = (dxx_plus - dxx_minus) / (2.0 * eps)
    np.testing.assert_allclose(derivs["d3phi_dx3"], fd_dxxx, rtol=2.0e-5, atol=2.0e-6)


def test_diffrax_augmented_solver_matches_closed_form():
    reservoir = LiquidReservoir(
        tau=np.asarray([0.8]),
        W_in=np.asarray([[0.4]]),
        b_in=np.asarray([0.1]),
        A=np.asarray([0.7]),
        W_h0=np.asarray([[-0.3]]),
        b_h0=np.asarray([0.2]),
        x_center=np.asarray([0.0]),
        radius=1.2,
    )
    x = 0.25
    t = 0.15
    g = local_tanh_1d(x, reservoir.W_in[:, 0], reservoir.b_in, 0.0, reservoir.radius)
    h0 = local_tanh_1d(x, reservoir.W_h0[:, 0], reservoir.b_h0, 0.0, reservoir.radius)
    y0 = np.stack([h0.value, h0.dx, h0.dxx], axis=0)

    ode = solve_augmented_ltc_1d_diffrax(
        y0=y0,
        t1=t,
        tau=reservoir.tau,
        g=g.value,
        dg_dx=g.dx,
        d2g_dx2=g.dxx,
        asymptote=reservoir.A,
    )
    closed = reservoir.features_with_derivs(np.asarray([x]), t, order=2)
    expected = np.stack([closed["phi"], closed["grad_x"][:, 0], closed["lap_x"]], axis=0)
    np.testing.assert_allclose(ode, expected, rtol=1.0e-5, atol=1.0e-6)


def test_diffrax_order_three_augmented_solver_matches_closed_form():
    reservoir = LiquidReservoir(
        tau=np.asarray([0.8]),
        W_in=np.asarray([[0.4]]),
        b_in=np.asarray([0.1]),
        A=np.asarray([0.7]),
        W_h0=np.asarray([[-0.3]]),
        b_h0=np.asarray([0.2]),
        x_center=np.asarray([0.0]),
        radius=1.2,
    )
    x = 0.25
    t = 0.15
    g = local_tanh_1d(x, reservoir.W_in[:, 0], reservoir.b_in, 0.0, reservoir.radius)
    h0 = local_tanh_1d(x, reservoir.W_h0[:, 0], reservoir.b_h0, 0.0, reservoir.radius)
    y0 = np.stack([h0.value, h0.dx, h0.dxx, h0.dxxx], axis=0)

    ode = solve_augmented_ltc_1d_diffrax(
        y0=y0,
        t1=t,
        tau=reservoir.tau,
        g=g.value,
        dg_dx=g.dx,
        d2g_dx2=g.dxx,
        d3g_dx3=g.dxxx,
        asymptote=reservoir.A,
    )
    closed = reservoir.features_with_derivs(np.asarray([x]), t, order=3)
    expected = np.stack(
        [closed["phi"], closed["grad_x"][:, 0], closed["lap_x"], closed["d3phi_dx3"]],
        axis=0,
    )
    np.testing.assert_allclose(ode, expected, rtol=1.0e-5, atol=1.0e-6)
