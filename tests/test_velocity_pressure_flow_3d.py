# 中文注释：检验三维速度-压力解析流动和残差。
from __future__ import annotations

import numpy as np

from lrfm.assembly.matrix import sample_collocation_3d
from lrfm.baselines.pielm import make_pielm_basis_3d
from lrfm.nonlinear.velocity_pressure_3d import (
    VelocityPressureSolution3D,
    assemble_velocity_pressure_picard_3d,
)
from lrfm.pdes.velocity_pressure_flow import (
    DecayingBeltramiFlow3D,
    ForcedDecayingBeltramiFlow3D,
    InteractingHelicalVortex3D,
)


class _ExactPrevious:
    def __init__(self, pde):
        self.pde = pde

    def velocity(self, x, t):
        return self.pde.exact_velocity(x, t)


class _ExactHelicalSpan:
    """Scalar Fourier-relaxation columns that contain the manufactured state."""

    modes = ((1, 0.5), (2, 4.0))

    @property
    def n_features(self):
        return 12

    def terms(self, x, t):
        point = np.asarray(x, dtype=float)
        values = []
        dts = []
        gradients = []
        laplacians = []
        for wavenumber, rate in self.modes:
            if abs(float(t)) < 1.0e-10:
                temporal = -rate
                temporal_dt = 0.5 * rate**2
            else:
                temporal = np.expm1(-rate * float(t)) / float(t)
                temporal_dt = (
                    1.0 - np.exp(-rate * float(t)) * (1.0 + rate * float(t))
                ) / float(t) ** 2
            for axis in range(3):
                phase = wavenumber * point[axis]
                for kind in ("sin", "cos"):
                    if kind == "sin":
                        spatial = np.sin(phase)
                        derivative = wavenumber * np.cos(phase)
                    else:
                        spatial = np.cos(phase)
                        derivative = -wavenumber * np.sin(phase)
                    gradient = np.zeros(3)
                    gradient[axis] = derivative * temporal
                    values.append(spatial * temporal)
                    dts.append(spatial * temporal_dt)
                    gradients.append(gradient)
                    laplacians.append(-(wavenumber**2) * spatial * temporal)
        return {
            "value": np.asarray(values),
            "dt": np.asarray(dts),
            "grad": np.asarray(gradients),
            "lap": np.asarray(laplacians),
        }


class _PressureAwareBasis:
    n_features = 1

    def terms(self, x, t):
        del x, t
        return {
            "value": np.asarray([1.0]),
            "dt": np.asarray([0.0]),
            "grad": np.zeros((1, 3)),
            "lap": np.asarray([0.0]),
        }

    def pressure_terms(self, x, t):
        del x
        return {
            "value": np.asarray([7.0 + float(t)]),
            "dt": np.asarray([1.0]),
            "grad": np.asarray([[11.0, 13.0, 17.0]]),
            "lap": np.asarray([0.0]),
        }


def _tiny_basis(pde, n_features=12, seed=701):
    return make_pielm_basis_3d(
        np.random.default_rng(seed),
        n_features=n_features,
        domain=pde.domain,
        T=pde.T,
        r_w=1.0,
        r_b=1.0,
    )


def test_helical_reference_is_divergence_free_and_has_active_transport():
    pde = InteractingHelicalVortex3D()
    points = np.asarray([[0.31, 0.77, 1.23], [2.17, 1.31, 0.41]])
    gradient = pde.exact_velocity_gradient(points, 0.37)
    velocity = pde.exact_velocity(points, 0.37)
    vorticity_from_gradient = np.stack(
        [
            gradient[..., 2, 1] - gradient[..., 1, 2],
            gradient[..., 0, 2] - gradient[..., 2, 0],
            gradient[..., 1, 0] - gradient[..., 0, 1],
        ],
        axis=-1,
    )
    transport = np.einsum("...j,...ij->...i", velocity, gradient)
    slow_mode = pde._mode(points, pde.slow_wavenumber)
    fast_mode = pde._mode(points, pde.fast_wavenumber)
    slow_gradient = pde._mode_gradient(points, pde.slow_wavenumber)
    fast_gradient = pde._mode_gradient(points, pde.fast_wavenumber)
    cross_transport = np.einsum("...j,...ij->...i", slow_mode, fast_gradient)
    cross_transport += np.einsum("...j,...ij->...i", fast_mode, slow_gradient)

    np.testing.assert_allclose(np.trace(gradient, axis1=-2, axis2=-1), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(vorticity_from_gradient, pde.exact_vorticity(points, 0.37))
    np.testing.assert_allclose(pde.exact_pressure(points, np.asarray([0.1, 0.2])), 0.0)
    assert np.linalg.norm(transport) > 0.1
    assert np.linalg.norm(cross_transport) > 0.1


def test_helical_forcing_matches_finite_difference_substitution():
    pde = InteractingHelicalVortex3D()
    point = np.asarray([0.41, 0.83, 1.37])
    time = 0.29
    step = 2.0e-5
    dt = (
        pde.exact_velocity(point, time + step)
        - pde.exact_velocity(point, time - step)
    ) / (2.0 * step)
    gradient = np.empty((3, 3), dtype=float)
    laplacian = np.zeros(3, dtype=float)
    center = pde.exact_velocity(point, time)
    for axis in range(3):
        offset = np.zeros(3, dtype=float)
        offset[axis] = step
        plus = pde.exact_velocity(point + offset, time)
        minus = pde.exact_velocity(point - offset, time)
        gradient[:, axis] = (plus - minus) / (2.0 * step)
        laplacian += (plus + minus - 2.0 * center) / step**2
    forcing_fd = dt + gradient @ center - pde.viscosity * laplacian

    np.testing.assert_allclose(pde.momentum_forcing(point, time), forcing_fd, rtol=2.0e-6, atol=2.0e-6)


def test_velocity_pressure_assembly_has_expected_coupled_blocks():
    pde = InteractingHelicalVortex3D(T=0.05)
    basis = _tiny_basis(pde)
    collocation = sample_collocation_3d(pde, n_int=5, n_ic=1, n_bc=2, seed=702)
    gauge_times = np.linspace(0.0, pde.T, 3)
    matrix, target = assemble_velocity_pressure_picard_3d(
        pde,
        basis,
        collocation,
        previous=_ExactPrevious(pde),
        gauge_times=gauge_times,
    )

    expected_rows = 4 * collocation.n_int + 24 * 2 + len(gauge_times)
    assert matrix.shape == (expected_rows, 4 * basis.n_features)
    assert target.shape == (expected_rows,)
    assert np.all(np.isfinite(matrix))
    first_momentum = matrix[0]
    assert np.any(np.abs(first_momentum[: basis.n_features]) > 0.0)
    assert np.any(np.abs(first_momentum[3 * basis.n_features :]) > 0.0)
    first_divergence = matrix[3]
    assert np.any(np.abs(first_divergence[: 3 * basis.n_features]) > 0.0)
    assert np.all(first_divergence[3 * basis.n_features :] == 0.0)


def test_velocity_pressure_solver_uses_pressure_specific_temporal_columns():
    pde = InteractingHelicalVortex3D(T=0.05)
    basis = _PressureAwareBasis()
    collocation = sample_collocation_3d(pde, n_int=1, n_ic=1, n_bc=1, seed=706)
    gauge_times = np.asarray([0.0, 0.05])
    matrix, _ = assemble_velocity_pressure_picard_3d(
        pde,
        basis,
        collocation,
        previous=_ExactPrevious(pde),
        gauge_times=gauge_times,
    )
    np.testing.assert_allclose(matrix[:3, 3], np.asarray([11.0, 13.0, 17.0]))
    np.testing.assert_allclose(matrix[-2:, 3], 7.0 + gauge_times)

    coefficients = np.zeros(4)
    coefficients[3] = 2.0
    solution = VelocityPressureSolution3D(pde, basis, coefficients)
    np.testing.assert_allclose(solution.pressure(np.asarray([0.2, 0.4, 0.6]), 0.03), 14.06)
    np.testing.assert_allclose(
        solution.pressure_gradient(np.asarray([0.2, 0.4, 0.6]), 0.03),
        np.asarray([22.0, 26.0, 34.0]),
    )


def test_exact_helical_coefficients_satisfy_picard_assembly():
    pde = InteractingHelicalVortex3D()
    basis = _ExactHelicalSpan()
    collocation = sample_collocation_3d(pde, n_int=80, n_ic=1, n_bc=5, seed=705)
    matrix, target = assemble_velocity_pressure_picard_3d(
        pde,
        basis,
        collocation,
        previous=_ExactPrevious(pde),
        gauge_times=np.linspace(0.0, pde.T, 5),
    )
    coefficients = np.zeros(4 * basis.n_features)
    # Per mode the scalar columns are [sin x, cos x, sin y, cos y, sin z, cos z].
    component_columns = ((4, 3), (0, 5), (2, 1))
    for component, columns in enumerate(component_columns):
        coefficients[component * basis.n_features + columns[0]] = 1.0
        coefficients[component * basis.n_features + columns[1]] = 1.0
        coefficients[component * basis.n_features + 6 + columns[0]] = pde.amplitude
        coefficients[component * basis.n_features + 6 + columns[1]] = pde.amplitude

    residual = matrix @ coefficients - target
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-13)


def test_hard_lift_exactly_preserves_initial_velocity():
    pde = InteractingHelicalVortex3D()
    basis = _tiny_basis(pde, n_features=8, seed=703)
    coefficients = np.random.default_rng(704).normal(size=4 * basis.n_features)
    solution = VelocityPressureSolution3D(pde, basis, coefficients)
    points = np.asarray([[0.2, 0.4, 0.6], [1.1, 2.2, 3.3]])

    np.testing.assert_allclose(solution.velocity(points, 0.0), pde.initial_velocity(points))
    np.testing.assert_allclose(solution.divergence(points, 0.0), 0.0, atol=1.0e-14)


def test_decaying_beltrami_flow_satisfies_unforced_velocity_pressure_equations():
    pde = DecayingBeltramiFlow3D()
    points = np.asarray([[0.31, 0.77, 1.23], [2.17, 1.31, 0.41]])
    time = 0.37
    velocity = pde.exact_velocity(points, time)
    gradient = pde.exact_velocity_gradient(points, time)
    residual = pde.residual(
        velocity,
        pde.exact_velocity_dt(points, time),
        gradient,
        pde.exact_velocity_laplacian(points, time),
        points,
        time,
        pde.exact_pressure_gradient(points, time),
    )

    np.testing.assert_allclose(residual, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(np.trace(gradient, axis1=-2, axis2=-1), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(pde.exact_vorticity(points, time), pde.wavenumber * velocity)
    gauge = np.asarray([bound[0] for bound in pde.domain])
    np.testing.assert_allclose(pde.exact_pressure(gauge, np.asarray([0.0, 0.3, 1.0])), 0.0)
    np.testing.assert_allclose(pde.momentum_forcing(points, time), 0.0)


def test_forced_decaying_beltrami_flow_has_active_transport_and_zero_pressure():
    pde = ForcedDecayingBeltramiFlow3D()
    points = np.asarray([[0.31, 0.77, 1.23], [2.17, 1.31, 0.41]])
    time = 0.37
    velocity = pde.exact_velocity(points, time)
    gradient = pde.exact_velocity_gradient(points, time)
    transport = np.einsum("...j,...ij->...i", velocity, gradient)
    residual = pde.residual(
        velocity,
        pde.exact_velocity_dt(points, time),
        gradient,
        pde.exact_velocity_laplacian(points, time),
        points,
        time,
        pde.exact_pressure_gradient(points, time),
    )

    assert np.linalg.norm(transport) > 0.1
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(pde.exact_pressure(points, time), 0.0)
