# 中文注释：检验二维涡量流动解析解、源项和评价指标。
import numpy as np

from lrfm.assembly.matrix import LRFMBasis2D, sample_collocation_2d
from lrfm.ltc.init import make_reservoirs_from_pou_2d
from lrfm.nonlinear.vorticity_streamfunction import (
    assemble_vorticity_streamfunction_picard_2d,
    default_vorticity_flow_weights,
    solve_vorticity_streamfunction_block_marching_2d,
    solve_vorticity_streamfunction_picard_2d,
)
from lrfm.pdes.vorticity_flow import (
    DecayingTurbulence2D,
    DoubleShearLayer2D,
    InteractingModeVortex2D,
    LinearInteractingModeStokes2D,
    MultiscaleShearDecay2D,
    TaylorGreenVortex2D,
    TwoRateVortex2D,
    solve_vorticity_reference_2d,
)
from lrfm.pou.decomposition import PartitionOfUnity2D


def test_decaying_turbulence_initial_field_is_periodic_normalized_and_nonlinear():
    pde = DecayingTurbulence2D()
    n = 128
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    points = np.stack([xx, yy], axis=-1)
    velocity = pde.initial_velocity(points)
    rms_velocity = np.sqrt(np.mean(np.sum(velocity**2, axis=-1)))
    assert np.isclose(rms_velocity, pde.target_rms_velocity, rtol=1.0e-12)

    probes = np.asarray([[0.3, 0.8], [1.7, 2.4], [4.1, 5.2]])
    shifted = probes + np.asarray([2.0 * np.pi, -2.0 * np.pi])
    assert np.allclose(pde.initial_velocity(probes), pde.initial_velocity(shifted))
    assert np.allclose(pde.initial_vorticity(probes), pde.initial_vorticity(shifted))

    transport = np.sum(
        pde.initial_velocity(probes) * pde.initial_vorticity_gradient(probes), axis=-1
    )
    assert np.linalg.norm(transport) > 1.0e-2
    assert np.allclose(pde.vorticity_forcing(probes, 0.2), 0.0)


def test_decaying_turbulence_reference_dissipates_energy_and_enstrophy():
    pde = DecayingTurbulence2D(viscosity=0.02, T=0.1)
    reference = solve_vorticity_reference_2d(
        pde,
        nx=32,
        ny=32,
        nt=4,
        dt_max=1.0e-3,
    )

    assert np.all(np.diff(reference.kinetic_energy) < 0.0)
    assert np.all(np.diff(reference.enstrophy) < 0.0)
    relative_change = np.linalg.norm(reference.omega[-1] - reference.omega[0])
    relative_change /= np.linalg.norm(reference.omega[0])
    assert relative_change > 5.0e-2


def _tiny_basis(pde, *, m_features: int = 2, seed: int = 401) -> LRFMBasis2D:
    pou = PartitionOfUnity2D.uniform(pde.domain, (2, 2))
    rng = np.random.default_rng(seed)
    reservoirs = make_reservoirs_from_pou_2d(
        rng,
        centers=pou.centers,
        radius=pou.radius,
        m_features=m_features,
        tau_min=0.05 * pde.T,
        tau_max=5.0 * pde.T,
        r_w=1.5,
    )
    return LRFMBasis2D(pou, reservoirs)


class _ExactTaylorGreen:
    def __init__(self, pde):
        self.pde = pde

    def velocity(self, x, t):
        return self.pde.exact_velocity(x, t)

    def vorticity(self, x, t):
        return self.pde.exact_vorticity(x, t)


def test_taylor_green_exact_fields_satisfy_vorticity_relations():
    pde = TaylorGreenVortex2D(viscosity=0.03)
    x = np.asarray([1.1, 2.2])
    t = 0.4
    psi = pde.exact_streamfunction(x, t)
    omega = pde.exact_vorticity(x, t)
    velocity = pde.exact_velocity(x, t)

    assert np.isclose(omega, 2.0 * psi)
    assert velocity.shape == (2,)
    assert np.all(np.isfinite(velocity))
    assert np.isclose(pde.vorticity_forcing(x, t), 0.0)


def test_two_rate_vortex_exact_fields_satisfy_forced_vorticity_equation():
    pde = TwoRateVortex2D()
    point = np.asarray([1.1, 2.2])
    time = 0.37
    step = 2.0e-5
    ex = np.asarray([step, 0.0])
    ey = np.asarray([0.0, step])

    psi = pde.exact_streamfunction(point, time)
    omega = pde.exact_vorticity(point, time)
    velocity = pde.exact_velocity(point, time)
    psi_x = (
        pde.exact_streamfunction(point + ex, time) - pde.exact_streamfunction(point - ex, time)
    ) / (2.0 * step)
    psi_y = (
        pde.exact_streamfunction(point + ey, time) - pde.exact_streamfunction(point - ey, time)
    ) / (2.0 * step)
    psi_lap = (
        pde.exact_streamfunction(point + ex, time)
        + pde.exact_streamfunction(point - ex, time)
        + pde.exact_streamfunction(point + ey, time)
        + pde.exact_streamfunction(point - ey, time)
        - 4.0 * psi
    ) / step**2
    omega_t = (
        pde.exact_vorticity(point, time + step) - pde.exact_vorticity(point, time - step)
    ) / (2.0 * step)
    omega_x = (
        pde.exact_vorticity(point + ex, time) - pde.exact_vorticity(point - ex, time)
    ) / (2.0 * step)
    omega_y = (
        pde.exact_vorticity(point + ey, time) - pde.exact_vorticity(point - ey, time)
    ) / (2.0 * step)
    omega_lap = (
        pde.exact_vorticity(point + ex, time)
        + pde.exact_vorticity(point - ex, time)
        + pde.exact_vorticity(point + ey, time)
        + pde.exact_vorticity(point - ey, time)
        - 4.0 * omega
    ) / step**2
    residual = omega_t + velocity @ np.asarray([omega_x, omega_y])
    residual -= pde.viscosity * omega_lap + pde.vorticity_forcing(point, time)

    assert np.isclose(omega, 2.0 * psi)
    np.testing.assert_allclose(velocity, np.asarray([psi_y, -psi_x]), rtol=2.0e-7, atol=2.0e-7)
    np.testing.assert_allclose(omega, -psi_lap, rtol=2.0e-6, atol=2.0e-6)
    assert abs(residual) < 2.0e-5


def test_interacting_mode_vortex_has_nonzero_transport_and_exact_forcing():
    pde = InteractingModeVortex2D()
    point = np.asarray([0.83, 1.37])
    time = 0.29
    step = 2.0e-5
    ex = np.asarray([step, 0.0])
    ey = np.asarray([0.0, step])

    psi = pde.exact_streamfunction(point, time)
    omega = pde.exact_vorticity(point, time)
    velocity = pde.exact_velocity(point, time)
    psi_x = (
        pde.exact_streamfunction(point + ex, time) - pde.exact_streamfunction(point - ex, time)
    ) / (2.0 * step)
    psi_y = (
        pde.exact_streamfunction(point + ey, time) - pde.exact_streamfunction(point - ey, time)
    ) / (2.0 * step)
    psi_lap = (
        pde.exact_streamfunction(point + ex, time)
        + pde.exact_streamfunction(point - ex, time)
        + pde.exact_streamfunction(point + ey, time)
        + pde.exact_streamfunction(point - ey, time)
        - 4.0 * psi
    ) / step**2
    omega_t = (
        pde.exact_vorticity(point, time + step) - pde.exact_vorticity(point, time - step)
    ) / (2.0 * step)
    omega_x = (
        pde.exact_vorticity(point + ex, time) - pde.exact_vorticity(point - ex, time)
    ) / (2.0 * step)
    omega_y = (
        pde.exact_vorticity(point + ey, time) - pde.exact_vorticity(point - ey, time)
    ) / (2.0 * step)
    omega_lap = (
        pde.exact_vorticity(point + ex, time)
        + pde.exact_vorticity(point - ex, time)
        + pde.exact_vorticity(point + ey, time)
        + pde.exact_vorticity(point - ey, time)
        - 4.0 * omega
    ) / step**2
    advection = float(velocity @ np.asarray([omega_x, omega_y]))
    residual = omega_t + advection - pde.viscosity * omega_lap
    residual -= pde.vorticity_forcing(point, time)

    assert abs(advection) > 1.0e-3
    np.testing.assert_allclose(velocity, np.asarray([psi_y, -psi_x]), rtol=3.0e-7, atol=3.0e-7)
    np.testing.assert_allclose(omega, -psi_lap, rtol=3.0e-6, atol=3.0e-6)
    assert abs(residual) < 3.0e-5


def test_linear_interacting_stokes_matches_exact_solution_without_transport():
    linear = LinearInteractingModeStokes2D()
    nonlinear = InteractingModeVortex2D()
    point = np.asarray([0.83, 1.37])
    time = 0.29
    step = 2.0e-5
    ex = np.asarray([step, 0.0])
    ey = np.asarray([0.0, step])

    omega_t = (
        linear.exact_vorticity(point, time + step)
        - linear.exact_vorticity(point, time - step)
    ) / (2.0 * step)
    omega = linear.exact_vorticity(point, time)
    omega_lap = (
        linear.exact_vorticity(point + ex, time)
        + linear.exact_vorticity(point - ex, time)
        + linear.exact_vorticity(point + ey, time)
        + linear.exact_vorticity(point - ey, time)
        - 4.0 * omega
    ) / step**2
    transport = float(
        linear.exact_velocity(point, time) @ linear.exact_vorticity_gradient(point, time)
    )

    assert linear.is_linear()
    assert not nonlinear.is_linear()
    assert abs(omega_t - linear.viscosity * omega_lap - linear.vorticity_forcing(point, time)) < 3.0e-5
    np.testing.assert_allclose(
        nonlinear.vorticity_forcing(point, time) - linear.vorticity_forcing(point, time),
        transport,
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_linear_interacting_stokes_uses_exactly_one_linear_solve():
    pde = LinearInteractingModeStokes2D(T=0.03)
    basis = _tiny_basis(pde, m_features=3, seed=433)
    collocation = sample_collocation_2d(pde, n_int=28, n_ic=14, n_bc=4, seed=434)
    result = solve_vorticity_streamfunction_picard_2d(
        pde,
        basis,
        collocation,
        gauge_times=np.linspace(0.0, pde.T, 3),
        max_iters=9,
        monitor_nx=5,
        monitor_ny=5,
        monitor_nt=2,
        ridge_lambda=1.0e-10,
        hard_initial=True,
    )

    assert result.iterations == 1
    assert result.converged
    assert len(result.relative_updates) == 1
    assert np.all(np.isfinite(result.coefficients))


def test_multiscale_shear_decay_is_an_unforced_exact_solution():
    pde = MultiscaleShearDecay2D()
    point = np.asarray([1.1, 2.2])
    time = 0.37
    step = 2.0e-5
    ex = np.asarray([step, 0.0])
    ey = np.asarray([0.0, step])

    psi = pde.exact_streamfunction(point, time)
    omega = pde.exact_vorticity(point, time)
    velocity = pde.exact_velocity(point, time)
    psi_x = (
        pde.exact_streamfunction(point + ex, time) - pde.exact_streamfunction(point - ex, time)
    ) / (2.0 * step)
    psi_y = (
        pde.exact_streamfunction(point + ey, time) - pde.exact_streamfunction(point - ey, time)
    ) / (2.0 * step)
    psi_lap = (
        pde.exact_streamfunction(point + ex, time)
        + pde.exact_streamfunction(point - ex, time)
        + pde.exact_streamfunction(point + ey, time)
        + pde.exact_streamfunction(point - ey, time)
        - 4.0 * psi
    ) / step**2
    omega_t = (
        pde.exact_vorticity(point, time + step) - pde.exact_vorticity(point, time - step)
    ) / (2.0 * step)
    omega_lap = (
        pde.exact_vorticity(point + ex, time)
        + pde.exact_vorticity(point - ex, time)
        + pde.exact_vorticity(point + ey, time)
        + pde.exact_vorticity(point - ey, time)
        - 4.0 * omega
    ) / step**2

    np.testing.assert_allclose(velocity, np.asarray([psi_y, -psi_x]), rtol=2.0e-7, atol=2.0e-7)
    np.testing.assert_allclose(omega, -psi_lap, rtol=3.0e-6, atol=3.0e-6)
    assert abs(omega_t - pde.viscosity * omega_lap) < 2.0e-5
    assert np.isclose(pde.vorticity_forcing(point, time), 0.0)


def test_double_shear_initial_streamfunction_matches_velocity_and_vorticity():
    pde = DoubleShearLayer2D()
    point = np.asarray([0.31, 0.37])
    step = 1.0e-6
    ex = np.asarray([step, 0.0])
    ey = np.asarray([0.0, step])
    psi_x = (pde.initial_streamfunction(point + ex) - pde.initial_streamfunction(point - ex)) / (2.0 * step)
    psi_y = (pde.initial_streamfunction(point + ey) - pde.initial_streamfunction(point - ey)) / (2.0 * step)
    velocity = pde.initial_velocity(point)
    v_x = (pde.initial_velocity(point + ex)[1] - pde.initial_velocity(point - ex)[1]) / (2.0 * step)
    u_y = (pde.initial_velocity(point + ey)[0] - pde.initial_velocity(point - ey)[0]) / (2.0 * step)

    np.testing.assert_allclose(velocity, np.asarray([psi_y, -psi_x]), rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(pde.initial_vorticity(point), v_x - u_y, rtol=2.0e-5, atol=2.0e-5)
    assert np.isclose(pde.initial_streamfunction(np.asarray([0.0, 0.0])), 0.0)


def test_vorticity_streamfunction_assembly_has_coupled_poisson_rows():
    pde = TaylorGreenVortex2D(T=0.05)
    basis = _tiny_basis(pde)
    collocation = sample_collocation_2d(pde, n_int=5, n_ic=4, n_bc=3, seed=402)
    gauge_times = np.linspace(0.0, pde.T, 3)
    weights = default_vorticity_flow_weights(collocation, n_gauge=len(gauge_times))
    matrix, target = assemble_vorticity_streamfunction_picard_2d(
        pde,
        basis,
        collocation,
        previous=_ExactTaylorGreen(pde),
        weights=weights,
        gauge_times=gauge_times,
    )

    expected_rows = 2 * collocation.n_int + 2 * collocation.n_ic + 2 * collocation.n_bc + len(gauge_times)
    assert matrix.shape == (expected_rows, 2 * basis.n_features)
    assert target.shape == (expected_rows,)
    poisson_row = matrix[1]
    assert np.any(np.abs(poisson_row[: basis.n_features]) > 0.0)
    assert np.any(np.abs(poisson_row[basis.n_features :]) > 0.0)
    assert np.all(np.isfinite(matrix))


def test_spectral_reference_tracks_short_taylor_green_decay():
    pde = TaylorGreenVortex2D(viscosity=0.02, T=0.02)
    reference = solve_vorticity_reference_2d(pde, nx=16, ny=16, nt=3, dt_max=1.0e-3)
    xx, yy = np.meshgrid(reference.x, reference.y, indexing="ij")
    xy = np.stack([xx, yy], axis=-1)
    exact = pde.exact_vorticity(xy, pde.T)
    relative_error = np.linalg.norm((reference.omega[-1] - exact).ravel()) / np.linalg.norm(exact.ravel())

    assert relative_error < 1.0e-8
    assert np.all(np.diff(reference.kinetic_energy) < 0.0)
    assert np.all(np.diff(reference.enstrophy) < 0.0)


def test_tiny_taylor_green_picard_solve_is_finite():
    pde = TaylorGreenVortex2D(viscosity=0.05, T=0.03)
    basis = _tiny_basis(pde, m_features=3, seed=403)
    collocation = sample_collocation_2d(pde, n_int=28, n_ic=14, n_bc=4, seed=404)
    result = solve_vorticity_streamfunction_picard_2d(
        pde,
        basis,
        collocation,
        gauge_times=np.linspace(0.0, pde.T, 3),
        max_iters=1,
        monitor_nx=5,
        monitor_ny=5,
        monitor_nt=2,
        ridge_lambda=1.0e-10,
    )
    points = np.asarray([[0.3, 0.7], [1.1, 2.0]])

    assert result.coefficients.shape == (2 * basis.n_features,)
    assert np.all(np.isfinite(result.solution.velocity(points, 0.01)))
    assert np.all(np.isfinite(result.solution.vorticity(points, 0.01)))


def test_double_shear_initial_vorticity_derivatives_are_consistent():
    pde = DoubleShearLayer2D()
    point = np.asarray([0.31, 0.37])
    step = 2.0e-5
    ex = np.asarray([step, 0.0])
    ey = np.asarray([0.0, step])
    grad_fd = np.asarray(
        [
            (pde.initial_vorticity(point + ex) - pde.initial_vorticity(point - ex)) / (2.0 * step),
            (pde.initial_vorticity(point + ey) - pde.initial_vorticity(point - ey)) / (2.0 * step),
        ]
    )
    lap_fd = (
        pde.initial_vorticity(point + ex)
        + pde.initial_vorticity(point - ex)
        + pde.initial_vorticity(point + ey)
        + pde.initial_vorticity(point - ey)
        - 4.0 * pde.initial_vorticity(point)
    ) / step**2

    np.testing.assert_allclose(pde.initial_vorticity_gradient(point), grad_fd, rtol=2.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(pde.initial_vorticity_laplacian(point), lap_fd, rtol=2.0e-4, atol=2.0e-3)


def test_hard_initial_lift_recovers_double_shear_initial_fields():
    pde = DoubleShearLayer2D(T=0.02)
    basis = _tiny_basis(pde, m_features=2, seed=405)
    collocation = sample_collocation_2d(pde, n_int=12, n_ic=6, n_bc=3, seed=406)
    result = solve_vorticity_streamfunction_picard_2d(
        pde,
        basis,
        collocation,
        max_iters=1,
        monitor_nx=4,
        monitor_ny=4,
        monitor_nt=2,
        ridge_lambda=1.0e-10,
        hard_initial=True,
    )
    points = np.asarray([[0.13, 0.24], [0.41, 0.77], [0.85, 0.52]])

    np.testing.assert_allclose(result.solution.velocity(points, 0.0), pde.initial_velocity(points))
    np.testing.assert_allclose(result.solution.vorticity(points, 0.0), pde.initial_vorticity(points))


def test_block_marching_preserves_terminal_trace_at_restart():
    pde = TaylorGreenVortex2D(viscosity=0.05, T=0.04)

    def basis_factory(block_pde, block_index):
        return _tiny_basis(block_pde, m_features=2, seed=410 + block_index)

    def collocation_factory(block_pde, block_index):
        return sample_collocation_2d(
            block_pde,
            n_int=20,
            n_ic=1,
            n_bc=3,
            seed=420 + block_index,
        )

    result = solve_vorticity_streamfunction_block_marching_2d(
        pde,
        basis_factory=basis_factory,
        collocation_factory=collocation_factory,
        n_blocks=2,
        n_gauge=3,
        max_iters=1,
        monitor_nx=4,
        monitor_ny=4,
        monitor_nt=2,
        ridge_lambda=1.0e-10,
    )
    points = np.asarray([[0.3, 0.7], [1.1, 2.0]])
    first, second = result.solution.block_solutions

    np.testing.assert_allclose(second.streamfunction(points, 0.0), first.streamfunction(points, 0.02))
    np.testing.assert_allclose(second.velocity(points, 0.0), first.velocity(points, 0.02))
    np.testing.assert_allclose(second.vorticity(points, 0.0), first.vorticity(points, 0.02))
    np.testing.assert_allclose(result.solution.velocity(points, 0.02), second.velocity(points, 0.0))
    assert len(result.block_results) == 2
    assert np.all(np.isfinite(result.solution.vorticity(points, pde.T)))
