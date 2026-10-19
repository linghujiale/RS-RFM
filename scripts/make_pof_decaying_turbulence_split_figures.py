# 中文注释：生成非制造衰减流的场对比图与积分量误差图。
"""Split the decaying-turbulence field and integral diagnostics into two figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SKILL_SCRIPTS = PROJECT_ROOT / "figure_style"
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from layout_tools import finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--paper", type=Path, default=Path("paper_pof"))
    parser.add_argument("--method-label", default="RS-RFM-Local")
    return parser.parse_args()


def configure_style() -> None:
    setup_style(journal="general", lang="en", use_sciplots=False, constrained_layout=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 20.0,
            "axes.labelsize": 20.0,
            "axes.titlesize": 20.0,
            "xtick.labelsize": 18.0,
            "ytick.labelsize": 18.0,
            "legend.fontsize": 18.0,
            "axes.linewidth": 1.15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def qa_and_export(fig, base: Path, size: tuple[float, float]) -> None:
    finalize_figure(fig)
    render_preview(fig, str(base.with_name(base.name + "_preview.png")), dpi=150)
    if print_report(audit_layout(fig)) == "FAIL":
        raise RuntimeError(f"layout audit failed for {base.name}")
    export_figure(
        fig,
        str(base),
        formats=("pdf", "svg", "png"),
        size_inches=size,
        dpi=300,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)


def load_fields(reference_path: Path, prediction_path: Path) -> dict[str, np.ndarray]:
    with np.load(reference_path) as archive:
        data = {
            "x": archive["x"].copy(),
            "y": archive["y"].copy(),
            "t": archive["t"].copy(),
            "reference_omega": archive["omega"].copy(),
            "reference_energy": archive["kinetic_energy"].copy(),
            "reference_enstrophy": archive["enstrophy"].copy(),
        }
    with np.load(prediction_path) as archive:
        prediction_t = archive["t"].copy()
        if not np.allclose(data["t"], prediction_t, rtol=0.0, atol=1.0e-14):
            raise ValueError("reference and prediction time grids differ")
        data.update(
            {
                "prediction_omega": archive["omega"].copy(),
                "prediction_energy": archive["kinetic_energy"].copy(),
                "prediction_enstrophy": archive["enstrophy"].copy(),
            }
        )
    if data["reference_omega"].shape != data["prediction_omega"].shape:
        raise ValueError("reference and prediction vorticity arrays differ in shape")
    return data


def time_label(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_field_axis(ax, *, show_x: bool, show_y: bool) -> None:
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    if show_x:
        ax.set_xlabel("$x$")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_y:
        ax.set_ylabel("$y$")
    else:
        ax.tick_params(axis="y", labelleft=False)


def make_field_figure(data: dict[str, np.ndarray], figure_dir: Path) -> None:
    x = data["x"]
    y = data["y"]
    time_values = data["t"]
    reference_omega = data["reference_omega"]
    prediction_omega = data["prediction_omega"]
    selected = (0, len(time_values) // 2, len(time_values) - 1)
    scale = float(np.max(np.abs(reference_omega)))
    normalized_errors = np.abs(prediction_omega[list(selected)] - reference_omega[list(selected)]) / scale
    error_max = max(float(normalized_errors.max()), 1.0e-6)
    extent = (
        float(x[0]),
        float(x[-1] + (x[1] - x[0])),
        float(y[0]),
        float(y[-1] + (y[1] - y[0])),
    )

    fig = plt.figure(figsize=(12.2, 7.75), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.050),
        wspace=0.035,
        hspace=0.08,
    )
    field_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    error_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    field_cax = fig.add_subplot(grid[0, 3])
    error_cax = fig.add_subplot(grid[1, 3])

    field_image = None
    error_image = None
    for column, index in enumerate(selected):
        field_image = field_axes[column].imshow(
            (reference_omega[index] / scale).T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        field_axes[column].set_title(
            rf"DNS, $t={time_label(float(time_values[index]))}$",
            loc="center", fontweight="normal", pad=8.0,
        )
        format_field_axis(field_axes[column], show_x=False, show_y=column == 0)

        error_image = error_axes[column].imshow(
            normalized_errors[column].T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=error_max,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        error_axes[column].set_title(
            rf"Normalized error, $t={time_label(float(time_values[index]))}$",
            loc="center",
            fontweight="normal", pad=8.0,
        )
        format_field_axis(error_axes[column], show_x=True, show_y=column == 0)

    field_colorbar = fig.colorbar(field_image, cax=field_cax)
    field_colorbar.set_label(
        r"$\omega/\max_{\mathbf{x},t}|\omega_{\rm DNS}|$", fontsize=20.0
    )
    error_colorbar = fig.colorbar(error_image, cax=error_cax)
    error_colorbar.set_label(
        r"$|\omega_P-\omega_{\rm DNS}|/\max_{\mathbf{x},t}|\omega_{\rm DNS}|$",
        fontsize=20.0,
    )
    for colorbar in (field_colorbar, error_colorbar):
        colorbar.ax.tick_params(direction="in", length=3.0, labelsize=17.0)

    qa_and_export(fig, figure_dir / "fig08_decaying_turbulence_fields_local", (12.2, 7.75))
    print("selected_time_indices", selected)
    print("selected_times", [float(time_values[index]) for index in selected])
    print("shared_error_max", error_max)


def kinetic_energy_spectrum(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = omega.shape[0]
    omega_hat = np.fft.fft2(omega) / (n * n)
    frequencies = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(frequencies, frequencies, indexing="ij")
    k_squared = kx**2 + ky**2
    shell = np.rint(np.sqrt(k_squared)).astype(int)
    density = np.zeros_like(k_squared, dtype=float)
    nonzero = k_squared > 0.0
    density[nonzero] = 0.5 * np.abs(omega_hat[nonzero]) ** 2 / k_squared[nonzero]
    spectrum = np.bincount(shell.ravel(), weights=density.ravel(), minlength=n // 2 + 1)
    wave_number = np.arange(len(spectrum))
    return wave_number[1 : n // 3 + 1], spectrum[1 : n // 3 + 1]


def make_integral_figure(data: dict[str, np.ndarray], figure_dir: Path, method_label: str) -> None:
    time_values = data["t"]
    reference_omega = data["reference_omega"]
    prediction_omega = data["prediction_omega"]
    reference_energy = data["reference_energy"]
    prediction_energy = data["prediction_energy"]
    reference_enstrophy = data["reference_enstrophy"]
    prediction_enstrophy = data["prediction_enstrophy"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.20),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.16},
    )
    energy_ax, spectrum_ax = axes
    short_label = method_label.replace("RS-RFM-", "")
    for reference, prediction, color, marker, quantity in (
        (reference_energy, prediction_energy, "#0072B2", "o", "$K$"),
        (reference_enstrophy, prediction_enstrophy, "#D55E00", "s", "$Z$"),
    ):
        energy_ax.plot(
            time_values,
            reference / reference[0],
            color=color,
            linewidth=2.85,
            label=f"DNS {quantity}",
        )
        energy_ax.plot(
            time_values,
            prediction / reference[0],
            color=color,
            linestyle="--",
            marker=marker,
            markersize=8.0,
            markevery=2,
            linewidth=2.55,
            label=f"{short_label} {quantity}",
        )
    integral_curves = (
        reference_energy / reference_energy[0],
        prediction_energy / reference_energy[0],
        reference_enstrophy / reference_enstrophy[0],
        prediction_enstrophy / reference_enstrophy[0],
    )
    energy_ax.set_xlabel("Time", fontsize=24.0)
    energy_ax.set_ylabel(r"$K(t)/K(0),\ Z(t)/Z(0)$", fontsize=24.0)
    energy_ax.set_title("Integral decay", loc="center", fontweight="normal", fontsize=24.0, pad=9.0)
    energy_ax.set_ylim(max(0.0, min(float(np.min(curve)) for curve in integral_curves) - 0.04), 1.025)
    energy_ax.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        fontsize=18.0,
        handlelength=1.55,
        columnspacing=0.90,
        handletextpad=0.55,
    )

    reference_initial_k, reference_initial_spectrum = kinetic_energy_spectrum(reference_omega[0])
    reference_final_k, reference_final_spectrum = kinetic_energy_spectrum(reference_omega[-1])
    prediction_final_k, prediction_final_spectrum = kinetic_energy_spectrum(prediction_omega[-1])
    spectrum_normalizer = float(reference_initial_spectrum.sum())
    for wave_number, spectrum, color, linestyle, marker, label in (
        (reference_initial_k, reference_initial_spectrum, "#7F7F7F", ":", None, "DNS, $t=0$"),
        (reference_final_k, reference_final_spectrum, "black", "-", None, "DNS, $t=T$"),
        (
            prediction_final_k,
            prediction_final_spectrum,
            "#0072B2",
            "--",
            "o",
            f"{short_label}, $t=T$",
        ),
    ):
        positive = spectrum > 1.0e-15 * spectrum_normalizer
        spectrum_ax.plot(
            wave_number[positive],
            spectrum[positive] / spectrum_normalizer,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=8.0,
            linewidth=2.55,
            label=label,
        )
    spectrum_ax.set_yscale("log")
    spectrum_ax.set_xlabel("Wave number $k$", fontsize=24.0)
    spectrum_ax.set_ylabel(r"$E(k,t)/K(0)$", fontsize=24.0)
    spectrum_ax.set_title(
        "Energy spectrum", loc="center", fontweight="normal", fontsize=24.0, pad=9.0
    )
    spectrum_ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=18.0,
        handlelength=1.55,
        handletextpad=0.55,
    )

    for axis in axes:
        axis.tick_params(axis="both", labelsize=21.5)
        axis.set_box_aspect(3.0 / 4.0)

    qa_and_export(fig, figure_dir / "fig09_decaying_turbulence_integrals_local", (12.2, 6.20))


def main() -> int:
    args = parse_args()
    reference_path = args.reference.resolve()
    prediction_path = args.prediction.resolve()
    paper = args.paper.resolve()
    figure_dir = paper / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = load_fields(reference_path, prediction_path)
    make_field_figure(data, figure_dir)
    make_integral_figure(data, figure_dir, args.method_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
