# 中文注释：从冻结场数据生成二维算例及三维螺旋涡的解场图。
"""Create per-case solution-field figures from the frozen PoF archives.

The four analytic two-dimensional panels read the cached held-out seed-10
reference and RS-RFM-Global fields.  The three-dimensional panel reads the
fixed-P=2048 seed-100 archive.  This script does not rerun a solver.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SKILL_SCRIPTS = PROJECT_ROOT / "figure_style"
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from layout_tools import add_panel_labels, finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


CASE_SPECS = (
    ("tgv_order1_decay", "Taylor--Green", "fig_taylor_green_solution"),
    ("two_rate_physics_range", "Forced two-rate vortex", "fig_two_rate_solution"),
    (
        "interacting_two_rate_vortex",
        "Interacting-mode vortex",
        "fig_interacting_modes_solution",
    ),
    (
        "multiscale_shear_decay_k12",
        "Multiscale shear diffusion",
        "fig_multiscale_shear_solution",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper",
        type=Path,
        default=PROJECT_ROOT / "paper_pof",
    )
    parser.add_argument(
        "--two-d-cache",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_pof"
            / "results"
            / "rsrfm_fluid_production_final"
            / "representative_fields_seed10.npz"
        ),
    )
    parser.add_argument(
        "--three-d-reference",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_pof"
            / "results"
            / "helical_vortex_3d_p2048_fixed_20260821"
            / "results"
            / "rs_ic_global"
            / "reference_helical_vortex_3d.npz"
        ),
    )
    parser.add_argument(
        "--three-d-prediction",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_pof"
            / "results"
            / "helical_vortex_3d_p2048_fixed_20260821"
            / "results"
            / "rs_ic_global"
            / "prediction_helical_vortex_3d_rs_ic_global_seed100.npz"
        ),
    )
    return parser.parse_args()


def configure_style() -> None:
    setup_style(journal="general", lang="en", use_sciplots=False, constrained_layout=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 16.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
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
    preview = base.with_name(base.name + "_preview.png")
    render_preview(fig, str(preview), dpi=150)
    verdict = print_report(audit_layout(fig))
    if verdict == "FAIL":
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


def periodic_extent(values: np.ndarray) -> tuple[float, float]:
    step = float(values[1] - values[0])
    return float(values[0]), float(values[-1] + step)


def make_two_dimensional_figures(cache: Path, figure_dir: Path) -> None:
    with np.load(cache) as archive:
        for case, title, filename in CASE_SPECS:
            x = archive[f"{case}_x"].copy()
            y = archive[f"{case}_y"].copy()
            reference = archive[f"{case}_reference"].copy()
            prediction = archive[f"{case}_prediction"].copy()

            scale = float(np.max(np.abs(reference)))
            reference_normalized = reference / scale
            prediction_normalized = prediction / scale
            error = np.abs(prediction - reference) / scale
            positive_error = error[error > 0.0]
            error_max = max(float(error.max()), np.finfo(float).tiny)
            error_vmax = 10.0 ** math.ceil(math.log10(error_max))
            error_floor = (
                float(np.quantile(positive_error, 0.01))
                if positive_error.size
                else error_vmax * 1.0e-4
            )
            error_vmin = max(min(error_floor, error_vmax * 1.0e-2), error_vmax * 1.0e-5)

            fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), constrained_layout=True)
            signed_image = None
            error_image = None
            for column, (field, panel_title) in enumerate(
                (
                    (reference_normalized, "Reference"),
                    (prediction_normalized, "RS-RFM-Global"),
                    (np.maximum(error, error_vmin), "Normalized absolute error"),
                )
            ):
                ax = axes[column]
                if column < 2:
                    signed_image = ax.pcolormesh(
                        x,
                        y,
                        field.T,
                        shading="auto",
                        cmap="RdBu_r",
                        vmin=-1.0,
                        vmax=1.0,
                    )
                else:
                    error_image = ax.pcolormesh(
                        x,
                        y,
                        field.T,
                        shading="auto",
                        cmap="magma",
                        norm=LogNorm(vmin=error_vmin, vmax=error_vmax),
                    )
                ax.set_aspect("equal")
                ax.set_title(panel_title, loc="left", fontweight="bold")
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(r"$y$" if column == 0 else "")
                ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
                ax.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])

            signed_bar = fig.colorbar(
                signed_image,
                ax=axes[:2],
                orientation="horizontal",
                shrink=0.74,
                pad=0.10,
            )
            signed_bar.solids.set_rasterized(False)
            signed_bar.set_label(r"Normalized vorticity $\omega/\max|\omega_*|$")
            error_bar = fig.colorbar(
                error_image,
                ax=axes[2],
                orientation="horizontal",
                shrink=0.88,
                pad=0.10,
            )
            error_bar.solids.set_rasterized(False)
            error_bar.set_label(r"$|\omega_P-\omega_*|/\max|\omega_*|$")
            labels = add_panel_labels(
                fig,
                axes=axes,
                style="paren",
                x_offset_pt=2.0,
                y_offset_pt=-2.0,
                ha="left",
                va="top",
            )
            for label in labels:
                label.set_bbox(
                    {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5}
                )
            fig.suptitle(title, fontsize=9.0, fontweight="bold")
            qa_and_export(fig, figure_dir / filename, (7.0, 2.55))


def make_three_dimensional_figure(
    reference_path: Path,
    prediction_path: Path,
    figure_dir: Path,
) -> None:
    title_size = 20.0
    label_size = 20.0
    tick_size = 18.0
    colorbar_size = 18.0

    with np.load(reference_path) as archive:
        x = archive["x"].copy()
        y = archive["y"].copy()
        z = archive["z"].copy()
        reference = archive["vorticity"][-1].copy()
    with np.load(prediction_path) as archive:
        prediction = archive["vorticity"][-1].copy()

    planes = (
        (
            "x",
            y,
            z,
            reference[len(x) // 2, :, :],
            prediction[len(x) // 2, :, :],
            r"$y$",
            r"$z$",
        ),
        (
            "y",
            x,
            z,
            reference[:, len(y) // 2, :],
            prediction[:, len(y) // 2, :],
            r"$x$",
            r"$z$",
        ),
        (
            "z",
            x,
            y,
            reference[:, :, len(z) // 2],
            prediction[:, :, len(z) // 2],
            r"$x$",
            r"$y$",
        ),
    )
    field_max = max(
        float(np.linalg.norm(field, axis=-1).max())
        for _, _, _, ref, pred, _, _ in planes
        for field in (ref, pred)
    )
    error_max = max(
        float(np.linalg.norm(pred - ref, axis=-1).max())
        for _, _, _, ref, pred, _, _ in planes
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.75), constrained_layout=True)
    field_image = None
    error_image = None
    for column, (
        normal_axis,
        axis_a,
        axis_b,
        ref,
        pred,
        horizontal_label,
        vertical_label,
    ) in enumerate(planes):
        fields = (
            (np.linalg.norm(pred, axis=-1), "RS-RFM-Global", False),
            (np.linalg.norm(pred - ref, axis=-1), "Pointwise error", True),
        )
        for row, (field, title, is_error) in enumerate(fields):
            ax = axes[row, column]
            image = ax.imshow(
                field.T,
                origin="lower",
                extent=(*periodic_extent(axis_a), *periodic_extent(axis_b)),
                cmap="magma" if is_error else "viridis",
                vmin=0.0,
                vmax=error_max if is_error else field_max,
                interpolation="bilinear",
                aspect="equal",
                rasterized=True,
            )
            if is_error:
                error_image = image
            else:
                field_image = image
            ax.set_title(
                f"{title}, ${normal_axis}=\\pi$",
                loc="center",
                fontsize=title_size,
                fontweight="bold",
                pad=8.0,
            )
            ax.set_xlabel(horizontal_label, fontsize=label_size, labelpad=2.0)
            ax.set_ylabel(vertical_label, fontsize=label_size, labelpad=2.0)
            ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
            ax.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=tick_size,
                length=5.0,
                width=1.15,
                pad=3.0,
            )
    field_bar = fig.colorbar(
        field_image, ax=axes[0, :], fraction=0.024, pad=0.015, aspect=28
    )
    field_bar.set_label(
        r"Vorticity magnitude $|\boldsymbol{\omega}|$",
        fontsize=colorbar_size,
        labelpad=8.0,
    )
    field_bar.locator = MaxNLocator(nbins=5)
    field_bar.update_ticks()
    field_bar.ax.tick_params(labelsize=colorbar_size, length=4.5, width=1.15)
    error_bar = fig.colorbar(
        error_image, ax=axes[1, :], fraction=0.024, pad=0.015, aspect=28
    )
    error_bar.set_label(
        r"$|\boldsymbol{\omega}_P-\boldsymbol{\omega}_*|$",
        fontsize=colorbar_size,
        labelpad=8.0,
    )
    error_bar.locator = MaxNLocator(nbins=5)
    error_bar.update_ticks()
    error_bar.ax.tick_params(labelsize=colorbar_size, length=4.5, width=1.15)
    qa_and_export(fig, figure_dir / "fig06_helical_vortex_3d_fields", (12.2, 7.75))


def main() -> int:
    args = parse_args()
    paper = args.paper.resolve()
    figure_dir = paper / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_two_dimensional_figures(args.two_d_cache.resolve(), figure_dir)
    make_three_dimensional_figure(
        args.three_d_reference.resolve(),
        args.three_d_prediction.resolve(),
        figure_dir,
    )
    print(f"wrote per-case solution figures to {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
