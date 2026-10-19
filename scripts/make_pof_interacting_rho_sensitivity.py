# 中文注释：生成静态锚定比例 rho_0 的敏感性图和机器可读汇总。
"""Build the interacting-vortex static-anchor sensitivity artifact.

The script combines the archived rho_0=0 and rho_0=1/4 records with the
newly completed rho_0=1/8 and rho_0=1/2 runs.  It validates the paired
four-by-five design, writes machine-readable summaries, and exports the
publication figure without rerunning the solver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SKILL_SCRIPTS = PROJECT_ROOT / "figure_style"
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from layout_tools import finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


EXPECTED_RHOS = (0.0, 0.125, 0.25, 0.5)
EXPECTED_SEEDS = (10, 11, 12, 13, 14)
METRICS = (
    ("rel_l2_velocity", r"Velocity error $\mathcal{E}_{\mathbf{u}}$", "#0072B2", "o"),
    ("rel_l2_vorticity", r"Vorticity error $\mathcal{E}_{\omega}$", "#D55E00", "s"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archived",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_pof"
            / "results"
            / "ablation_revision_20260824"
            / "aggregated"
            / "ablation_metrics_all.csv"
        ),
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_pof"
            / "results"
            / "interacting_rho_sensitivity_20260825"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_pof" / "figures",
    )
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def load_and_validate(archived_path: Path, new_root: Path) -> pd.DataFrame:
    archived = pd.read_csv(archived_path)
    archived = archived[
        (archived["case"] == "interacting_two_rate_vortex")
        & archived["variant"].isin(["rho_0", "full"])
    ].copy()

    new_paths = sorted(new_root.rglob("metrics_raw.csv"))
    if len(new_paths) != 10:
        raise ValueError(f"expected 10 new metric files, found {len(new_paths)}")
    new = pd.concat([pd.read_csv(path) for path in new_paths], ignore_index=True)
    frame = pd.concat([archived, new], ignore_index=True, sort=False)
    frame["rho_static"] = frame["rho_static"].astype(float)
    frame["seed"] = frame["seed"].astype(int)
    frame["P"] = frame["P"].astype(int)
    frame = frame.sort_values(["rho_static", "seed"]).reset_index(drop=True)

    observed = list(frame.groupby(["rho_static", "seed"], observed=True).size().items())
    expected = [((rho, seed), 1) for rho in EXPECTED_RHOS for seed in EXPECTED_SEEDS]
    if observed != expected:
        raise ValueError(f"unexpected rho/seed grid: {observed}")
    if set(frame["case"]) != {"interacting_two_rate_vortex"}:
        raise ValueError("mixed cases in rho sensitivity data")
    checks = {
        "P": {384},
        "tau_min_over_T": {0.08},
        "tau_max_over_T": {3.0},
        "sampling_mode": {"log_stratified"},
        "n_int": {1800},
        "n_bc_side": {70},
    }
    for column, expected_values in checks.items():
        values = set(frame[column].dropna())
        if values != expected_values:
            raise ValueError(f"{column} mismatch: {values}")
    converged = frame["picard_converged"].astype(str).str.lower().isin(["true", "1"])
    if not converged.all():
        failed = frame.loc[~converged, ["rho_static", "seed"]].to_dict("records")
        raise ValueError(f"nonconverged runs in sensitivity grid: {failed}")
    return frame


def write_summaries(frame: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    combined_path = output_root / "combined_metrics.csv"
    frame.to_csv(combined_path, index=False)

    summary = (
        frame.groupby("rho_static", observed=True)
        .agg(
            n=("seed", "count"),
            velocity_mean=("rel_l2_velocity", "mean"),
            velocity_sd=("rel_l2_velocity", "std"),
            vorticity_mean=("rel_l2_vorticity", "mean"),
            vorticity_sd=("rel_l2_vorticity", "std"),
            converged_count=("picard_converged", "count"),
        )
        .reset_index()
    )
    summary.to_csv(output_root / "summary.csv", index=False)
    return summary


def configure_style() -> None:
    setup_style(journal="general", lang="en", use_sciplots=False, constrained_layout=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 24.0,
            "axes.titlesize": 24.0,
            "axes.labelsize": 24.0,
            "xtick.labelsize": 21.5,
            "ytick.labelsize": 21.5,
            "legend.fontsize": 20.0,
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


def make_figure(frame: pd.DataFrame, figure_dir: Path) -> None:
    configure_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.15),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.16},
    )
    jitter = np.linspace(-0.012, 0.012, len(EXPECTED_SEEDS))

    all_values = frame[[metric for metric, _, _, _ in METRICS]].to_numpy(float)
    # Keep a shared log scale while avoiding a largely empty upper decade.
    lower = all_values.min() / 1.35
    upper = all_values.max() * 1.70
    y_ticks = (1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2)
    y_ticklabels = (
        r"$10^{-3}$",
        r"$2\times10^{-3}$",
        r"$5\times10^{-3}$",
        r"$10^{-2}$",
        r"$2\times10^{-2}$",
    )

    for ax, (metric, title, color, marker) in zip(axes, METRICS):
        means = []
        standard_deviations = []
        for rho in EXPECTED_RHOS:
            values = (
                frame[np.isclose(frame["rho_static"], rho)]
                .sort_values("seed")[metric]
                .to_numpy(float)
            )
            ax.scatter(
                rho + jitter,
                values,
                s=96,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.75,
                alpha=0.45,
                zorder=2,
            )
            means.append(float(values.mean()))
            standard_deviations.append(float(values.std(ddof=1)))
        ax.errorbar(
            EXPECTED_RHOS,
            means,
            yerr=standard_deviations,
            color=color,
            marker=marker,
            markersize=10.0,
            markeredgecolor="black",
            markeredgewidth=0.75,
            linewidth=2.85,
            elinewidth=2.15,
            capsize=5.35,
            capthick=2.15,
            zorder=3,
        )
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlim(-0.04, 0.54)
        ax.set_box_aspect(1.0)
        ax.set_xticks(EXPECTED_RHOS, ["0", r"$1/8$", r"$1/4$", r"$1/2$"])
        ax.set_xlabel(r"Static-anchor fraction $\rho_0$")
        ax.set_title(title, loc="center", fontweight="normal", pad=9.0)
    axes[0].set_ylabel("Relative error")
    for ax, (_, _, color, marker) in zip(axes, METRICS):
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.75,
                markersize=np.sqrt(96),
                alpha=0.45,
                label="Individual runs",
            ),
            Line2D(
                [0],
                [0],
                marker=marker,
                color=color,
                linewidth=2.85,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.75,
                markersize=10.0,
                label=r"Mean $\pm$ sample SD",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.015, 0.985),
            frameon=False,
            handlelength=1.6,
            borderaxespad=0.0,
        )

    finalize_figure(fig)

    base = figure_dir / "fig04_interacting_rho_sensitivity"
    preview = base.with_name(base.name + "_preview.png")
    render_preview(fig, str(preview), dpi=150)
    verdict = print_report(audit_layout(fig))
    if verdict == "FAIL":
        raise RuntimeError("layout audit failed for interacting rho sensitivity")
    export_figure(
        fig,
        str(base),
        formats=("pdf", "svg", "png"),
        size_inches=(12.2, 6.15),
        dpi=450,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    frame = load_and_validate(args.archived.resolve(), args.new_root.resolve())
    summary = write_summaries(frame, args.new_root.resolve())
    print(summary.to_string(index=False))
    if not args.data_only:
        make_figure(frame, args.figure_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
