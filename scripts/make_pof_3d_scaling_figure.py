# 中文注释：生成PoF 论文 三维规模分析图，并输出可复核的实验或制图结果。
"""Build the three-dimensional feature-scaling figure from completed archives.

The script reads the fixed P=1536 and P=2048 production archives together
with the P=3072 and P=4096 scaling runs.  It validates five independent runs
for RS-RFM-Global and Sep-RFM-Local at every feature count before plotting.
No solver is rerun and no metric is recomputed from prediction fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SKILL_SCRIPTS = PROJECT_ROOT / "figure_style"
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from layout_tools import add_panel_labels, finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


FEATURE_COUNTS = (1536, 2048, 3072, 4096)
METHODS = ("rs_ic_global", "strfm_sov")
METHOD_LABELS = {
    "rs_ic_global": "RS-RFM-Global",
    "strfm_sov": "Sep-RFM-Local",
}
COLORS = {
    "rs_ic_global": "#0072B2",
    "strfm_sov": "#009E73",
}
MARKERS = {
    "rs_ic_global": "o",
    "strfm_sov": "D",
}
LINESTYLES = {
    "rs_ic_global": "-",
    "strfm_sov": "--",
}
METRICS = (
    ("rel_l2_velocity", r"Velocity error $\mathcal{E}_{\mathbf{u}}$"),
    ("rel_l2_vorticity", r"Vorticity error $\mathcal{E}_{\omega}$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p1536-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_pof"
            / "results"
            / "helical_vortex_3d_p1536_allmethods_20260821"
        ),
    )
    parser.add_argument(
        "--p2048-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_pof"
            / "results"
            / "helical_vortex_3d_p2048_fixed_20260821"
        ),
    )
    parser.add_argument(
        "--scaling-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_pof"
            / "results"
            / "helical_vortex_3d_scaling_20260825"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_pof" / "figures",
    )
    return parser.parse_args()


def read_metric_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("no metric files found")
    return pd.concat(frames, ignore_index=True, sort=False)


def collect_records(p1536_root: Path, p2048_root: Path, scaling_root: Path) -> pd.DataFrame:
    paths: list[Path] = []
    for method in METHODS:
        paths.append(p1536_root / "results" / method / "metrics_raw.csv")
        paths.append(p2048_root / "results" / method / "metrics_raw.csv")
        paths.extend(sorted((p2048_root / "results_parallel").glob(f"{method}_seed*/metrics_raw.csv")))
    paths.extend(sorted(scaling_root.rglob("metrics_raw.csv")))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing metric files: {missing}")

    frame = read_metric_files(paths)
    frame = frame[
        frame["method_key"].isin(METHODS)
        & frame["features_per_field"].astype(int).isin(FEATURE_COUNTS)
    ].copy()
    frame["features_per_field"] = frame["features_per_field"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    frame = frame.sort_values("source_file").drop_duplicates(
        ["features_per_field", "method_key", "seed"], keep="last"
    )

    counts = frame.groupby(["features_per_field", "method_key"]).size()
    expected_index = pd.MultiIndex.from_product(
        [FEATURE_COUNTS, METHODS], names=["features_per_field", "method_key"]
    )
    counts = counts.reindex(expected_index, fill_value=0)
    if not (counts == 5).all():
        raise ValueError(f"expected five runs per feature count and method:\n{counts}")
    for metric, _ in METRICS:
        if frame[metric].isna().any() or (frame[metric] <= 0.0).any():
            raise ValueError(f"{metric} must be complete and positive")
    return frame.reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["features_per_field", "method_key"], as_index=False)
        .agg(
            n=("seed", "count"),
            velocity_mean=("rel_l2_velocity", "mean"),
            velocity_sd=("rel_l2_velocity", "std"),
            vorticity_mean=("rel_l2_vorticity", "mean"),
            vorticity_sd=("rel_l2_vorticity", "std"),
        )
        .sort_values(["method_key", "features_per_field"])
    )


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
            "legend.fontsize": 13.5,
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


def make_figure(frame: pd.DataFrame, figure_dir: Path) -> pd.DataFrame:
    summary = summarize(frame)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.20), constrained_layout=True)
    seed_jitter = np.linspace(-0.010, 0.010, 5)
    method_offset = {"rs_ic_global": 0.985, "strfm_sov": 1.015}

    for ax, (metric, title) in zip(axes, METRICS):
        mean_column = "velocity_mean" if metric == "rel_l2_velocity" else "vorticity_mean"
        sd_column = "velocity_sd" if metric == "rel_l2_velocity" else "vorticity_sd"
        for method in METHODS:
            method_raw = frame[frame["method_key"] == method]
            method_summary = summary[summary["method_key"] == method]
            for feature_count in FEATURE_COUNTS:
                values = (
                    method_raw[method_raw["features_per_field"] == feature_count]
                    .sort_values("seed")[metric]
                    .to_numpy(float)
                )
                ax.scatter(
                    feature_count * method_offset[method] * (1.0 + seed_jitter),
                    values,
                    s=42,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    edgecolor="white",
                    linewidth=0.65,
                    alpha=0.32,
                    zorder=2,
                )
            x = method_summary["features_per_field"].to_numpy(float)
            mean = method_summary[mean_column].to_numpy(float)
            sd = method_summary[sd_column].to_numpy(float)
            ax.errorbar(
                x * method_offset[method],
                mean,
                yerr=sd,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.0,
                markersize=7.0,
                markeredgecolor="black",
                markeredgewidth=0.65,
                elinewidth=1.4,
                capsize=3.8,
                capthick=1.4,
                label=METHOD_LABELS[method],
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(FEATURE_COUNTS, [str(value) for value in FEATURE_COUNTS])
        ax.set_xlabel("Features per field $P$")
        ax.set_ylabel("Relative error")
        ax.set_title(title, loc="center", fontweight="bold", pad=7.0)
    axes[1].legend(frameon=False, loc="upper right")

    labels = add_panel_labels(
        fig, axes=axes, style="paren", x_offset_pt=4.0, y_offset_pt=-4.0
    )
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.45})

    figure_dir.mkdir(parents=True, exist_ok=True)
    base = figure_dir / "fig07_helical_vortex_3d_scaling"
    finalize_figure(fig)
    render_preview(fig, str(base.with_name(base.name + "_preview.png")), dpi=160)
    if print_report(audit_layout(fig)) == "FAIL":
        raise RuntimeError("layout audit failed for three-dimensional scaling figure")
    export_figure(
        fig,
        str(base),
        formats=("pdf", "svg", "png"),
        size_inches=(12.2, 4.20),
        dpi=300,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)
    return summary


def main() -> int:
    args = parse_args()
    frame = collect_records(
        args.p1536_root.resolve(), args.p2048_root.resolve(), args.scaling_root.resolve()
    )
    configure_style()
    summary = make_figure(frame, args.figure_dir.resolve())
    output = args.scaling_root.resolve() / "aggregated"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics_all.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
