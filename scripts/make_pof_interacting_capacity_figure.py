# 中文注释：生成PoF 论文 interacting capacity figure，并输出可复核的实验或制图结果。
"""Plot the interacting-vortex feature-capacity study for the PoF manuscript."""

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


METHODS = [
    "rs_ic_global",
    "rs_ic_local",
    "rfm_global",
    "rfm_local",
    "sep_rfm_global",
    "sep_rfm_local",
]
LABELS = {
    "rs_ic_global": "RS-RFM-Global",
    "rs_ic_local": "RS-RFM-Local",
    "rfm_global": "RFM-Global",
    "rfm_local": "RFM-Local",
    "sep_rfm_global": "Sep-RFM-Global",
    "sep_rfm_local": "Sep-RFM-Local",
}
COLORS = {
    "rs_ic_global": "#0072B2",
    "rs_ic_local": "#0072B2",
    "rfm_global": "#D55E00",
    "rfm_local": "#D55E00",
    "sep_rfm_global": "#009E73",
    "sep_rfm_local": "#009E73",
}
MARKERS = {
    "rs_ic_global": "o",
    "rs_ic_local": "s",
    "rfm_global": "^",
    "rfm_local": "v",
    "sep_rfm_global": "D",
    "sep_rfm_local": "P",
}
LINESTYLES = {
    "rs_ic_global": "-",
    "rs_ic_local": "--",
    "rfm_global": "-",
    "rfm_local": "--",
    "sep_rfm_global": "-",
    "sep_rfm_local": "--",
}
METRICS = [
    ("rel_l2_velocity", r"$\mathcal{E}_{\mathbf{u}}$", "Velocity error"),
    ("rel_l2_vorticity", r"$\mathcal{E}_{\omega}$", "Vorticity error"),
]
FEATURE_COUNTS = [96, 128, 192, 256, 384, 512, 768]
SEEDS = [10, 11, 12, 13, 14]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    setup_style(journal="general", lang="en", constrained_layout=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def load_and_validate(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = raw[
        (raw["study"] == "feature_budget")
        & (raw["case"] == "interacting_vortex")
        & raw["method_key"].isin(METHODS)
    ].copy()
    expected_rows = len(METHODS) * len(FEATURE_COUNTS) * len(SEEDS)
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} records, found {len(frame)}")
    if sorted(frame["features_per_field"].astype(int).unique().tolist()) != FEATURE_COUNTS:
        raise ValueError("feature-count grid is incomplete")
    if sorted(frame["seed"].astype(int).unique().tolist()) != SEEDS:
        raise ValueError("seed grid is incomplete")
    counts = frame.groupby(["features_per_field", "method_key"]).size()
    if not (counts == len(SEEDS)).all():
        raise ValueError("each method and feature count must contain five runs")
    for metric, _, _ in METRICS:
        if frame[metric].isna().any() or (frame[metric] <= 0.0).any():
            raise ValueError(f"{metric} must be complete and positive for log plotting")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["features_per_field", "method_key"], as_index=False)
        .agg(
            mean_rel_l2_velocity=("rel_l2_velocity", "mean"),
            std_rel_l2_velocity=("rel_l2_velocity", "std"),
            mean_rel_l2_vorticity=("rel_l2_vorticity", "mean"),
            std_rel_l2_vorticity=("rel_l2_vorticity", "std"),
        )
        .sort_values(["method_key", "features_per_field"])
    )


def add_method_curve(
    ax: plt.Axes,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    method: str,
) -> None:
    raw_method = raw[raw["method_key"] == method]
    mean_method = summary[summary["method_key"] == method]
    x_raw = raw_method["features_per_field"].to_numpy(dtype=float)
    y_raw = raw_method[metric].to_numpy(dtype=float)
    ax.scatter(
        x_raw,
        y_raw,
        s=8,
        marker=MARKERS[method],
        color=COLORS[method],
        edgecolor="none",
        alpha=0.22,
        zorder=1,
    )
    x = mean_method["features_per_field"].to_numpy(dtype=float)
    mean = mean_method[f"mean_{metric}"].to_numpy(dtype=float)
    sd = mean_method[f"std_{metric}"].to_numpy(dtype=float)
    display_floor = 0.5 * raw.loc[raw[metric] > 0.0, metric].min()
    lower = np.maximum(mean - sd, display_floor)
    upper = mean + sd
    ax.fill_between(x, lower, upper, color=COLORS[method], alpha=0.08, linewidth=0.0)
    ax.plot(
        x,
        mean,
        color=COLORS[method],
        marker=MARKERS[method],
        linestyle=LINESTYLES[method],
        linewidth=1.05,
        markersize=3.4,
        markeredgecolor="black",
        markeredgewidth=0.25,
        label=LABELS[method],
        zorder=3,
    )


def make_figure(raw: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize(raw)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for ax, (metric, ylabel, title) in zip(axes, METRICS, strict=True):
        for method in METHODS:
            add_method_curve(ax, raw, summary, metric, method)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(FEATURE_COUNTS)
        ax.set_xticklabels([str(value) for value in FEATURE_COUNTS])
        ax.set_xlabel("Features per field $P$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    panel_labels = add_panel_labels(
        fig,
        axes=axes,
        style="paren",
        x_offset_pt=2.0,
        y_offset_pt=-2.0,
    )
    for label in panel_labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.4})

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig03_interacting_capacity"
    finalize_figure(fig)
    render_preview(fig, str(base.with_name(base.name + "_preview.png")), dpi=170)
    if print_report(audit_layout(fig)) == "FAIL":
        raise RuntimeError("layout audit failed")
    export_figure(
        fig,
        str(base),
        formats=("pdf", "svg", "png"),
        size_inches=(7.0, 3.0),
        dpi=300,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_style()
    raw = load_and_validate(args.metrics.resolve())
    make_figure(raw, args.paper.resolve() / "figures")
    print(f"validated_records={len(raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
