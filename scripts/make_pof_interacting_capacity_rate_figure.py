# 中文注释：生成相互作用涡的字典容量与松弛率分辨率联合敏感性图。
"""Plot interacting-vortex capacity and physical-rate-separation studies."""

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
from layout_tools import finalize_figure  # noqa: E402
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
LEGACY_TO_CANONICAL = {
    "rs_ic_global": "rs_ic_global",
    "rs_ic_local": "rs_ic_local",
    "pielm": "rfm_global",
    "strfm_stc": "rfm_local",
    "sep_rfm_global": "sep_rfm_global",
    "strfm_sov": "sep_rfm_local",
}
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
RATE_RATIOS = [1, 2, 4, 8, 16]
SEEDS = [10, 11, 12, 13, 14]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-metrics", type=Path, required=True)
    parser.add_argument("--rate-metrics", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    setup_style(journal="general", lang="en", use_sciplots=False, constrained_layout=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 24.0,
            "axes.labelsize": 24.0,
            "axes.titlesize": 24.0,
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


def validate_grid(
    frame: pd.DataFrame,
    *,
    x_col: str,
    x_values: list[int],
    expected_rows: int,
) -> None:
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} records, found {len(frame)}")
    observed_x = sorted(frame[x_col].astype(int).unique().tolist())
    if observed_x != x_values:
        raise ValueError(f"incomplete {x_col} grid: {observed_x}")
    if sorted(frame["seed"].astype(int).unique().tolist()) != SEEDS:
        raise ValueError("seed grid is incomplete")
    counts = frame.groupby([x_col, "method_key"]).size()
    if len(counts) != len(x_values) * len(METHODS) or not (counts == len(SEEDS)).all():
        raise ValueError(f"each method and {x_col} value must contain five runs")
    for metric, _, _ in METRICS:
        if frame[metric].isna().any() or (frame[metric] <= 0.0).any():
            raise ValueError(f"{metric} must be complete and positive for log plotting")


def load_capacity(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = raw[
        (raw["study"] == "feature_budget")
        & (raw["case"] == "interacting_vortex")
        & raw["method_key"].isin(METHODS)
    ].copy()
    frame["features_per_field"] = frame["features_per_field"].astype(int)
    validate_grid(
        frame,
        x_col="features_per_field",
        x_values=FEATURE_COUNTS,
        expected_rows=len(METHODS) * len(FEATURE_COUNTS) * len(SEEDS),
    )
    return frame


def load_rates(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = raw[
        (raw["study"] == "scale_separation")
        & (raw["case"] == "interacting_vortex")
        & raw["method_key"].isin(LEGACY_TO_CANONICAL)
    ].copy()
    frame["legacy_method_key"] = frame["method_key"]
    frame["method_key"] = frame["method_key"].map(LEGACY_TO_CANONICAL)
    frame["point"] = frame["point"].astype(int)
    validate_grid(
        frame,
        x_col="point",
        x_values=RATE_RATIOS,
        expected_rows=len(METHODS) * len(RATE_RATIOS) * len(SEEDS),
    )
    for metric in ("early_rel_l2_velocity", "early_rel_l2_vorticity"):
        if metric not in frame or frame[metric].isna().any() or (frame[metric] <= 0.0).any():
            raise ValueError(f"missing early-time check: {metric}")
    return frame


def summarize(frame: pd.DataFrame, x_col: str) -> pd.DataFrame:
    return (
        frame.groupby([x_col, "method_key"], as_index=False)
        .agg(
            mean_rel_l2_velocity=("rel_l2_velocity", "mean"),
            std_rel_l2_velocity=("rel_l2_velocity", "std"),
            mean_rel_l2_vorticity=("rel_l2_vorticity", "mean"),
            std_rel_l2_vorticity=("rel_l2_vorticity", "std"),
        )
        .sort_values(["method_key", x_col])
    )


def add_method_curve(
    ax: plt.Axes,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metric: str,
    method: str,
    x_col: str,
) -> None:
    raw_method = raw[raw["method_key"] == method]
    mean_method = summary[summary["method_key"] == method]
    ax.scatter(
        raw_method[x_col].to_numpy(dtype=float),
        raw_method[metric].to_numpy(dtype=float),
        s=96,
        marker=MARKERS[method],
        facecolor=COLORS[method],
        edgecolor="white",
        linewidth=0.75,
        alpha=0.45,
        zorder=2,
    )
    x = mean_method[x_col].to_numpy(dtype=float)
    mean = mean_method[f"mean_{metric}"].to_numpy(dtype=float)
    sd = mean_method[f"std_{metric}"].to_numpy(dtype=float)
    display_floor = 0.5 * raw.loc[raw[metric] > 0.0, metric].min()
    lower = np.maximum(mean - sd, display_floor)
    upper = mean + sd
    ax.fill_between(x, lower, upper, color=COLORS[method], alpha=0.10, linewidth=0.0)
    ax.plot(
        x,
        mean,
        color=COLORS[method],
        marker=MARKERS[method],
        linestyle=LINESTYLES[method],
        linewidth=2.85,
        markersize=10.0,
        markeredgecolor="black",
        markeredgewidth=0.75,
        label=LABELS[method],
        zorder=3,
    )


def style_axis(
    ax: plt.Axes,
    *,
    x_values: list[int],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="center", fontweight="normal", pad=9.0)
    ax.tick_params(axis="y", labelleft=True)
    ax.set_box_aspect(3.0 / 4.0)


def make_figure(capacity: pd.DataFrame, rates: pd.DataFrame, output_dir: Path) -> None:
    capacity_summary = summarize(capacity, "features_per_field")
    rate_summary = summarize(rates, "point")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 10.4),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.16, "hspace": 0.14},
    )

    for col, (metric, ylabel, metric_title) in enumerate(METRICS):
        for method in METHODS:
            add_method_curve(
                axes[0, col], capacity, capacity_summary,
                metric=metric, method=method, x_col="features_per_field",
            )
            add_method_curve(
                axes[1, col], rates, rate_summary,
                metric=metric, method=method, x_col="point",
            )
        style_axis(
            axes[0, col],
            x_values=FEATURE_COUNTS,
            xlabel="Features per field $P$",
            ylabel="Relative error" if col == 0 else "",
            title=f"Capacity: {metric_title.lower()}",
        )
        style_axis(
            axes[1, col],
            x_values=RATE_RATIOS,
            xlabel=r"Physical rate ratio $\gamma_2/\gamma_1$",
            ylabel="Relative error" if col == 0 else "",
            title=f"Rate separation: {metric_title.lower()}",
        )

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        fontsize=18.0,
        frameon=False,
        handlelength=1.55,
        columnspacing=1.20,
        handletextpad=0.55,
        borderaxespad=0.35,
    )

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
        size_inches=(12.2, 10.4),
        dpi=450,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_style()
    capacity = load_capacity(args.capacity_metrics.resolve())
    rates = load_rates(args.rate_metrics.resolve())
    make_figure(capacity, rates, args.paper.resolve() / "figures")
    print(f"validated_capacity_records={len(capacity)}")
    print(f"validated_rate_records={len(rates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
