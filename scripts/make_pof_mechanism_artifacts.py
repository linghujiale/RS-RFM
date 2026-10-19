# 中文注释：生成PoF 论文 机制实验论文制品，并输出可复核的实验或制图结果。
"""Generate Section-IV mechanism figures and LaTeX from audited raw records."""

from __future__ import annotations

import argparse
import json
import math
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
    "rs_fixed_tau",
]
SCALING_METHODS = [
    "rs_ic_global",
    "rs_ic_local",
    "pielm",
    "strfm_stc",
    "strfm_sov",
]
LABELS = {
    "rs_ic_global": "RS-RFM-Global",
    "rs_ic_local": "RS-RFM-Local",
    "rfm_global": "RFM-Global",
    "rfm_local": "RFM-Local",
    "sep_rfm_global": "Sep-RFM-Global",
    "sep_rfm_local": "Sep-RFM-Local",
    "rs_fixed_tau": "Legacy Fixed-$\\tau$",
    "pielm": "RFM-Global",
    "strfm_stc": "RFM-Local",
    "strfm_sov": "Sep-RFM-Local",
}
PLAIN_LABELS = {
    "rs_ic_global": "RS-RFM-Global",
    "rs_ic_local": "RS-RFM-Local",
    "rfm_global": "RFM-Global",
    "rfm_local": "RFM-Local",
    "sep_rfm_global": "Sep-RFM-Global",
    "sep_rfm_local": "Sep-RFM-Local",
    "rs_fixed_tau": r"Legacy Fixed-$\tau$",
    "pielm": "RFM-Global",
    "strfm_stc": "RFM-Local",
    "strfm_sov": "Sep-RFM-Local",
}
COLORS = {
    "rs_ic_global": "#0072B2",
    "rs_ic_local": "#0072B2",
    "rfm_global": "#D55E00",
    "rfm_local": "#D55E00",
    "sep_rfm_global": "#009E73",
    "sep_rfm_local": "#009E73",
    "rs_fixed_tau": "#666666",
    "pielm": "#D55E00",
    "strfm_stc": "#D55E00",
    "strfm_sov": "#009E73",
}
MARKERS = {
    "rs_ic_global": "o",
    "rs_ic_local": "s",
    "rfm_global": "^",
    "rfm_local": "v",
    "sep_rfm_global": "D",
    "sep_rfm_local": "P",
    "rs_fixed_tau": "X",
    "pielm": "^",
    "strfm_stc": "v",
    "strfm_sov": "D",
}
LINESTYLES = {
    "rs_ic_global": "-",
    "rs_ic_local": "--",
    "rfm_global": "-",
    "rfm_local": "--",
    "sep_rfm_global": "-",
    "sep_rfm_local": "--",
    "rs_fixed_tau": ":",
    "pielm": "-",
    "strfm_stc": "--",
    "strfm_sov": "-.",
}
METRIC_LABELS = {
    "rel_l2_velocity": r"$\mathcal{E}_{\mathbf{u}}$",
    "rel_l2_vorticity": r"$\mathcal{E}_{\omega}$",
}
SEEDS = [10, 11, 12, 13, 14]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    setup_style(journal="general", lang="en", constrained_layout=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def export_with_qa(fig, base: Path, size: tuple[float, float]) -> None:
    finalize_figure(fig)
    render_preview(fig, str(base.with_name(base.name + "_preview.png")), dpi=170)
    if print_report(audit_layout(fig)) == "FAIL":
        raise RuntimeError(f"layout audit failed: {base.name}")
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


def add_method_curve(
    ax,
    frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    metric: str,
    x_name: str,
    method: str,
) -> None:
    subset = frame[frame.method_key == method].sort_values(x_name)
    if subset.empty:
        return
    raw_subset = raw_frame[raw_frame.method_key == method]
    ax.scatter(
        raw_subset[x_name].to_numpy(dtype=float),
        raw_subset[metric].to_numpy(dtype=float),
        s=7,
        marker=MARKERS[method],
        color=COLORS[method],
        edgecolor="none",
        alpha=0.20,
        zorder=1,
    )
    x = subset[x_name].to_numpy(dtype=float)
    mean = subset[f"mean_{metric}"].to_numpy(dtype=float)
    sd = subset[f"std_{metric}"].to_numpy(dtype=float)
    positive_raw = raw_frame.loc[raw_frame[metric] > 0.0, metric].to_numpy(dtype=float)
    display_floor = 0.5 * positive_raw.min()
    lower = np.maximum(mean - sd, display_floor)
    upper = mean + sd
    ax.fill_between(x, lower, upper, color=COLORS[method], alpha=0.10, linewidth=0.0)
    ax.plot(
        x,
        mean,
        color=COLORS[method],
        marker=MARKERS[method],
        linestyle=LINESTYLES[method],
        linewidth=1.0,
        markersize=3.2,
        markeredgecolor="black",
        markeredgewidth=0.25,
        label=PLAIN_LABELS[method],
    )


def make_scaling_figure(raw: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path) -> None:
    feature = summary[summary.study == "feature_budget"].copy()
    separation = summary[summary.study == "scale_separation"].copy()
    feature_raw = raw[raw.study == "feature_budget"].copy()
    separation_raw = raw[raw.study == "scale_separation"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.55), constrained_layout=True)
    columns = [
        (
            feature[feature.case == "two_rate_vortex"],
            feature_raw[feature_raw.case == "two_rate_vortex"],
            "features_per_field",
            "Two-rate vortex",
        ),
        (
            feature[feature.case == "interacting_vortex"],
            feature_raw[feature_raw.case == "interacting_vortex"],
            "features_per_field",
            "Interacting vortex",
        ),
        (separation, separation_raw, "point", "Scale separation"),
    ]
    for column, (frame, raw_frame, x_name, title) in enumerate(columns):
        for row, metric in enumerate(("rel_l2_velocity", "rel_l2_vorticity")):
            ax = axes[row, column]
            for method in SCALING_METHODS:
                add_method_curve(ax, frame, raw_frame, metric, x_name, method)
            ax.set_yscale("log")
            if x_name == "features_per_field":
                ax.set_xscale("log", base=2)
                ax.set_xlabel("Features per field $P$")
                ax.set_xticks([96, 192, 384, 768])
                ax.set_xticklabels(["96", "192", "384", "768"])
            else:
                ax.set_xscale("log", base=2)
                ax.set_xlabel(r"Rate ratio $\lambda_2/\lambda_1$")
                ax.set_xticks([1, 2, 4, 8, 16])
                ax.set_xticklabels(["1", "2", "4", "8", "16"])
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{title}: {METRIC_LABELS[metric]}", loc="left", fontweight="bold")
            ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.75)
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    panel_labels = add_panel_labels(fig, axes=axes.flat, style="paren", x_offset_pt=2.0, y_offset_pt=-2.0)
    for label in panel_labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.4})
    export_with_qa(fig, figure_dir / "fig03_mechanism_scaling", (7.0, 4.55))


def m3_categories(raw: pd.DataFrame) -> pd.DataFrame:
    window = raw[raw.study == "spectral_window"].copy()
    sampling = raw[(raw.study == "sampling_ablation") & (raw.sampling_mode == "iid_log_uniform")].copy()
    window["category"] = window.point_label.map(
        {"W1": "W1-S", "W2": "W2-S", "W3": "W3-S", "W4": "W4-S"}
    )
    sampling["category"] = "W1-IID"
    combined = pd.concat([window, sampling], ignore_index=True)
    order = ["W1-S", "W1-IID", "W2-S", "W3-S", "W4-S"]
    combined["category"] = pd.Categorical(combined.category, categories=order, ordered=True)
    return combined.sort_values("category")


def validate_w1_replicate(raw: pd.DataFrame) -> None:
    """Require the duplicated stratified-W1 tasks to reproduce numerically."""
    left = raw[(raw.study == "spectral_window") & (raw.point_label == "W1")]
    right = raw[
        (raw.study == "sampling_ablation") & (raw.sampling_mode == "log_stratified")
    ]
    keys = ["method_key", "seed"]
    merged = left.merge(right, on=keys, suffixes=("_window", "_sampling"), validate="one_to_one")
    if len(merged) != 10:
        raise ValueError(f"expected ten duplicated W1 records, found {len(merged)}")
    fields = [
        "rel_l2_velocity",
        "rel_l2_vorticity",
        "rank_fraction",
        "condition_number",
        "picard_iterations",
        "final_relative_update",
    ]
    for field in fields:
        a = merged[f"{field}_window"].to_numpy(dtype=float)
        b = merged[f"{field}_sampling"].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1.0e-10, atol=1.0e-13, equal_nan=True):
            raise ValueError(f"duplicated W1 result drift in {field}")


def make_window_figure(raw: pd.DataFrame, figure_dir: Path) -> None:
    frame = m3_categories(raw)
    categories = ["W1-S", "W1-IID", "W2-S", "W3-S", "W4-S"]
    quantities = [
        ("rel_l2_velocity", r"$\mathcal{E}_{\mathbf{u}}$", True),
        ("rel_l2_vorticity", r"$\mathcal{E}_{\omega}$", True),
        ("rank_fraction", r"Effective-rank fraction", False),
        ("condition_number", r"Truncated condition estimate", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.55), constrained_layout=True)
    jitter = np.linspace(-0.08, 0.08, len(SEEDS))
    method_offsets = {"rs_ic_global": -0.17, "rs_ic_local": 0.17}
    for ax, (quantity, ylabel, log_scale) in zip(axes.flat, quantities, strict=True):
        for method in ("rs_ic_global", "rs_ic_local"):
            for category_index, category in enumerate(categories):
                subset = frame[
                    (frame.method_key == method) & (frame.category.astype(str) == category)
                ].sort_values("seed")
                values = subset[quantity].to_numpy(dtype=float)
                x = category_index + method_offsets[method] + jitter[: len(values)]
                ax.scatter(
                    x,
                    values,
                    s=14,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.22,
                    alpha=0.72,
                    zorder=3,
                )
                if values.size:
                    center = category_index + method_offsets[method]
                    ax.hlines(values.mean(), center - 0.10, center + 0.10, color="black", linewidth=1.0)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
    axes[0, 0].set_title("Field accuracy", loc="left", fontweight="bold")
    axes[0, 1].set_title("Field accuracy", loc="left", fontweight="bold")
    axes[1, 0].set_title("Retained column space", loc="left", fontweight="bold")
    axes[1, 0].set_ylim(0.90, 1.002)
    axes[1, 1].set_title("Linear-system conditioning", loc="left", fontweight="bold")
    handles = [
        plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle="", markeredgecolor="black", markeredgewidth=0.25)
        for m in ("rs_ic_global", "rs_ic_local")
    ]
    fig.legend(handles, ["RS-RFM-Global", "RS-RFM-Local"], loc="outside lower center", ncol=2, frameon=False)
    panel_labels = add_panel_labels(fig, axes=axes.flat, style="paren", x_offset_pt=2.0, y_offset_pt=-2.0)
    for label in panel_labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.4})
    export_with_qa(fig, figure_dir / "fig04_mechanism_window", (7.0, 4.55))


def sci(value: float) -> str:
    exponent = int(math.floor(math.log10(abs(value))))
    return f"${value / 10.0**exponent:.2f}\\times10^{{{exponent}}}$"


def write_efficiency_table(efficiency: pd.DataFrame) -> str:
    lookup = {
        (row.case, row.method_key): row.minimum_P_below_threshold
        for row in efficiency.itertuples()
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{达到预注册阈值 $\overline{\mathcal E}_{\bu}<10^{-2}$ 的最小特征数；“--”表示在 $P\leq768$ 内未达到。}",
        r"\label{tab:mechanism-feature-efficiency}",
        r"\TableFont",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"方法 & 双速率涡 & 相互作用涡 \\",
        r"\midrule",
    ]
    for method in METHODS:
        values = []
        for case in ("two_rate_vortex", "interacting_vortex"):
            value = lookup[(case, method)]
            values.append("--" if pd.isna(value) else str(int(value)))
        lines.append(" & ".join([LABELS[method], *values]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_tex(raw: pd.DataFrame, summary: pd.DataFrame, efficiency: pd.DataFrame, audit: dict) -> str:
    if audit["expected_seed_runs"] != 725 or audit["records"] != 725 or audit["failures"]:
        raise ValueError(f"mechanism archive failed completeness gate: {audit}")
    feature = summary[summary.study == "feature_budget"]
    separation = summary[summary.study == "scale_separation"]
    m3 = m3_categories(raw)
    nonconverged = (
        raw.loc[~raw.picard_converged.astype(bool)]
        .groupby("method_key")
        .size()
        .reindex(METHODS, fill_value=0)
    )
    nonconverged_text = "、".join(
        f"{LABELS[method]} 为 {int(count)} 次"
        for method, count in nonconverged.items()
        if count
    )
    convergence_note = (
        "全部逐种子运行均进入统计；其中达到 Picard 上限而未满足停止阈值的运行共 "
        f"{int(nonconverged.sum())} 次（{nonconverged_text}），未以新种子替换。"
        if nonconverged.sum()
        else "全部逐种子运行均满足 Picard 停止阈值。"
    )

    def summary_row(frame: pd.DataFrame, method: str, point: float) -> pd.Series:
        rows = frame[(frame.method_key == method) & (frame.point.astype(float) == float(point))]
        if len(rows) != 1:
            raise ValueError(f"expected one summary row for {method}, {point}; got {len(rows)}")
        return rows.iloc[0]

    p768_two = feature[(feature.case == "two_rate_vortex") & (feature.features_per_field == 768)]
    p768_inter = feature[(feature.case == "interacting_vortex") & (feature.features_per_field == 768)]
    best_two_u = p768_two.loc[p768_two.mean_rel_l2_velocity.idxmin()]
    best_two_w = p768_two.loc[p768_two.mean_rel_l2_vorticity.idxmin()]
    best_inter_u = p768_inter.loc[p768_inter.mean_rel_l2_velocity.idxmin()]
    best_inter_w = p768_inter.loc[p768_inter.mean_rel_l2_vorticity.idxmin()]

    def budget_winner_text(best_u: pd.Series, best_w: pd.Series) -> str:
        if best_u.method_key == best_w.method_key:
            return (
                f"速度和涡量最低均值均由 {LABELS[best_u.method_key]} 取得，分别为 "
                f"{sci(best_u.mean_rel_l2_velocity)} 和 {sci(best_w.mean_rel_l2_vorticity)}"
            )
        return (
            f"最低速度均值由 {LABELS[best_u.method_key]} 取得（{sci(best_u.mean_rel_l2_velocity)}），"
            f"最低涡量均值由 {LABELS[best_w.method_key]} 取得（{sci(best_w.mean_rel_l2_vorticity)}）"
        )

    two_budget_winners = budget_winner_text(best_two_u, best_two_w)
    interacting_budget_winners = budget_winner_text(best_inter_u, best_inter_w)
    ratio1_global = summary_row(separation, "rs_ic_global", 1)
    ratio16_global = summary_row(separation, "rs_ic_global", 16)
    ratio1_local = summary_row(separation, "rs_ic_local", 1)
    ratio16_local = summary_row(separation, "rs_ic_local", 16)
    ratio1_fixed = summary_row(separation, "rs_fixed_tau", 1)
    ratio16_fixed = summary_row(separation, "rs_fixed_tau", 16)
    ratio16 = separation[separation.point.astype(float) == 16.0]
    best_ratio16_u = ratio16.loc[ratio16.mean_rel_l2_velocity.idxmin()]
    best_ratio16_w = ratio16.loc[ratio16.mean_rel_l2_vorticity.idxmin()]
    if best_ratio16_u.method_key == best_ratio16_w.method_key:
        ratio16_winners = f"两项最低均值均由 {LABELS[best_ratio16_u.method_key]} 取得"
    else:
        ratio16_winners = (
            f"最低速度与涡量均值分别由 {LABELS[best_ratio16_u.method_key]} 和 "
            f"{LABELS[best_ratio16_w.method_key]} 取得"
        )

    m3_means = (
        m3.groupby(["category", "method_key"], observed=True)[
            [
                "rel_l2_velocity",
                "rel_l2_vorticity",
                "rank_fraction",
                "condition_number",
                "max_log_tau_gap",
            ]
        ]
        .mean()
        .reset_index()
    )
    baseline = m3_means[m3_means.category.astype(str) == "W1-S"]
    iid = m3_means[m3_means.category.astype(str) == "W1-IID"]
    stratified = m3_means[m3_means.category.astype(str).isin(["W1-S", "W2-S", "W3-S", "W4-S"])]
    window_sentences = []
    sampling_sentences = []
    for method in ("rs_ic_global", "rs_ic_local"):
        b = baseline[baseline.method_key == method].iloc[0]
        q = iid[iid.method_key == method].iloc[0]
        w = stratified[stratified.method_key == method]
        tex_window = {"W1-S": "W_1", "W2-S": "W_2", "W3-S": "W_3", "W4-S": "W_4"}
        best_u = tex_window[str(w.loc[w.rel_l2_velocity.idxmin(), "category"])]
        best_w = tex_window[str(w.loc[w.rel_l2_vorticity.idxmin(), "category"])]
        wide = w[w.category.astype(str) == "W4-S"].iloc[0]
        wide_u = 100.0 * (wide.rel_l2_velocity / b.rel_l2_velocity - 1.0)
        wide_w = 100.0 * (wide.rel_l2_vorticity / b.rel_l2_vorticity - 1.0)
        window_sentences.append(
            f"{LABELS[method]} 的最低速度与涡量均值分别出现在 ${best_u}$ 和 ${best_w}$，"
            f"$W_4$ 相对 $W_1$ 分别变化 {wide_u:+.1f}\\% 和 {wide_w:+.1f}\\%"
        )
        du = 100.0 * (q.rel_l2_velocity / b.rel_l2_velocity - 1.0)
        dw = 100.0 * (q.rel_l2_vorticity / b.rel_l2_vorticity - 1.0)
        gap_ratio = q.max_log_tau_gap / b.max_log_tau_gap
        sampling_sentences.append(
            f"{LABELS[method]} 的最大对数尺度间隙增至 {gap_ratio:.2f} 倍，"
            f"速度和涡量均值分别变化 {du:+.1f}\\% 和 {dw:+.1f}\\%"
        )
    blocks = [
        (
            "机制实验保持五个固定种子和匹配容量。M1 在双速率涡与相互作用涡上取 "
            "$P=96,128,192,256,384,512,768$，并按 $N_{\\rm int}=\\operatorname{round}(75P/16)$、"
            "$N_{\\rm bc}=\\operatorname{round}(35P/192)$ 同步增加配点；M2 固定 $P=384$，"
            "令 $\\lambda_1=0.5$、$\\lambda_2=0.5,1,2,4,8$；M3 固定 $P=384$，比较 "
            "$W_1=[0.08,3]$、$W_2=[0.5,3]$、$W_3=[0.08,0.5]$、$W_4=[0.01,100]$，"
            "并在 $W_1$ 上比较对数分层采样与独立对数均匀采样。M1--M2 比较六种 3$\\times$2 RFM 与 Fixed-$\\tau$ RFM，"
            "M3 只比较 Global 与 Local；所有区间均以 $T$ 归一化。" + convergence_note
        ),
        write_efficiency_table(efficiency),
        (
            f"在最大预算 $P=768$ 处，双速率涡的{two_budget_winners}；"
            f"相互作用涡的{interacting_budget_winners}。"
            "表~\\ref{tab:mechanism-feature-efficiency} 用预注册阈值进一步比较达到给定精度所需的最小容量。"
        ),
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/fig03_mechanism_scaling.pdf}",
        (
            r"\caption{特征预算与速率分离实验。前两列分别给出双速率涡和相互作用涡随 $P$ 的"
            r"速度、涡量误差；第三列给出固定 $P=384$ 时随 $\lambda_2/\lambda_1$ 的变化。"
            r"浅色点为逐种子结果，曲线和阴影分别表示均值与样本标准差。}"
        ),
        r"\label{fig:mechanism-scaling}",
        r"\end{figure}",
        (
            f"当速率比由 $1$ 增至 $16$ 时，RS-RFM-Global 的平均速度误差由 "
            f"{sci(ratio1_global.mean_rel_l2_velocity)} 变为 {sci(ratio16_global.mean_rel_l2_velocity)}，"
            f"平均涡量误差由 {sci(ratio1_global.mean_rel_l2_vorticity)} 变为 "
            f"{sci(ratio16_global.mean_rel_l2_vorticity)}。在速率比 $16$ 处，{ratio16_winners}。"
            f"Fixed-$\\tau$/RS-RFM-Local 的速度、涡量误差比从速率比 $1$ 时的 "
            f"{ratio1_fixed.mean_rel_l2_velocity / ratio1_local.mean_rel_l2_velocity:.2f} 和 "
            f"{ratio1_fixed.mean_rel_l2_vorticity / ratio1_local.mean_rel_l2_vorticity:.2f}，"
            f"变为速率比 $16$ 时的 "
            f"{ratio16_fixed.mean_rel_l2_velocity / ratio16_local.mean_rel_l2_velocity:.2f} 和 "
            f"{ratio16_fixed.mean_rel_l2_vorticity / ratio16_local.mean_rel_l2_vorticity:.2f}。"
        ),
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/fig04_mechanism_window.pdf}",
        (
            r"\caption{松弛谱区间与采样策略敏感性。W1-S、W1-IID 分别表示区间 "
            r"$[0.08,3]$ 上的对数分层和独立对数均匀采样，W2-S--W4-S 表示其余三个区间"
            r"上的对数分层采样。彩色点为五个固定种子，黑色短线为均值；(a)、(b) 为场误差，"
            r"(c)、(d) 为有效秩比例与截断条件数；(c) 的纵轴放大显示 $[0.90,1.00]$ 区间。}"
        ),
        r"\label{fig:mechanism-window}",
        r"\end{figure}",
        (
            "在四个分层采样区间中，" + "；".join(window_sentences) + "。"
            "在相同的 $W_1$ 区间内，" + "；".join(sampling_sentences) + "。"
            "$W_3$ 虽未显式覆盖慢尺度 $2T$，却给出两种变体的最低场误差，表明本算例中的"
            "静态锚点已能承担缓慢分量；这也说明窗口敏感性不能简化为‘同时覆盖全部目标尺度即最优’。"
            "独立采样扩大了有限字典中的尺度空隙，但没有在两个误差指标上产生一致的劣化，"
            "因此本实验只支持分层采样改善覆盖规则性，而不支持其必然降低平均误差。"
            "区间改变同时影响误差、保留秩和截断条件数，图~\\ref{fig:mechanism-window} 因而把"
            "精度变化与残差矩阵诊断并列，而不把误差差异归因于单一条件数。"
        ),
    ]
    return "\n\n".join(blocks) + "\n"


def main() -> int:
    args = parse_args()
    summary_dir = args.summary_dir.resolve()
    audit = json.loads((summary_dir / "audit.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(summary_dir / "metrics_all.csv")
    summary = pd.read_csv(summary_dir / "summary.csv")
    efficiency = pd.read_csv(summary_dir / "feature_efficiency.csv")
    if len(raw) != 725 or sorted(raw.seed.unique().tolist()) != SEEDS:
        raise ValueError("mechanism raw archive is incomplete")
    validate_w1_replicate(raw)
    configure_style()
    figure_dir = args.paper / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    make_scaling_figure(raw, summary, figure_dir)
    make_window_figure(raw, figure_dir)
    output = args.paper / "sections_cn" / "generated" / "mechanism_results.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_tex(raw, summary, efficiency, audit), encoding="utf-8")
    print(json.dumps({"records": len(raw), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
