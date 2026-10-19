# 中文注释：从消融与机制实验记录生成正文消融表格和图。
"""Generate the compact Section-IV.C ablation figures and prose."""

from __future__ import annotations

import argparse
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


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"
METRICS = {
    "rel_l2_velocity": r"$\mathcal{E}_{\mathbf{u}}$",
    "rel_l2_vorticity": r"$\mathcal{E}_{\omega}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
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
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def export_with_qa(fig, base: Path, size: tuple[float, float]) -> None:
    finalize_figure(fig)
    add_panel_labels(fig, style="nature")
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


def jitter(count: int, width: float = 0.018) -> np.ndarray:
    if count == 1:
        return np.zeros(1)
    return np.linspace(-width, width, count)


def raw_mean_sd(ax, x, values, *, color, marker, offset=0.0, label=None, connect=False):
    values = np.asarray(values, dtype=float)
    x_value = float(x) + offset
    ax.scatter(
        x_value + jitter(values.size, width=0.010),
        values,
        s=12,
        color=color,
        marker=marker,
        alpha=0.35,
        edgecolor="none",
        zorder=2,
    )
    return {
        "x": x_value,
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "color": color,
        "marker": marker,
        "label": label,
        "connect": connect,
    }


def draw_summaries(ax, summaries, *, connect=False):
    if connect:
        ax.plot(
            [row["x"] for row in summaries],
            [row["mean"] for row in summaries],
            color=summaries[0]["color"],
            linewidth=0.9,
            linestyle="-",
            zorder=3,
        )
    for row in summaries:
        ax.errorbar(
            row["x"],
            row["mean"],
            yerr=row["sd"],
            fmt=row["marker"],
            color=row["color"],
            ecolor=row["color"],
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            markersize=4.0,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=row["label"],
            zorder=4,
        )


def make_sensitivity_figure(ablation: pd.DataFrame, mechanism: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.65), constrained_layout=True)
    rho = ablation[
        (ablation.case == "two_rate_physics_range")
        & ablation.variant.isin(["rho_0", "rho_0p125", "full", "rho_0p5"])
    ].copy()
    for ax, metric, title in zip(
        axes[0],
        METRICS,
        ("Static-anchor sensitivity: velocity", "Static-anchor sensitivity: vorticity"),
        strict=True,
    ):
        summaries = []
        for value in (0.0, 0.125, 0.25, 0.5):
            values = rho[np.isclose(rho.rho_static.astype(float), value)][metric]
            if len(values) != 5:
                raise ValueError(f"rho={value} lacks five seeds for {metric}")
            summaries.append(raw_mean_sd(ax, value, values, color=BLUE, marker="o"))
        draw_summaries(ax, summaries, connect=True)
        ax.axvline(0.25, color=GRAY, linestyle=":", linewidth=0.8, zorder=1)
        ax.set_yscale("log")
        ax.set_xticks([0.0, 0.125, 0.25, 0.5], ["0", "1/8", "1/4", "1/2"])
        ax.set_xlabel(r"Static fraction $\rho_0$")
        ax.set_ylabel(METRICS[metric])
        ax.set_title(title)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, linestyle="--")

    window = mechanism[
        (mechanism.study == "spectral_window")
        & (mechanism.method_key == "rs_ic_global")
    ].copy()
    ax = axes[1, 0]
    for metric, color, marker, offset, label in (
        ("rel_l2_velocity", BLUE, "o", -0.09, r"$\mathcal{E}_{\mathbf{u}}$"),
        ("rel_l2_vorticity", ORANGE, "s", 0.09, r"$\mathcal{E}_{\omega}$"),
    ):
        summaries = []
        for index, point in enumerate(("W1", "W2", "W3", "W4")):
            values = window[window.point_label == point][metric]
            if len(values) != 5:
                raise ValueError(f"window {point} lacks five seeds for {metric}")
            summaries.append(
                raw_mean_sd(
                    ax, index, values, color=color, marker=marker, offset=offset,
                    label=label if index == 0 else None,
                )
            )
        draw_summaries(ax, summaries)
    ax.set_yscale("log")
    ax.set_xticks(range(4), [r"$W_1$", r"$W_2$", r"$W_3$", r"$W_4$"])
    ax.set_xlabel("Relaxation window")
    ax.set_ylabel("Relative error")
    ax.set_title("Relaxation-window sensitivity")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, linestyle="--")

    sampling = mechanism[
        (mechanism.study == "sampling_ablation")
        & (mechanism.method_key == "rs_ic_global")
    ].copy()
    ax = axes[1, 1]
    for metric, color, marker, offset, label in (
        ("rel_l2_velocity", BLUE, "o", -0.09, r"$\mathcal{E}_{\mathbf{u}}$"),
        ("rel_l2_vorticity", ORANGE, "s", 0.09, r"$\mathcal{E}_{\omega}$"),
    ):
        summaries = []
        for index, mode in enumerate(("log_stratified", "iid_log_uniform")):
            values = sampling[sampling.sampling_mode == mode][metric]
            if len(values) != 5:
                raise ValueError(f"sampling {mode} lacks five seeds for {metric}")
            summaries.append(
                raw_mean_sd(
                    ax, index, values, color=color, marker=marker, offset=offset,
                    label=label if index == 0 else None,
                )
            )
        draw_summaries(ax, summaries)
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Stratified", "IID"])
    ax.set_xlabel("Log-scale sampling")
    ax.set_ylabel("Relative error")
    ax.set_title("Sampling-strategy sensitivity")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, linestyle="--")

    export_with_qa(fig, figure_dir / "fig04_ablation_sensitivity", (7.0, 4.65))


def sci(value: float) -> str:
    if value == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


def mean_sd(frame: pd.DataFrame, case: str, variant: str, metric: str):
    values = frame[(frame.case == case) & (frame.variant == variant)][metric].to_numpy(float)
    if values.size == 0:
        return None
    if values.size != 5:
        raise ValueError(f"{case}/{variant}/{metric} has {values.size} seeds")
    return float(values.mean()), float(values.std(ddof=1))


def factor(frame: pd.DataFrame, case: str, numerator: str, denominator: str, metric: str) -> float:
    top = mean_sd(frame, case, numerator, metric)[0]
    bottom = mean_sd(frame, case, denominator, metric)[0]
    return top / bottom


def window_best(mechanism: pd.DataFrame, metric: str) -> tuple[str, float]:
    frame = mechanism[
        (mechanism.study == "spectral_window") & (mechanism.method_key == "rs_ic_global")
    ]
    means = frame.groupby("point_label", observed=True)[metric].mean()
    label = str(means.idxmin())
    if label.startswith("W") and label[1:].isdigit():
        label = f"W_{{{label[1:]}}}"
    return label, float(means.min())


def sampling_change(mechanism: pd.DataFrame, metric: str) -> float:
    frame = mechanism[
        (mechanism.study == "sampling_ablation") & (mechanism.method_key == "rs_ic_global")
    ]
    means = frame.groupby("sampling_mode", observed=True)[metric].mean()
    return float(means["iid_log_uniform"] / means["log_stratified"] - 1.0)


def write_latex(ablation: pd.DataFrame, mechanism: pd.DataFrame, output: Path) -> None:
    cases = ["two_rate_physics_range", "interacting_two_rate_vortex"]
    lines = [
        (
            "核心消融统一采用双速率受迫涡：该算例具有 $0.125T$ 与 $2T$ 两个明确分离的松弛时间，"
            "且两个空间模态具有相同的 Laplace 特征值，因而可在简化非线性输运影响的同时考察时间试探空间。"
            "相互作用涡仅用于检验关键现象在 $\\bu\\cdot\\nabla\\omega\\neq0$ 时是否保留。"
        ),
        "",
        r"\noindent\textbf{多松弛率核心证据。}",
        (
            "主消融采用 RS-RFM-Global，并固定 $P=384$、$\\rho_0=1/4$、谱窗口 "
            "$[0.08T,3T]$ 和对数分层采样。朴素指数对照只将动态列的时间因子从 "
            "$q_\\tau(t)=(e^{-t/\\tau}-1)/t$ 改为 $e^{-t/\\tau}$；相同的硬提升使二者均严格满足初值。"
            "单松弛尺度对照保留相同空间随机特征、商函数和静态比例，只令全部动态列取 $\\tau_0=T$。"
        ),
        "",
    ]

    q_two_u = factor(ablation, cases[0], "naive_exponential", "full", "rel_l2_velocity")
    q_two_w = factor(ablation, cases[0], "naive_exponential", "full", "rel_l2_vorticity")
    q_int_u = factor(ablation, cases[1], "naive_exponential", "full", "rel_l2_velocity")
    q_int_w = factor(ablation, cases[1], "naive_exponential", "full", "rel_l2_vorticity")
    single_u = factor(ablation, cases[0], "single_rate", "full", "rel_l2_velocity")
    single_w = factor(ablation, cases[0], "single_rate", "full", "rel_l2_vorticity")
    anchor_u = factor(ablation, cases[0], "rho_0", "full", "rel_l2_velocity")
    anchor_w = factor(ablation, cases[0], "rho_0", "full", "rel_l2_vorticity")
    single = ablation[
        (ablation.case == cases[0]) & (ablation.variant == "single_rate")
    ]
    capped_single = int((~single.picard_converged.astype(bool)).sum())
    update_min = float(single.final_relative_update.min())
    update_max = float(single.final_relative_update.max())
    lines.append(
        "与默认多松弛率构造相比，严格单松弛尺度在双速率涡上的速度和涡量平均误差分别增至 "
        f"{single_u:.2f} 倍和 {single_w:.2f} 倍，直接支持在该算例中以多个松弛率表示分离的时间尺度。"
        "相比之下，商函数和静态锚点的精度作用具有算例依赖性：朴素指数特征在双速率涡上的两项平均误差分别为"
        f"默认构造的 {q_two_u:.2f} 倍和 {q_two_w:.2f} 倍，在相互作用涡上则为 {q_int_u:.2f} 倍和 {q_int_w:.2f} 倍；"
        f"去除静态锚点后，双速率涡的两项误差分别为默认构造的 {anchor_u:.2f} 倍和 {anchor_w:.2f} 倍。"
        "因此，现有消融明确支持多松弛率时间字典，但不支持将商函数或静态锚点解释为在所有有限特征问题上均能降低误差。"
        f"单松弛尺度的 {capped_single} 个运行均达到 10 步上限，其最终相对更新量为 "
        f"{sci(update_min)}--{sci(update_max)}；所有运行均原样保留。"
    )
    lines.extend(
        [
            "",
            r"\noindent\textbf{参数敏感性。}",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{figures/fig04_ablation_sensitivity.pdf}",
            (
                r"\caption{RS-RFM-Global 的参数敏感性。彩色小点为五次独立运行，实心标记及误差棒为均值 $\pm$ 样本标准差。"
                r"(a)、(b) 分别给出双速率受迫涡的速度和涡量误差随静态比例 $\rho_0$ 的变化，竖直虚线标出默认值 $1/4$；"
                r"(c) 比较 $W_1=[0.08,3]$、$W_2=[0.5,3]$、$W_3=[0.08,0.5]$ 和 $W_4=[0.01,100]$；"
                r"(d) 在 $W_1$ 上比较对数分层与独立对数均匀采样。窗口均以 $T$ 归一化。}"
            ),
            r"\label{fig:ablation-sensitivity}",
            r"\end{figure}",
        ]
    )
    rho_means = (
        ablation[
            (ablation.case == cases[0])
            & ablation.variant.isin(["rho_0", "rho_0p125", "full", "rho_0p5"])
        ]
        .groupby("rho_static", observed=True)[["rel_l2_velocity", "rel_l2_vorticity"]]
        .mean()
    )
    best_u = float(rho_means.rel_l2_velocity.idxmin())
    best_w = float(rho_means.rel_l2_vorticity.idxmin())
    w_u, _ = window_best(mechanism, "rel_l2_velocity")
    w_w, _ = window_best(mechanism, "rel_l2_vorticity")
    iid_u = 100.0 * sampling_change(mechanism, "rel_l2_velocity")
    iid_w = 100.0 * sampling_change(mechanism, "rel_l2_vorticity")
    if w_u == w_w:
        window_result = f"四个谱窗口中，两项最低均值均出现在 ${w_u}$。"
    else:
        window_result = f"四个谱窗口中，两项最低均值分别出现在 ${w_u}$ 和 ${w_w}$。"
    if math.isclose(best_u, best_w):
        rho_result = f"四个静态比例中，速度和涡量平均误差均在 $\\rho_0={best_u:g}$ 处最低；"
    else:
        rho_result = (
            f"四个静态比例中，速度和涡量平均误差分别在 $\\rho_0={best_u:g}$ "
            f"与 $\\rho_0={best_w:g}$ 处最低；"
        )
    lines.append(
        "图~\\ref{fig:ablation-sensitivity} 显示，"
        + rho_result
        + "本扫描及相互作用涡的二点核查均不支持将 $1/4$ 解释为精度最优值；论文仍将其作为预先固定的结构性折中，"
        + "同时明确静态锚点并非对所有问题都必需。"
        + window_result
        + f"相对分层采样，IID 采样的速度与涡量均值变化分别为 {iid_u:+.1f}\\% 和 {iid_w:+.1f}\\%；"
        + "因此分层采样在有限样本中的作用是避免对数区间出现空层，并不能等同于本算例上更低的平均误差。"
    )
    lines.extend(
        [
            "",
            r"\noindent\textbf{有限特征与尺度分离行为。}",
            (
                "图~\\ref{fig:mechanism-scaling} 比较特征容量增加和物理衰减尺度分离对六种随机特征方法的影响。"
                "前两列检验有限字典扩展时的误差变化，第三列固定 $P=384$ 并改变目标解的 "
                "$\\lambda_2/\\lambda_1$，因而不属于算法参数调优。"
            ),
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{figures/fig03_mechanism_scaling.pdf}",
            (
                r"\caption{特征预算与尺度分离实验。前两列分别给出双速率受迫涡和相互作用涡随 $P$ 的速度、涡量误差；"
                r"第三列给出固定 $P=384$ 时随 $\lambda_2/\lambda_1$ 的变化。浅色点为五次独立运行，曲线和阴影为均值与样本标准差；"
                r"图中仅比较“时间构造 $\times$ 空间支撑”体系中的六种随机特征方法。}"
            ),
            r"\label{fig:mechanism-scaling}",
            r"\end{figure}",
            (
                "随着特征预算增加，各方法误差总体下降但排序并非处处固定。RS-RFM-Global 在双速率涡和相互作用涡上"
                "分别从 $P=192$ 和 $P=384$ 起达到预设阈值 $\\overline{\\mathcal E}_{\\bu}<10^{-2}$。"
                "尺度比从 $1$ 增至 $16$ 时误差未呈简单单调的相对优势扩张，说明该扫描刻画的是目标时间尺度分离与有限字典覆盖之间的共同作用。"
            ),
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ablation = pd.read_csv(args.ablation)
    mechanism = pd.read_csv(args.mechanism)
    if len(ablation) != 45:
        raise ValueError(f"expected 45 merged ablation records, got {len(ablation)}")
    configure_style()
    figure_dir = args.paper / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    make_sensitivity_figure(ablation, mechanism, figure_dir)
    output = args.paper / "sections_cn" / "generated" / "ablation_results.tex"
    write_latex(ablation, mechanism, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
