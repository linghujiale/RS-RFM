# 中文注释：从冻结结果生成四个受控二维流动的汇总表和代表性场图。
"""Build the compact controlled-2D evidence from frozen production archives.

The six-method archive and the later RS-RFM-Local archive are read without
alteration.  The script checks the full four-case, seven-method, five-seed
grid before emitting one consolidated table and one representative field
figure.  It never reruns or replaces a seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SKILL_SCRIPTS = PROJECT_ROOT / "figure_style"
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from layout_tools import finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


CASES = [
    "tgv_order1_decay",
    "two_rate_physics_range",
    "interacting_two_rate_vortex",
    "multiscale_shear_decay_k12",
]
CASE_LABELS = {
    "tgv_order1_decay": "Taylor--Green",
    "two_rate_physics_range": "双速率受迫涡",
    "interacting_two_rate_vortex": "相互作用涡",
    "multiscale_shear_decay_k12": "多尺度剪切流",
}
FIGURE_CASE_LABELS = {
    "tgv_order1_decay": "Taylor--Green",
    "two_rate_physics_range": "Two-rate vortex",
    "interacting_two_rate_vortex": "Interacting vortex",
    "multiscale_shear_decay_k12": "Multiscale shear",
}
METHODS = [
    "rs_ic_global",
    "rs_ic_local",
    "rs_fixed_tau",
    "rs_static",
    "strfm_sov",
    "strfm_stc",
    "pielm",
]
METHOD_LABELS = {
    "rs_ic_global": "RS-RFM-Global",
    "rs_ic_local": "RS-RFM-Local",
    "rs_fixed_tau": "Fixed-$\\tau$ RFM",
    "rs_static": "Static-RFM-Local",
    "strfm_sov": "ST-RFM-SoV",
    "strfm_stc": "ST-RFM-STC",
    "pielm": "PIELM",
}
SEEDS = [10, 11, 12, 13, 14]
METRICS = ["rel_l2_velocity", "rel_l2_vorticity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-results", type=Path, required=True)
    parser.add_argument("--local-results", type=Path, required=True)
    parser.add_argument("--local-field", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_records(global_results: Path, local_results: Path) -> list[dict]:
    rows = read_csv(global_results / "metrics_raw.csv")
    for csv_path in sorted(local_results.rglob("metrics_raw.csv")):
        rows.extend(read_csv(csv_path))

    records: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        candidate = row["candidate_id"]
        method = row["method_key"]
        seed = int(row["seed"])
        if candidate not in CASES or method not in METHODS or seed not in SEEDS:
            raise ValueError(f"unexpected record {(candidate, method, seed)}")
        key = (candidate, method, seed)
        if key in seen:
            raise ValueError(f"duplicate record {key}")
        seen.add(key)
        item = dict(row)
        item["seed"] = seed
        for name in METRICS + ["condition_number", "rank", "features_per_field"]:
            item[name] = float(row[name])
        records.append(item)

    expected = {(case, method, seed) for case in CASES for method in METHODS for seed in SEEDS}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"incomplete controlled grid: missing={missing}, extra={extra}")
    return records


def summarize(records: list[dict]) -> dict:
    summary: dict = {}
    for case in CASES:
        summary[case] = {}
        for method in METHODS:
            subset = [
                row for row in records
                if row["candidate_id"] == case and row["method_key"] == method
            ]
            summary[case][method] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in subset])),
                    "sample_sd": float(np.std([row[metric] for row in subset], ddof=1)),
                }
                for metric in METRICS
            }
    return summary


def mean_sd_tex(stats: dict, bold: bool) -> str:
    mean = stats["mean"]
    sd = stats["sample_sd"]
    exponent = int(math.floor(math.log10(abs(mean))))
    body = (
        f"({mean / 10.0**exponent:.2f}\\pm{sd / 10.0**exponent:.2f})"
        f"\\times10^{{{exponent}}}"
    )
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def write_table(summary: dict, output: Path) -> None:
    best = {
        (case, metric): min(METHODS, key=lambda method: summary[case][method][metric]["mean"])
        for case in CASES for metric in METRICS
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{四个受控二维算例的时空相对误差。数值为五次独立运行的均值 "
            r"$\pm$ 样本标准差；所有方法在同一算例中使用相同特征数和配点。"
            r"粗体表示相应误差列的最低均值。}"
        ),
        r"\label{tab:all-flow-results}",
        r"\WideTableFont",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l*{4}{cc}@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Taylor--Green}"
        r"& \multicolumn{2}{c}{双速率受迫涡}"
        r"& \multicolumn{2}{c}{相互作用涡}"
        r"& \multicolumn{2}{c}{多尺度剪切流} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
        r"方法 & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$"
        r" & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$"
        r" & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$"
        r" & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = [METHOD_LABELS[method]]
        for case in CASES:
            for metric in METRICS:
                cells.append(mean_sd_tex(summary[case][method][metric], best[(case, metric)] == method))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])

    winners = {
        case: (best[(case, METRICS[0])], best[(case, METRICS[1])]) for case in CASES
    }
    if any(pair[0] != pair[1] for pair in winners.values()):
        raise ValueError(f"velocity/vorticity winners disagree: {winners}")
    winner_text = "、".join(
        f"{CASE_LABELS[case]}（{METHOD_LABELS[pair[0]]}）" for case, pair in winners.items()
    )
    paragraph = (
        "表~\\ref{tab:all-flow-results} 给出四个受控二维算例的时空相对误差。"
        f"两项场误差的最低均值分别出现在：{winner_text}。"
        "这里的均值与样本标准差仅用于描述五次独立运行，不作显著性检验。"
        "图~\\ref{fig:controlled-2d-fields} 从场分布角度补充表~\\ref{tab:all-flow-results} 的误差比较。"
        "四个算例的终时预测均重现了解析涡量场的主导结构与相位；前两个算例的归一化逐点误差"
        "整体较低，而相互作用涡和多尺度剪切流中的局部误差更明显，这与表中较高的涡量误差一致。"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(paragraph + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


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


def export_with_qa(fig, base: Path) -> None:
    finalize_figure(fig)
    render_preview(fig, str(base.with_name(base.name + "_preview.png")), dpi=160)
    if print_report(audit_layout(fig)) == "FAIL":
        raise RuntimeError("controlled-field layout audit failed")
    export_figure(
        fig,
        str(base),
        formats=("pdf", "svg", "png"),
        size_inches=(12.2, 12.45),
        dpi=450,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)


def make_field_figure(global_results: Path, local_field: Path, figure_dir: Path) -> None:
    with np.load(global_results / "representative_fields_seed10.npz") as old:
        fields = {}
        for case in CASES[:3]:
            fields[case] = (
                old[f"{case}_x"].copy(),
                old[f"{case}_y"].copy(),
                old[f"{case}_reference"].copy(),
                old[f"{case}_prediction"].copy(),
                "RS-RFM-Global",
            )
    with np.load(local_field) as local:
        fields[CASES[3]] = (
            local["x"].copy(), local["y"].copy(), local["reference"].copy(),
            local["prediction"].copy(), "RS-RFM-Local",
        )

    prepared = []
    max_error = 0.0
    min_positive = math.inf
    for case in CASES:
        x, y, reference, prediction, method = fields[case]
        scale = float(np.max(np.abs(reference)))
        error = np.abs(prediction - reference) / scale
        positive = error[error > 0.0]
        max_error = max(max_error, float(error.max()))
        if positive.size:
            min_positive = min(min_positive, float(positive.min()))
        prepared.append((case, x, y, reference / scale, prediction / scale, error, method))
    error_vmax = 10.0 ** math.ceil(math.log10(max_error))
    error_vmin = max(10.0 ** math.floor(math.log10(min_positive)), error_vmax * 1.0e-5)

    fig = plt.figure(figsize=(12.2, 12.45), constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        6,
        width_ratios=(0.20, 1.0, 1.0, 0.055, 1.0, 0.055),
        wspace=0.025,
        hspace=0.08,
    )
    axes = np.empty((4, 3), dtype=object)
    row_label_axes = []
    signed_caxes = []
    error_caxes = []
    for row in range(4):
        row_label_axes.append(fig.add_subplot(grid[row, 0]))
        axes[row, 0] = fig.add_subplot(grid[row, 1])
        axes[row, 1] = fig.add_subplot(grid[row, 2])
        axes[row, 2] = fig.add_subplot(grid[row, 4])
        signed_caxes.append(fig.add_subplot(grid[row, 3]))
        error_caxes.append(fig.add_subplot(grid[row, 5]))
    for row, (case, x, y, reference, prediction, error, method) in enumerate(prepared):
        row_label_ax = row_label_axes[row]
        row_label_ax.axis("off")
        row_label_ax.text(
            0.18,
            0.96,
            f"({chr(ord('a') + row)})",
            ha="center",
            va="top",
            fontsize=20.0,
            fontweight="bold",
        )
        row_label_ax.text(
            0.72,
            0.46,
            FIGURE_CASE_LABELS[case],
            ha="center",
            va="center",
            fontsize=19.0,
            rotation=90,
        )
        signed_image = None
        error_image = None
        extent = (0.0, 2.0 * math.pi, 0.0, 2.0 * math.pi)
        pi_ticks = (0.0, math.pi, 2.0 * math.pi)
        pi_ticklabels = (r"$0$", r"$\pi$", r"$2\pi$")
        for column, field in enumerate((reference, prediction, np.maximum(error, error_vmin))):
            ax = axes[row, column]
            if column < 2:
                signed_image = ax.imshow(
                    field.T,
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-1.0,
                    vmax=1.0,
                    interpolation="bicubic",
                    interpolation_stage="rgba",
                    resample=True,
                    rasterized=True,
                )
            else:
                error_image = ax.imshow(
                    field.T,
                    origin="lower",
                    extent=extent,
                    cmap="magma",
                    norm=LogNorm(vmin=error_vmin, vmax=error_vmax),
                    interpolation="bicubic",
                    interpolation_stage="rgba",
                    resample=True,
                    rasterized=True,
                )
            ax.set_aspect("equal")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xticks(pi_ticks)
            ax.set_xticklabels(pi_ticklabels)
            ax.set_yticks(pi_ticks)
            ax.set_yticklabels(pi_ticklabels if column == 0 else ("", "", ""))
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$" if column == 0 else "")
            if row == 0:
                ax.set_title(
                    ("Reference", "RS-RFM", "Absolute error")[column],
                    pad=7.0,
                )
        signed_bar = fig.colorbar(signed_image, cax=signed_caxes[row])
        signed_bar.solids.set_rasterized(False)
        signed_bar.set_ticks((-1.0, 0.0, 1.0))
        error_bar = fig.colorbar(error_image, cax=error_caxes[row])
        error_bar.solids.set_rasterized(False)
        error_bar.set_ticks((error_vmin, 1.0e-4, 1.0e-2, error_vmax))
        for colorbar in (signed_bar, error_bar):
            colorbar.ax.tick_params(direction="in", length=3.0, labelsize=17.0)
    export_with_qa(fig, figure_dir / "fig02_controlled_2d_fields")


def write_audit(records: list[dict], summary: dict, output_data: Path) -> None:
    output_data.mkdir(parents=True, exist_ok=True)
    keys = sorted(records[0])
    with (output_data / "metrics_all.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (output_data / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_data / "audit.json").write_text(
        json.dumps(
            {
                "planned_records": 140,
                "observed_records": len(records),
                "cases": CASES,
                "methods": METHODS,
                "seeds": SEEDS,
                "replacement_seeds": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    records = load_records(args.global_results, args.local_results)
    summary = summarize(records)
    write_audit(records, summary, args.output_data)
    configure_style()
    make_field_figure(args.global_results, args.local_field, args.paper / "figures")
    write_table(summary, args.paper / "sections_cn" / "generated" / "controlled_2d_results.tex")
    print(json.dumps({"records": len(records), "output": str(args.output_data)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
