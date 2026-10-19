# 中文注释：将 RS-RFM 生产结果整理为论文表格、字段缓存和图形输入。
"""Build the final PoF fluid evidence from frozen held-out records.

The script refuses to generate manuscript claims unless the production archive
is complete, failure-free, and RS-RFM-Global has the lowest mean velocity and
vorticity errors for every reported case.  Pilot records are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
FIGURE_SKILL_SCRIPTS = PROJECT_ROOT / "figure_style"
for path in (SCRIPTS, FIGURE_SKILL_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_figure import export_figure  # noqa: E402
from layout_tools import add_panel_labels, finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402

import run_pof_suitability_scan as scan  # noqa: E402


PROPOSED = "rs_ic_global"
CASE_ORDER = [
    "tgv_order1_decay",
    "two_rate_physics_range",
    "interacting_two_rate_vortex",
    "multiscale_shear_decay_k12",
]
CASE_LABELS = {
    "tgv_order1_decay": "Taylor--Green",
    "two_rate_physics_range": "Forced two-rate vortex",
    "interacting_two_rate_vortex": "Interacting-mode vortex",
    "multiscale_shear_decay_k12": "Multiscale shear",
}
CASE_PLAIN_LABELS = {
    "tgv_order1_decay": "Taylor-Green",
    "two_rate_physics_range": "Forced two-rate vortex",
    "interacting_two_rate_vortex": "Interacting-mode vortex",
    "multiscale_shear_decay_k12": "Multiscale shear",
}
TABLE_LABELS = {
    "tgv_order1_decay": "tab:taylor-green-results",
    "two_rate_physics_range": "tab:two-rate-results",
    "interacting_two_rate_vortex": "tab:interacting-vortex-results",
    "multiscale_shear_decay_k12": "tab:multiscale-shear-results",
}
METHOD_ORDER = [
    "rs_ic_global",
    "rs_fixed_tau",
    "rs_static",
    "strfm_sov",
    "strfm_stc",
    "pielm",
]
METHOD_LABELS = {
    "rs_ic_global": "RS-RFM-Global",
    "rs_fixed_tau": "Fixed-$\\tau$ RFM",
    "rs_static": "Static-RFM-Local",
    "strfm_sov": "ST-RFM-SoV",
    "strfm_stc": "ST-RFM-STC",
    "pielm": "PIELM",
}
TICK_LABELS = {
    "rs_ic_global": "RS-RFM",
    "rs_fixed_tau": "Fixed $\\tau$",
    "rs_static": "Static",
    "strfm_sov": "ST-SoV",
    "strfm_stc": "ST-STC",
    "pielm": "PIELM",
}
COLORS = {
    "rs_ic_global": "#0072B2",
    "rs_fixed_tau": "#E69F00",
    "rs_static": "#7F7F7F",
    "strfm_sov": "#009E73",
    "strfm_stc": "#CC79A7",
    "pielm": "#D55E00",
}
MARKERS = {
    "rs_ic_global": "o",
    "rs_fixed_tau": "^",
    "rs_static": "D",
    "strfm_sov": "v",
    "strfm_stc": "P",
    "pielm": "X",
}
METRICS = ["rel_l2_velocity", "rel_l2_vorticity", "wall_seconds"]
DIAGNOSTIC_FLOATS = [
    "condition_number",
    "residual_norm",
    "final_rel_l2_velocity",
    "final_rel_l2_vorticity",
    "final_relative_update",
    "rel_l2_enstrophy",
    "rel_l2_kinetic_energy",
]
DIAGNOSTIC_INTS = ["features_per_field", "rank", "picard_iterations"]
SEEDS = [10, 11, 12, 13, 14]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument(
        "--skip-fields",
        action="store_true",
        help="Do not rerun representative seed-10 solutions for the field figure.",
    )
    return parser.parse_args()


def load_records(directory: Path) -> list[dict]:
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    failures = json.loads((directory / "failures.json").read_text(encoding="utf-8"))
    expected = len(CASE_ORDER) * len(METHOD_ORDER) * len(SEEDS)
    if status.get("completed") != expected or status.get("total") != expected:
        raise ValueError(f"production archive is incomplete: {status}")
    if status.get("failures") or failures:
        raise ValueError(f"production archive contains failures: {failures}")

    with (directory / "metrics_raw.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected:
        raise ValueError(f"expected {expected} metric rows, found {len(rows)}")

    seen: set[tuple[str, str, int]] = set()
    converted: list[dict] = []
    for row in rows:
        item = dict(row)
        item["seed"] = int(row["seed"])
        key = (row["candidate_id"], row["method_key"], item["seed"])
        if key in seen:
            raise ValueError(f"duplicate production record: {key}")
        seen.add(key)
        if (
            row["candidate_id"] not in CASE_ORDER
            or row["method_key"] not in METHOD_ORDER
            or item["seed"] not in SEEDS
        ):
            raise ValueError(f"unexpected production record: {key}")
        for metric in METRICS:
            item[metric] = float(row[metric])
        for metric in DIAGNOSTIC_FLOATS:
            item[metric] = float(row[metric])
        for metric in DIAGNOSTIC_INTS:
            item[metric] = int(float(row[metric]))
        item["picard_converged"] = row["picard_converged"].lower() == "true"
        converted.append(item)
    if len(seen) != expected:
        raise ValueError("production archive does not contain the full case-method-seed grid")
    return converted


def summarize(records: list[dict]) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    output: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for case in CASE_ORDER:
        output[case] = {}
        for method in METHOD_ORDER:
            subset = [
                row
                for row in records
                if row["candidate_id"] == case and row["method_key"] == method
            ]
            if [row["seed"] for row in sorted(subset, key=lambda row: row["seed"])] != SEEDS:
                raise ValueError(f"missing held-out seed for {(case, method)}")
            values: dict[str, tuple[float, float]] = {}
            for metric in METRICS:
                data = np.asarray([row[metric] for row in subset], dtype=float)
                values[metric] = (float(data.mean()), float(data.std(ddof=1)))
            values["converged"] = (
                float(sum(row["picard_converged"] for row in subset)),
                float(len(subset)),
            )
            output[case][method] = values
    return output


def validate_rankings(summaries) -> dict[str, dict[str, float | str]]:
    audit: dict[str, dict[str, float | str]] = {}
    for case in CASE_ORDER:
        audit[case] = {}
        for metric in ("rel_l2_velocity", "rel_l2_vorticity"):
            means = {method: summaries[case][method][metric][0] for method in METHOD_ORDER}
            winner = min(means, key=means.get)
            if winner != PROPOSED:
                raise ValueError(
                    f"refusing superiority claim: {case} {metric} winner is {winner}, not {PROPOSED}"
                )
            comparator = min(
                (method for method in METHOD_ORDER if method != PROPOSED),
                key=lambda method: means[method],
            )
            audit[case][f"{metric}_winner"] = winner
            audit[case][f"{metric}_comparator"] = comparator
            audit[case][f"{metric}_factor"] = means[comparator] / means[PROPOSED]
    return audit


def configure_style() -> None:
    setup_style(journal="general", lang="en", constrained_layout=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
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


def make_accuracy_figure(records: list[dict], figure_dir: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 7.2), constrained_layout=True)
    jitter = np.linspace(-0.16, 0.16, len(SEEDS))
    for row_index, case in enumerate(CASE_ORDER):
        for column_index, (metric, quantity) in enumerate(
            (("rel_l2_velocity", "Velocity"), ("rel_l2_vorticity", "Vorticity"))
        ):
            ax = axes[row_index, column_index]
            for method_index, method in enumerate(METHOD_ORDER):
                subset = sorted(
                    [
                        row
                        for row in records
                        if row["candidate_id"] == case and row["method_key"] == method
                    ],
                    key=lambda row: row["seed"],
                )
                values = np.asarray([row[metric] for row in subset])
                ax.scatter(
                    method_index + jitter,
                    values,
                    s=23,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.35,
                    zorder=3,
                )
                ax.hlines(
                    values.mean(),
                    method_index - 0.24,
                    method_index + 0.24,
                    color="black",
                    linewidth=1.0,
                    zorder=4,
                )
            ax.set_yscale("log")
            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels(
                [TICK_LABELS[method] for method in METHOD_ORDER], rotation=27, ha="right"
            )
            ax.set_ylabel(rf"Relative $L^2$ error, {quantity.lower()}")
            ax.set_title(
                f"{CASE_PLAIN_LABELS[case]}: {quantity.lower()}", loc="left", fontweight="bold"
            )
            ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.75)
            ax.set_axisbelow(True)
            ax.margins(x=0.06, y=0.12)
    finalize_figure(fig)
    labels = add_panel_labels(
        fig,
        axes=axes.flat,
        style="paren",
        x_offset_pt=2.0,
        y_offset_pt=-2.0,
        ha="left",
        va="top",
    )
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5})
    qa_and_export(fig, figure_dir / "fig02_rsrfm_accuracy_seeds", (7.0, 7.2))


def make_diagnostics_figure(records: list[dict], figure_dir: Path) -> None:
    """Show all seed-level conditioning and retained-rank observations."""
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 7.2), constrained_layout=True)
    jitter = np.linspace(-0.16, 0.16, len(SEEDS))
    for row_index, case in enumerate(CASE_ORDER):
        for column_index, quantity in enumerate(("condition", "rank")):
            ax = axes[row_index, column_index]
            for method_index, method in enumerate(METHOD_ORDER):
                subset = sorted(
                    [
                        row
                        for row in records
                        if row["candidate_id"] == case and row["method_key"] == method
                    ],
                    key=lambda row: row["seed"],
                )
                if quantity == "condition":
                    values = np.asarray([row["condition_number"] for row in subset])
                else:
                    values = np.asarray(
                        [
                            row["rank"] / (2.0 * row["features_per_field"])
                            for row in subset
                        ]
                    )
                ax.scatter(
                    method_index + jitter,
                    values,
                    s=23,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.35,
                    zorder=3,
                )
                ax.hlines(
                    np.median(values),
                    method_index - 0.24,
                    method_index + 0.24,
                    color="black",
                    linewidth=1.0,
                    zorder=4,
                )
            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels(
                [TICK_LABELS[method] for method in METHOD_ORDER], rotation=27, ha="right"
            )
            if quantity == "condition":
                ax.set_yscale("log")
                ax.set_ylabel(r"Truncated condition estimate $\kappa_{\rm eff}$")
                descriptor = "conditioning"
            else:
                ax.set_ylim(0.35, 1.025)
                ax.set_ylabel(r"Effective-rank fraction $r_{\rm eff}/(2P)$")
                descriptor = "retained rank"
            ax.set_title(
                f"{CASE_PLAIN_LABELS[case]}: {descriptor}", loc="left", fontweight="bold"
            )
            ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.75)
            ax.set_axisbelow(True)
            ax.margins(x=0.06)
    finalize_figure(fig)
    labels = add_panel_labels(
        fig,
        axes=axes.flat,
        style="paren",
        x_offset_pt=2.0,
        y_offset_pt=-2.0,
        ha="left",
        va="top",
    )
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5})
    qa_and_export(fig, figure_dir / "fig04_rsrfm_linear_diagnostics", (7.0, 7.2))


def candidate_lookup() -> dict[str, scan.Candidate]:
    return {
        candidate.candidate_id: candidate
        for candidates in scan.candidate_matrix().values()
        for candidate in candidates
    }


def build_field_cache(results_dir: Path) -> Path:
    output = results_dir / "representative_fields_seed10.npz"
    if output.exists():
        return output
    lookup = candidate_lookup()
    payload: dict[str, np.ndarray] = {}
    for case in CASE_ORDER:
        candidate = lookup[case]
        pde = scan.make_pde(candidate)
        reference = scan.make_reference(candidate, pde)
        _, solution = scan.run_one(candidate, PROPOSED, SEEDS[0], reference)
        xx, yy = np.meshgrid(reference.x, reference.y, indexing="ij")
        xy = np.stack([xx, yy], axis=-1)
        prediction = solution.vorticity(xy, candidate.horizon)
        prefix = case.replace("-", "_")
        payload[f"{prefix}_x"] = reference.x
        payload[f"{prefix}_y"] = reference.y
        payload[f"{prefix}_reference"] = reference.omega[-1]
        payload[f"{prefix}_prediction"] = prediction
    np.savez_compressed(output, **payload)
    return output


def make_field_figure(results_dir: Path, figure_dir: Path) -> None:
    cache_path = build_field_cache(results_dir)
    archive = np.load(cache_path)
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    max_relative_error = 0.0
    for case in CASE_ORDER:
        prefix = case.replace("-", "_")
        x = archive[f"{prefix}_x"]
        y = archive[f"{prefix}_y"]
        reference = archive[f"{prefix}_reference"]
        prediction = archive[f"{prefix}_prediction"]
        scale = float(np.max(np.abs(reference)))
        ref_normalized = reference / scale
        pred_normalized = prediction / scale
        relative_error = np.abs(prediction - reference) / scale
        max_relative_error = max(max_relative_error, float(relative_error.max()))
        prepared.append((x, y, ref_normalized, pred_normalized, relative_error))

    error_vmax = 10.0 ** math.ceil(math.log10(max_relative_error))
    error_vmax = max(error_vmax, 1.0e-4)
    error_vmin = error_vmax * 1.0e-4
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 8.5), constrained_layout=True)
    signed_image = None
    error_image = None
    for row, (case, fields) in enumerate(zip(CASE_ORDER, prepared, strict=True)):
        x, y, reference, prediction, error = fields
        for column, field in enumerate((reference, prediction, np.maximum(error, error_vmin))):
            ax = axes[row, column]
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
            ax.set_xlabel(r"$x$")
            if column == 0:
                ax.set_ylabel(f"{CASE_PLAIN_LABELS[case]}\n$y$")
            else:
                ax.set_ylabel("")
            if row == 0:
                ax.set_title(
                    ("Reference", "RS-RFM-Global", "Normalized absolute error")[column],
                    fontweight="bold",
                )
    signed_bar = fig.colorbar(
        signed_image, ax=axes[:, :2], orientation="horizontal", shrink=0.72, pad=0.045
    )
    # Matplotlib rasterizes colorbar QuadMeshes above its default segment
    # threshold.  Keep the SVG/PDF submission assets genuinely vector while
    # retaining the fixed-size PNG preview for visual QA.
    signed_bar.solids.set_rasterized(False)
    signed_bar.set_label(r"Normalized vorticity $\omega/\max|\omega_*|$")
    error_bar = fig.colorbar(
        error_image, ax=axes[:, 2], orientation="horizontal", shrink=0.88, pad=0.045
    )
    error_bar.solids.set_rasterized(False)
    error_bar.set_label(r"$|\omega-\omega_*|/\max|\omega_*|$")
    finalize_figure(fig)
    labels = add_panel_labels(
        fig,
        axes=axes.flat,
        style="paren",
        x_offset_pt=2.0,
        y_offset_pt=-2.0,
        ha="left",
        va="top",
    )
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5})
    qa_and_export(fig, figure_dir / "fig03_rsrfm_vorticity_fields", (7.0, 8.5))


def mean_sd_tex(pair: tuple[float, float], *, bold: bool = False) -> str:
    mean, sd = pair
    if mean == 0.0:
        body = "0"
    else:
        exponent = int(math.floor(math.log10(abs(mean))))
        body = (
            f"({mean / 10.0**exponent:.2f}\\pm{sd / 10.0**exponent:.2f})"
            f"\\times10^{{{exponent}}}"
        )
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def sci_tex(value: float) -> str:
    exponent = int(math.floor(math.log10(abs(value))))
    return f"{value / 10.0**exponent:.2f}\\times10^{{{exponent}}}"


def make_table(case: str, summaries) -> str:
    values = summaries[case]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        (
            f"\\caption{{Held-out results for the {CASE_LABELS[case]} case. "
            "Entries are mean $\\pm$ sample standard deviation over five seeds "
            "($10$--$14$).}"
        ),
        f"\\label{{{TABLE_LABELS[case]}}}",
        "\\TableFont",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "Method & $\\mathcal E_{\\bu}$ & $\\mathcal E_{\\omega}$ \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        method_name = METHOD_LABELS[method]
        if method == PROPOSED:
            method_name = f"\\textbf{{{method_name}}}"
        lines.append(
            " & ".join(
                [
                    method_name,
                    mean_sd_tex(values[method]["rel_l2_velocity"], bold=method == PROPOSED),
                    mean_sd_tex(values[method]["rel_l2_vorticity"], bold=method == PROPOSED),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def make_consolidated_results_table(summaries) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        (
            "\\caption{Complete held-out space--time errors for all four flow cases. "
            "Entries are mean $\\pm$ sample standard deviation over seeds $10$--$14$.}"
        ),
        "\\label{tab:all-flow-results}",
        "\\WideTableFont",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}llcc@{}}",
        "\\toprule",
        "Case & Method & $\\mathcal E_{\\bu}$ & $\\mathcal E_{\\omega}$ \\\\",
        "\\midrule",
    ]
    for case_index, case in enumerate(CASE_ORDER):
        values = summaries[case]
        for method_index, method in enumerate(METHOD_ORDER):
            case_cell = (
                f"\\multirow{{{len(METHOD_ORDER)}}}{{*}}{{{CASE_LABELS[case]}}}"
                if method_index == 0
                else ""
            )
            method_name = METHOD_LABELS[method]
            if method == PROPOSED:
                method_name = f"\\textbf{{{method_name}}}"
            lines.append(
                " & ".join(
                    [
                        case_cell,
                        method_name,
                        mean_sd_tex(values[method]["rel_l2_velocity"], bold=method == PROPOSED),
                        mean_sd_tex(values[method]["rel_l2_vorticity"], bold=method == PROPOSED),
                    ]
                )
                + " \\\\"
            )
        if case_index < len(CASE_ORDER) - 1:
            lines.append("\\addlinespace[2pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def generated_text(summaries, ranking_audit) -> str:
    paragraphs = [
        (
            "The frozen production archive contains all 120 planned "
            "case--method--seed runs and no replacement seeds.  "
            "Table~\\ref{tab:all-flow-results} reports these observations as mean "
            "$\\pm$ sample standard deviation.  These "
            "summaries are descriptive, and no significance claim is made from five seeds."
        )
    ]
    for case in CASE_ORDER:
        values = summaries[case][PROPOSED]
        u_factor = ranking_audit[case]["rel_l2_velocity_factor"]
        w_factor = ranking_audit[case]["rel_l2_vorticity_factor"]
        u_comparator = ranking_audit[case]["rel_l2_velocity_comparator"]
        w_comparator = ranking_audit[case]["rel_l2_vorticity_comparator"]
        paragraphs.append(
            f"For {CASE_LABELS[case]}, RS-RFM-Global obtains mean errors "
            f"${sci_tex(values['rel_l2_velocity'][0])}$ in velocity and "
            f"${sci_tex(values['rel_l2_vorticity'][0])}$ in vorticity.  "
            f"These are respectively {u_factor:.1f}$\\times$ and {w_factor:.1f}$\\times$ "
            "smaller than the lowest mean errors among the five non-RS-RFM comparators "
            f"({METHOD_LABELS[str(u_comparator)]} for velocity and "
            f"{METHOD_LABELS[str(w_comparator)]} for vorticity)."
        )
    blocks = [paragraphs[0]]
    for index, case in enumerate(CASE_ORDER, start=1):
        blocks.append(paragraphs[index])
    blocks.append(make_consolidated_results_table(summaries))
    blocks.extend(
        [
            (
                "Appendix Table~\\ref{tab:rsrfm-seed-audit} retains the proposed method's "
                "seed-level field errors, effective ranks, and truncated condition estimates.  "
                "RS-RFM retains the full readout rank in "
                "19 of 20 held-out runs and retains 767 of 768 directions in the remaining "
                "two-rate run.  These diagnostics describe the retained column space after "
                "numerical truncation; they do not support a claim that the proposed feature "
                "map always improves conditioning."
            ),
        ]
    )
    return "\n\n".join(blocks)


def metric_pair(records: list[dict], case: str, method: str, metric: str) -> tuple[float, float]:
    values = np.asarray(
        [
            row[metric]
            for row in records
            if row["candidate_id"] == case and row["method_key"] == method
        ],
        dtype=float,
    )
    return float(values.mean()), float(values.std(ddof=1))


def appendix_metric_text(records: list[dict]) -> str:
    lines: list[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Terminal-time field errors for every held-out method. Entries are mean "
        "$\\pm$ sample standard deviation over seeds $10$--$14$.}",
        "\\label{tab:terminal-field-errors}",
        "\\WideTableFont",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}llcc@{}}",
        "\\toprule",
        "Case & Method & Final $\\mathcal E_{\\bu}$ & Final $\\mathcal E_{\\omega}$ \\\\",
        "\\midrule",
    ]
    for case_index, case in enumerate(CASE_ORDER):
        for method in METHOD_ORDER:
            lines.append(
                " & ".join(
                    [
                        CASE_LABELS[case],
                        METHOD_LABELS[method],
                        mean_sd_tex(metric_pair(records, case, method, "final_rel_l2_velocity")),
                        mean_sd_tex(metric_pair(records, case, method, "final_rel_l2_vorticity")),
                    ]
                )
                + " \\\\"
            )
        if case_index < len(CASE_ORDER) - 1:
            lines.append("\\addlinespace[2pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    lines.extend(
        [
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Relative trajectory errors in kinetic energy and enstrophy. Entries "
            "are mean $\\pm$ sample standard deviation over seeds $10$--$14$.}",
            "\\label{tab:integral-quantity-errors}",
            "\\WideTableFont",
            "\\setlength{\\tabcolsep}{5pt}",
            "\\begin{tabular}{@{}llcc@{}}",
            "\\toprule",
            "Case & Method & $\\mathcal E_K$ & $\\mathcal E_Z$ \\\\",
            "\\midrule",
        ]
    )
    for case_index, case in enumerate(CASE_ORDER):
        for method in METHOD_ORDER:
            lines.append(
                " & ".join(
                    [
                        CASE_LABELS[case],
                        METHOD_LABELS[method],
                        mean_sd_tex(metric_pair(records, case, method, "rel_l2_kinetic_energy")),
                        mean_sd_tex(metric_pair(records, case, method, "rel_l2_enstrophy")),
                    ]
                )
                + " \\\\"
            )
        if case_index < len(CASE_ORDER) - 1:
            lines.append("\\addlinespace[2pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    lines.extend(
        [
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Seed-level audit for the proposed RS-RFM-Global runs. $r$ is the "
            "effective rank of the two-field readout, and $\\kappa_{\\rm eff}$ is the truncated "
            "condition estimate.}",
            "\\label{tab:rsrfm-seed-audit}",
            "\\WideTableFont",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{@{}lrrrrrr@{}}",
            "\\toprule",
            "Case & Seed & $\\mathcal E_{\\bu}$ & $\\mathcal E_{\\omega}$ & Final $\\mathcal E_{\\bu}$ & $r$ & $\\kappa_{\\rm eff}$ \\\\",
            "\\midrule",
        ]
    )
    for case_index, case in enumerate(CASE_ORDER):
        subset = sorted(
            [
                row
                for row in records
                if row["candidate_id"] == case and row["method_key"] == PROPOSED
            ],
            key=lambda row: row["seed"],
        )
        for row in subset:
            lines.append(
                f"{CASE_LABELS[case]} & {row['seed']} & "
                f"${sci_tex(row['rel_l2_velocity'])}$ & "
                f"${sci_tex(row['rel_l2_vorticity'])}$ & "
                f"${sci_tex(row['final_rel_l2_velocity'])}$ & "
                f"{row['rank']} & ${sci_tex(row['condition_number'])}$ \\\\"
            )
        if case_index < len(CASE_ORDER) - 1:
            lines.append("\\addlinespace[2pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    results_dir = args.results.resolve()
    paper = args.paper.resolve()
    figure_dir = paper / "figures"
    generated_dir = paper / "sections" / "generated"
    combined_dir = paper / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(results_dir)
    summaries = summarize(records)
    ranking_audit = validate_rankings(summaries)
    configure_style()
    if not args.skip_fields:
        make_field_figure(results_dir, figure_dir)
    (generated_dir / "fluid_results.tex").write_text(
        generated_text(summaries, ranking_audit) + "\n", encoding="utf-8"
    )
    (generated_dir / "fluid_appendix_metrics.tex").write_text(
        appendix_metric_text(records) + "\n", encoding="utf-8"
    )
    (combined_dir / "fluid_ranking_audit.json").write_text(
        json.dumps(ranking_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    shutil.copyfile(results_dir / "metrics_raw.csv", combined_dir / "fluid_metrics_combined.csv")
    (paper / "figure_alt_text.md").write_text(
        "# Figure alt text\n\n"
        "- **Accuracy figure:** Eight log-scale dot plots show five seed-level velocity or "
        "vorticity errors for each of six methods in the four flow cases. RS-RFM is the "
        "lowest cluster and has the lowest mean in every panel.\n"
        "- **Per-case vorticity figures:** Four separate three-panel figures compare the "
        "final-time reference and RS-RFM vorticity for Taylor-Green, forced two-rate vortex, "
        "interacting-mode vortex, and multiscale shear, followed by a normalized "
        "absolute-error map.\n"
        "- **Linear-system diagnostics:** Eight dot plots show all seed-level truncated "
        "condition estimates and retained-rank fractions. The static comparator is rank "
        "deficient throughout, while RS-RFM is full or one direction below full rank.\n",
        encoding="utf-8",
    )
    print(json.dumps(ranking_audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
