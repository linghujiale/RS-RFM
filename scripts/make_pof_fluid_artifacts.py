# 中文注释：生成PoF 论文 二维流动论文制品，并输出可复核的实验或制图结果。
"""Generate PoF tables, figures, and alt text from archived flow records only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "rs_local",
    "rs_global",
    "rs_fixed_tau",
    "rs_static",
    "strfm_sov",
    "strfm_stc",
    "pielm",
]
METHOD_LABELS = {
    "rs_local": "RS-RFM-Local",
    "rs_global": "RS-RFM-Global",
    "rs_fixed_tau": "Fixed-$\\tau$ RFM",
    "rs_static": "Static-RFM-Local",
    "strfm_sov": "ST-RFM-SoV",
    "strfm_stc": "ST-RFM-STC",
    "pielm": "PIELM",
}
SHORT_LABELS = {
    "rs_local": "RS local",
    "rs_global": "RS global",
    "rs_fixed_tau": "Fixed $\\tau$",
    "rs_static": "Static",
    "strfm_sov": "ST-SoV",
    "strfm_stc": "ST-STC",
    "pielm": "PIELM",
}
COLORS = {
    "rs_local": "#0072B2",
    "rs_global": "#56B4E9",
    "rs_fixed_tau": "#E69F00",
    "rs_static": "#999999",
    "strfm_sov": "#009E73",
    "strfm_stc": "#CC79A7",
    "pielm": "#D55E00",
}
MARKERS = {
    "rs_local": "o",
    "rs_global": "s",
    "rs_fixed_tau": "^",
    "rs_static": "D",
    "strfm_sov": "v",
    "strfm_stc": "P",
    "pielm": "X",
}
CASE_ORDER = ["taylor_green", "two_rate_vortex", "double_shear"]
CASE_LABELS = {
    "taylor_green": "Taylor--Green",
    "two_rate_vortex": "forced two-rate vortex",
    "double_shear": "early double shear",
}
PLOT_CASE_LABELS = {
    "taylor_green": "Taylor--Green",
    "two_rate_vortex": "Forced two-rate vortex",
    "double_shear": "Early double shear",
}
METRICS = [
    "rel_l2_velocity",
    "rel_l2_vorticity",
    "final_rel_l2_velocity",
    "final_rel_l2_vorticity",
    "wall_seconds",
]
DIAGNOSTIC_METRICS = [
    "rel_l2_kinetic_energy",
    "rel_l2_enstrophy",
    "condition_number",
    "rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgv", type=Path, required=True)
    parser.add_argument("--two-rate", type=Path, required=True)
    parser.add_argument("--shear", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    return parser.parse_args()


def load_records(directory: Path, expected_case: str) -> list[dict]:
    with (directory / "metrics_raw.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    failures = json.loads((directory / "failures.json").read_text(encoding="utf-8"))
    if failures:
        raise ValueError(f"production directory {directory} contains failures: {failures}")
    if len(rows) != len(METHOD_ORDER) * 5:
        raise ValueError(f"expected 35 rows for {expected_case}, found {len(rows)}")
    seen: set[tuple[str, str, int]] = set()
    converted: list[dict] = []
    for row in rows:
        key = (row["case"], row["method_key"], int(row["seed"]))
        if key in seen:
            raise ValueError(f"duplicate production key: {key}")
        seen.add(key)
        if row["case"] != expected_case or row["method_key"] not in METHOD_ORDER:
            raise ValueError(f"unexpected production key: {key}")
        item = dict(row)
        item["seed"] = int(row["seed"])
        for metric in METRICS + DIAGNOSTIC_METRICS + ["final_relative_update"]:
            item[metric] = float(row[metric])
        item["picard_converged"] = row["picard_converged"].lower() == "true"
        converted.append(item)
    return converted


def summarize(records: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    result: dict[str, dict[str, tuple[float, float]]] = {}
    for method in METHOD_ORDER:
        subset = [row for row in records if row["method_key"] == method]
        result[method] = {}
        for metric in METRICS + DIAGNOSTIC_METRICS:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            result[method][metric] = (float(values.mean()), float(values.std(ddof=1)))
        result[method]["converged"] = (float(sum(row["picard_converged"] for row in subset)), 5.0)
    return result


def sci_tex(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return f"{mantissa:.2f}\\times10^{{{exponent}}}"


def mean_sd_tex(pair: tuple[float, float]) -> str:
    mean, sd = pair
    if mean == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(mean))))
    scaled_mean = mean / 10.0**exponent
    scaled_sd = sd / 10.0**exponent
    return f"$({scaled_mean:.2f}\\pm{scaled_sd:.2f})\\times10^{{{exponent}}}$"


def set_style() -> None:
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
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def grayscale_copy(png_path: Path) -> None:
    image = plt.imread(png_path)
    rgb = image[..., :3]
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    plt.imsave(png_path.with_name(png_path.stem + "_grayscale.png"), gray, cmap="gray", vmin=0, vmax=1)


def save_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    grayscale_copy(base.with_suffix(".png"))
    plt.close(fig)


def accuracy_figure(records_by_case: dict[str, list[dict]], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(6.69, 7.0), constrained_layout=True)
    jitter = np.linspace(-0.16, 0.16, 5)
    for row_index, case in enumerate(CASE_ORDER):
        for column_index, (metric, field_label) in enumerate(
            [("rel_l2_velocity", "Velocity"), ("rel_l2_vorticity", "Vorticity")]
        ):
            ax = axes[row_index, column_index]
            for method_index, method in enumerate(METHOD_ORDER):
                subset = sorted(
                    [row for row in records_by_case[case] if row["method_key"] == method],
                    key=lambda row: row["seed"],
                )
                values = np.asarray([row[metric] for row in subset])
                ax.scatter(
                    method_index + jitter,
                    values,
                    s=17,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
                ax.hlines(
                    values.mean(),
                    method_index - 0.23,
                    method_index + 0.23,
                    color="black",
                    linewidth=0.8,
                    zorder=4,
                )
            ax.set_yscale("log")
            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels([SHORT_LABELS[key] for key in METHOD_ORDER], rotation=31, ha="right")
            ax.set_ylabel(f"Relative $L^2$ error ({field_label.lower()})")
            ax.set_title(
                f"{PLOT_CASE_LABELS[case]}: {field_label}", loc="left", fontweight="bold"
            )
            ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.7)
            ax.set_axisbelow(True)
    for ax, label in zip(axes.flat, "abcdef", strict=True):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom")
    save_figure(fig, output / "fig03_fluid_accuracy_seeds")


def analytic_solution_figure(directories: dict[str, Path], output: Path) -> None:
    """Show final-time vorticity solutions and spatial errors for the two analytic cases."""

    cases = [
        ("taylor_green", "Taylor--Green", 2.0),
        ("two_rate_vortex", "Forced two-rate vortex", 1.0),
    ]
    column_titles = [
        "Reference",
        "RS-RFM-Local",
        "RS-RFM-Global",
        "Local $|\\omega-\\omega_*|$",
        "Global $|\\omega-\\omega_*|$",
    ]
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(6.69, 3.55),
        constrained_layout=True,
        sharex="row",
        sharey="row",
    )
    panel_labels = iter("abcdefghij")
    for row, (case, case_label, final_time) in enumerate(cases):
        directory = directories[case]
        with np.load(directory / f"reference_{case}.npz") as archive:
            x = archive["x"].copy()
            y = archive["y"].copy()
            reference = archive["omega"][-1].copy()
        predictions = []
        for method in ("rs_local", "rs_global"):
            with np.load(directory / f"prediction_{case}_{method}_seed0.npz") as archive:
                predictions.append(archive["omega"][-1].copy())
        signed_fields = [reference, *predictions]
        error_fields = [np.abs(field - reference) for field in predictions]
        signed_limit = max(float(np.max(np.abs(field))) for field in signed_fields)
        error_limit = max(float(np.max(field)) for field in error_fields)
        signed_image = None
        error_image = None
        for column, field in enumerate([*signed_fields, *error_fields]):
            ax = axes[row, column]
            if column < 3:
                signed_image = ax.pcolormesh(
                    x,
                    y,
                    field.T,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-signed_limit,
                    vmax=signed_limit,
                    rasterized=True,
                )
            else:
                error_image = ax.pcolormesh(
                    x,
                    y,
                    field.T,
                    shading="auto",
                    cmap="magma",
                    vmin=0.0,
                    vmax=error_limit,
                    rasterized=True,
                )
            ax.set_aspect("equal")
            ax.set_xlabel("$x$")
            if column == 0:
                ax.set_ylabel(f"{case_label}\n$y$")
                ax.text(
                    0.04,
                    0.05,
                    f"$t={final_time:g}$",
                    transform=ax.transAxes,
                    color="black",
                    fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
                )
            if row == 0:
                ax.set_title(column_titles[column], pad=3)
            ax.text(
                0.02,
                0.98,
                next(panel_labels),
                transform=ax.transAxes,
                fontweight="bold",
                va="top",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.66, "pad": 0.6},
            )
        solution_bar = fig.colorbar(signed_image, ax=axes[row, :3], shrink=0.78, pad=0.01)
        solution_bar.set_label("Vorticity $\\omega$")
        error_bar = fig.colorbar(error_image, ax=axes[row, 3:], shrink=0.78, pad=0.01)
        error_bar.set_label("Absolute vorticity error")
    save_figure(fig, output / "fig02_analytic_solution_fields")


def best_external(records: list[dict]) -> str:
    candidates = ["strfm_sov", "strfm_stc", "pielm"]
    scores = {}
    for method in candidates:
        subset = [row for row in records if row["method_key"] == method]
        scores[method] = float(
            np.mean(
                [
                    math.log10(row["rel_l2_velocity"]) + math.log10(row["rel_l2_vorticity"])
                    for row in subset
                ]
            )
        )
    return min(scores, key=scores.get)


def shear_field_figure(shear_dir: Path, shear_records: list[dict], output: Path) -> str:
    baseline = best_external(shear_records)
    reference = np.load(shear_dir / "reference_double_shear.npz")
    keys = ["reference", "rs_local", "rs_global", baseline]
    titles = ["Reference", METHOD_LABELS["rs_local"], METHOD_LABELS["rs_global"], METHOD_LABELS[baseline]]
    fields = [reference["omega"][-1]]
    for method in keys[1:]:
        prediction = np.load(shear_dir / f"prediction_double_shear_{method}_seed0.npz")
        fields.append(prediction["omega"][-1])
    limit = max(float(np.max(np.abs(field))) for field in fields)
    fig, axes = plt.subplots(1, 4, figsize=(6.69, 1.9), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for index, (ax, field, title) in enumerate(zip(axes, fields, titles, strict=True)):
        image = ax.pcolormesh(
            reference["x"],
            reference["y"],
            field.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(title, pad=3)
        ax.set_xlabel("$x$")
        if index == 0:
            ax.set_ylabel("$y$")
        ax.text(0.02, 0.98, "abcd"[index], transform=ax.transAxes, fontweight="bold", va="top")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Vorticity $\\omega$")
    save_figure(fig, output / "fig04_double_shear_vorticity")
    return baseline


def trajectory_figure(
    directories: dict[str, Path], records_by_case: dict[str, list[dict]], output: Path
) -> str:
    baseline = best_external(records_by_case["double_shear"])
    selected = ["rs_local", "rs_global", "rs_fixed_tau", "rs_static", baseline]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    fig = plt.figure(figsize=(6.69, 6.3), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[0.12, 1.0, 1.0, 1.0])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    axes = np.asarray(
        [[fig.add_subplot(grid[row + 1, column]) for column in range(2)] for row in range(3)]
    )
    for row_index, case in enumerate(CASE_ORDER):
        directory = directories[case]
        reference = np.load(directory / f"reference_{case}.npz")
        for column_index, (quantity, label) in enumerate(
            [("kinetic_energy", "Kinetic energy"), ("enstrophy", "Enstrophy")]
        ):
            ax = axes[row_index, column_index]
            reference_values = reference[quantity] / reference[quantity][0]
            ax.plot(reference["t"], reference_values, color="black", linewidth=1.7, label="Reference")
            for method, line_style in zip(selected, line_styles, strict=True):
                prediction = np.load(directory / f"prediction_{case}_{method}_seed0.npz")
                values = prediction[quantity] / reference[quantity][0]
                ax.plot(
                    prediction["t"],
                    values,
                    color=COLORS[method],
                    linestyle=line_style,
                    marker=MARKERS[method],
                    markersize=2.8,
                    label=SHORT_LABELS[method],
                )
            ax.set_xlabel("Time")
            ax.set_ylabel(f"Normalized {label.lower()}")
            ax.set_title(f"{PLOT_CASE_LABELS[case]}: {label}", loc="left", fontweight="bold")
            ax.grid(axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.7)
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=6,
        frameon=False,
    )
    for ax, label in zip(axes.flat, "abcdef", strict=True):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom")
    save_figure(fig, output / "fig05_energy_enstrophy")
    return baseline


def make_table(case: str, summary: dict[str, dict[str, tuple[float, float]]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Matched-capacity results for the {CASE_LABELS[case]} case. Values are mean $\\pm$ sample standard deviation over five fixed seeds.}}",
        f"\\label{{tab:{case}-results}}",
        "\\TableFont",
        "\\setlength{\\tabcolsep}{3.8pt}",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Method & $\\mathcal E_{\\bu}$ & $\\mathcal E_{\\omega}$ & Final $\\mathcal E_{\\bu}$ & Final $\\mathcal E_{\\omega}$ & Wall time (s) \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        values = summary[method]
        lines.append(
            " & ".join(
                [
                    METHOD_LABELS[method],
                    mean_sd_tex(values["rel_l2_velocity"]),
                    mean_sd_tex(values["rel_l2_vorticity"]),
                    mean_sd_tex(values["final_rel_l2_velocity"]),
                    mean_sd_tex(values["final_rel_l2_vorticity"]),
                    f"${values['wall_seconds'][0]:.1f}\\pm{values['wall_seconds'][1]:.1f}$",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def make_diagnostic_table(
    case: str,
    summary: dict[str, dict[str, tuple[float, float]]],
) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Integral-quantity errors and final weighted-matrix diagnostics for the {CASE_LABELS[case]} case. Values are mean $\\pm$ sample standard deviation over five fixed seeds; $\\kappa/r$ denotes condition number and numerical rank.}}",
        f"\\label{{tab:{case}-diagnostics}}",
        "\\TableFont",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "Method & $\\mathcal E_K$ & $\\mathcal E_Z$ & $\\kappa/r$ \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        values = summary[method]
        cells = [
            METHOD_LABELS[method],
            mean_sd_tex(values["rel_l2_kinetic_energy"]),
            mean_sd_tex(values["rel_l2_enstrophy"]),
            f"{mean_sd_tex(values['condition_number'])} / ${values['rank'][0]:.0f}\\pm{values['rank'][1]:.0f}$",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def generated_text(
    summaries: dict[str, dict[str, dict[str, tuple[float, float]]]],
    shear_records: list[dict],
    shear_dir: Path,
    baseline: str,
) -> str:
    shear = summaries["double_shear"]
    tgv = summaries["taylor_green"]
    two_rate = summaries["two_rate_vortex"]
    local_shear = shear["rs_local"]
    global_shear = shear["rs_global"]
    baseline_shear = shear[baseline]
    local_tgv = tgv["rs_local"]
    local_two_rate = two_rate["rs_local"]
    global_two_rate = two_rate["rs_global"]
    fixed_two_rate = two_rate["rs_fixed_tau"]
    static_two_rate = two_rate["rs_static"]
    validation_path = shear_dir / "reference_validation_double_shear.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    best_tgv_u = min(METHOD_ORDER, key=lambda method: tgv[method]["rel_l2_velocity"][0])
    best_tgv_w = min(METHOD_ORDER, key=lambda method: tgv[method]["rel_l2_vorticity"][0])
    best_two_rate_u = min(METHOD_ORDER, key=lambda method: two_rate[method]["rel_l2_velocity"][0])
    best_two_rate_w = min(METHOD_ORDER, key=lambda method: two_rate[method]["rel_l2_vorticity"][0])
    best_shear_u = min(METHOD_ORDER, key=lambda method: shear[method]["rel_l2_velocity"][0])
    best_shear_w = min(METHOD_ORDER, key=lambda method: shear[method]["rel_l2_vorticity"][0])
    paragraphs = [
        "The archived production records contain all 105 frozen case--method--seed combinations and no replacement seeds. "
        f"For Taylor--Green verification, RS-RFM-Local gives mean velocity and vorticity errors of ${sci_tex(local_tgv['rel_l2_velocity'][0])}$ and ${sci_tex(local_tgv['rel_l2_vorticity'][0])}$, respectively. "
        f"The smallest means in this verification are {METHOD_LABELS[best_tgv_u]} for velocity and {METHOD_LABELS[best_tgv_w]} for vorticity; no superiority claim is attached to this single-rate analytic case. "
        "Figure~\\ref{fig:analytic-solutions} shows final-time reference, RS-RFM-Local, and RS-RFM-Global vorticity fields together with their pointwise absolute errors. "
        "Figure~\\ref{fig:fluid-accuracy} displays every seed rather than only aggregate bars, and Table~\\ref{tab:taylor_green-results} reports the corresponding sample standard deviations.",
        f"For the targeted two-rate vortex, RS-RFM-Local gives mean velocity and vorticity errors of ${sci_tex(local_two_rate['rel_l2_velocity'][0])}$ and ${sci_tex(local_two_rate['rel_l2_vorticity'][0])}$. "
        f"RS-RFM-Global gives ${sci_tex(global_two_rate['rel_l2_velocity'][0])}$ and ${sci_tex(global_two_rate['rel_l2_vorticity'][0])}$ on this smooth domain. "
        f"The fixed-relaxation ablation gives ${sci_tex(fixed_two_rate['rel_l2_velocity'][0])}$ and ${sci_tex(fixed_two_rate['rel_l2_vorticity'][0])}$, while the static ablation gives ${sci_tex(static_two_rate['rel_l2_velocity'][0])}$ and ${sci_tex(static_two_rate['rel_l2_vorticity'][0])}$. "
        f"The lowest mean velocity and vorticity errors are obtained by {METHOD_LABELS[best_two_rate_u]} and {METHOD_LABELS[best_two_rate_w]}, respectively. "
        "The second row of Figure~\\ref{fig:analytic-solutions} shows where the two RS-RFM variants incur their final-time vorticity errors. "
        "This manufactured comparison is used only to diagnose representation of the prescribed decay rates; it is not pooled with the unforced double-shear result.",
        f"For the early double-shear interval, RS-RFM-Local gives mean velocity and vorticity errors of ${sci_tex(local_shear['rel_l2_velocity'][0])}$ and ${sci_tex(local_shear['rel_l2_vorticity'][0])}$. "
        f"The matched global variant gives ${sci_tex(global_shear['rel_l2_velocity'][0])}$ and ${sci_tex(global_shear['rel_l2_vorticity'][0])}$, while the external baseline with the smallest descriptive joint log-error summary is {METHOD_LABELS[baseline]} at ${sci_tex(baseline_shear['rel_l2_velocity'][0])}$ and ${sci_tex(baseline_shear['rel_l2_vorticity'][0])}$. "
        f"Across all methods, the smallest mean velocity error is given by {METHOD_LABELS[best_shear_u]}, whereas {METHOD_LABELS[best_shear_w]} gives the smallest mean vorticity error. "
        "These comparisons are interpreted metric by metric; a method is not called uniformly superior when the velocity and vorticity rankings differ.",
        f"The nested pseudo-spectral check gives relative discrepancies of ${sci_tex(float(validation['rel_l2_velocity']))}$ in velocity and ${sci_tex(float(validation['rel_l2_vorticity']))}$ in vorticity between the archived fine trajectory and the coarser calculation. "
        "The representative seed-0 vorticity fields in Figure~\\ref{fig:shear-vorticity} use one shared color scale, and Figure~\\ref{fig:integral-trajectories} compares normalized kinetic-energy and enstrophy trajectories. "
        "Only this early interval is evaluated; the data do not support a claim about fully developed vortex roll-up.",
    ]
    figures = [
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{figures/fig02_analytic_solution_fields.pdf}\n\\caption{Final-time vorticity solutions for the Taylor--Green vortex at $t=2$ (top) and the forced two-rate vortex at $t=1$ (bottom), shown for seed 0. The first three columns compare the reference, RS-RFM-Local, and RS-RFM-Global on a common symmetric vorticity scale within each row. The last two columns show pointwise absolute errors on a second common scale within each row. The panels are representative solution fields; five-seed statistics are reported separately in Figure~\\ref{fig:fluid-accuracy} and Tables~\\ref{tab:taylor_green-results}--\\ref{tab:two_rate_vortex-results}.}\n\\label{fig:analytic-solutions}\n\\end{figure*}",
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{figures/fig03_fluid_accuracy_seeds.pdf}\n\\caption{Space--time relative errors for all five fixed random seeds in the Taylor--Green, forced two-rate-vortex, and early double-shear cases. Dots are individual runs and black horizontal ticks are arithmetic means; all ordinates are logarithmic.}\n\\label{fig:fluid-accuracy}\n\\end{figure*}",
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{figures/fig04_double_shear_vorticity.pdf}\n\\caption{Final vorticity at $t=0.1$ for the early double-shear benchmark, shown for seed 0. All panels use the same diverging color scale. The displayed external baseline is selected descriptively by the joint five-seed velocity--vorticity log-error summary stated in the text; the full comparison is retained in Table~\\ref{tab:double_shear-results}.}\n\\label{fig:shear-vorticity}\n\\end{figure*}",
        "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{figures/fig05_energy_enstrophy.pdf}\n\\caption{Normalized kinetic energy and enstrophy for representative seed-0 predictions in all three flow cases. Curves are normalized by the reference initial value. The full seven-method error comparison is retained in the tables; trajectories are limited to the reference, the two RS-RFM variants, two ablations, and the strongest external double-shear baseline to keep the panels legible.}\n\\label{fig:integral-trajectories}\n\\end{figure*}",
    ]
    return "\n\n".join(
        [paragraphs[0], paragraphs[1], figures[0], figures[1], make_table("taylor_green", summaries["taylor_green"]), make_table("two_rate_vortex", summaries["two_rate_vortex"]), paragraphs[2], figures[2], make_table("double_shear", summaries["double_shear"]), paragraphs[3], figures[3]]
    )


def main() -> int:
    args = parse_args()
    paper = args.paper.resolve()
    figure_dir = paper / "figures"
    generated_dir = paper / "sections" / "generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    records_by_case = {
        "taylor_green": load_records(args.tgv.resolve(), "taylor_green"),
        "two_rate_vortex": load_records(args.two_rate.resolve(), "two_rate_vortex"),
        "double_shear": load_records(args.shear.resolve(), "double_shear"),
    }
    set_style()
    analytic_solution_figure(
        {
            "taylor_green": args.tgv.resolve(),
            "two_rate_vortex": args.two_rate.resolve(),
        },
        figure_dir,
    )
    accuracy_figure(records_by_case, figure_dir)
    baseline = shear_field_figure(args.shear.resolve(), records_by_case["double_shear"], figure_dir)
    trajectory_figure(
        {
            "taylor_green": args.tgv.resolve(),
            "two_rate_vortex": args.two_rate.resolve(),
            "double_shear": args.shear.resolve(),
        },
        records_by_case,
        figure_dir,
    )
    summaries = {case: summarize(records) for case, records in records_by_case.items()}
    tex = generated_text(summaries, records_by_case["double_shear"], args.shear.resolve(), baseline)
    (generated_dir / "fluid_results.tex").write_text(tex + "\n", encoding="utf-8")
    (generated_dir / "fluid_diagnostics.tex").write_text(
        "\n\n".join(make_diagnostic_table(case, summaries[case]) for case in CASE_ORDER) + "\n",
        encoding="utf-8",
    )
    alt_text = (
        "# Figure alt text\n\n"
        "- **Figure 1:** Workflow diagram showing frozen random parameters and sampled relaxation times feeding closed-form temporal responses, local or global spatial support, analytic PDE derivatives, and a linear least-squares readout.\n"
        "- **Figure 2:** Two rows of final-time vorticity fields compare the Taylor--Green and forced two-rate references with RS-RFM-Local and RS-RFM-Global; adjacent panels show pointwise absolute vorticity errors on shared within-row scales.\n"
        "- **Figure 3:** Six log-scale dot plots compare velocity and vorticity errors for seven methods on Taylor--Green, forced two-rate-vortex, and double-shear flows. Each method has five seed-level dots and a horizontal mean marker.\n"
        f"- **Figure 4:** Four equally scaled vorticity maps at double-shear final time compare the reference, RS-RFM-Local, RS-RFM-Global, and {METHOD_LABELS[baseline]}.\n"
        "- **Figure 5:** Six trajectory panels compare normalized kinetic energy and enstrophy for the three flow cases against the reference.\n"
    )
    (paper / "figure_alt_text.md").write_text(alt_text, encoding="utf-8")
    combined = [row for case in CASE_ORDER for row in records_by_case[case]]
    with (paper / "results" / "fluid_metrics_combined.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in combined for key in row}))
        writer.writeheader()
        writer.writerows(combined)
    print(f"generated artifacts; external field comparator={baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
