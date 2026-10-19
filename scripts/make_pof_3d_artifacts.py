# 中文注释：生成PoF 论文 三维流动论文制品，并输出可复核的实验或制图结果。
"""Generate the 3D PoF table and figures from frozen production records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SKILL_SCRIPTS = PROJECT_ROOT / "figure_style"
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from layout_tools import add_panel_labels, finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


PROPOSED = "rs_ic_global"
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
    "strfm_sov": "SoV",
    "strfm_stc": "STC",
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
SEEDS = [10, 11, 12, 13, 14]
FLOAT_FIELDS = {
    "rel_l2_velocity",
    "rel_l2_vorticity",
    "relative_divergence",
    "pressure_rms_over_u2",
    "final_rel_l2_velocity",
    "final_rel_l2_vorticity",
    "rel_l2_kinetic_energy",
    "rel_l2_enstrophy",
    "final_relative_update",
    "condition_number",
    "residual_norm",
    "wall_seconds",
    "viscosity",
    "horizon",
    "lambda_slow",
    "lambda_fast",
    "tau_min_scale",
    "tau_max_scale",
    "static_fraction",
}
INT_FIELDS = {
    "seed",
    "rank",
    "features_per_field",
    "total_unknowns",
    "n_int",
    "n_bc_face",
    "n_rows",
    "picard_iterations",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    return parser.parse_args()


def load_records(results: Path) -> list[dict]:
    status = json.loads((results / "status.json").read_text(encoding="utf-8"))
    failures = json.loads((results / "failures.json").read_text(encoding="utf-8"))
    metadata = json.loads((results / "run_metadata.json").read_text(encoding="utf-8"))
    expected_status = {"completed": 30, "records": 30, "failures": 0, "total": 30}
    if any(status.get(key) != value for key, value in expected_status.items()):
        raise ValueError(f"incomplete 3D exploratory status: {status}")
    if failures:
        raise ValueError(f"3D production contains failures: {failures}")
    integrity = metadata.get("integrity", {})
    if metadata.get("stage") != "exploratory":
        raise ValueError(f"unexpected 3D evidence stage: {metadata.get('stage')}")
    if integrity.get("confirmatory_eligible") is not False:
        raise ValueError("3D exploratory records must not be marked confirmatory-eligible")
    if integrity.get("failed_gate_job_id") != "6783779":
        raise ValueError("3D exploratory metadata does not identify the preserved failed gate")
    expected_profile = {
        "total_features": 256,
        "n_int": 1600,
        "n_bc_face": 50,
        "eval_n": 12,
        "eval_nt": 7,
        "max_picard": 10,
    }
    profile = metadata.get("profile", {})
    if any(profile.get(key) != value for key, value in expected_profile.items()):
        raise ValueError(f"3D exploratory profile drift: {profile}")
    with (results / "metrics_raw.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for field in FLOAT_FIELDS:
            row[field] = float(row[field])
            if not np.isfinite(row[field]):
                raise ValueError(f"nonfinite {field} for {row.get('method_key')} seed {row.get('seed')}")
        for field in INT_FIELDS:
            row[field] = int(row[field])
        row["picard_converged"] = row["picard_converged"].lower() == "true"
    keys = {(row["method_key"], row["seed"]) for row in records}
    expected = {(method, seed) for method in METHOD_ORDER for seed in SEEDS}
    if keys != expected or len(records) != len(expected):
        raise ValueError(f"3D method-seed grid mismatch: missing={expected - keys}, extra={keys - expected}")
    expected_constants = {
        "candidate_id": "interacting_helical_vortex_3d",
        "case": "helical_vortex_3d",
        "features_per_field": 256,
        "total_unknowns": 1024,
        "n_int": 1600,
        "n_bc_face": 50,
        "n_rows": 7607,
        "viscosity": 0.02,
        "horizon": 1.0,
        "lambda_slow": 0.5,
        "lambda_fast": 4.0,
        "tau_min_scale": 0.08,
        "tau_max_scale": 3.0,
    }
    for field, expected_value in expected_constants.items():
        if any(row.get(field) != expected_value for row in records):
            raise ValueError(f"3D exploratory constant drift in {field}")
    return records


def configure_style() -> None:
    setup_style(journal="general", lang="en", use_sciplots=False)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
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


def grouped(records: list[dict]) -> dict[str, list[dict]]:
    return {
        method: sorted(
            [row for row in records if row["method_key"] == method], key=lambda row: row["seed"]
        )
        for method in METHOD_ORDER
    }


def summarize(records: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for method, rows in grouped(records).items():
        summary[method] = {}
        for metric in (
            "rel_l2_velocity",
            "rel_l2_vorticity",
            "relative_divergence",
            "pressure_rms_over_u2",
            "wall_seconds",
        ):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[method][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        summary[method]["converged"] = int(sum(row["picard_converged"] for row in rows))
        summary[method]["rank_min"] = int(min(row["rank"] for row in rows))
    return summary


def make_accuracy_figure(records: list[dict], figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), constrained_layout=True)
    jitter = np.linspace(-0.16, 0.16, len(SEEDS))
    panels = (
        ("rel_l2_velocity", r"Velocity error $\mathcal{E}_{\mathbf{u}}$"),
        ("rel_l2_vorticity", r"Vorticity error $\mathcal{E}_{\omega}$"),
        ("relative_divergence", r"Divergence $\mathcal{E}_{\mathrm{div}}$"),
    )
    groups = grouped(records)
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for method_index, method in enumerate(METHOD_ORDER):
            values = np.asarray([row[metric] for row in groups[method]])
            ax.scatter(
                method_index + jitter,
                values,
                s=25,
                marker=MARKERS[method],
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
            ax.hlines(values.mean(), method_index - 0.24, method_index + 0.24, color="black", lw=1.0)
        ax.set_yscale("log")
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(
            [TICK_LABELS[method] for method in METHOD_ORDER],
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        ax.margins(x=0.06, y=0.15)
    labels = add_panel_labels(
        fig,
        axes=axes,
        style="paren",
        x_offset_pt=3.0,
        y_offset_pt=3.0,
        ha="left",
        va="bottom",
    )
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5})
    qa_and_export(fig, figure_dir / "fig05_helical_vortex_3d_accuracy", (7.0, 2.55))


def make_field_figure(results: Path, figure_dir: Path) -> None:
    with np.load(results / "reference_helical_vortex_3d.npz") as archive:
        x = archive["x"].copy()
        y = archive["y"].copy()
        z = archive["z"].copy()
        reference = archive["vorticity"][-1].copy()
    with np.load(results / "prediction_helical_vortex_3d_rs_ic_global_seed10.npz") as archive:
        prediction = archive["vorticity"][-1].copy()
    planes = [
        ("z", len(z) // 2, x, y, reference[:, :, len(z) // 2], prediction[:, :, len(z) // 2]),
        ("y", len(y) // 2, x, z, reference[:, len(y) // 2, :], prediction[:, len(y) // 2, :]),
    ]
    field_max = max(
        float(np.linalg.norm(field, axis=-1).max())
        for _, _, _, _, ref, pred in planes
        for field in (ref, pred)
    )
    error_max = max(
        float(np.linalg.norm(pred - ref, axis=-1).max())
        for _, _, _, _, ref, pred in planes
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.45), constrained_layout=True)
    field_image = None
    error_image = None
    for row, (normal_axis, index, axis_a, axis_b, ref, pred) in enumerate(planes):
        fields = (
            (np.linalg.norm(ref, axis=-1), "Reference"),
            (np.linalg.norm(pred, axis=-1), "RS-RFM-Global"),
            (np.linalg.norm(pred - ref, axis=-1), "Pointwise error"),
        )
        for column, (field, title) in enumerate(fields):
            ax = axes[row, column]
            is_error = column == 2
            image = ax.imshow(
                field.T,
                origin="lower",
                extent=(axis_a[0], axis_a[-1] + (axis_a[1] - axis_a[0]), axis_b[0], axis_b[-1] + (axis_b[1] - axis_b[0])),
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
            ax.set_title(f"{title}, ${normal_axis}=\\pi$", loc="left", fontweight="bold")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$" if normal_axis == "z" else "$z$")
            ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
            ax.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    if field_image is not None:
        colorbar = fig.colorbar(field_image, ax=axes[:, :2], fraction=0.026, pad=0.02)
        colorbar.set_label(r"Vorticity magnitude $|\boldsymbol{\omega}|$")
    if error_image is not None:
        colorbar = fig.colorbar(error_image, ax=axes[:, 2], fraction=0.052, pad=0.03)
        colorbar.set_label(r"$|\boldsymbol{\omega}_P-\boldsymbol{\omega}_\star|$")
    labels = add_panel_labels(
        fig,
        axes=axes.flat,
        style="paren",
        x_offset_pt=3.0,
        y_offset_pt=-3.0,
        ha="left",
        va="top",
    )
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5})
    qa_and_export(fig, figure_dir / "fig06_helical_vortex_3d_fields", (7.0, 4.45))


def sci_tex(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2f}\times10^{{{exponent}}}"


def mean_std_tex(stats: dict[str, float]) -> str:
    mean = stats["mean"]
    std = stats["sample_std"]
    exponent = int(np.floor(np.log10(abs(mean)))) if mean else 0
    scale = 10.0**exponent
    return rf"({mean / scale:.2f}\pm{std / scale:.2f})\times10^{{{exponent}}}"


def write_generated_tex(summary: dict[str, dict], output: Path) -> None:
    best_velocity = min(METHOD_ORDER, key=lambda method: summary[method]["rel_l2_velocity"]["mean"])
    best_vorticity = min(METHOD_ORDER, key=lambda method: summary[method]["rel_l2_vorticity"]["mean"])
    best_divergence = min(METHOD_ORDER, key=lambda method: summary[method]["relative_divergence"]["mean"])
    best_pressure = min(METHOD_ORDER, key=lambda method: summary[method]["pressure_rms_over_u2"]["mean"])
    proposed = summary[PROPOSED]
    comparator_velocity = min(
        (method for method in METHOD_ORDER if method != PROPOSED),
        key=lambda method: summary[method]["rel_l2_velocity"]["mean"],
    )
    comparator_vorticity = min(
        (method for method in METHOD_ORDER if method != PROPOSED),
        key=lambda method: summary[method]["rel_l2_vorticity"]["mean"],
    )
    velocity_factor = summary[comparator_velocity]["rel_l2_velocity"]["mean"] / proposed["rel_l2_velocity"]["mean"]
    vorticity_factor = summary[comparator_vorticity]["rel_l2_vorticity"]["mean"] / proposed["rel_l2_vorticity"]["mean"]

    def comparison_phrase(metric: str, factor: float, comparator: str) -> str:
        label = METHOD_LABELS[comparator]
        if factor >= 1.0:
            return f"Its mean {metric} error is smaller than that of {label}, the strongest non-RS comparator, by a factor of {factor:.1f}."
        return f"Its mean {metric} error exceeds that of {label}, the strongest non-RS comparator, by a factor of {1.0 / factor:.1f}."

    lines = [
        r"\subsection{Exploratory three-dimensional interacting helical vortex}",
        r"\label{subsec:helical-vortex-3d-results}",
        "",
        (
            "The exploratory three-dimensional extension contains all 30 planned method--seed runs "
            "and no replacement seed. Its frozen seed-0 accuracy gate was not met, so these results "
            "are descriptive and are not pooled with the confirmatory two-dimensional suite. "
            "Figure~\\ref{fig:helical-3d-accuracy} displays the five observations for velocity "
            "error, vorticity error, and normalized divergence."
        ),
        "",
        (
            f"RS-RFM-Global gives mean space--time errors ${sci_tex(proposed['rel_l2_velocity']['mean'])}$ "
            f"in velocity and ${sci_tex(proposed['rel_l2_vorticity']['mean'])}$ in vorticity. "
            f"{comparison_phrase('velocity', velocity_factor, comparator_velocity)} "
            f"{comparison_phrase('vorticity', vorticity_factor, comparator_vorticity)} "
            f"The lowest means are obtained by {METHOD_LABELS[best_velocity]} for velocity, "
            f"{METHOD_LABELS[best_vorticity]} for vorticity, and {METHOD_LABELS[best_divergence]} "
            f"for the divergence diagnostic. {METHOD_LABELS[best_pressure]} gives the lowest "
            "normalized pressure RMS."
        ),
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/fig05_helical_vortex_3d_accuracy.pdf}",
        r"\caption{Exploratory held-out errors for the forced interacting helical vortex on $[0,2\pi]^3$ with $\nu=0.02$ and $T=1$. The frozen seed-0 accuracy gate was not met; these results therefore characterize three-dimensional behavior without supporting a confirmatory superiority claim. Panels report space--time relative velocity error, relative vorticity error, and divergence normalized by the reference velocity-gradient norm. Each colored symbol is one held-out seed ($10$--$14$, $n=5$ per method); black ticks are arithmetic means. All methods use $P=256$ scalar features per field and identical collocation points for a given seed.}",
        r"\label{fig:helical-3d-accuracy}",
        r"\end{figure*}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Exploratory three-dimensional helical-vortex results. Entries are mean $\pm$ sample standard deviation over seeds $10$--$14$; Div. is the normalized divergence, and $\mathcal E_p^{(0)}$ is the normalized pressure RMS in Eq.~\eqref{eq:pressure-rms-3d}.}",
        r"\label{tab:helical-3d-results}",
        r"\TableFont",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & $\mathcal E_{\bu}$ & $\mathcal E_{\boldsymbol{\omega}}$ & Div. & $\mathcal E_p^{(0)}$ \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        values = summary[method]
        cells = []
        for metric, best in (
            ("rel_l2_velocity", best_velocity),
            ("rel_l2_vorticity", best_vorticity),
            ("relative_divergence", best_divergence),
        ):
            text = f"${mean_std_tex(values[metric])}$"
            cells.append(rf"\textbf{{{text}}}" if method == best else text)
        label = METHOD_LABELS[method]
        if method == PROPOSED:
            label = rf"\textbf{{{label}}}"
        pressure = f"${mean_std_tex(values['pressure_rms_over_u2'])}$"
        if method == best_pressure:
            pressure = rf"\textbf{{{pressure}}}"
        lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & {pressure} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{figure*}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{figures/fig06_helical_vortex_3d_fields.pdf}",
            r"\caption{Final-time vorticity-magnitude slices for held-out seed 10 in the three-dimensional interacting helical vortex. The top and bottom rows show the orthogonal planes $z=\pi$ and $y=\pi$; stored values on the $12\times12$ slice grids are bilinearly rendered for display. Reference and RS-RFM-Global panels share one magnitude scale; error panels show $|\boldsymbol{\omega}_P-\boldsymbol{\omega}_\star|$ on a second shared scale. Quantitative conclusions use all five seeds in Table~\ref{tab:helical-3d-results}.}",
            r"\label{fig:helical-3d-fields}",
            r"\end{figure*}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results = args.results.resolve()
    paper = args.paper.resolve()
    figure_dir = paper / "figures"
    generated_dir = paper / "sections" / "generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(results)
    summary = summarize(records)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    configure_style()
    make_accuracy_figure(records, figure_dir)
    make_field_figure(results, figure_dir)
    write_generated_tex(summary, generated_dir / "fluid_3d_results.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
