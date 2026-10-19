# 中文注释：生成PoF 论文 三维 Beltrami 流论文制品，并输出可复核的实验或制图结果。
"""Generate audited artifacts for the low-error 3D Beltrami verification."""

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


SEEDS = [30, 31, 32, 33, 34]
METRICS = (
    "rel_l2_velocity",
    "rel_l2_vorticity",
    "relative_divergence",
    "pressure_rms_over_u2",
)
FLOAT_FIELDS = set(METRICS) | {
    "final_relative_update",
    "wall_seconds",
    "condition_number",
    "residual_norm",
}
INT_FIELDS = {
    "seed",
    "features_per_field",
    "total_unknowns",
    "n_int",
    "n_bc_face",
    "n_rows",
    "rank",
    "picard_iterations",
}
PROFILE = {
    "total_features": 384,
    "n_int": 2400,
    "n_bc_face": 75,
    "n_gauge": 9,
    "eval_n": 16,
    "eval_nt": 9,
    "max_picard": 15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Root of the Beltrami archive")
    parser.add_argument("--paper", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(results: Path) -> list[dict]:
    """Load only the frozen high-capacity held-out records after auditing the gates."""

    standard_gate = results / "results" / "gate"
    high_gate = results / "results" / "high_gate"
    production = results / "results" / "high_production"
    if _read_json(standard_gate / "gate_report.json").get("passed") is not False:
        raise ValueError("the preserved P=256 standard gate must remain failed")
    high_report = _read_json(high_gate / "gate_report.json")
    if high_report.get("passed") is not True:
        raise ValueError("the P=384 continuation gate did not pass")
    status = _read_json(production / "status.json")
    expected_status = {"completed": 5, "records": 5, "failures": 0, "total": 5}
    if any(status.get(key) != value for key, value in expected_status.items()):
        raise ValueError(f"incomplete Beltrami production status: {status}")
    failures = _read_json(production / "failures.json")
    if failures:
        raise ValueError(f"Beltrami production contains failures: {failures}")
    metadata = _read_json(production / "run_metadata.json")
    integrity = metadata.get("integrity", {})
    if metadata.get("stage") != "production" or metadata.get("profile_name") != "high":
        raise ValueError("unexpected Beltrami evidence stage or profile")
    if integrity.get("confirmatory_eligible") is not False:
        raise ValueError("post-hoc Beltrami records must not be marked confirmatory-eligible")
    if integrity.get("replacement_runs") is not False:
        raise ValueError("replacement Beltrami runs are not allowed")
    if integrity.get("held_out_seeds") != SEEDS:
        raise ValueError("Beltrami held-out seed list drift")
    if integrity.get("failed_standard_gate_job_id") != "6784011":
        raise ValueError("the preserved standard gate is not identified")
    profile = metadata.get("profile", {})
    if any(profile.get(key) != value for key, value in PROFILE.items()):
        raise ValueError(f"Beltrami high profile drift: {profile}")

    with (production / "metrics_raw.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for field in FLOAT_FIELDS:
            row[field] = float(row[field])
            if not np.isfinite(row[field]):
                raise ValueError(f"nonfinite {field} for seed {row.get('seed')}")
        for field in INT_FIELDS:
            row[field] = int(row[field])
        row["picard_converged"] = row["picard_converged"].lower() == "true"
    if len(records) != 5 or sorted(row["seed"] for row in records) != SEEDS:
        raise ValueError("Beltrami production seed grid is incomplete")
    expected_constants = {
        "candidate_id": "forced_decaying_beltrami_3d",
        "case": "forced_decaying_beltrami_3d",
        "method_key": "rs_ic_global",
        "features_per_field": 384,
        "total_unknowns": 1536,
        "n_int": 2400,
        "n_bc_face": 75,
        "n_rows": 11409,
    }
    for field, expected in expected_constants.items():
        if any(row.get(field) != expected for row in records):
            raise ValueError(f"Beltrami production constant drift in {field}")
    if not all(row["picard_converged"] for row in records):
        raise ValueError("a Beltrami held-out Picard solve did not converge")
    return sorted(records, key=lambda row: row["seed"])


def summarize(records: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in METRICS + ("wall_seconds", "final_relative_update"):
        values = np.asarray([row[metric] for row in records], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary["runs"] = {"completed": len(records), "converged": sum(row["picard_converged"] for row in records)}
    return summary


def configure_style() -> None:
    setup_style(journal="general", lang="en", use_sciplots=False)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 8.4,
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
    render_preview(fig, str(base.with_name(base.name + "_preview.png")), dpi=150)
    if print_report(audit_layout(fig)) == "FAIL":
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
    panels = (
        ("rel_l2_velocity", r"$\mathcal{E}_{\mathbf{u}}$ (\%)"),
        ("rel_l2_vorticity", r"$\mathcal{E}_{\omega}$ (\%)"),
        ("relative_divergence", r"$\mathcal{E}_{\mathrm{div}}$ (\%)"),
        ("pressure_rms_over_u2", r"$\mathcal{E}_{p}^{(0)}$ (\%)"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.15), constrained_layout=True)
    x = np.arange(len(records))
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        values = 100.0 * np.asarray([row[metric] for row in records])
        ax.scatter(x, values, s=32, marker="o", color="#0072B2", edgecolor="black", linewidth=0.4, zorder=3)
        ax.axhline(values.mean(), color="black", linewidth=1.0, zorder=2)
        ax.set_xticks(x, [str(row["seed"]) for row in records])
        ax.set_xlabel("Held-out seed")
        ax.set_ylabel(r"Error (\%)")
        ax.set_title(ylabel, pad=5.0)
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        ax.margins(x=0.12, y=0.24)
    labels = add_panel_labels(fig, axes=axes, style="paren", x_offset_pt=3.0, y_offset_pt=3.0, ha="left", va="bottom")
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5})
    qa_and_export(fig, figure_dir / "fig05_beltrami_3d_accuracy", (7.0, 2.15))


def make_field_figure(results: Path, figure_dir: Path) -> None:
    production = results / "results" / "high_production"
    with np.load(production / "reference_helical_vortex_3d.npz") as archive:
        x, y, z = archive["x"].copy(), archive["y"].copy(), archive["z"].copy()
        reference = archive["vorticity"][-1].copy()
    prediction_path = production / "prediction_forced_decaying_beltrami_3d_rs_ic_global_seed30.npz"
    with np.load(prediction_path) as archive:
        prediction = archive["vorticity"][-1].copy()
    planes = (
        ("z", x, y, reference[:, :, len(z) // 2], prediction[:, :, len(z) // 2]),
        ("y", x, z, reference[:, len(y) // 2, :], prediction[:, len(y) // 2, :]),
    )
    field_max = max(float(np.linalg.norm(field, axis=-1).max()) for _, _, _, ref, pred in planes for field in (ref, pred))
    error_max = max(float(np.linalg.norm(pred - ref, axis=-1).max()) for _, _, _, ref, pred in planes)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.35), constrained_layout=True)
    field_image = error_image = None
    for row, (normal, axis_a, axis_b, ref, pred) in enumerate(planes):
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
                extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
                cmap="magma" if is_error else "viridis",
                vmin=0.0,
                vmax=error_max if is_error else field_max,
                interpolation="bilinear",
                aspect="equal",
                rasterized=True,
            )
            error_image = image if is_error else error_image
            field_image = image if not is_error else field_image
            ax.set_title(f"{title}, ${normal}=\\pi$", loc="left", fontweight="bold")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$" if normal == "z" else "$z$")
            ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
            ax.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    if field_image is not None:
        colorbar = fig.colorbar(field_image, ax=axes[:, :2], fraction=0.026, pad=0.02)
        colorbar.set_label(r"Vorticity magnitude $|\boldsymbol{\omega}|$")
    if error_image is not None:
        colorbar = fig.colorbar(error_image, ax=axes[:, 2], fraction=0.052, pad=0.03)
        colorbar.set_label(r"$|\boldsymbol{\omega}_P-\boldsymbol{\omega}_\star|$")
    labels = add_panel_labels(fig, axes=axes.flat, style="paren", x_offset_pt=3.0, y_offset_pt=-3.0, ha="left", va="top")
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5})
    qa_and_export(fig, figure_dir / "fig06_beltrami_3d_fields", (7.0, 4.35))


def mean_std_tex(stats: dict[str, float]) -> str:
    mean, std = stats["mean"], stats["sample_std"]
    exponent = int(np.floor(np.log10(abs(mean)))) if mean else 0
    scale = 10.0**exponent
    return rf"({mean / scale:.2f}\pm{std / scale:.2f})\times10^{{{exponent}}}"


def write_generated_tex(summary: dict[str, dict[str, float]], output: Path) -> None:
    eu, ew = summary["rel_l2_velocity"], summary["rel_l2_vorticity"]
    lines = [
        r"\subsection{Low-error three-dimensional Beltrami verification}",
        r"\label{subsec:beltrami-3d-results}",
        "",
        (
            "The post-hoc high-capacity continuation contains all five frozen held-out seeds "
            "$30$--$34$, with no failure or replacement.  Its independently seeded $P=384$ gate "
            "passed all five declared thresholds.  The mean space--time velocity and vorticity "
            f"errors are ${mean_std_tex(eu)}$ and ${mean_std_tex(ew)}$, respectively.  "
            "These runs verify that the three-dimensional assembly can attain low field error on a smooth single-mode "
            "flow; because the case and capacity were introduced after inspecting the harder "
            "two-mode result, they remain exploratory rather than confirmatory evidence."
        ),
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/fig05_beltrami_3d_accuracy.pdf}",
        r"\caption{Held-out accuracy for the low-error forced Beltrami verification on $[0,2\pi]^3$ with $\nu=0.02$, $T=1$, and $P=384$ scalar features per field.  Panels show all five independent seeds; the black line is the arithmetic mean.  Metrics are displayed as percentages.  The reference has zero pressure, so panel (d) reports the normalized pressure RMS in Eq.~\eqref{eq:pressure-rms-3d}.}",
        r"\label{fig:beltrami-3d-accuracy}",
        r"\end{figure*}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Low-error three-dimensional Beltrami verification.  Entries are mean $\pm$ sample standard deviation over held-out seeds $30$--$34$.}",
        r"\label{tab:beltrami-3d-results}",
        r"\TableFont",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & $\mathcal E_{\bu}$ & $\mathcal E_{\boldsymbol{\omega}}$ & $\mathcal E_{\rm div}$ & $\mathcal E_p^{(0)}$ \\",
        r"\midrule",
        (
            r"RS-RFM ($P=384$) & "
            f"${mean_std_tex(summary['rel_l2_velocity'])}$ & "
            f"${mean_std_tex(summary['rel_l2_vorticity'])}$ & "
            f"${mean_std_tex(summary['relative_divergence'])}$ & "
            f"${mean_std_tex(summary['pressure_rms_over_u2'])}$ \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/fig06_beltrami_3d_fields.pdf}",
        r"\caption{Final-time vorticity-magnitude slices for held-out seed 30 in the low-error Beltrami verification.  The rows show $z=\pi$ and $y=\pi$.  Reference and prediction panels share one magnitude scale; pointwise-error panels share a separate scale.  The stored $16\times16$ slices are bilinearly rendered for display, while Table~\ref{tab:beltrami-3d-results} uses all five seeds and the full $16^3\times9$ evaluation grid.}",
        r"\label{fig:beltrami-3d-fields}",
        r"\end{figure*}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results, paper = args.results.resolve(), args.paper.resolve()
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
    write_generated_tex(summary, generated_dir / "fluid_3d_beltrami_results.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
