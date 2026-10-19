# 中文注释：生成PoF 论文 二维衰减流实验 论文制品，并输出可复核的实验或制图结果。
"""Generate audited manuscript artifacts for the physical decaying-turbulence case."""

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
SEEDS = [50, 51, 52, 53, 54]
METRICS = [
    "rel_l2_velocity",
    "rel_l2_vorticity",
    "rel_l2_kinetic_energy",
    "rel_l2_enstrophy",
]
FLOAT_FIELDS = set(METRICS) | {
    "final_relative_update",
    "final_rel_l2_velocity",
    "final_rel_l2_vorticity",
    "residual_norm",
    "condition_number",
    "wall_seconds",
}
INT_FIELDS = {"seed", "features_per_field", "n_int", "n_bc_total", "picard_iterations", "rank"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_archive(results: Path) -> tuple[Path, list[dict], dict, dict, dict]:
    selected = (results / "selected_profile.txt").read_text(encoding="utf-8").strip()
    if selected != "continuation":
        raise ValueError(f"unexpected selected profile: {selected}")
    development = results / "development" / selected
    gate = results / "gate" / selected
    production = results / "production" / selected
    for directory, mode in ((development, "development"), (gate, "gate"), (production, "production")):
        validation = read_json(directory / f"validation_{mode}.json")
        if validation.get("passed") is not True:
            raise ValueError(f"{mode} validation did not pass: {validation}")
    status = read_json(production / "status.json")
    expected_runs = len(METHOD_ORDER) * len(SEEDS)
    if any(
        status.get(key) != value
        for key, value in {"completed": expected_runs, "total": expected_runs, "records": expected_runs, "failures": 0}.items()
    ):
        raise ValueError(f"incomplete production status: {status}")
    if read_json(production / "failures.json"):
        raise ValueError("production failure list is not empty")
    metadata = read_json(production / "run_metadata.json")
    if metadata.get("stage") != "production" or metadata.get("profile") != "continuation":
        raise ValueError("production stage/profile drift")
    if metadata.get("seeds") != SEEDS or metadata.get("methods") != METHOD_ORDER:
        raise ValueError("production method or seed grid drift")
    profile = metadata.get("profile_parameters", {})
    expected_profile = {
        "total_features": 1280,
        "n_int": 4800,
        "n_bc_side": 140,
        "n_gauge": 11,
        "eval_n": 64,
        "eval_nt": 11,
        "reference_n": 128,
        "reference_dt": 5.0e-4,
        "max_picard": 20,
    }
    if any(profile.get(key) != value for key, value in expected_profile.items()):
        raise ValueError(f"continuation profile drift: {profile}")
    integrity = metadata.get("integrity", {})
    if not integrity.get("physical_initial_seed_fixed") or integrity.get("body_force") != 0.0:
        raise ValueError("physical-case integrity metadata drift")
    reference_validation = read_json(production / "reference_validation.json")
    if max(reference_validation["rel_l2_velocity"], reference_validation["rel_l2_vorticity"]) >= 5.0e-3:
        raise ValueError(f"reference grid validation failed: {reference_validation}")
    gate_validation = read_json(gate / "validation_gate.json")

    with (production / "metrics_raw.csv").open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    expected = {(method, seed) for method in METHOD_ORDER for seed in SEEDS}
    observed: set[tuple[str, int]] = set()
    for row in records:
        for field in FLOAT_FIELDS:
            row[field] = float(row[field])
            if not np.isfinite(row[field]):
                raise ValueError(f"nonfinite {field} in {row}")
        for field in INT_FIELDS:
            row[field] = int(float(row[field]))
        row["picard_converged"] = row["picard_converged"].lower() == "true"
        observed.add((row["method_key"], row["seed"]))
    if len(records) != expected_runs or observed != expected:
        raise ValueError("production metric grid is incomplete or contains duplicates")
    return production, records, metadata, reference_validation, gate_validation


def summarize(records: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    output: dict[str, dict[str, tuple[float, float]]] = {}
    for method in METHOD_ORDER:
        rows = sorted((row for row in records if row["method_key"] == method), key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != SEEDS:
            raise ValueError(f"missing seed for {method}")
        output[method] = {}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            output[method][metric] = (float(values.mean()), float(values.std(ddof=1)))
        output[method]["converged"] = (
            float(sum(row["picard_converged"] for row in rows)),
            float(len(rows)),
        )
    return output


def configure_style() -> None:
    setup_style(journal="general", lang="en", constrained_layout=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.4,
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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)
    jitter = np.linspace(-0.16, 0.16, len(SEEDS))
    for ax, (metric, title, gate_limit) in zip(
        axes,
        (
            ("rel_l2_velocity", "Velocity", 3.0e-2),
            ("rel_l2_vorticity", "Vorticity", 5.0e-2),
        ),
        strict=True,
    ):
        for method_index, method in enumerate(METHOD_ORDER):
            rows = sorted(
                (row for row in records if row["method_key"] == method),
                key=lambda row: row["seed"],
            )
            values = np.asarray([row[metric] for row in rows], dtype=float)
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
            ax.hlines(values.mean(), method_index - 0.24, method_index + 0.24, color="black", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([TICK_LABELS[method] for method in METHOD_ORDER], rotation=25, ha="right")
        ax.set_ylabel("Space-time relative $L^2$ error")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(gate_limit, color="#4D4D4D", linestyle=":", linewidth=0.8, zorder=1)
        ax.text(
            0.99,
            gate_limit,
            " gate",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="#4D4D4D",
            fontsize=6.5,
        )
        ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
        ax.margins(x=0.07, y=0.18)
    labels = add_panel_labels(fig, axes=axes, style="paren", x_offset_pt=2.0, y_offset_pt=-2.0)
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4})
    qa_and_export(fig, figure_dir / "fig07_decaying_turbulence_accuracy", (7.0, 2.45))


def make_physics_figure(
    production: Path,
    figure_dir: Path,
    prediction_path: Path | None = None,
    primary_label: str = "RS-RFM-Global",
    output_name: str = "fig08_decaying_turbulence_physics",
) -> None:
    with np.load(production / "reference_decaying_turbulence.npz") as archive:
        x = archive["x"].copy()
        y = archive["y"].copy()
        time_values = archive["t"].copy()
        reference_omega = archive["omega"].copy()
        reference_energy = archive["kinetic_energy"].copy()
        reference_enstrophy = archive["enstrophy"].copy()
    if prediction_path is None:
        prediction_path = production / "prediction_decaying_turbulence_rs_ic_global_seed50.npz"
    with np.load(prediction_path) as archive:
        prediction_omega = archive["omega"].copy()
        prediction_energy = archive["kinetic_energy"].copy()
        prediction_enstrophy = archive["enstrophy"].copy()

    scale = float(np.max(np.abs(reference_omega)))
    normalized_error = np.abs(prediction_omega[-1] - reference_omega[-1]) / scale
    error_max = max(float(normalized_error.max()), 1.0e-6)
    extent = (float(x[0]), float(x[-1] + (x[1] - x[0])), float(y[0]), float(y[-1] + (y[1] - y[0])))
    fig = plt.figure(figsize=(7.0, 4.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=(1.0, 1.0, 1.0, 0.055))
    field_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    error_ax = fig.add_subplot(grid[1, 0])
    energy_ax = fig.add_subplot(grid[1, 1])
    spectrum_ax = fig.add_subplot(grid[1, 2])
    field_cax = fig.add_subplot(grid[0, 3])
    error_cax = fig.add_subplot(grid[1, 3])

    fields = (
        (reference_omega[0] / scale, r"DNS, $t=0$"),
        (reference_omega[-1] / scale, rf"DNS, $t={time_values[-1]:.2f}$"),
        (prediction_omega[-1] / scale, rf"{primary_label}, $t={time_values[-1]:.2f}$"),
    )
    field_image = None
    for ax, (field, title) in zip(field_axes, fields, strict=True):
        field_image = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="bilinear",
            aspect="equal",
            rasterized=True,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        ax.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    colorbar = fig.colorbar(field_image, cax=field_cax)
    colorbar.set_label(r"$\omega/\max|\omega_{\rm DNS}|$")

    error_image = error_ax.imshow(
        normalized_error.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=error_max,
        interpolation="bilinear",
        aspect="equal",
        rasterized=True,
    )
    error_ax.set_title("Final normalized error", loc="left", fontweight="bold")
    error_ax.set_xlabel("$x$")
    error_ax.set_ylabel("$y$")
    error_ax.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    error_ax.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    error_colorbar = fig.colorbar(error_image, cax=error_cax)
    error_colorbar.set_label(r"$|\omega_P-\omega_{\rm DNS}|/\max|\omega_{\rm DNS}|$")

    for reference, prediction, color, marker, quantity in (
        (reference_energy, prediction_energy, "#0072B2", "o", "$K$"),
        (reference_enstrophy, prediction_enstrophy, "#D55E00", "s", "$Z$"),
    ):
        energy_ax.plot(
            time_values,
            reference / reference[0],
            color=color,
            linewidth=1.3,
            label=f"DNS {quantity}",
        )
        energy_ax.plot(
            time_values,
            prediction / reference[0],
            color=color,
            linestyle="--",
            marker=marker,
            markersize=2.8,
            markevery=2,
            linewidth=1.0,
            label=f"{primary_label.replace('RS-RFM-', '')} {quantity}",
        )
    integral_curves = (
        reference_energy / reference_energy[0],
        prediction_energy / reference_energy[0],
        reference_enstrophy / reference_enstrophy[0],
        prediction_enstrophy / reference_enstrophy[0],
    )
    energy_ax.set_xlabel("Time")
    energy_ax.set_ylabel(r"$Q(t)/Q(0)$")
    energy_ax.set_title("Integral decay", loc="left", fontweight="bold")
    energy_ax.set_ylim(max(0.0, min(float(np.min(curve)) for curve in integral_curves) - 0.04), 1.025)
    energy_ax.grid(axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.75)
    energy_ax.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.5)

    def kinetic_energy_spectrum(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = omega.shape[0]
        omega_hat = np.fft.fft2(omega) / (n * n)
        frequencies = np.fft.fftfreq(n, d=1.0 / n)
        kx, ky = np.meshgrid(frequencies, frequencies, indexing="ij")
        k_squared = kx**2 + ky**2
        shell = np.rint(np.sqrt(k_squared)).astype(int)
        density = np.zeros_like(k_squared, dtype=float)
        nonzero = k_squared > 0.0
        density[nonzero] = 0.5 * np.abs(omega_hat[nonzero]) ** 2 / k_squared[nonzero]
        spectrum = np.bincount(shell.ravel(), weights=density.ravel(), minlength=n // 2 + 1)
        wave_number = np.arange(len(spectrum))
        return wave_number[1 : n // 3 + 1], spectrum[1 : n // 3 + 1]

    reference_initial_k, reference_initial_spectrum = kinetic_energy_spectrum(reference_omega[0])
    reference_final_k, reference_final_spectrum = kinetic_energy_spectrum(reference_omega[-1])
    prediction_final_k, prediction_final_spectrum = kinetic_energy_spectrum(prediction_omega[-1])
    spectrum_normalizer = float(reference_initial_spectrum.sum())
    for wave_number, spectrum, color, linestyle, marker, label in (
        (reference_initial_k, reference_initial_spectrum, "#7F7F7F", ":", None, "DNS, $t=0$"),
        (reference_final_k, reference_final_spectrum, "black", "-", None, "DNS, $t=T$"),
        (
            prediction_final_k,
            prediction_final_spectrum,
            "#0072B2",
            "--",
            "o",
            f"{primary_label.replace('RS-RFM-', '')}, $t=T$",
        ),
    ):
        positive = spectrum > 1.0e-15 * spectrum_normalizer
        spectrum_ax.plot(
            wave_number[positive],
            spectrum[positive] / spectrum_normalizer,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=2.6,
            linewidth=1.1,
            label=label,
        )
    spectrum_ax.set_yscale("log")
    spectrum_ax.set_xlabel("Wave number $k$")
    spectrum_ax.set_ylabel(r"$E(k,t)/K(0)$")
    spectrum_ax.set_title("Energy spectrum", loc="left", fontweight="bold")
    spectrum_ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.45, alpha=0.75)
    spectrum_ax.legend(frameon=False, loc="upper right")
    panel_axes = field_axes + [error_ax, energy_ax, spectrum_ax]
    labels = add_panel_labels(fig, axes=panel_axes, style="paren", x_offset_pt=2.0, y_offset_pt=-2.0)
    for label in labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4})
    qa_and_export(fig, figure_dir / output_name, (7.0, 4.35))


def sci(value: float) -> str:
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.2f}\times10^{{{exponent}}}$"


def mean_sd(mean: float, sd: float) -> str:
    if mean == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(mean))))
    return rf"$({mean / 10**exponent:.2f}\pm{sd / 10**exponent:.2f})\times10^{{{exponent}}}$"


def make_latex(summaries, metadata, reference_validation, gate_validation) -> str:
    rs = summaries["rs_ic_global"]
    fixed = summaries["rs_fixed_tau"]
    pde = metadata["pde"]
    profile = metadata["profile_parameters"]
    lines = [
        (
            "The physical decaying-turbulence archive contains all 30 declared method--seed "
            "records and no replacement seeds.  The selected post-hoc capacity continuation "
            f"uses $P={profile['total_features']}$ features per field.  Its independent seed-5 "
            "gate passed the frozen field, integral-trajectory, and Picard-update thresholds."
        ),
        "",
        (
            "The nested DNS comparison gives space--time relative discrepancies "
            f"{sci(reference_validation['rel_l2_velocity'])} in velocity and "
            f"{sci(reference_validation['rel_l2_vorticity'])} in vorticity.  "
            f"Over $T={pde['T']}$, the DNS kinetic energy and enstrophy fall to "
            f"{metadata['reference_energy_decay_ratio']:.3f} and "
            f"{metadata['reference_enstrophy_decay_ratio']:.3f} of their initial values."
        ),
        "",
        (
            "Across the five untouched feature seeds, RS-RFM-Global obtains mean velocity and "
            f"vorticity errors {sci(rs['rel_l2_velocity'][0])} and "
            f"{sci(rs['rel_l2_vorticity'][0])}.  Its kinetic-energy and enstrophy trajectory "
            f"errors are {sci(rs['rel_l2_kinetic_energy'][0])} and "
            f"{sci(rs['rel_l2_enstrophy'][0])}.  These values establish a low-error solve of an "
            "unforced initial-value problem; method rankings below are reported without turning "
            "this post-hoc capacity continuation into a fifth confirmatory comparison."
        ),
        "",
        (
            "Fixed-$\\tau$ RFM has lower velocity and vorticity means, "
            f"{sci(fixed['rel_l2_velocity'][0])} and "
            f"{sci(fixed['rel_l2_vorticity'][0])}, respectively.  The retained reversal shows "
            "that the broad spectrum is not uniformly optimal here."
        ),
        "",
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Held-out errors for the unforced decaying-turbulence case.  Entries are "
            r"mean $\pm$ sample standard deviation over seeds $50$--$54$.  Bold indicates the lowest descriptive "
            r"mean in each error column; no statistical significance is implied.  This post-hoc "
            r"case is excluded from the confirmatory ranking.}"
        ),
        r"\label{tab:decaying-turbulence-results}",
        r"\TableFont",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ & $\mathcal E_K$ & $\mathcal E_Z$ \\",
        r"\midrule",
    ]
    error_keys = (
        "rel_l2_velocity",
        "rel_l2_vorticity",
        "rel_l2_kinetic_energy",
        "rel_l2_enstrophy",
    )
    best_method = {
        key: min(METHOD_ORDER, key=lambda method: summaries[method][key][0])
        for key in error_keys
    }
    for method in METHOD_ORDER:
        values = summaries[method]
        formatted_errors = []
        for key in error_keys:
            formatted = mean_sd(*values[key])
            if method == best_method[key]:
                formatted = rf"$\mathbf{{{formatted[1:-1]}}}$"
            formatted_errors.append(formatted)
        lines.append(" & ".join([METHOD_LABELS[method], *formatted_errors]) + " \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\begin{figure*}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{figures/fig08_decaying_turbulence_physics.pdf}",
            (
                r"\caption{Physical evolution in the non-manufactured case.  Panels (a)--(c) "
                r"show normalized DNS vorticity initially, DNS vorticity at $T=0.75$, and the "
                r"RS-RFM-Global prediction for representative held-out seed 50.  Panel (d) is "
                r"the corresponding normalized absolute error.  Panel (e) compares normalized "
                r"kinetic energy and "
                r"enstrophy.  Panel (f) shows the shell-summed "
                r"kinetic-energy spectrum and the nonlinear transfer beyond the initial band.  "
                r"Quantitative claims use all five "
                r"held-out seeds rather than this representative field.}"
            ),
            r"\label{fig:decaying-turbulence-physics}",
            r"\end{figure*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    results = args.results.resolve()
    paper = args.paper.resolve()
    production, records, metadata, reference_validation, gate_validation = audit_archive(results)
    summaries = summarize(records)
    configure_style()
    figure_dir = paper / "figures"
    generated_dir = paper / "sections" / "generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    make_physics_figure(production, figure_dir)
    (generated_dir / "fluid_decaying_turbulence_results.tex").write_text(
        make_latex(summaries, metadata, reference_validation, gate_validation),
        encoding="utf-8",
    )
    audit = {
        "archive_complete": True,
        "production_records": len(records),
        "methods": METHOD_ORDER,
        "seeds": SEEDS,
        "reference_validation": reference_validation,
        "gate_validation": gate_validation,
        "summaries": {
            method: {
                metric: (
                    {"count": int(values[0]), "total": int(values[1])}
                    if metric == "converged"
                    else {"mean": values[0], "sample_std": values[1]}
                )
                for metric, values in summaries[method].items()
            }
            for method in METHOD_ORDER
        },
    }
    (results / "artifact_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
