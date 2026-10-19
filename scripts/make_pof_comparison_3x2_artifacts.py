# 中文注释：汇总六类随机特征方法和 PINN 的二维、衰减流及三维比较结果。
"""Merge archived and newly authorized results into the 3x2+PINN comparison.

The script is deliberately strict: it accepts only the frozen method/case/seed
grid, preserves source-file provenance, rejects duplicate records, and writes
all manuscript tables from per-seed data.  It never launches a solver.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
    "rs_ic_global",
    "rs_ic_local",
    "rfm_global",
    "rfm_local",
    "sep_rfm_global",
    "sep_rfm_local",
    "pinn",
]
RS_METHODS = METHODS[:2]
NON_RS_METHODS = METHODS[2:]
RFM_METHODS = METHODS[:-1]
MECHANISM_METHODS = [*RFM_METHODS, "rs_fixed_tau"]
METHOD_LABELS = {
    "rs_ic_global": "RS-RFM-Global",
    "rs_ic_local": "RS-RFM-Local",
    "rfm_global": "RFM-Global",
    "rfm_local": "RFM-Local",
    "sep_rfm_global": "Sep-RFM-Global",
    "sep_rfm_local": "Sep-RFM-Local",
    "pinn": "PINN",
    "rs_fixed_tau": "Fixed-$\\tau$ RFM",
    "rs_static": "Static-RFM-Local",
}
LEGACY_MAP = {
    "rs_ic_global": "rs_ic_global",
    "rs_ic_local": "rs_ic_local",
    "pielm": "rfm_global",
    "strfm_stc": "rfm_local",
    "strfm_sov": "sep_rfm_local",
    "sep_rfm_global": "sep_rfm_global",
    "pinn": "pinn",
    "rs_fixed_tau": "rs_fixed_tau",
    "rs_static": "rs_static",
}
CONTROLLED_CASES = [
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
    "decaying_turbulence": "二维衰减流",
    "interacting_helical_vortex_3d": "三维螺旋涡",
}
CASE_FEATURES = {
    "tgv_order1_decay": 256,
    "two_rate_physics_range": 384,
    "interacting_two_rate_vortex": 384,
    "multiscale_shear_decay_k12": 384,
    "decaying_turbulence": 1280,
    "interacting_helical_vortex_3d": 2048,
}
SEEDS = {
    "controlled2d": [10, 11, 12, 13, 14],
    "decay2d": [50, 51, 52, 53, 54],
    "helical3d": [100, 101, 102, 103, 104],
}
MAIN_METRICS = [
    "rel_l2_velocity",
    "rel_l2_vorticity",
    "final_rel_l2_velocity",
    "final_rel_l2_vorticity",
    "relative_divergence",
    "pressure_rms_over_u2",
    "rel_l2_kinetic_energy",
    "rel_l2_enstrophy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mechanism-only",
        action="store_true",
        help="validate and merge the mechanism archive without requiring the main PINN grid",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["source_file"] = str(path.resolve())
    return frame


def read_tree(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("metrics_raw.csv"))
    if not paths:
        raise FileNotFoundError(f"no metrics_raw.csv below {root}")
    frames = []
    for path in paths:
        status_path = path.parent / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("failures", 0) != 0 or status.get("records", 0) < 1:
                raise ValueError(f"incomplete result beside {path}: {status}")
        frames.append(read_csv(path))
    return pd.concat(frames, ignore_index=True, sort=False)


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "candidate_id" not in output:
        output["candidate_id"] = output.get("case", "")
    output["candidate_id"] = output["candidate_id"].fillna(output.get("case", ""))
    output.loc[output.candidate_id == "helical_vortex_3d", "candidate_id"] = (
        "interacting_helical_vortex_3d"
    )
    output["legacy_method_key"] = output["method_key"]
    unknown = sorted(set(output.method_key.astype(str)) - set(LEGACY_MAP))
    if unknown:
        raise ValueError(f"unmapped method keys: {unknown}")
    output["method_key"] = output.method_key.map(LEGACY_MAP)
    output["method"] = output.method_key.map(METHOD_LABELS)
    output["seed"] = pd.to_numeric(output.seed, errors="raise").astype(int)
    for field in set(MAIN_METRICS) | {
        "features_per_field",
        "wall_seconds",
        "training_seconds",
        "evaluation_seconds",
        "trainable_parameters",
        "condition_number",
        "rank",
        "picard_iterations",
        "picard_limit",
        "rank_fraction",
        "max_log_tau_gap",
        "point",
        "rho0",
        "tau_min_over_T",
        "tau_max_over_T",
    }:
        if field in output:
            output[field] = pd.to_numeric(output[field], errors="coerce")
    if "picard_converged" in output:
        output["picard_converged"] = output.picard_converged.astype(str).str.lower().eq("true")
    return output


def tagged(frame: pd.DataFrame, suite: str, source_kind: str) -> pd.DataFrame:
    output = canonicalize(frame)
    output["suite"] = suite
    output["source_kind"] = source_kind
    return output


def enrich_pinn_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach optimizer and realized-iteration provenance to PINN seed rows.

    The solver deliberately keeps the immutable metric row separate from the
    training history and run metadata.  This function joins those files during
    aggregation without rewriting any raw server output.
    """
    output = frame.copy()
    records: list[dict] = []
    for row in output.itertuples(index=False):
        run_dir = Path(row.source_file).parent
        metadata_path = run_dir / "run_metadata.json"
        history_path = run_dir / "training_history.csv"
        if not metadata_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"missing PINN provenance beside {row.source_file}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training = metadata["training"]
        history = pd.read_csv(history_path)
        adam = history.loc[history.phase.eq("Adam"), "step"]
        lbfgs = history.loc[history.phase.eq("L-BFGS"), "step"]
        records.append(
            {
                "optimizer": "Adam + L-BFGS",
                "learning_rate": float(training["learning_rate"]),
                "maximum_training_iterations": int(training["adam_steps"])
                + int(training["lbfgs_steps"]),
                "recorded_adam_iterations": int(adam.max()) if len(adam) else 0,
                "recorded_lbfgs_evaluations": int(lbfgs.max()) if len(lbfgs) else 0,
                "training_history_records": int(len(history)),
                "precision": "float32",
                "training_host": metadata["environment"]["host"],
                "training_device": metadata["environment"]["device"],
                "scheduler_job_id": metadata["environment"].get("scheduler_job_id"),
                "scheduler_array_task_id": metadata["environment"].get(
                    "scheduler_array_task_id"
                ),
            }
        )
    provenance = pd.DataFrame(records, index=output.index)
    for column in provenance:
        output[column] = provenance[column]
    return output


def load_main(paper: Path, new_results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = paper / "results"
    controlled_all = read_csv(results / "controlled_2d_combined_20260823" / "metrics_all.csv")
    controlled = tagged(
        controlled_all[controlled_all.method_key.isin(["rs_ic_global", "rs_ic_local", "pielm", "strfm_stc", "strfm_sov"])],
        "controlled2d",
        "archived",
    )
    decay_all = read_csv(
        results / "fluid_decaying_turbulence" / "production" / "continuation" / "metrics_raw.csv"
    )
    decay = tagged(
        decay_all[decay_all.method_key.isin(["rs_ic_global", "pielm", "strfm_stc", "strfm_sov"])],
        "decay2d",
        "archived",
    )
    decay_local = tagged(
        read_csv(
            results
            / "rsrfm_local_extension_20260823"
            / "results_retry1"
            / "decaying_turbulence"
            / "metrics_raw.csv"
        ),
        "decay2d",
        "archived",
    )
    helical_all = read_csv(
        results / "helical_vortex_3d_p2048_fixed_20260821" / "metrics_unique_30.csv"
    )
    helical = tagged(
        helical_all[helical_all.method_key.isin(["rs_ic_global", "pielm", "strfm_stc", "strfm_sov"])],
        "helical3d",
        "archived",
    )
    local_paths = sorted(
        (
            results
            / "rsrfm_local_extension_20260823"
            / "results"
            / "helical_vortex_3d"
        ).rglob("metrics_raw.csv")
    )
    helical_local = tagged(
        pd.concat([read_csv(path) for path in local_paths], ignore_index=True, sort=False),
        "helical3d",
        "archived",
    )

    new_rfm = canonicalize(read_tree(new_results / "rfm"))
    main_run = new_rfm.study.isna() if "study" in new_rfm else pd.Series(True, index=new_rfm.index)
    new_rfm = new_rfm[new_rfm.method_key.eq("sep_rfm_global") & main_run]
    new_rfm["source_kind"] = "new"
    new_rfm["suite"] = np.select(
        [
            new_rfm.candidate_id.isin(CONTROLLED_CASES),
            new_rfm.candidate_id.eq("decaying_turbulence"),
            new_rfm.candidate_id.eq("interacting_helical_vortex_3d"),
        ],
        ["controlled2d", "decay2d", "helical3d"],
        default="unknown",
    )
    new_pinn = enrich_pinn_provenance(canonicalize(read_tree(new_results / "pinn")))
    new_pinn = new_pinn[new_pinn.method_key.eq("pinn")]
    new_pinn["source_kind"] = "new"
    new_pinn["suite"] = np.select(
        [
            new_pinn.candidate_id.isin(CONTROLLED_CASES),
            new_pinn.candidate_id.eq("decaying_turbulence"),
            new_pinn.candidate_id.eq("interacting_helical_vortex_3d"),
        ],
        ["controlled2d", "decay2d", "helical3d"],
        default="unknown",
    )
    main = pd.concat(
        [controlled, decay, decay_local, helical, helical_local, new_rfm, new_pinn],
        ignore_index=True,
        sort=False,
    )
    main.loc[main.suite.eq("decay2d"), "candidate_id"] = "decaying_turbulence"
    main.loc[main.suite.eq("helical3d"), "candidate_id"] = "interacting_helical_vortex_3d"

    controls = pd.concat(
        [
            tagged(controlled_all[controlled_all.method_key.isin(["rs_fixed_tau", "rs_static"])], "controlled2d", "archived"),
            tagged(decay_all[decay_all.method_key.isin(["rs_fixed_tau", "rs_static"])], "decay2d", "archived"),
            tagged(helical_all[helical_all.method_key.isin(["rs_fixed_tau", "rs_static"])], "helical3d", "archived"),
        ],
        ignore_index=True,
        sort=False,
    )
    controls.loc[controls.suite.eq("decay2d"), "candidate_id"] = "decaying_turbulence"
    controls.loc[controls.suite.eq("helical3d"), "candidate_id"] = "interacting_helical_vortex_3d"
    return main, controls


def validate_main(frame: pd.DataFrame) -> None:
    expected: set[tuple[str, str, str, int]] = set()
    for case in CONTROLLED_CASES:
        expected |= {("controlled2d", case, method, seed) for method in METHODS for seed in SEEDS["controlled2d"]}
    expected |= {
        ("decay2d", "decaying_turbulence", method, seed)
        for method in METHODS
        for seed in SEEDS["decay2d"]
    }
    expected |= {
        ("helical3d", "interacting_helical_vortex_3d", method, seed)
        for method in METHODS
        for seed in SEEDS["helical3d"]
    }
    observed = set(
        frame[["suite", "candidate_id", "method_key", "seed"]]
        .itertuples(index=False, name=None)
    )
    duplicates = frame.duplicated(["suite", "candidate_id", "method_key", "seed"], keep=False)
    if duplicates.any() or observed != expected or len(frame) != 210:
        raise ValueError(
            f"main grid mismatch: rows={len(frame)}, duplicates={duplicates.sum()}, "
            f"missing={sorted(expected-observed)[:8]}, extra={sorted(observed-expected)[:8]}"
        )
    for row in frame.itertuples():
        required = ["rel_l2_velocity", "rel_l2_vorticity"]
        if row.suite == "decay2d":
            required += ["rel_l2_kinetic_energy", "rel_l2_enstrophy"]
        if row.suite == "helical3d":
            required += [
                "final_rel_l2_velocity",
                "final_rel_l2_vorticity",
                "relative_divergence",
                "pressure_rms_over_u2",
            ]
        for metric in required:
            value = getattr(row, metric)
            if not np.isfinite(value):
                raise ValueError(f"nonfinite {metric}: {row.suite}, {row.method_key}, {row.seed}")
        if row.method_key in RFM_METHODS:
            expected_p = CASE_FEATURES[row.candidate_id]
            if int(row.features_per_field) != expected_p:
                raise ValueError(f"feature-budget drift: {row.candidate_id}, {row.method_key}")


def stat(frame: pd.DataFrame, case: str, method: str, metric: str) -> tuple[float, float]:
    values = frame.loc[
        frame.candidate_id.eq(case) & frame.method_key.eq(method), metric
    ].to_numpy(dtype=float)
    if len(values) != 5:
        raise ValueError(f"expected five values for {case}, {method}, {metric}")
    return float(values.mean()), float(values.std(ddof=1))


def tex_mean_sd(values: tuple[float, float], *, bold: bool = False) -> str:
    mean, sd = values
    scale = max(abs(mean), abs(sd))
    if scale == 0.0:
        body = "0"
    else:
        exponent = int(math.floor(math.log10(scale)))
        body = f"({mean/10**exponent:.2f}\\pm{sd/10**exponent:.2f})\\times10^{{{exponent}}}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def tex_value(value: float) -> str:
    exponent = int(math.floor(math.log10(abs(value))))
    return f"${value / 10**exponent:.2f}\\times10^{{{exponent}}}$"


def winners(frame: pd.DataFrame, case: str, metrics: list[str]) -> dict[str, str]:
    return {
        metric: min(METHODS, key=lambda method: stat(frame, case, method, metric)[0])
        for metric in metrics
    }


def reduction_vs_best_non_rs(frame: pd.DataFrame, case: str, metric: str) -> float:
    """Error reduction of the best RS-RFM relative to the strongest comparator."""
    best_rs = min(stat(frame, case, method, metric)[0] for method in RS_METHODS)
    best_non_rs = min(stat(frame, case, method, metric)[0] for method in NON_RS_METHODS)
    return 100.0 * (1.0 - best_rs / best_non_rs)


def write_controlled(frame: pd.DataFrame, path: Path) -> dict:
    metrics = ["rel_l2_velocity", "rel_l2_vorticity"]
    best = {case: winners(frame, case, metrics) for case in CONTROLLED_CASES}
    reductions = {
        case: {
            metric: reduction_vs_best_non_rs(frame, case, metric) for metric in metrics
        }
        for case in CONTROLLED_CASES
    }
    velocity_reductions = [
        reductions[case]["rel_l2_velocity"] for case in CONTROLLED_CASES
    ]
    vorticity_reductions = [
        reductions[case]["rel_l2_vorticity"] for case in CONTROLLED_CASES
    ]
    two_rate = reductions["two_rate_physics_range"]
    lines = [
        (
            "表~\\ref{tab:all-flow-results} 显示，两种 RS-RFM 变体在四个受控二维算例中的"
            "速度和涡量误差均低于五种对照方法，整体处于 $10^{-4}$--$10^{-3}$ 量级。"
            "RS-RFM-Global 在 Taylor--Green、双速率受迫涡和相互作用涡中取得最低均值，"
            "而 RS-RFM-Local 在多尺度剪切流中取得最低均值。"
            "以每个算例中相应误差均值最低的非 RS-RFM 方法为参照，最优 RS-RFM 的速度误差降低 "
            f"${min(velocity_reductions):.1f}\\%$--${max(velocity_reductions):.1f}\\%$，涡量误差降低 "
            f"${min(vorticity_reductions):.1f}\\%$--${max(vorticity_reductions):.1f}\\%$；其中双速率"
            f"受迫涡的两项误差分别降低 ${two_rate['rel_l2_velocity']:.1f}\\%$ 和 "
            f"${two_rate['rel_l2_vorticity']:.1f}\\%$。该结果表明，RS-RFM 的精度"
            "优势在不同流动结构和衰减尺度下保持一致，同时最优空间支撑形式会随算例发生变化。"
            "图~\\ref{fig:controlled-2d-fields} 从场分布角度补充表~\\ref{tab:all-flow-results} 的误差比较。"
            "四个算例的终时预测均重现了解析涡量场的主导结构与相位；前两个算例的归一化逐点误差"
            "整体较低，而相互作用涡和多尺度剪切流中的局部误差更明显，这与表中较高的涡量误差一致。"
        ),
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{四个受控二维算例的时空相对误差。数值为五次独立运行的均值 $\pm$ 样本标准差；六种 RFM 在同一算例中使用相同特征数和配点，PINN 使用相同 PDE、配点与评价网格。粗体表示七种主方法中的最低均值。}",
        r"\label{tab:all-flow-results}",
        r"\WideTableFont",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l*{4}{cc}@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Taylor--Green}& \multicolumn{2}{c}{双速率受迫涡}& \multicolumn{2}{c}{相互作用涡}& \multicolumn{2}{c}{多尺度剪切流} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
        r"方法 & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = [METHOD_LABELS[method]]
        for case in CONTROLLED_CASES:
            for metric in metrics:
                cells.append(tex_mean_sd(stat(frame, case, method, metric), bold=best[case][metric] == method))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best


def write_decay(frame: pd.DataFrame, _controls: pd.DataFrame, path: Path) -> dict:
    case = "decaying_turbulence"
    metrics = [
        "rel_l2_velocity",
        "rel_l2_vorticity",
        "rel_l2_kinetic_energy",
        "rel_l2_enstrophy",
    ]
    best = winners(frame, case, metrics)
    local_u = tex_value(stat(frame, case, "rs_ic_local", "rel_l2_velocity")[0])
    local_w = tex_value(stat(frame, case, "rs_ic_local", "rel_l2_vorticity")[0])
    local_z = tex_value(stat(frame, case, "rs_ic_local", "rel_l2_enstrophy")[0])
    global_k = tex_value(stat(frame, case, "rs_ic_global", "rel_l2_kinetic_energy")[0])
    reductions = [reduction_vs_best_non_rs(frame, case, metric) for metric in metrics]
    lines = [
        (
            "表~\\ref{tab:decaying-turbulence-results} 显示，在非制造衰减流中，"
            f"RS-RFM-Local 的速度、涡量和拟涡能误差最低，分别为 {local_u}、{local_w} 和 {local_z}；"
            f"RS-RFM-Global 则取得最低动能误差 {global_k}。两种 RS-RFM 在四项指标上均低于"
            "其余五种方法。以各指标中误差均值最低的非 RS-RFM 方法为参照，速度、涡量、动能和"
            f"拟涡能误差分别降低 ${reductions[0]:.1f}\\%$、${reductions[1]:.1f}\\%$、"
            f"${reductions[2]:.1f}\\%$ 和 ${reductions[3]:.1f}\\%$，说明松弛谱表示的优势在"
            "无解析未来状态参与训练时仍然保留。"
        ),
        "",
        r"\begin{table*}[htb]",
        r"\centering",
        r"\caption{固定容量 $P=1280$ 下无外力衰减流的七方法比较。数值为五次独立运行的均值 $\pm$ 样本标准差；粗体表示七种主方法中的最低均值。}",
        r"\label{tab:decaying-turbulence-results}",
        r"\TableFont",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"方法 & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ & $\mathcal E_K$ & $\mathcal E_Z$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = [METHOD_LABELS[method]]
        for metric in metrics:
            cells.append(tex_mean_sd(stat(frame, case, method, metric), bold=best[metric] == method))
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        (
            "场重构结果表明，RS-RFM-Local 在三个评价时刻均保持了 DNS 的主要涡结构。"
            "初始时刻误差为零，随后较大逐点误差主要集中在细丝状高梯度区域，见图~\\ref{fig:decaying-turbulence-fields}。"
            "动能与拟涡能衰减轨迹与 DNS 基本一致；终时能谱在低波数范围吻合较好，而高波数能量存在一定低估。"
            "这表明在当前固定特征容量下，RS-RFM-Local 能够恢复主导大尺度结构，但空间高频与小尺度信息"
            "仍较难准确捕捉，见图~\\ref{fig:decaying-turbulence-integrals}。"
        ),
        "",
    ]
    lines += [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/fig08_decaying_turbulence_fields_local.pdf}",
        r"\caption{非制造二维衰减流的涡量演化与逐点误差。上排为 DNS 在 $t=0$、$0.375$ 和 $0.75$ 时的归一化涡量，下排为 RS-RFM-Local 在相同时刻的归一化逐点绝对误差。三幅涡量图和三幅误差图分别共用色标。}",
        r"\label{fig:decaying-turbulence-fields}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/fig09_decaying_turbulence_integrals_local.pdf}",
        r"\caption{非制造二维衰减流的整体物理量演化。左图比较 DNS 与 RS-RFM-Local 的归一化动能和拟涡能衰减，右图比较初始 DNS、终时 DNS 与终时 RS-RFM-Local 的壳层求和动能谱。}",
        r"\label{fig:decaying-turbulence-integrals}",
        r"\end{figure*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best


def write_helical(frame: pd.DataFrame, path: Path) -> dict:
    case = "interacting_helical_vortex_3d"
    metrics = [
        "rel_l2_velocity",
        "rel_l2_vorticity",
        "final_rel_l2_velocity",
        "final_rel_l2_vorticity",
        "relative_divergence",
        "pressure_rms_over_u2",
    ]
    best = winners(frame, case, metrics)
    reductions = [reduction_vs_best_non_rs(frame, case, metric) for metric in metrics]
    lines = [
        (
            "表~\\ref{tab:helical-3d-p2048-comparison} 给出三维算例的完整六指标结果。"
            "RS-RFM-Local 在时空速度、时空涡量和终时速度误差上取得最低均值，RS-RFM-Global "
            "则在终时涡量、归一化散度和压力幅值上取得最低均值。以各指标中均值最低的非 "
            f"RS-RFM 方法为参照，前五项指标降低 ${min(reductions[:5]):.1f}\\%$--"
            f"${max(reductions[:5]):.1f}\\%$，压力幅值指标降低 ${reductions[5]:.1f}\\%$。"
            "这说明松弛谱表示的精度优势能够延伸到三维速度--压力格式，但压力指标上的差异有限；"
            "Global 与 Local 支撑在场误差和不可压缩约束上各有侧重。"
        ),
        "",
        r"\begin{table*}[htb]",
        r"\centering",
        r"\caption{固定 $P=2048$ 的三维螺旋涡六指标比较。数值为五次独立运行的均值 $\pm$ 样本标准差；粗体表示七种主方法中的最低均值。}",
        r"\label{tab:helical-3d-p2048-comparison}",
        r"\WideTableFont",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"方法 & $\mathcal E_{\bu}$ & $\mathcal E_{\boldsymbol\omega}$ & $\mathcal E_{\bu}(T)$ & $\mathcal E_{\boldsymbol\omega}(T)$ & $\mathcal E_{\rm div}$ & $\mathcal E_p^{(0)}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = [METHOD_LABELS[method]]
        for metric in metrics:
            cells.append(tex_mean_sd(stat(frame, case, method, metric), bold=best[metric] == method))
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
        (
            "RS-RFM-Global 的预测场在 $x=\\pi$、$y=\\pi$ 和 $z=\\pi$ 三个正交截面上均重现了"
            "解析涡量模的主要空间结构，较大逐点误差集中于少量局部区域，具体结果见"
            "图~\\ref{fig:helical-3d-fields}。选择 Global 是因为其在"
            "表~\\ref{tab:helical-3d-p2048-comparison} 中的终时涡量、归一化散度和压力幅值均值较低。"
        ),
        "",
    ]
    lines += [
        "",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/fig06_helical_vortex_3d_fields.pdf}",
        r"\caption{三维螺旋涡在 $t=T=1$ 的涡量模切片。上排依次为 RS-RFM-Global 在 $x=\pi$、$y=\pi$ 和 $z=\pi$ 截面的预测，下排为相应的逐点误差；三幅预测图和三幅误差图分别共用色标。}",
        r"\label{fig:helical-3d-fields}",
        r"\end{figure*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best


def load_mechanism(paper: Path, new_results: Path, output: Path) -> pd.DataFrame:
    old_path = (
        paper
        / "results"
        / "mechanism"
        / "rsrfm_pof_mechanism_20260823_v1"
        / "summary"
        / "metrics_all.csv"
    )
    old = canonicalize(read_csv(old_path))
    new = canonicalize(read_tree(new_results / "rfm" / "mechanism"))
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    combined.to_csv(output / "mechanism_metrics_preserved_all.csv", index=False)
    desired = combined[
        (
            combined.study.isin(["feature_budget", "scale_separation"])
            & combined.method_key.isin(MECHANISM_METHODS)
        )
        |
        (
            combined.study.isin(["spectral_window", "sampling_ablation"])
            & combined.method_key.isin(["rs_ic_global", "rs_ic_local"])
        )
    ].copy()
    key = ["study", "case", "point_label", "method_key", "seed"]
    if len(desired) != 725 or desired.duplicated(key).any():
        raise ValueError(f"mechanism grid mismatch: rows={len(desired)}, duplicates={desired.duplicated(key).sum()}")
    desired.to_csv(output / "mechanism_metrics_all.csv", index=False)
    mechanism_dir = output / "mechanism_summary"
    mechanism_dir.mkdir(parents=True, exist_ok=True)
    desired.to_csv(mechanism_dir / "metrics_all.csv", index=False)

    summary_rows = []
    group_cols = [
        "study", "case", "point", "point_label", "method_key", "method",
        "features_per_field", "rho0", "sampling_mode", "tau_min_over_T", "tau_max_over_T",
    ]
    numeric = [
        "condition_number", "max_log_tau_gap", "picard_iterations", "rank",
        "rank_fraction", "rel_l2_velocity", "rel_l2_vorticity", "wall_seconds",
    ]
    for keys, group in desired.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        row["n"] = len(group)
        row["converged_count"] = int(group.picard_converged.fillna(False).sum())
        if "picard_limit" in group:
            at_limit = group.picard_iterations.eq(group.picard_limit)
            row["picard_at_limit_count"] = int(at_limit.sum())
            row["picard_limit_nonconverged_count"] = int((at_limit & ~group.picard_converged.fillna(False)).sum())
        for field in numeric:
            values = pd.to_numeric(group[field], errors="coerce")
            row[f"mean_{field}"] = values.mean()
            row[f"std_{field}"] = values.std(ddof=1)
            row[f"min_{field}"] = values.min()
            row[f"max_{field}"] = values.max()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "mechanism_summary.csv", index=False)
    summary.to_csv(mechanism_dir / "summary.csv", index=False)
    efficiency_rows = []
    feature = summary[summary.study.eq("feature_budget")]
    for (case_name, method), group in feature.groupby(["case", "method_key"]):
        passed = group[group.mean_rel_l2_velocity < 1.0e-2]
        minimum = np.nan if passed.empty else int(passed.features_per_field.min())
        efficiency_rows.append(
            {
                "case": case_name,
                "method": METHOD_LABELS[method],
                "method_key": method,
                "minimum_P_below_threshold": minimum,
                "threshold_crossed": not passed.empty,
                "velocity_error_threshold": 1.0e-2,
            }
        )
    efficiency = pd.DataFrame(efficiency_rows)
    efficiency.to_csv(output / "mechanism_feature_efficiency.csv", index=False)
    efficiency.to_csv(mechanism_dir / "feature_efficiency.csv", index=False)
    mechanism_audit = {
        "expected_seed_runs": 725,
        "records": len(desired),
        "preserved_records": len(combined),
        "new_sep_rfm_global_records": len(new),
        "failures": [],
    }
    audit_text = json.dumps(mechanism_audit, indent=2)
    (output / "mechanism_audit.json").write_text(audit_text, encoding="utf-8")
    (mechanism_dir / "audit.json").write_text(audit_text, encoding="utf-8")
    return desired


def ranking_payload(frame: pd.DataFrame) -> dict:
    output: dict = {}
    case_metrics = {
        **{case: ["rel_l2_velocity", "rel_l2_vorticity"] for case in CONTROLLED_CASES},
        "decaying_turbulence": ["rel_l2_velocity", "rel_l2_vorticity", "rel_l2_kinetic_energy", "rel_l2_enstrophy"],
        "interacting_helical_vortex_3d": [
            "rel_l2_velocity", "rel_l2_vorticity", "final_rel_l2_velocity",
            "final_rel_l2_vorticity", "relative_divergence", "pressure_rms_over_u2",
        ],
    }
    for case, metrics in case_metrics.items():
        output[case] = {}
        for metric in metrics:
            ranked = sorted(
                (
                    {
                        "method_key": method,
                        "method": METHOD_LABELS[method],
                        "mean": stat(frame, case, method, metric)[0],
                        "sample_sd": stat(frame, case, method, metric)[1],
                    }
                    for method in METHODS
                ),
                key=lambda item: item["mean"],
            )
            output[case][metric] = ranked
    return output


def write_pairwise_summary(frame: pd.DataFrame, path: Path) -> None:
    """Write the prespecified comparisons that the 3x2 design is meant to answer."""
    case_metrics = {
        **{case: ["rel_l2_velocity", "rel_l2_vorticity"] for case in CONTROLLED_CASES},
        "decaying_turbulence": [
            "rel_l2_velocity", "rel_l2_vorticity",
            "rel_l2_kinetic_energy", "rel_l2_enstrophy",
        ],
        "interacting_helical_vortex_3d": [
            "rel_l2_velocity", "rel_l2_vorticity", "final_rel_l2_velocity",
            "final_rel_l2_vorticity", "relative_divergence", "pressure_rms_over_u2",
        ],
    }
    pairs = [
        ("time_global_rs_vs_joint", "rs_ic_global", "rfm_global"),
        ("time_local_rs_vs_joint", "rs_ic_local", "rfm_local"),
        ("time_global_rs_vs_separable", "rs_ic_global", "sep_rfm_global"),
        ("time_local_rs_vs_separable", "rs_ic_local", "sep_rfm_local"),
        ("support_rs_global_vs_local", "rs_ic_global", "rs_ic_local"),
        ("support_joint_global_vs_local", "rfm_global", "rfm_local"),
        ("support_separable_global_vs_local", "sep_rfm_global", "sep_rfm_local"),
    ]
    rows = []
    for case, metrics in case_metrics.items():
        for metric in metrics:
            for comparison, method_a, method_b in pairs:
                mean_a, sd_a = stat(frame, case, method_a, metric)
                mean_b, sd_b = stat(frame, case, method_b, metric)
                rows.append(
                    {
                        "case": case,
                        "metric": metric,
                        "comparison": comparison,
                        "method_a": METHOD_LABELS[method_a],
                        "method_b": METHOD_LABELS[method_b],
                        "mean_a": mean_a,
                        "sample_sd_a": sd_a,
                        "mean_b": mean_b,
                        "sample_sd_b": sd_b,
                        "mean_b_over_mean_a": mean_b / mean_a,
                        "lower_mean": METHOD_LABELS[method_a if mean_a < mean_b else method_b],
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    paper = args.paper.resolve()
    new_results = args.new_results.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mechanism_only:
        mechanism = load_mechanism(paper, new_results, output)
        print(json.dumps({"mechanism_records": len(mechanism), "passed": True}, indent=2))
        return 0
    main_frame, controls = load_main(paper, new_results)
    validate_main(main_frame)
    main_frame.to_csv(output / "main_comparison_per_seed.csv", index=False)
    pinn_columns = [
        "suite", "candidate_id", "method", "seed", "trainable_parameters",
        "n_int", "n_bc_total", "optimizer", "learning_rate",
        "maximum_training_iterations", "recorded_adam_iterations",
        "recorded_lbfgs_evaluations", "training_history_records", "precision",
        "training_host", "training_device", "scheduler_job_id",
        "scheduler_array_task_id", "training_seconds", "evaluation_seconds",
        "final_training_loss", "final_pde_loss", "source_file",
    ]
    main_frame.loc[main_frame.method_key.eq("pinn"), pinn_columns].to_csv(
        output / "pinn_training_manifest.csv", index=False
    )
    controls.to_csv(output / "additional_controls_per_seed.csv", index=False)
    generated = paper / "sections_cn" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    controlled_best = write_controlled(main_frame, generated / "controlled_2d_results.tex")
    decay_best = write_decay(main_frame, controls, generated / "fluid_decaying_turbulence_results.tex")
    helical_best = write_helical(main_frame, generated / "fluid_3d_results.tex")
    load_mechanism(paper, new_results, output)
    write_pairwise_summary(main_frame, output / "pairwise_comparison_summary.csv")
    rankings = ranking_payload(main_frame)
    rankings["table_winners"] = {
        "controlled": controlled_best,
        "decay": decay_best,
        "helical": helical_best,
    }
    (output / "ranking_summary.json").write_text(
        json.dumps(rankings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    audit = {
        "main_records": len(main_frame),
        "archived_main_records": int(main_frame.source_kind.eq("archived").sum()),
        "new_main_records": int(main_frame.source_kind.eq("new").sum()),
        "additional_control_records": len(controls),
        "methods": METHODS,
        "cases": [*CONTROLLED_CASES, "decaying_turbulence", "interacting_helical_vortex_3d"],
        "passed": True,
    }
    (output / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
