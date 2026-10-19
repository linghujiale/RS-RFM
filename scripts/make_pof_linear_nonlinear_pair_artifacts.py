# 中文注释：生成同一解析流场下线性 Stokes 与非线性 Navier-Stokes 的配对结果表。
"""Validate and summarize the matched linear/nonlinear fluid benchmark.

The input directory must contain the complete frozen grid of 2 cases, 7
methods, and 5 seeds.  The script rejects missing, duplicate, or non-finite
runs before emitting the manuscript table and auditable per-seed CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CASES = ["linear_interacting_stokes", "interacting_two_rate_vortex"]
CASE_LABELS = {
    "linear_interacting_stokes": "线性 Stokes",
    "interacting_two_rate_vortex": "非线性 Navier--Stokes",
}
METHODS = [
    "rs_ic_global",
    "rs_ic_local",
    "rfm_global",
    "rfm_local",
    "sep_rfm_global",
    "sep_rfm_local",
    "pinn",
]
METHOD_LABELS = {
    "rs_ic_global": "RS-RFM-Global",
    "rs_ic_local": "RS-RFM-Local",
    "rfm_global": "RFM-Global",
    "rfm_local": "RFM-Local",
    "sep_rfm_global": "Sep-RFM-Global",
    "sep_rfm_local": "Sep-RFM-Local",
    "pinn": "PINN",
}
LEGACY_METHOD_MAP = {
    "rs_ic_global": "rs_ic_global",
    "rs_ic_local": "rs_ic_local",
    "pielm": "rfm_global",
    "strfm_stc": "rfm_local",
    "sep_rfm_global": "sep_rfm_global",
    "strfm_sov": "sep_rfm_local",
    "pinn": "pinn",
}
SEEDS = [10, 11, 12, 13, 14]
METRICS = ["rel_l2_velocity", "rel_l2_vorticity", "wall_seconds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    return parser.parse_args()


def _read_one(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}, found {len(rows)}")
    row: dict[str, Any] = dict(rows[0])
    row["source_file"] = str(path.resolve())
    return row


def _canonical_case(row: dict[str, Any]) -> str:
    candidate = str(row.get("candidate_id", ""))
    if candidate not in CASES:
        raise ValueError(f"unexpected candidate_id {candidate!r}")
    return candidate


def _canonical_method(row: dict[str, Any]) -> str:
    raw = str(row.get("method_key", ""))
    if raw not in LEGACY_METHOD_MAP:
        raise ValueError(f"unexpected method_key {raw!r}")
    return LEGACY_METHOD_MAP[raw]


def _float(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {row['source_file']}")
    return value


def load_records(result_root: Path) -> list[dict[str, Any]]:
    paths = sorted(result_root.rglob("metrics_raw.csv"))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        row = _read_one(path)
        case = _canonical_case(row)
        method = _canonical_method(row)
        seed = int(row["seed"])
        key = (case, method, seed)
        if key in seen:
            raise ValueError(f"duplicate run {key}: {path}")
        seen.add(key)
        if seed not in SEEDS:
            raise ValueError(f"unexpected seed {seed} in {path}")
        normalized = {
            **row,
            "candidate_id": case,
            "case_variant": "linear" if case == CASES[0] else "nonlinear",
            "method_key": method,
            "method": METHOD_LABELS[method],
            "seed": seed,
        }
        for metric in METRICS:
            normalized[metric] = _float(row, metric)
        if method == "pinn":
            normalized["least_squares_solves"] = ""
            normalized["picard_iterations"] = ""
        else:
            normalized["least_squares_solves"] = int(row["least_squares_solves"])
            normalized["picard_iterations"] = int(row["picard_iterations"])
        records.append(normalized)

    expected = {(case, method, seed) for case in CASES for method in METHODS for seed in SEEDS}
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise ValueError(f"incomplete paired archive: missing={missing}, extra={extra}")
    return sorted(records, key=lambda row: (CASES.index(row["candidate_id"]), METHODS.index(row["method_key"]), row["seed"]))


def validate_pairing(records: list[dict[str, Any]]) -> None:
    indexed = {
        (row["candidate_id"], row["method_key"], row["seed"]): row for row in records
    }
    for method in METHODS[:-1]:
        for seed in SEEDS:
            linear = indexed[(CASES[0], method, seed)]
            nonlinear = indexed[(CASES[1], method, seed)]
            for key in (
                "features_per_field",
                "viscosity",
                "horizon",
                "tau_min_scale",
                "tau_max_scale",
                "static_fraction",
                "picard_relaxation",
            ):
                if linear.get(key) != nonlinear.get(key):
                    raise ValueError(f"unpaired {key} for method={method}, seed={seed}")
            if int(linear["least_squares_solves"]) != 1 or int(linear["picard_iterations"]) != 0:
                raise ValueError(f"linear run did not use one-shot assembly: method={method}, seed={seed}")
            if str(nonlinear.get("picard_converged", "")).lower() != "true":
                raise ValueError(f"nonlinear Picard solve did not converge: method={method}, seed={seed}")
    for seed in SEEDS:
        linear = indexed[(CASES[0], "pinn", seed)]
        nonlinear = indexed[(CASES[1], "pinn", seed)]
        metadata: list[dict[str, Any]] = []
        for row in (linear, nonlinear):
            path = Path(row["source_file"]).parent / "run_metadata.json"
            if not path.exists():
                raise FileNotFoundError(f"missing PINN metadata: {path}")
            metadata.append(json.loads(path.read_text(encoding="utf-8")))
        left, right = metadata
        for key in ("profile", "training", "network", "sampling"):
            if left.get(key) != right.get(key):
                raise ValueError(f"unpaired PINN {key} for seed={seed}")
        left_device = left.get("environment", {}).get("device")
        right_device = right.get("environment", {}).get("device")
        if left_device != right_device:
            raise ValueError(f"unpaired PINN device for seed={seed}")


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for case in CASES:
        for method in METHODS:
            rows = [
                row for row in records
                if row["candidate_id"] == case and row["method_key"] == method
            ]
            item: dict[str, Any] = {
                "candidate_id": case,
                "case": CASE_LABELS[case],
                "method_key": method,
                "method": METHOD_LABELS[method],
                "n": len(rows),
            }
            for metric in METRICS:
                item[f"mean_{metric}"], item[f"std_{metric}"] = _mean_std(rows, metric)
            if method == "pinn":
                item["mean_least_squares_solves"] = ""
                item["std_least_squares_solves"] = ""
                item["mean_picard_iterations"] = ""
                item["std_picard_iterations"] = ""
            else:
                for metric in ("least_squares_solves", "picard_iterations"):
                    item[f"mean_{metric}"], item[f"std_{metric}"] = _mean_std(rows, metric)
            summary.append(item)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sci_tex(mean: float, std: float, *, bold: bool = False) -> str:
    if mean == 0.0:
        return f"$0\\pm{std:.2g}$"
    exponent = int(math.floor(math.log10(abs(mean))))
    scale = 10.0**exponent
    mantissa = f"({mean / scale:.2f}\\pm{std / scale:.2f})"
    if bold:
        mantissa = f"\\mathbf{{{mantissa}}}"
    return f"${mantissa}\\times10^{{{exponent}}}$"


def make_tex(summary: list[dict[str, Any]]) -> str:
    indexed = {(row["candidate_id"], row["method_key"]): row for row in summary}
    best = {
        (case, metric): min(float(indexed[(case, method)][f"mean_{metric}"]) for method in METHODS)
        for case in CASES
        for metric in ("rel_l2_velocity", "rel_l2_vorticity")
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{相同解析流场下线性非定常 Stokes 与非线性 Navier--Stokes 的配对比较。数值为五次配对独立运行的均值 $\pm$ 样本标准差；粗体表示七种主方法中的最低误差均值。}",
        r"\label{tab:linear-nonlinear-pair}",
        r"\TableFont",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lcc@{\qquad}cc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{线性非定常 Stokes} & \multicolumn{2}{c}{非线性 Navier--Stokes} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"方法 & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ & $\mathcal E_{\bu}$ & $\mathcal E_{\omega}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        linear = indexed[(CASES[0], method)]
        nonlinear = indexed[(CASES[1], method)]
        lines.append(
            " & ".join(
                [
                    METHOD_LABELS[method],
                    _sci_tex(
                        linear["mean_rel_l2_velocity"],
                        linear["std_rel_l2_velocity"],
                        bold=float(linear["mean_rel_l2_velocity"]) == best[(CASES[0], "rel_l2_velocity")],
                    ),
                    _sci_tex(
                        linear["mean_rel_l2_vorticity"],
                        linear["std_rel_l2_vorticity"],
                        bold=float(linear["mean_rel_l2_vorticity"]) == best[(CASES[0], "rel_l2_vorticity")],
                    ),
                    _sci_tex(
                        nonlinear["mean_rel_l2_velocity"],
                        nonlinear["std_rel_l2_velocity"],
                        bold=float(nonlinear["mean_rel_l2_velocity"]) == best[(CASES[1], "rel_l2_velocity")],
                    ),
                    _sci_tex(
                        nonlinear["mean_rel_l2_vorticity"],
                        nonlinear["std_rel_l2_vorticity"],
                        bold=float(nonlinear["mean_rel_l2_vorticity"]) == best[(CASES[1], "rel_l2_vorticity")],
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    records = load_records(args.result_root.resolve())
    validate_pairing(records)
    summary = summarize(records)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "linear_nonlinear_pair_per_seed.csv", records)
    _write_csv(output / "linear_nonlinear_pair_summary.csv", summary)
    audit = {
        "cases": CASES,
        "methods": METHODS,
        "seeds": SEEDS,
        "expected_runs": len(CASES) * len(METHODS) * len(SEEDS),
        "observed_runs": len(records),
        "strict_pairing_checks_passed": True,
        "pinn_training_and_device_pairing_checked": True,
        "all_nonlinear_rfm_picard_runs_converged": True,
    }
    (output / "linear_nonlinear_pair_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.tex_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.resolve().write_text(make_tex(summary), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
