# 中文注释：汇总PoF 论文 机制研究，并输出可复核的实验或制图结果。
"""Aggregate raw PoF mechanism-study records without hand-entered numbers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PRIMARY_METRICS = ("rel_l2_velocity", "rel_l2_vorticity")
SUPPLEMENTARY_METRICS = ("early_rel_l2_velocity", "early_rel_l2_vorticity")
DIAGNOSTIC_METRICS = (
    "rank",
    "rank_fraction",
    "condition_number",
    "max_log_tau_gap",
    "wall_seconds",
    "picard_iterations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def convert_record(row: dict[str, str]) -> dict:
    integer_fields = {
        "seed",
        "basis_seed",
        "collocation_seed",
        "features_per_field",
        "total_unknowns",
        "n_int",
        "n_bc_side",
        "n_gauge",
        "eval_n",
        "eval_nt",
        "picard_limit",
        "picard_iterations",
        "rank",
        "dynamic_features",
        "static_features",
    }
    boolean_fields = {"picard_converged"}
    text_fields = {
        "study",
        "case",
        "point_label",
        "method",
        "method_key",
        "sampling_mode",
    }
    converted: dict = {}
    for field, value in row.items():
        if field in text_fields:
            converted[field] = value
        elif field in boolean_fields:
            converted[field] = parse_bool(value)
        elif value in {"", None, "None"}:
            converted[field] = None
        elif field in integer_fields:
            converted[field] = int(float(value))
        else:
            try:
                converted[field] = float(value)
            except (TypeError, ValueError):
                converted[field] = value
    return converted


def collect(
    *, manifest: Path, results: Path, expected_seeds: list[int]
) -> tuple[list[dict], list[dict], list[dict]]:
    tasks = read_csv(manifest, delimiter="\t")
    records: list[dict] = []
    failures: list[dict] = []
    task_audit: list[dict] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for task in tasks:
        task_dir = results / task["output_rel"]
        status_path = task_dir / "status.json"
        raw_path = task_dir / "metrics_raw.csv"
        failure_path = task_dir / "failures.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        task_failures = (
            json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else []
        )
        task_records = [convert_record(row) for row in read_csv(raw_path)] if raw_path.exists() else []
        for record in task_records:
            key = (
                record["study"],
                record["case"],
                record["point_label"],
                record["method_key"],
                record["seed"],
            )
            if key in seen:
                raise ValueError(f"duplicate mechanism record: {key}")
            seen.add(key)
            records.append(record)
        failures.extend(task_failures)
        observed_seeds = sorted(record["seed"] for record in task_records)
        complete = (
            status.get("completed") == len(expected_seeds)
            and status.get("total") == len(expected_seeds)
            and observed_seeds == expected_seeds
            and not task_failures
        )
        task_audit.append(
            {
                "task_id": int(task["task_id"]),
                "study": task["study"],
                "case": task["case"],
                "point": task["point"],
                "method": task["method"],
                "output_rel": task["output_rel"],
                "records": len(task_records),
                "failures": len(task_failures),
                "observed_seeds": observed_seeds,
                "complete": complete,
            }
        )
    return records, failures, task_audit


def finite_values(rows: list[dict], field: str) -> np.ndarray:
    values = [row.get(field) for row in rows]
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=float,
    )


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for record in records:
        key = (
            record["study"],
            record["case"],
            record["point_label"],
            record["method_key"],
        )
        grouped.setdefault(key, []).append(record)
    summaries: list[dict] = []
    for key, rows in grouped.items():
        study, case, point_label, method_key = key
        ordered = sorted(rows, key=lambda row: row["seed"])
        summary = {
            "study": study,
            "case": case,
            "point": ordered[0]["point"],
            "point_label": point_label,
            "method_key": method_key,
            "method": ordered[0]["method"],
            "features_per_field": ordered[0]["features_per_field"],
            "sampling_mode": ordered[0]["sampling_mode"],
            "tau_min_over_T": ordered[0]["tau_min_over_T"],
            "tau_max_over_T": ordered[0]["tau_max_over_T"],
            "rho0": ordered[0]["rho0"],
            "n": len(ordered),
            "converged_count": sum(row["picard_converged"] for row in ordered),
            "picard_at_limit_count": sum(
                row["picard_iterations"] >= row["picard_limit"] for row in ordered
            ),
            "picard_limit_nonconverged_count": sum(
                row["picard_iterations"] >= row["picard_limit"]
                and not row["picard_converged"]
                for row in ordered
            ),
        }
        for metric in PRIMARY_METRICS + SUPPLEMENTARY_METRICS + DIAGNOSTIC_METRICS:
            values = finite_values(ordered, metric)
            if not values.size:
                continue
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"std_{metric}"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            summary[f"min_{metric}"] = float(np.min(values))
            summary[f"max_{metric}"] = float(np.max(values))
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            row["study"],
            row["case"],
            float(row["point"]) if isinstance(row["point"], (int, float)) else str(row["point"]),
            row["method_key"],
        ),
    )


def feature_efficiency(summary: list[dict], threshold: float) -> list[dict]:
    rows = [row for row in summary if row["study"] == "feature_budget"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["case"], row["method_key"]), []).append(row)
    output: list[dict] = []
    for (case, method), method_rows in sorted(grouped.items()):
        ordered = sorted(method_rows, key=lambda row: int(row["features_per_field"]))
        crossing = next(
            (
                row
                for row in ordered
                if row.get("mean_rel_l2_velocity", float("inf")) < threshold
            ),
            None,
        )
        output.append(
            {
                "case": case,
                "method_key": method,
                "method": ordered[0]["method"],
                "velocity_error_threshold": threshold,
                "minimum_P_below_threshold": (
                    int(crossing["features_per_field"]) if crossing is not None else None
                ),
                "threshold_crossed": crossing is not None,
            }
        )
    return output


def main() -> int:
    args = parse_args()
    config_raw = args.config.read_bytes()
    config = json.loads(config_raw.decode("utf-8"))
    manifest_raw = args.manifest.read_bytes()
    expected_seeds = [int(seed) for seed in config["seeds"]]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records, failures, task_audit = collect(
        manifest=args.manifest.resolve(),
        results=args.results.resolve(),
        expected_seeds=expected_seeds,
    )
    summaries = summarize(records)
    if "feature_budget" in config:
        threshold = float(config["feature_budget"]["predeclared_velocity_error_threshold"])
        efficiency = feature_efficiency(summaries, threshold)
    else:
        efficiency = []
    write_csv(output / "metrics_all.csv", records)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "feature_efficiency.csv", efficiency)
    audit = {
        "study_id": config["study_id"],
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "expected_tasks": len(task_audit),
        "complete_tasks": sum(row["complete"] for row in task_audit),
        "expected_seed_runs": len(task_audit) * len(expected_seeds),
        "records": len(records),
        "failures": failures,
        "incomplete_tasks": [row for row in task_audit if not row["complete"]],
        "picard_limit_nonconverged_records": [
            {
                "study": row["study"],
                "case": row["case"],
                "point_label": row["point_label"],
                "method_key": row["method_key"],
                "seed": row["seed"],
                "iterations": row["picard_iterations"],
                "limit": row["picard_limit"],
            }
            for row in records
            if row["picard_iterations"] >= row["picard_limit"]
            and not row["picard_converged"]
        ],
    }
    write_json(output / "audit.json", audit)
    write_json(output / "task_audit.json", task_audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if not audit["incomplete_tasks"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
