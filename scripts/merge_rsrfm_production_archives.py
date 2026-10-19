# 中文注释：合并并校验不同批次的 RS-RFM 冻结实验记录。
"""Merge disjoint RS-RFM production archives with strict integrity checks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pof_suitability_scan as scan  # noqa: E402


CASES = {
    "tgv_order1_decay",
    "two_rate_physics_range",
    "interacting_two_rate_vortex",
    "multiscale_shear_decay_k12",
}
METHODS = {
    "rs_ic_local",
    "rs_ic_global",
    "rs_fixed_tau",
    "rs_static",
    "strfm_sov",
    "strfm_stc",
    "pielm",
}
SEEDS = {10, 11, 12, 13, 14}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_archive(directory: Path) -> tuple[list[dict], dict]:
    failures = json.loads((directory / "failures.json").read_text(encoding="utf-8"))
    if failures:
        raise ValueError(f"archive {directory} contains failures: {failures}")
    with (directory / "metrics_raw.csv").open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    return records, metadata


def convert(row: dict) -> dict:
    item = dict(row)
    for key, value in list(item.items()):
        if key in {"candidate_id", "case", "method", "method_key"}:
            continue
        if key == "picard_converged":
            item[key] = str(value).lower() == "true"
        elif key in {"seed", "features_per_field", "picard_iterations", "rank"}:
            item[key] = int(float(value))
        else:
            item[key] = float(value)
    return item


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records_by_key: dict[tuple[str, str, int], dict] = {}
    consistent_duplicates: list[tuple[str, str, int]] = []
    metadata_sources: list[dict] = []
    candidate_definitions: dict[str, dict] = {}
    for archive in args.archive:
        source = archive.resolve()
        source_records, metadata = load_archive(source)
        for candidate in metadata.get("candidates", []):
            candidate_id = candidate["candidate_id"]
            previous_candidate = candidate_definitions.get(candidate_id)
            if previous_candidate is not None and previous_candidate != candidate:
                raise ValueError(f"inconsistent candidate definition for {candidate_id}")
            candidate_definitions[candidate_id] = candidate
        for raw_row in source_records:
            row = convert(raw_row)
            key = (row["candidate_id"], row["method_key"], row["seed"])
            if key in records_by_key:
                previous = records_by_key[key]
                ignored = {"wall_seconds"}
                for field in sorted(set(previous) | set(row)):
                    if field in ignored:
                        continue
                    left = previous.get(field)
                    right = row.get(field)
                    if isinstance(left, float) or isinstance(right, float):
                        if abs(float(left) - float(right)) > 1.0e-12 * max(
                            1.0, abs(float(left)), abs(float(right))
                        ):
                            raise ValueError(f"inconsistent duplicate {key} in field {field}")
                    elif left != right:
                        raise ValueError(f"inconsistent duplicate {key} in field {field}")
                consistent_duplicates.append(key)
                continue
            records_by_key[key] = row
        metadata_sources.append(
            {
                "path": str(source),
                "records": len(source_records),
                "started_unix": metadata.get("started_unix"),
                "completed_unix": metadata.get("completed_unix"),
            }
        )

    expected_keys = {
        (case, method, seed) for case in CASES for method in METHODS for seed in SEEDS
    }
    actual_keys = set(records_by_key)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(f"archive grid mismatch; missing={missing}, unexpected={unexpected}")

    records = sorted(
        records_by_key.values(),
        key=lambda row: (row["candidate_id"], row["method_key"], row["seed"]),
    )
    scan.write_csv(output / "metrics_raw.csv", records)
    scan.write_csv(output / "config_rankings.csv", scan.summarize(records))
    scan.write_json(output / "failures.json", [])
    scan.write_json(
        output / "status.json",
        {
            "completed": len(expected_keys),
            "total": len(expected_keys),
            "records": len(records),
            "failures": 0,
            "updated_unix": time.time(),
        },
    )
    scan.write_json(
        output / "run_metadata.json",
        {
            "stage": "production",
            "cases": sorted(CASES),
            "methods": sorted(METHODS),
            "seeds": sorted(SEEDS),
            "records": len(records),
            "failures": 0,
            "candidates": [candidate_definitions[key] for key in sorted(candidate_definitions)],
            "completed_unix": time.time(),
            "source_archives": metadata_sources,
            "integrity": {
                "sources_disjoint": not consistent_duplicates,
                "consistent_duplicate_keys": [list(key) for key in consistent_duplicates],
                "duplicate_accuracy_consistency_verified": True,
                "full_case_method_seed_grid": True,
                "configuration_frozen_before_held_out_seeds": True,
                "replacement_runs": False,
            },
        },
    )
    print(f"merged {len(records)} unique held-out records into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
