# 中文注释：汇总PoF 论文 消融实验结果，并输出可复核的实验或制图结果。
"""Merge new strict ablations with archived full RS-RFM records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [10, 11, 12, 13, 14]
CASE_MAP = {
    "two_rate_vortex": "two_rate_physics_range",
    "interacting_vortex": "interacting_two_rate_vortex",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_new(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("metrics_raw.csv"))
    if not files:
        raise FileNotFoundError(f"no new metrics under {root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def archived_full(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    full = raw[
        (raw.study == "feature_budget")
        & (raw.method_key == "rs_ic_global")
        & (raw.features_per_field == 384)
        & raw.case.isin(CASE_MAP)
    ].copy()
    full["case"] = full.case.map(CASE_MAP)
    full["variant"] = "full"
    full["variant_label"] = "Full RS-RFM"
    full["temporal_mode"] = "relaxation_quotient"
    full["implemented_temporal_mode"] = "relaxation_quotient"
    full["P"] = full.features_per_field.astype(int)
    full["rho_static"] = 0.25
    full["tau0"] = np.nan
    full["source"] = "archived_feature_budget_P384"
    return full


def validate(new: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    expected = manifest[["case", "variant", "seed"]].drop_duplicates()
    actual = new[["case", "variant", "seed"]].drop_duplicates()
    merged = expected.merge(actual, on=["case", "variant", "seed"], how="outer", indicator=True)
    missing = merged[merged._merge == "left_only"]
    unexpected = merged[merged._merge == "right_only"]
    duplicate_count = int(new.duplicated(["case", "variant", "seed"]).sum())
    capacity_failures = int(
        np.count_nonzero(
            new.dynamic_features.to_numpy(dtype=int)
            + new.static_features.to_numpy(dtype=int)
            != new.P.to_numpy(dtype=int)
        )
    )
    spatial_hash_counts = (
        new.groupby(["case", "seed"], observed=True).spatial_draw_sha256.nunique().to_dict()
    )
    pairing_failures = int(sum(count != 1 for count in spatial_hash_counts.values()))
    audit = {
        "expected_new_records": int(len(expected)),
        "actual_new_records": int(len(new)),
        "missing_records": int(len(missing)),
        "unexpected_records": int(len(unexpected)),
        "duplicate_records": duplicate_count,
        "capacity_failures": capacity_failures,
        "spatial_pairing_failures": pairing_failures,
        "all_seeds_present": sorted(new.seed.astype(int).unique().tolist()) == SEEDS,
        "passed": False,
    }
    audit["passed"] = all(
        [
            audit["actual_new_records"] == 35,
            audit["missing_records"] == 0,
            audit["unexpected_records"] == 0,
            audit["duplicate_records"] == 0,
            audit["capacity_failures"] == 0,
            audit["spatial_pairing_failures"] == 0,
            audit["all_seeds_present"],
        ]
    )
    if not audit["passed"]:
        raise ValueError(f"ablation archive failed completeness gate: {audit}")
    return audit


def main() -> int:
    args = parse_args()
    new = read_new(args.new_root)
    manifest = pd.read_csv(args.manifest)
    audit = validate(new, manifest)
    full = archived_full(args.archived)
    combined = pd.concat([full, new], ignore_index=True, sort=False)
    if len(full) != 10 or len(combined) != 45:
        raise ValueError(f"unexpected archived/full counts: full={len(full)}, all={len(combined)}")

    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / "ablation_metrics_all.csv", index=False)
    summary = (
        combined.groupby(
            ["case", "variant", "variant_label", "P", "rho_static", "temporal_mode"],
            dropna=False,
            observed=True,
        )[["rel_l2_velocity", "rel_l2_vorticity", "wall_seconds"]]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "ablation_summary.csv", index=False)
    (args.output / "ablation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
