# 中文注释：构建PoF 论文 消融实验任务清单，并输出可复核的实验或制图结果。
"""Build the immutable manifest containing only confirmed-missing ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-original-task-ids",
        default="",
        help="comma-separated original task IDs already completed in a partial submission",
    )
    return parser.parse_args()


def safe(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def variant_key(temporal_mode: str, rho_static: float) -> str:
    if temporal_mode == "naive_exponential":
        return "naive_exponential"
    if temporal_mode == "single_rate":
        return "single_rate"
    return f"rho_{safe(rho_static)}"


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    for specification in config["missing_variants"]:
        for seed in config["seeds"]:
            variant = variant_key(specification["temporal_mode"], specification["rho_static"])
            rows.append(
                {
                    "task_id": len(rows),
                    "experiment": "core_ablation" if specification["temporal_mode"] != "relaxation_quotient" else "rho_sensitivity",
                    "case": specification["case"],
                    "variant": variant,
                    "temporal_mode": specification["temporal_mode"],
                    "rho_static": specification["rho_static"],
                    "tau_min": config["tau_window_over_T"][0],
                    "tau_max": config["tau_window_over_T"][1],
                    "sampling_mode": config["sampling_mode"],
                    "tau0": config["tau0_over_T"] if specification["temporal_mode"] == "single_rate" else "",
                    "seed": seed,
                    "P": config["P"],
                    "status": "missing",
                    "output_rel": f"{specification['case']}/{variant}/seed{seed}",
                }
            )
    excluded = {
        int(value) for value in args.exclude_original_task_ids.split(",") if value.strip()
    }
    rows = [row for row in rows if int(row["task_id"]) not in excluded]
    for task_id, row in enumerate(rows):
        row["task_id"] = task_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(json.dumps({"missing_seed_runs": len(rows), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
