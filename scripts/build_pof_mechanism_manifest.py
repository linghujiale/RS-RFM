# 中文注释：构建PoF 论文 机制实验任务清单，并输出可复核的实验或制图结果。
"""Create the immutable Slurm task manifest for PoF mechanism studies."""

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
    return parser.parse_args()


def safe_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, str | int | float]] = []

    feature = config.get("feature_budget")
    if feature is not None:
        for case in feature["cases"]:
            for features in feature["feature_budgets"]:
                for method in feature["methods"]:
                    rows.append(
                        {
                            "task_id": len(rows),
                            "study": "feature_budget",
                            "case": case,
                            "point": features,
                            "method": method,
                            "output_rel": f"feature_budget/{case}/P{features:04d}/{method}",
                        }
                    )

    separation = config.get("scale_separation")
    if separation is not None:
        symbol = str(separation.get("physical_rate_symbol", "lambda"))
        for fast_rate in separation["fast_rates"]:
            for method in separation["methods"]:
                rows.append(
                    {
                        "task_id": len(rows),
                        "study": "scale_separation",
                        "case": separation["case"],
                        "point": fast_rate,
                        "method": method,
                        "output_rel": (
                            f"scale_separation/{symbol}2_{safe_token(float(fast_rate))}/{method}"
                        ),
                    }
                )

    windows = config.get("spectral_window")
    if windows is not None:
        for window_id in windows["windows_over_T"]:
            for method in windows["methods"]:
                rows.append(
                    {
                        "task_id": len(rows),
                        "study": "spectral_window",
                        "case": windows["case"],
                        "point": window_id,
                        "method": method,
                        "output_rel": f"spectral_window/{window_id}/{method}",
                    }
                )

    sampling = config.get("sampling_ablation")
    if sampling is not None:
        for sampling_mode in sampling["sampling_modes"]:
            for method in sampling["methods"]:
                rows.append(
                    {
                        "task_id": len(rows),
                        "study": "sampling_ablation",
                        "case": sampling["case"],
                        "point": sampling_mode,
                        "method": method,
                        "output_rel": f"sampling_ablation/{sampling_mode}/{method}",
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["task_id", "study", "case", "point", "method", "output_rel"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "tasks": len(rows),
                "planned_seed_runs": len(rows) * len(config["seeds"]),
                "manifest": str(output),
                "sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
