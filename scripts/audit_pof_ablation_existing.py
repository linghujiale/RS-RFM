# 中文注释：审核PoF 论文 已有消融结果，并输出可复核的实验或制图结果。
"""Inventory archived mechanism records before any new ablation is submitted."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def main() -> int:
    args = parse_args()
    raw = pd.read_csv(args.input)
    required = {
        "study", "case", "method_key", "seed", "features_per_field", "rho0",
        "tau_min", "tau_max", "sampling_mode", "rel_l2_velocity",
        "rel_l2_vorticity", "wall_seconds", "source_file",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"archive lacks required fields: {missing}")
    inventory = pd.DataFrame(
        {
            "experiment": raw.study,
            "case": raw.case,
            "variant": raw.method_key,
            "seed": raw.seed.astype(int),
            "P": raw.features_per_field.astype(int),
            "rho0": raw.rho0,
            "tau_min": raw.tau_min,
            "tau_max": raw.tau_max,
            "sampling": raw.sampling_mode,
            "tau0": raw.apply(
                lambda row: 1.0 if row.method_key == "rs_fixed_tau" else "", axis=1
            ),
            "Eu": raw.rel_l2_velocity,
            "Eomega": raw.rel_l2_vorticity,
            "runtime_s": raw.wall_seconds,
            "source": raw.source_file,
            "reusable": raw.method_key.map(
                lambda key: "context-only; not strict single-rate" if key == "rs_fixed_tau" else "yes"
            ),
        }
    ).sort_values(["experiment", "case", "variant", "P", "seed"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Existing RS-RFM mechanism results",
        "",
        f"Audited source: `{args.input}`",
        "",
        f"Records: {len(inventory)}. No archived run was resubmitted.",
        "",
        "The archived `rs_fixed_tau` implementation uses the legacy local liquid-feature basis; it is retained as context but is not reusable as the strict RS-RFM-SingleRate ablation.",
        "",
        "| experiment | case | variant | seed | P | rho0 | tau_min | tau_max | sampling | tau0 | Eu | Eomega | runtime_s | source | reusable |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in inventory.itertuples(index=False):
        lines.append("| " + " | ".join(text(value) for value in row) + " |")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(inventory)} archived records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
