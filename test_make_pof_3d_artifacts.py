# 中文注释：检验三维结果表格和图形生成脚本。
from __future__ import annotations

import csv
import json

import pytest

from scripts.make_pof_3d_artifacts import (
    FLOAT_FIELDS,
    INT_FIELDS,
    METHOD_ORDER,
    SEEDS,
    load_records,
    summarize,
    write_generated_tex,
)


def _write_complete_archive(path):
    status = {"completed": 30, "records": 30, "failures": 0, "total": 30}
    metadata = {
        "stage": "exploratory",
        "profile": {
            "total_features": 256,
            "n_int": 1600,
            "n_bc_face": 50,
            "eval_n": 12,
            "eval_nt": 7,
            "max_picard": 10,
        },
        "integrity": {
            "confirmatory_eligible": False,
            "failed_gate_job_id": "6783779",
        },
    }
    (path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (path / "failures.json").write_text("[]", encoding="utf-8")
    (path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    records = []
    for method in METHOD_ORDER:
        for seed in SEEDS:
            row = {field: 0.1 for field in FLOAT_FIELDS}
            row.update({field: 1 for field in INT_FIELDS})
            row.update(
                {
                    "candidate_id": "interacting_helical_vortex_3d",
                    "case": "helical_vortex_3d",
                    "method_key": method,
                    "seed": seed,
                    "features_per_field": 256,
                    "total_unknowns": 1024,
                    "n_int": 1600,
                    "n_bc_face": 50,
                    "n_rows": 7607,
                    "viscosity": 0.02,
                    "horizon": 1.0,
                    "lambda_slow": 0.5,
                    "lambda_fast": 4.0,
                    "tau_min_scale": 0.08,
                    "tau_max_scale": 3.0,
                    "picard_converged": "True",
                }
            )
            records.append(row)
    with (path / "metrics_raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def test_3d_artifact_loader_accepts_only_complete_exploratory_grid(tmp_path):
    _write_complete_archive(tmp_path)
    records = load_records(tmp_path)
    assert len(records) == 30
    assert {(row["method_key"], row["seed"]) for row in records} == {
        (method, seed) for method in METHOD_ORDER for seed in SEEDS
    }


def test_3d_artifact_loader_rejects_confirmatory_relabeling(tmp_path):
    _write_complete_archive(tmp_path)
    metadata_path = tmp_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["integrity"]["confirmatory_eligible"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="must not be marked confirmatory"):
        load_records(tmp_path)


def test_generated_3d_table_has_no_paragraph_breaks_inside_tabular(tmp_path):
    _write_complete_archive(tmp_path)
    summary = summarize(load_records(tmp_path))
    output = tmp_path / "fluid_3d_results.tex"
    write_generated_tex(summary, output)
    text = output.read_text(encoding="utf-8")
    table = text.split(r"\begin{tabular}", 1)[1].split(r"\end{tabular}", 1)[0]

    assert "\n\n" not in table
    assert "Exploratory three-dimensional helical-vortex results" in text
    assert table.count(r"\\") == 7  # one header and six method rows
