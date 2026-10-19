# 中文注释：检验 Beltrami 三维结果制品生成脚本。
from __future__ import annotations

import csv
import json

import pytest

from scripts.make_pof_3d_beltrami_artifacts import SEEDS, load_records, summarize, write_generated_tex


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_archive(root):
    _write_json(root / "results/gate/gate_report.json", {"passed": False})
    _write_json(root / "results/high_gate/gate_report.json", {"passed": True})
    production = root / "results/high_production"
    _write_json(production / "status.json", {"completed": 5, "records": 5, "failures": 0, "total": 5})
    _write_json(production / "failures.json", [])
    _write_json(
        production / "run_metadata.json",
        {
            "stage": "production",
            "profile_name": "high",
            "profile": {
                "total_features": 384,
                "n_int": 2400,
                "n_bc_face": 75,
                "n_gauge": 9,
                "eval_n": 16,
                "eval_nt": 9,
                "max_picard": 15,
            },
            "integrity": {
                "confirmatory_eligible": False,
                "replacement_runs": False,
                "held_out_seeds": SEEDS,
                "failed_standard_gate_job_id": "6784011",
            },
        },
    )
    rows = []
    for seed in SEEDS:
        rows.append(
            {
                "candidate_id": "forced_decaying_beltrami_3d",
                "case": "forced_decaying_beltrami_3d",
                "method_key": "rs_ic_global",
                "seed": seed,
                "features_per_field": 384,
                "total_unknowns": 1536,
                "n_int": 2400,
                "n_bc_face": 75,
                "n_rows": 11409,
                "rank": 1536,
                "picard_iterations": 9,
                "picard_converged": "True",
                "rel_l2_velocity": 0.004,
                "rel_l2_vorticity": 0.01,
                "relative_divergence": 0.0035,
                "pressure_rms_over_u2": 0.013,
                "final_relative_update": 1e-8,
                "wall_seconds": 38.0,
                "condition_number": 2e5,
                "residual_norm": 0.6,
            }
        )
    with (production / "metrics_raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_beltrami_artifact_loader_audits_gates_and_heldout_grid(tmp_path):
    _write_archive(tmp_path)
    records = load_records(tmp_path)
    assert [row["seed"] for row in records] == SEEDS
    assert summarize(records)["runs"] == {"completed": 5, "converged": 5}


def test_beltrami_artifact_loader_rejects_confirmatory_relabeling(tmp_path):
    _write_archive(tmp_path)
    metadata_path = tmp_path / "results/high_production/run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["integrity"]["confirmatory_eligible"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be marked confirmatory"):
        load_records(tmp_path)


def test_generated_beltrami_table_omits_convergence_indicator(tmp_path):
    _write_archive(tmp_path)
    output = tmp_path / "fluid_3d_beltrami_results.tex"
    write_generated_tex(summarize(load_records(tmp_path)), output)
    text = output.read_text(encoding="utf-8")
    table = text.split(r"\begin{tabular}", 1)[1].split(r"\end{tabular}", 1)[0]
    assert "Conv." not in table
    assert "5/5" not in table
    assert "post-hoc" in text
    assert "confirmatory" in text
    assert "\n\n" not in table
