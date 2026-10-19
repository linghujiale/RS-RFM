# 中文注释：检验线性与非线性配对表格生成脚本。
import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_pof_linear_nonlinear_pair_artifacts.py"
SPEC = importlib.util.spec_from_file_location("make_pof_linear_nonlinear_pair_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


RAW_METHODS = {
    "rs_ic_global": "rs_ic_global",
    "rs_ic_local": "rs_ic_local",
    "rfm_global": "pielm",
    "rfm_local": "strfm_stc",
    "sep_rfm_global": "sep_rfm_global",
    "sep_rfm_local": "strfm_sov",
    "pinn": "pinn",
}


def test_complete_pair_archive_is_validated_and_rendered(tmp_path):
    for case in MODULE.CASES:
        for method in MODULE.METHODS:
            for seed in MODULE.SEEDS:
                output = tmp_path / case / method / f"seed{seed}"
                output.mkdir(parents=True)
                linear = case == MODULE.CASES[0]
                row = {
                    "candidate_id": case,
                    "method_key": RAW_METHODS[method],
                    "seed": seed,
                    "rel_l2_velocity": 1.0e-3 * (1 + seed - 10),
                    "rel_l2_vorticity": 2.0e-3 * (1 + seed - 10),
                    "wall_seconds": 1.0,
                }
                if method != "pinn":
                    row.update(
                        {
                            "features_per_field": 384,
                            "viscosity": 0.02,
                            "horizon": 1.0,
                            "tau_min_scale": 0.08,
                            "tau_max_scale": 3.0,
                            "static_fraction": 0.25 if method.startswith("rs_ic") else 0.0,
                            "picard_relaxation": 1.0,
                            "least_squares_solves": 1 if linear else 4,
                            "picard_iterations": 0 if linear else 4,
                            "picard_converged": True,
                        }
                    )
                with (output / "metrics_raw.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                    writer.writerow(row)
                if method == "pinn":
                    metadata = {
                        "profile": "production",
                        "training": {"adam_steps": 15000, "lbfgs_steps": 300},
                        "network": {"width": 128, "depth": 5},
                        "sampling": {"collocation_seed": 100_000 + seed},
                        "environment": {"device": "cpu"},
                    }
                    (output / "run_metadata.json").write_text(
                        json.dumps(metadata), encoding="utf-8"
                    )

    records = MODULE.load_records(tmp_path)
    MODULE.validate_pairing(records)
    summary = MODULE.summarize(records)
    tex = MODULE.make_tex(summary)

    assert len(records) == 70
    assert len(summary) == 14
    assert "RS-RFM-Global" in tex
    assert "N_{\\rm LS}" not in tex
    assert "lcc@{\\qquad}cc" in tex
    assert "线性非定常 Stokes" in tex
    assert "非线性 Navier--Stokes" in tex
