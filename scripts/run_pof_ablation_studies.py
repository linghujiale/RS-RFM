# 中文注释：运行严格单松弛率、朴素指数和静态特征等消融实验。
"""Run one strictly paired RS-RFM ablation task.

The script deliberately reuses the controlled-flow solver and the canonical
RS-RFM-Global random dictionary.  One invocation evaluates one production
seed, which makes the missing grid suitable for a Slurm job array without
rerunning archived records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_pof_mechanism_studies as mechanism  # noqa: E402
import run_pof_suitability_scan as scan  # noqa: E402
from lrfm.assembly.matrix import sample_collocation_2d  # noqa: E402
from lrfm.nonlinear.vorticity_streamfunction import (  # noqa: E402
    solve_vorticity_streamfunction_picard_2d,
)
from lrfm.pdes.vorticity_flow import (  # noqa: E402
    make_exact_vorticity_reference_2d,
    vorticity_reference_grid_errors,
)


CASES = {"two_rate_physics_range", "interacting_two_rate_vortex"}
TEMPORAL_MODES = {"relaxation_quotient", "naive_exponential", "single_rate"}
SAMPLING_MODES = {"log_stratified", "iid_log_uniform"}
DEFAULT_RHO = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--temporal-mode", choices=sorted(TEMPORAL_MODES), required=True)
    parser.add_argument("--rho-static", type=float, required=True)
    parser.add_argument("--tau-min", type=float, default=0.08)
    parser.add_argument("--tau-max", type=float, default=3.0)
    parser.add_argument("--sampling-mode", choices=sorted(SAMPLING_MODES), default="log_stratified")
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--P", type=int, default=384)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def candidate_by_id(case_id: str) -> scan.Candidate:
    for candidates in scan.candidate_matrix().values():
        for candidate in candidates:
            if candidate.candidate_id == case_id:
                return candidate
    raise KeyError(case_id)


def build_problem(args: argparse.Namespace):
    if not 0.0 <= args.rho_static < 1.0:
        raise ValueError("rho_static must lie in [0, 1)")
    if args.P <= 1:
        raise ValueError("P must exceed one")
    if not 0.0 < args.tau_min < args.tau_max:
        raise ValueError("tau bounds must satisfy 0 < tau_min < tau_max")
    if args.temporal_mode == "single_rate" and args.tau0 <= 0.0:
        raise ValueError("tau0 must be positive for single_rate")
    if args.seed not in {10, 11, 12, 13, 14}:
        raise ValueError("formal ablations use the frozen seeds 10--14")

    base = candidate_by_id(args.case)
    cfg = replace(
        base.cfg,
        total_features=int(args.P),
        n_int=round_half_up((75.0 / 16.0) * args.P),
        n_bc_side=round_half_up((35.0 / 192.0) * args.P),
        tau_min_scale=float(args.tau_min),
        tau_max_scale=float(args.tau_max),
    )
    candidate = replace(base, cfg=cfg, static_fraction=float(args.rho_static))
    return candidate, scan.make_pde(candidate)


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _extra_tau_pool(
    *,
    basis_seed: int,
    count: int,
    tau_min: float,
    tau_max: float,
    sampling_mode: str,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence([basis_seed, 0xA61A]))
    log_min = float(np.log(tau_min))
    log_max = float(np.log(tau_max))
    if sampling_mode == "iid_log_uniform":
        return np.exp(rng.uniform(log_min, log_max, size=count))
    edges = np.linspace(log_min, log_max, count + 1)
    return np.exp(rng.uniform(edges[:-1], edges[1:]))


def make_paired_basis(args: argparse.Namespace, candidate: scan.Candidate, pde):
    """Create an ablation basis paired to the archived rho=1/4 dictionary.

    At rho=1/4 this reproduces the archived RS-RFM-Global draws byte for byte.
    Other anchor fractions retain those spatial draws and use nested static
    masks.  Previously static positions receive deterministic extra tau values
    only when they become dynamic.
    """

    basis_seed = 10_000 + int(args.seed)  # archived rs_ic_global seed rule
    canonical = replace(candidate, static_fraction=DEFAULT_RHO)
    base = mechanism.make_proposed_basis(
        "rs_ic_global", pde, canonical, basis_seed, args.sampling_mode
    )
    tau = np.asarray(base.tau, dtype=float).copy()
    base_static = np.asarray(base.is_static, dtype=bool).copy()
    if tau.shape[0] != 1:
        raise ValueError("the primary ablation must use the global one-patch basis")

    target_static_count = round_half_up(float(args.rho_static) * args.P)
    if not 0 <= target_static_count < args.P:
        raise ValueError("static feature count must leave at least one dynamic feature")
    base_static_indices = np.flatnonzero(base_static[0])
    base_dynamic_indices = np.flatnonzero(~base_static[0])
    all_tau = tau[0].copy()
    all_tau[base_static_indices] = _extra_tau_pool(
        basis_seed=basis_seed,
        count=base_static_indices.size,
        tau_min=args.tau_min * pde.T,
        tau_max=args.tau_max * pde.T,
        sampling_mode=args.sampling_mode,
    )

    target_static = np.zeros(args.P, dtype=bool)
    if target_static_count <= base_static_indices.size:
        target_static[base_static_indices[:target_static_count]] = True
    else:
        target_static[base_static_indices] = True
        additional = target_static_count - base_static_indices.size
        target_static[base_dynamic_indices[:additional]] = True

    tau[0] = all_tau
    tau[0, target_static] = np.inf
    if args.temporal_mode == "single_rate":
        tau[0, ~target_static] = float(args.tau0) * pde.T
        implemented_mode = "relaxation_quotient"
    else:
        implemented_mode = args.temporal_mode

    basis = scan.ICCompatibleRelaxationBasis2D(
        pou=base.pou,
        W_space=np.asarray(base.W_space, dtype=float).copy(),
        b_space=np.asarray(base.b_space, dtype=float).copy(),
        tau=tau,
        is_static=target_static[None, :],
        name=variant_label(args),
        temporal_mode=implemented_mode,
    )
    diagnostics = {
        "basis_seed": basis_seed,
        "spatial_draw_sha256": _array_hash(basis.W_space, basis.b_space),
        "tau_draw_sha256": _array_hash(basis.tau),
        "static_mask_sha256": _array_hash(basis.is_static),
        "base_spatial_draw_sha256": _array_hash(base.W_space, base.b_space),
        "base_tau_draw_sha256": _array_hash(base.tau),
        "base_static_mask_sha256": _array_hash(base.is_static),
        "dynamic_features": int(np.count_nonzero(~basis.is_static)),
        "static_features": int(np.count_nonzero(basis.is_static)),
        "paired_to_archived_rho_quarter": True,
    }
    if diagnostics["dynamic_features"] + diagnostics["static_features"] != args.P:
        raise AssertionError("P_dyn + P_sta != P")
    return basis, diagnostics


def variant_key(args: argparse.Namespace) -> str:
    if args.temporal_mode == "naive_exponential":
        return "naive_exponential"
    if args.temporal_mode == "single_rate":
        return "single_rate"
    if np.isclose(args.rho_static, DEFAULT_RHO):
        return "full"
    token = f"{args.rho_static:g}".replace(".", "p")
    return f"rho_{token}"


def variant_label(args: argparse.Namespace) -> str:
    labels = {
        "full": "Full RS-RFM",
        "naive_exponential": "Naive exponential",
        "single_rate": "Single-rate",
        "rho_0": "No static anchor",
        "rho_0p125": "rho=1/8",
        "rho_0p5": "rho=1/2",
    }
    return labels.get(variant_key(args), variant_key(args))


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(record))
        writer.writeheader()
        writer.writerow(record)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    candidate, pde = build_problem(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protected = [output / "run_metadata.json", output / "status.json", output / "metrics_raw.csv"]
    if any(path.exists() for path in protected):
        raise FileExistsError(f"refusing to overwrite an existing task archive: {output}")

    basis, basis_audit = make_paired_basis(args, candidate, pde)
    collocation_seed = 100_000 + int(args.seed)
    collocation = sample_collocation_2d(
        pde,
        n_int=candidate.cfg.n_int,
        n_ic=max(candidate.cfg.n_ic, 1),
        n_bc=candidate.cfg.n_bc_side,
        seed=collocation_seed,
    )
    if candidate.cfg.n_ic == 0:
        collocation = replace(collocation, x_ic=np.empty((0, 2), dtype=float))
    reference = make_exact_vorticity_reference_2d(
        pde, nx=candidate.cfg.eval_n, ny=candidate.cfg.eval_n, nt=candidate.cfg.eval_nt
    )
    metadata = {
        "schema_version": 1,
        "study_id": "rsrfm_ablation_20260824_v1",
        "arguments": vars(args) | {"output": str(output)},
        "candidate": {**asdict(candidate), "cfg": asdict(candidate.cfg)},
        "variant": variant_key(args),
        "basis_audit": basis_audit,
        "host": platform.node(),
        "python": sys.version,
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "scheduler_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_unix": time.time(),
        "integrity": {
            "hard_initial_condition": True,
            "paired_spatial_draws": True,
            "archived_full_not_rerun": True,
            "shared_collocation_rule": "seed=100000+production_seed",
            "same_total_features": True,
            "no_seed_replacement": True,
        },
    }
    write_json(output / "run_metadata.json", metadata)
    np.savez_compressed(
        output / "reference.npz",
        x=reference.x,
        y=reference.y,
        t=reference.t,
        psi=reference.psi,
        omega=reference.omega,
        velocity=reference.velocity,
    )

    started = time.perf_counter()
    result = solve_vorticity_streamfunction_picard_2d(
        pde,
        basis,
        collocation,
        gauge_times=np.linspace(0.0, pde.T, candidate.cfg.n_gauge),
        max_iters=candidate.cfg.max_picard,
        tol=candidate.cfg.picard_tol,
        monitor_nx=candidate.cfg.monitor_n,
        monitor_ny=candidate.cfg.monitor_n,
        monitor_nt=candidate.cfg.monitor_nt,
        cond_threshold=1.0e-12,
        hard_initial=True,
        relaxation=candidate.cfg.picard_relaxation,
    )
    wall_seconds = time.perf_counter() - started
    metrics = vorticity_reference_grid_errors(result.solution, reference)
    record = {
        **metrics,
        "study": "core_ablation" if args.temporal_mode != "relaxation_quotient" else "rho_sensitivity",
        "case": args.case,
        "variant": variant_key(args),
        "variant_label": variant_label(args),
        "seed": int(args.seed),
        "P": int(args.P),
        "features_per_field": int(args.P),
        "rho_static": float(args.rho_static),
        "rho0": float(args.rho_static),
        "temporal_mode": args.temporal_mode,
        "implemented_temporal_mode": basis.temporal_mode,
        "tau_min": float(args.tau_min * pde.T),
        "tau_max": float(args.tau_max * pde.T),
        "tau_min_over_T": float(args.tau_min),
        "tau_max_over_T": float(args.tau_max),
        "tau0": float(args.tau0 * pde.T) if args.temporal_mode == "single_rate" else "",
        "sampling_mode": args.sampling_mode,
        "basis_seed": basis_audit["basis_seed"],
        "collocation_seed": collocation_seed,
        "spatial_draw_sha256": basis_audit["spatial_draw_sha256"],
        "tau_draw_sha256": basis_audit["tau_draw_sha256"],
        "static_mask_sha256": basis_audit["static_mask_sha256"],
        "dynamic_features": basis_audit["dynamic_features"],
        "static_features": basis_audit["static_features"],
        "n_int": int(candidate.cfg.n_int),
        "n_bc_side": int(candidate.cfg.n_bc_side),
        "eval_n": int(candidate.cfg.eval_n),
        "eval_nt": int(candidate.cfg.eval_nt),
        "picard_limit": int(candidate.cfg.max_picard),
        "picard_iterations": int(result.iterations),
        "picard_converged": bool(result.converged),
        "final_relative_update": float(result.relative_updates[-1]),
        "residual_norm": float(result.last_lstsq.residual_norm),
        "rank": int(result.last_lstsq.rank),
        "condition_number": float(result.last_lstsq.condition_number),
        "wall_seconds": float(wall_seconds),
        "source": "new_missing_ablation",
    }
    write_csv(output / "metrics_raw.csv", record)
    metadata["completed_unix"] = time.time()
    metadata["metrics_sha256"] = hashlib.sha256((output / "metrics_raw.csv").read_bytes()).hexdigest()
    write_json(output / "run_metadata.json", metadata)
    write_json(
        output / "status.json",
        {"completed": 1, "total": 1, "records": 1, "failures": 0, "updated_unix": time.time()},
    )
    print(
        json.dumps(
            {
                "case": args.case,
                "variant": variant_key(args),
                "seed": args.seed,
                "Eu": metrics["rel_l2_velocity"],
                "Eomega": metrics["rel_l2_vorticity"],
                "wall_seconds": wall_seconds,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
