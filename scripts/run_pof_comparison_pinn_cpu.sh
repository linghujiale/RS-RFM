#!/usr/bin/env bash
# 中文注释：运行PoF 论文 PINN 对比任务 CPU 版本，用于论文实验的批处理或远程执行。
#SBATCH --job-name=pof-pinn-cpu
#SBATCH --partition=dzacnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-14%15

set -euo pipefail

code_root=${RSRFM_CODE_ROOT:?RSRFM_CODE_ROOT is required}
result_root=${RSRFM_COMPARISON_ROOT:?RSRFM_COMPARISON_ROOT is required}
decay_reference=${RSRFM_DECAY_REFERENCE:?RSRFM_DECAY_REFERENCE is required}
array_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
python_bin=${RSRFM_PYTHON:-/work/home/zhaoteng_sciai/miniconda3/bin/python3}
export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export RSRFM_ALLOW_CPU_PRODUCTION=1
cd "$code_root"

controlled_ids=(
  tgv_order1_decay
  two_rate_physics_range
  interacting_two_rate_vortex
  multiscale_shear_decay_k12
)

run_one() {
  local task_id=$1
  if (( task_id < 20 )); then
    local case_index=$((task_id / 5))
    local seed=$((10 + task_id % 5))
    local candidate=${controlled_ids[$case_index]}
    local output="$result_root/pinn/controlled2d/$candidate/pinn_seed$seed"
    "$python_bin" scripts/run_pof_pinn_baseline.py \
      --suite controlled2d \
      --case "$candidate" \
      --seed "$seed" \
      --profile production \
      --output "$output"
  elif (( task_id < 25 )); then
    local seed=$((50 + task_id - 20))
    local output="$result_root/pinn/decay2d/decaying_turbulence/pinn_seed$seed"
    "$python_bin" scripts/run_pof_pinn_baseline.py \
      --suite decay2d \
      --case decaying_turbulence \
      --seed "$seed" \
      --profile production \
      --reference "$decay_reference" \
      --output "$output"
  else
    local seed=$((100 + task_id - 25))
    local output="$result_root/pinn/helical3d/interacting_helical_vortex_3d/pinn_seed$seed"
    "$python_bin" scripts/run_pof_pinn_baseline.py \
      --suite helical3d \
      --case interacting_helical_vortex_3d \
      --seed "$seed" \
      --profile production \
      --output "$output"
  fi
}

run_one $((2 * array_id))
run_one $((2 * array_id + 1))
