#!/usr/bin/env bash
# 中文注释：运行PoF 论文 缺失随机特征对比任务，用于论文实验的批处理或远程执行。
#SBATCH --job-name=pof-sepg
#SBATCH --partition=dzacnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-29%8

set -euo pipefail

code_root=${RSRFM_CODE_ROOT:?RSRFM_CODE_ROOT is required}
result_root=${RSRFM_COMPARISON_ROOT:?RSRFM_COMPARISON_ROOT is required}
array_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
python_bin=${RSRFM_PYTHON:-/work/home/zhaoteng_sciai/miniconda3/bin/python3}
export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export RSRFM_REMOTE_FS="$result_root"
cd "$code_root"

controlled_ids=(
  tgv_order1_decay
  two_rate_physics_range
  interacting_two_rate_vortex
  multiscale_shear_decay_k12
)
controlled_cases=(
  taylor_green
  two_rate_vortex
  interacting_vortex
  multiscale_shear_k12
)

run_one() {
  local task_id=$1
if (( task_id < 20 )); then
  case_index=$((task_id / 5))
  seed=$((10 + task_id % 5))
  candidate=${controlled_ids[$case_index]}
  case_name=${controlled_cases[$case_index]}
  output="$result_root/rfm/controlled2d/$candidate/sep_rfm_global_seed$seed"
  "$python_bin" scripts/run_pof_suitability_scan.py \
    --stage production \
    --cases "$case_name" \
    --candidate-ids "$candidate" \
    --methods sep_rfm_global \
    --seeds "$seed" \
    --output "$output"
elif (( task_id < 25 )); then
  seed=$((50 + task_id - 20))
  output="$result_root/rfm/decay2d/decaying_turbulence/sep_rfm_global_seed$seed"
  "$python_bin" scripts/run_pof_decaying_turbulence.py \
    --profile fixed_1280 \
    --stage production \
    --methods sep_rfm_global \
    --seeds "$seed" \
    --output "$output"
else
  seed=$((100 + task_id - 25))
  output="$result_root/rfm/helical3d/interacting_helical_vortex_3d/sep_rfm_global_seed$seed"
  "$python_bin" scripts/run_pof_fluid_3d_experiment.py \
    --stage fixed \
    --methods sep_rfm_global \
    --seeds "$seed" \
    --output "$output"
fi
}

if [[ ${RSRFM_GROUPED:-0} == 1 ]]; then
  if (( array_id < 4 )); then
    for seed_offset in 0 1 2 3 4; do
      run_one $((5 * array_id + seed_offset))
    done
  elif (( array_id == 4 )); then
    for task_id in 20 21 22 23 24; do run_one "$task_id"; done
  elif (( array_id == 5 )); then
    for task_id in 25 26 27 28 29; do run_one "$task_id"; done
  else
    echo "invalid grouped array id: $array_id" >&2
    exit 2
  fi
else
  run_one "$array_id"
fi
