#!/usr/bin/env bash
# 中文注释：运行PoF 论文 缺失机制对比任务，用于论文实验的批处理或远程执行。
#SBATCH --job-name=pof-sepg-mech
#SBATCH --partition=dzacnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3

set -euo pipefail

code_root=${RSRFM_CODE_ROOT:?RSRFM_CODE_ROOT is required}
result_root=${RSRFM_COMPARISON_ROOT:?RSRFM_COMPARISON_ROOT is required}
array_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
python_bin=${RSRFM_PYTHON:-/work/home/zhaoteng_sciai/miniconda3/bin/python3}
config="$code_root/configs/pof_comparison_mechanism_sep_global_20260824.json"
export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$code_root"

feature_budgets=(96 128 192 256 384 512 768)
feature_cases=(two_rate_vortex interacting_vortex)
fast_rates=(0.5 1.0 2.0 4.0 8.0)

run_point() {
  local point_id=$1
  if (( point_id < 14 )); then
    local case_index=$((point_id / 7))
    local budget_index=$((point_id % 7))
    local case_name=${feature_cases[$case_index]}
    local budget=${feature_budgets[$budget_index]}
    local output="$result_root/rfm/mechanism/feature_budget/$case_name/P$(printf '%04d' "$budget")/sep_rfm_global"
    "$python_bin" scripts/run_pof_mechanism_studies.py \
      --config "$config" \
      --study feature_budget \
      --case "$case_name" \
      --point "$budget" \
      --method sep_rfm_global \
      --seeds 10,11,12,13,14 \
      --output "$output"
  else
    local rate=${fast_rates[$((point_id - 14))]}
    local label=${rate/./p}
    local output="$result_root/rfm/mechanism/scale_separation/lambda2_${label}/sep_rfm_global"
    "$python_bin" scripts/run_pof_mechanism_studies.py \
      --config "$config" \
      --study scale_separation \
      --case two_rate_vortex \
      --point "$rate" \
      --method sep_rfm_global \
      --seeds 10,11,12,13,14 \
      --output "$output"
  fi
}

case "$array_id" in
  0) start=0; stop=7 ;;
  1) start=7; stop=13 ;;
  2) start=13; stop=19 ;;
  *) echo "invalid grouped array id: $array_id" >&2; exit 2 ;;
esac
for ((point_id=start; point_id<stop; point_id++)); do
  run_point "$point_id"
done
