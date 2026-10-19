#!/usr/bin/env bash
# 中文注释：运行PoF 论文 单个消融实验任务，用于论文实验的批处理或远程执行。
#SBATCH --job-name=rsrfm-ablate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

manifest=${1:?manifest path is required}
results_root=${2:?results root is required}
task_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

line=$(awk -F ',' -v target="$task_index" 'NR > 1 && $1 == target {print; exit}' "$manifest")
if [[ -z "$line" ]]; then
  echo "task $task_index not found in $manifest" >&2
  exit 2
fi
IFS=',' read -r manifest_task_id experiment case_name variant temporal_mode rho_static tau_min tau_max sampling_mode tau0 seed features status output_rel <<< "$line"
output_rel=${output_rel%$'\r'}
if [[ "$status" != "missing" ]]; then
  echo "refusing non-missing manifest row: $line" >&2
  exit 2
fi
output_dir=${results_root}/${output_rel}
mkdir -p "$output_dir"

{
  date --iso-8601=seconds
  hostname
  lscpu
  free -h
  python3 --version
  python3 -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
  printf 'SLURM_ARRAY_TASK_ID=%s\n' "$task_index"
  printf 'manifest_task_id=%s\n' "$manifest_task_id"
} > "$output_dir/hardware_and_environment.txt"

common_args=(
  --case "$case_name"
  --temporal-mode "$temporal_mode"
  --rho-static "$rho_static"
  --tau-min "$tau_min"
  --tau-max "$tau_max"
  --sampling-mode "$sampling_mode"
  --seed "$seed"
  --P "$features"
  --output "$output_dir"
)
if [[ -n "$tau0" ]]; then
  python3 scripts/run_pof_ablation_studies.py "${common_args[@]}" --tau0 "$tau0"
else
  python3 scripts/run_pof_ablation_studies.py "${common_args[@]}"
fi
