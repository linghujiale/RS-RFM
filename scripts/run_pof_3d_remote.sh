#!/usr/bin/env bash
# 中文注释：运行PoF 论文 三维远程实验，用于论文实验的批处理或远程执行。
#SBATCH --job-name=rsrfm-3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

stage=${1:?stage is required}
output_dir=${2:?output directory is required}
methods=${3:-}
seeds=${4:-}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export RSRFM_REMOTE_FS=${RSRFM_REMOTE_FS:-unknown}

mkdir -p "$output_dir"
{
  date --iso-8601=seconds
  hostname
  lscpu
  free -h
  df -h "$PWD"
  quota -s 2>&1 || true
  python3 --version
  python3 -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
  printf 'RSRFM_REMOTE_FS=%s\n' "$RSRFM_REMOTE_FS"
} > "$output_dir/hardware_and_environment.txt"

command=(
  python3 scripts/run_pof_fluid_3d_experiment.py
  --stage "$stage"
  --output "$output_dir"
)
if [[ -n "$methods" ]]; then
  command+=(--methods "$methods")
fi
if [[ -n "$seeds" ]]; then
  command+=(--seeds "$seeds")
fi
"${command[@]}"
