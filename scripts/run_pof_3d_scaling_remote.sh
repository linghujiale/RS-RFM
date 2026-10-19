#!/usr/bin/env bash
# 中文注释：运行PoF 论文 三维规模实验 远程计算任务，用于论文实验的批处理或远程执行。
#SBATCH --job-name=rsrfm-3d-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

set -euo pipefail

features=${1:?feature count is required}
method=${2:?method is required}
seed=${3:?seed is required}
output_dir=${4:?output directory is required}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export RSRFM_REMOTE_FS=${RSRFM_REMOTE_FS:-unknown}

mkdir -p "$output_dir"
{
  date --iso-8601=seconds
  hostname
  lscpu
  free -h
  python3 --version
  python3 -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
  printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK:-unknown}"
  printf 'RSRFM_REMOTE_FS=%s\n' "$RSRFM_REMOTE_FS"
} > "$output_dir/hardware_and_environment.txt"

/usr/bin/time -v -o "$output_dir/resource_usage.txt" \
  python3 scripts/run_pof_fluid_3d_capacity_scan.py \
    --features "$features" \
    --method "$method" \
    --seed "$seed" \
    --output "$output_dir" \
    --skip-prediction
