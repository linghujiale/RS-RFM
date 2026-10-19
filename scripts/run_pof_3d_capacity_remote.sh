#!/usr/bin/env bash
# 中文注释：运行PoF 论文 三维容量实验 远程计算任务，用于论文实验的批处理或远程执行。
#SBATCH --job-name=rsrfm-cap3d
#SBATCH --partition=dzacnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=24:00:00

set -euo pipefail

features=${1:?feature count is required}
output_dir=${2:?output directory is required}
seed=${3:-0}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export RSRFM_REMOTE_FS=${RSRFM_REMOTE_FS:-unknown}

mkdir -p "$output_dir"
python3 scripts/run_pof_fluid_3d_capacity_scan.py \
    --features "$features" \
    --seed "$seed" \
    --output "$output_dir"
