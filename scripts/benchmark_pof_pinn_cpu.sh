#!/usr/bin/env bash
# 中文注释：评测PoF 论文 pinn CPU 版本，用于论文实验的批处理或远程执行。
#SBATCH --job-name=pof-pinn-cpubench
#SBATCH --partition=dzacnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00

set -euo pipefail
code_root=${RSRFM_CODE_ROOT:?RSRFM_CODE_ROOT is required}
result_root=${RSRFM_COMPARISON_ROOT:?RSRFM_COMPARISON_ROOT is required}
python_bin=${RSRFM_PYTHON:-/work/home/zhaoteng_sciai/miniconda3/bin/python3}
export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export RSRFM_ALLOW_CPU_PRODUCTION=1
cd "$code_root"

/usr/bin/time -v "$python_bin" scripts/run_pof_pinn_baseline.py \
  --suite controlled2d \
  --case tgv_order1_decay \
  --seed 999 \
  --profile production \
  --adam-steps 10 \
  --lbfgs-steps 0 \
  --output "$result_root/benchmark_only/pinn_cpu_tgv_10steps"

/usr/bin/time -v "$python_bin" scripts/run_pof_pinn_baseline.py \
  --suite helical3d \
  --case interacting_helical_vortex_3d \
  --seed 999 \
  --profile production \
  --adam-steps 3 \
  --lbfgs-steps 0 \
  --output "$result_root/benchmark_only/pinn_cpu_helical3d_3steps"
