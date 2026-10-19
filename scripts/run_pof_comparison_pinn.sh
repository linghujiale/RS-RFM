#!/usr/bin/env bash
# 中文注释：运行PoF 论文 PINN 对比任务，用于论文实验的批处理或远程执行。
#SBATCH --job-name=pof-pinn
#SBATCH --partition=dzagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-29%6

set -euo pipefail

code_root=${RSRFM_CODE_ROOT:?RSRFM_CODE_ROOT is required}
result_root=${RSRFM_COMPARISON_ROOT:?RSRFM_COMPARISON_ROOT is required}
decay_reference=${RSRFM_DECAY_REFERENCE:?RSRFM_DECAY_REFERENCE is required}
array_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
python_bin=${RSRFM_PYTHON:-/work/home/zhaoteng_sciai/miniconda3/bin/python3}
export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd "$code_root"

nvidia-smi
"$python_bin" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

controlled_ids=(
  tgv_order1_decay
  two_rate_physics_range
  interacting_two_rate_vortex
  multiscale_shear_decay_k12
)

run_one() {
  local task_id=$1
if (( task_id < 20 )); then
  case_index=$((task_id / 5))
  seed=$((10 + task_id % 5))
  candidate=${controlled_ids[$case_index]}
  output="$result_root/pinn/controlled2d/$candidate/pinn_seed$seed"
  "$python_bin" scripts/run_pof_pinn_baseline.py \
    --suite controlled2d \
    --case "$candidate" \
    --seed "$seed" \
    --profile production \
    --output "$output"
elif (( task_id < 25 )); then
  seed=$((50 + task_id - 20))
  output="$result_root/pinn/decay2d/decaying_turbulence/pinn_seed$seed"
  "$python_bin" scripts/run_pof_pinn_baseline.py \
    --suite decay2d \
    --case decaying_turbulence \
    --seed "$seed" \
    --profile production \
    --reference "$decay_reference" \
    --output "$output"
else
  seed=$((100 + task_id - 25))
  output="$result_root/pinn/helical3d/interacting_helical_vortex_3d/pinn_seed$seed"
  "$python_bin" scripts/run_pof_pinn_baseline.py \
    --suite helical3d \
    --case interacting_helical_vortex_3d \
    --seed "$seed" \
    --profile production \
    --output "$output"
fi
}

if [[ ${RSRFM_GROUPED:-0} == 1 ]]; then
  run_one $((2 * array_id))
  run_one $((2 * array_id + 1))
else
  run_one "$array_id"
fi
