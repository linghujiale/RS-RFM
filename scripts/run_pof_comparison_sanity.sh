#!/usr/bin/env bash
# 中文注释：运行PoF 论文 对比实验快速检查，用于论文实验的批处理或远程执行。
#SBATCH --job-name=pof-sanity
#SBATCH --partition=dzagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
code_root=${RSRFM_CODE_ROOT:?RSRFM_CODE_ROOT is required}
result_root=${RSRFM_COMPARISON_ROOT:?RSRFM_COMPARISON_ROOT is required}
python_bin=${RSRFM_PYTHON:-/work/home/zhaoteng_sciai/miniconda3/bin/python3}
export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$code_root"

"$python_bin" -m pytest \
  tests/test_baselines.py \
  tests/test_shrinking_circle.py::test_2d_baseline_derivatives_match_finite_differences \
  tests/test_grayscott_3d.py::test_3d_feature_derivatives_match_finite_differences \
  tests/test_pof_suitability_scan.py -q
"$python_bin" scripts/run_pof_pinn_baseline.py \
  --suite sanity \
  --profile smoke \
  --output "$result_root/sanity"
"$python_bin" scripts/run_pof_pinn_baseline.py \
  --suite controlled2d \
  --case tgv_order1_decay \
  --seed 0 \
  --profile smoke \
  --output "$result_root/smoke/pinn_tgv_seed0"

