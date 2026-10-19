#!/usr/bin/env bash
# 中文注释：运行PoF 论文 消融实验快速检查，用于论文实验的批处理或远程执行。
#SBATCH --job-name=rsrfm-ablate-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
python3 -m pytest tests/test_pof_suitability_scan.py tests/test_pof_ablation_studies.py -q
