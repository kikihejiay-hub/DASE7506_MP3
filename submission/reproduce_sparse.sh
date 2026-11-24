#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
split="${1:-test}"
case "$split" in test|validation) ;; *) echo 'Use test or validation' >&2; exit 2;; esac
python_bin="${PYTHON_BIN:-python3}"
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MP3_BASE_MODEL="${MP3_BASE_MODEL:-$PWD/models/Qwen3-0.6B}"
export MP3_ADAPTER_DIR="${MP3_ADAPTER_DIR:-$PWD/adapters/qwen-sql}"
export MP3_QUANTIZE=0 MP3_RETRIEVAL=0
export MP3_MODEL_LOGDIR="${MP3_MODEL_LOGDIR:-$PWD/outputs/reproduce-sparse-$(date +%Y%m%d-%H%M%S)-logs}"
"$python_bin" -m mp3_sql.predict --split "$split" --workers 1 --method experiments/sparse_lora_v1_frozen/sparse_lora_agent.py --output "outputs/$split.json"
"$python_bin" -m mp3_sql.score --split "$split" --predictions "outputs/$split.json" --output "outputs/$split-score.json"
