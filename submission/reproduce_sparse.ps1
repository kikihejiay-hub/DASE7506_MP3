param([ValidateSet('validation','test')][string]$Split='test')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $python=Join-Path $projectRoot '.venv/Scripts/python.exe'
    $env:PYTHONUTF8='1'
    $env:HF_HUB_OFFLINE='1'
    $env:TRANSFORMERS_OFFLINE='1'
    $env:OMP_NUM_THREADS='4'
    $env:MKL_NUM_THREADS='4'
    $env:MP3_BASE_MODEL=Join-Path $projectRoot 'models/Qwen3-0.6B'
    $env:MP3_ADAPTER_DIR=Join-Path $projectRoot 'adapters/qwen-sql'
    $env:MP3_QUANTIZE='0'
    $env:MP3_RETRIEVAL='0'
    $env:MP3_MODEL_LOGDIR=Join-Path $projectRoot ('outputs/reproduce-sparse-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-logs')
    & $python -m mp3_sql.predict --split $Split --workers 1 --method experiments/sparse_lora_v1_frozen/sparse_lora_agent.py --output "outputs/$Split.json"
    if($LASTEXITCODE -ne 0){throw 'Prediction failed'}
    & $python -m mp3_sql.score --split $Split --predictions "outputs/$Split.json" --output "outputs/$Split-score.json"
    if($LASTEXITCODE -ne 0){throw 'Scoring failed'}
} finally {Pop-Location}
