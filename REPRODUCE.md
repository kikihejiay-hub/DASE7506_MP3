# Reproduction instructions

FINAL SUBMISSION: 195/240 (81.25%), validation 117/120 (97.50%).

1. Extract the source ZIP into the unchanged course starter package ROOT.
   Do not replace mp3_sql/ or data/. The source export contains neither.
2. Use Python 3.11 or 3.12 and install requirements.txt, the torch CPU wheel,
   and requirements-model.txt, as documented in report.md.
3. Before offline execution, run python experiments/download_model.py to
   provision the declared immutable Qwen3-0.6B revision. Base weights are
   deliberately excluded from both upload ZIPs.
4. Extract the separate adapter ZIP into adapters/qwen-sql/. The resulting
   directory must contain adapter_config.json and adapter_model.safetensors.
5. Run python -m mp3_sql.check_data, then:
   Windows: powershell -File submission/reproduce_sparse.ps1 -Split test
   Linux:   bash submission/reproduce_sparse.sh test
   Linux Python override: PYTHON_BIN=/path/to/python bash submission/reproduce_sparse.sh test
   These commands overwrite outputs/test.json and outputs/test-score.json;
   copy the recorded originals elsewhere first if comparing reproductions.
6. For validation, replace test with validation in the corresponding command.

Run with four CPU threads and workers=1. The live-model route permits 16 GiB
peak resident RAM, 7200 seconds prediction and 180 seconds scoring. Resource
limits should be enforced externally by the reproducer. Original measurements
are from Windows; Linux has not been independently reproduced by this work.

config.json describes model identity and settings; it is metadata, not an
extra checkpoint override. The entry is sparse_lora_agent.py in the frozen
directory. Do not run submission/submission.py alone to reproduce 81.25%:
that sibling file is a CPU dependency.

Official predictions and complete score JSON are in outputs/. Two final-source
controlled ablation configurations and results are included. Training records,
code, train-only tool trajectories, adapter provenance and generation logs are
included for inspection. outputs/train-candidates.json contains development
predictions/tool feedback on TRAIN only, not a benchmark database or gold bank.

To reproduce an ablation from the course root, set the same model environment
variables as the shell script/report, then run:
python experiments/evaluate_model.py --split validation --workers 1 --method experiments/sparse_lora_v1_frozen/sparse_lora_agent.py --checkpoint experiments/final-ablations/no-entities.json --tag reproduced-no-entities
python experiments/evaluate_model.py --split validation --workers 1 --method experiments/sparse_lora_v1_frozen/sparse_lora_agent.py --checkpoint experiments/final-ablations/no-values.json --tag reproduced-no-values

Additional profiling dependency for evaluate_model.py only: psutil==7.2.2.
Main official prediction/scoring entry does not require psutil.

The source and adapter packages have been uploaded to the versioned GitHub Release.
Creating the course submission issue is a separate step. This is a self-reported
public-benchmark score, subject to instructor reproduction and compliance review.
