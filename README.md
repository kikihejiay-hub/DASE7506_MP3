# DASE7506 MP3 - SQL Repair Agent

Public test: **195/240 (81.25%)**. Frozen validation: **117/120 (97.50%)**.
Each correct task passes all three official grading database instances. This is a self-reported result, subject to instructor reproduction and compliance review.

## Editable report

[report.md](report.md) is the editable English report, including the results, training configuration, two controlled ablations and reproduction commands.

The PDF inside the published source ZIP is the frozen submission snapshot. Editing report.md does not change that PDF or the Release. If you revise the final submission report, regenerate its PDF and publish a new source-package version; keep previously submitted Release files unchanged.

## Submission downloads

- [Source ZIP](https://github.com/kikihejiay-hub/DASE7506_MP3/releases/download/mp3-81_25-v1/final-source-81_25.zip)
- [LoRA adapter ZIP](https://github.com/kikihejiay-hub/DASE7506_MP3/releases/download/mp3-81_25-v1/final-adapter-81_25.zip)
- [Versioned Release](https://github.com/kikihejiay-hub/DASE7506_MP3/releases/tag/mp3-81_25-v1)
- [Complete official score JSON](test-score.json)

The source ZIP contains the frozen implementation, configuration, dependencies, PDF report, reproduction commands, training records, generation logs, complete predictions and score evidence. The adapter ZIP contains exactly adapter_config.json and adapter_model.safetensors. Base weights, course databases and environments are excluded. The final source and evidence are also expanded in this repository for direct inspection. Download the adapter and fixed submission ZIPs from the Release above.

## Model and reproduction

Base: Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca. Own train-only LoRA: rank 8, q_proj/v_proj A/B matrices. Total actual parameters: 597,196,800.

Extract the source ZIP into the unchanged course package root, provision the pinned base separately, and extract the adapter ZIP into adapters/qwen-sql/. Follow REPRODUCE.txt inside the source ZIP. Windows: `powershell -File submission/reproduce_sparse.ps1 -Split test`. Linux: `bash submission/reproduce_sparse.sh test`.

Use four CPU threads and workers=1, float32, no quantization. Independent Linux reproduction has not been completed. Data-use and external-assistance disclosures are retained in the report.

Uploading repository files does not create a course submission. Select test-score.json on the course project page, provide the two fixed ZIP links and your student ID, then actually create the prepared submission issue.

## Repository contents

- submission/: final agent modules and Windows/Linux reproduction scripts.
- xperiments/sparse_lora_v1_frozen/: frozen implementation and SHA-256 manifest used for the reported result.
- xperiments/: training code, provenance, training log, audit, ablation configurations and evaluation helpers.
- outputs/: recorded predictions, official scores, validation and ablation results, generation logs and train-only development trajectories.
- [REPRODUCE.md](REPRODUCE.md): setup and reproduction instructions; config.json and the two requirements files declare configuration and dependencies.
- [report.md](report.md): editable report. Base model weights and course databases must be provisioned separately; the adapter is in the Release.

The expanded source files match the published source ZIP. Later edits to this repository do not modify the existing versioned Release.
