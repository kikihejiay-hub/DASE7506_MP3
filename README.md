# DASE7506 MP3 - SQL Repair Agent

Final public test result: **195/240 (81.25%)**. Frozen validation: **117/120 (97.50%)**.
A task is correct only when it passes all three official grading database instances.
This is a self-reported result, subject to instructor reproduction and compliance review.

## Submission files

- `final-source-81_25.zip`: frozen implementation, configuration, dependencies, English report, reproduction commands, training records, generation logs, complete predictions and score evidence.
- `final-adapter-81_25.zip`: exactly `adapter_config.json` and `adapter_model.safetensors`.
- `test-score.json`: complete unchanged official score document, selected locally on the course submission page.
- `report.pdf`: English final report. `report-zh.pdf`: Chinese translation for reference.
- `package-audit.json`: local packaging checks, not instructor verification.

## Model and reproduction

Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
The own train-only LoRA uses rank 8 on q_proj/v_proj A/B matrices.
Actual base-plus-adapter parameters: 597,196,800. One model only; offline local safetensors loading.
Four CPU threads, workers=1; float32; no quantization. Independent Linux reproduction has not been completed.

Extract the source ZIP into the unchanged official course package root. Provision the pinned base separately, then extract the adapter ZIP into `adapters/qwen-sql/`.
Follow `REPRODUCE.txt` inside the source ZIP or page 4 of the report.
Windows: `powershell -File submission/reproduce_sparse.ps1 -Split test`
Linux: `bash submission/reproduce_sparse.sh test`

The entry point is `experiments/sparse_lora_v1_frozen/sparse_lora_agent.py`.
The original course framework/data, base weights, virtual environments and caches are excluded.
External assistance and data-use limitations are disclosed in the report.

Uploading this repository does not submit a course score. Use the course project page to select the score JSON, provide the two immutable ZIP download links and your student ID, and actually create the prepared GitHub issue.

## Fixed Release download links

- [Source ZIP](https://github.com/kikihejiay-hub/DASE7506_MP3/releases/download/mp3-81_25-v1/final-source-81_25.zip)
- [Adapter ZIP](https://github.com/kikihejiay-hub/DASE7506_MP3/releases/download/mp3-81_25-v1/final-adapter-81_25.zip)
- [Release page](https://github.com/kikihejiay-hub/DASE7506_MP3/releases/tag/mp3-81_25-v1)

The Release assets match the SHA-256 values in package-audit.json. This repository upload and Release publication do not create the course submission issue.
