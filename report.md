# SQL Repair Agent

DASE7506 Mini-project 3 | Final method report | 24 November 2026

## Result and evaluation protocol

The final agent achieves **195/240 (81.25%)** on the complete public test benchmark and **117/120 (97.50%)** on validation. Each correct task must pass all three official grading database instances. Scores are produced by the unchanged official scorer, which independently replays tool traces.

| Configuration | Validation | Test |
| --- | --- | --- |
| CPU controller without live-model rescue | 117/120 (97.50%) | 192/240 (80.00%) |
| Final sparse own-LoRA rescue | 117/120 (97.50%) | 195/240 (81.25%) |

The final method combines a CPU SQL repair controller with an optional, train-only Qwen3 LoRA completion. The model is invoked only when the controller is about to submit a query with a genuine failed check and no checked public match exists. The test run used 14 completions in total, at most one per task; most tasks used only the CPU controller.

## Perceive, plan, act and update

A fresh episode first checks the exact original SQL. It requests cached repair hints and schema through official tools, parses candidate queries, constructs repairs, and ranks them. Candidate generation covers column ownership, foreign-key paths, output shape, entity counting, grouping, set conditions and literals. Each check supplies real execution status and public-example agreement, which determine the next action.

The controller can spend one execute call to discover values of relevant fields. Values are used to propose a corrected condition and a subsequent check, with budget reserved for that check. Candidate queries are deduplicated by parsed form while retaining literal case. Public matches are selected using structural and question-based scores rather than treated as semantic proof.

When the CPU controller would finish without a match, a conservative text-anchor repair may be attempted. If its final proposed SQL has an actual failed check, the optional LoRA can generate one fresh SQL query. A duplicate or unparsable generation is rejected. If a paid call remains it is checked; otherwise it is submitted without fabricating feedback. A failed generated check does not establish correctness; the current policy still submits that generated query as its final hypothesis.

All schema, hint, execute and check actions share the eight-call budget. Submit is free. The official test traces satisfy the exact-original-first requirement, have no repeated identical charged actions and use at most eight charged calls.

## Architecture and model training

### Context and memory

Episode state consists of the actual history, a ranked candidate pool, query keys, discovered values and a one-completion flag. It resets for every task. The parser cache stores syntax trees, copied before editing; it is not an answer cache. Immutable model/tokenizer objects are reused across episodes. Retrieval is disabled and no cross-task solution memory is used.

### Repair operators

| Operator family | Purpose |
| --- | --- |
| Relations and scope | Qualify columns, repair FK paths and avoid alias-binding mistakes. |
| Entity and set semantics | Count complete entity keys; use set intersection for cross-record conditions; construct absence queries and handle nullable NOT IN inputs. |
| Output and grouping | Align projection shape, maintain referenced aggregate expressions and use entity identity when appropriate. |
| Public value evidence | Propose field/literal corrections and consistent categorical spellings from real tool output. |
| Failure-only rescue | Preserve checked matches; attempt one model completion only after real failure feedback. |

### Exact base and adapter

Base: **Qwen/Qwen3-0.6B**, immutable revision c1899de289a04d12100db370d81485cdf75e47ca. The unchanged base has 596,049,920 parameters. The unmerged adapter adds 1,146,880 parameters, giving **597,196,800 actual parameters** including embeddings. Only this one model and adapter are loaded.

| Training setting | Value |
| --- | --- |
| Data | 160 distinct tasks sampled from the official training split; 480 eligible examples |
| Optimization | 160 steps; AdamW; learning rate 0.0002; seed 7506 |
| LoRA | Rank 8, alpha 16, dropout 0.05; q_proj and v_proj A/B only |
| Sequence and precision | Maximum 640 training tokens; float32; gradient checkpointing |
| Frozen parameters | Base, embeddings, lm_head and biases; no modules_to_save |
| Training cost | 1671.2 wall seconds; 5005.3 process CPU seconds |

Targets are training-reference SQL. Prompts use the training question, draft, schema, hint candidate and public feedback. No validation/test examples are fitted. Training IDs, source-bank hash, logs and adapter hash are retained. The final adapter is the same train-only checkpoint; the sparse controller integration performs no further weight updates.

## Experiments, cost and reliability

### Controlled final-source ablations

| Single change from final method | Validation | Mean charged calls |
| --- | --- | --- |
| Full method | 117/120 (97.50%) | 4.258 |
| Disable entity_edits operator family | 112/120 (93.33%) | 4.275 |
| Disable public value probing/repair | 115/120 (95.83%) | 4.125 |

Both ablations use the same frozen eight source files, base/adapter, split and runtime settings. Each changes one boolean checkpoint flag. Disabling entity_edits leaves legacy semantic transformations and predicate operators active. Disabling values prevents the probing path; consequently no probe-derived literal update is available. The model rescue remains enabled in both experiments. These are controlled component removals, not comparisons of differently trained weights.

### Measured final test cost

| Measurement | Observed value |
| --- | --- |
| Prediction wall time | 179.24 seconds |
| Official scoring wall time | 34.20 seconds |
| Prediction process-tree CPU time | 352.23 sampled seconds |
| Prediction peak process-tree RSS | 4.11 GiB |
| Tool calls | Mean 4.533; maximum 8 |
| Model generation | 14 total; maximum one per task; actual maximum 46 new tokens |

Inference uses CPU float32, four PyTorch threads, workers=1, no quantization and local-only safetensors loading with trust_remote_code=False. The implemented generation ceiling is 384 tokens, below the course limit. Prediction took approximately three minutes, within the live-model two-hour allowance; scoring is below 180 seconds. RSS/CPU are sampled every 0.2 seconds for the prediction process tree, not total scoring CPU or Linux virtual address space. These are Windows measurements; independent Linux reproduction remains unverified.

### Failure modes and prompt injection

Remaining failures include wrong entities or relationships, duplicate-sensitive joins, incomplete questions, incorrect aggregate scope and misleading empty public matches. Failure-only model routing cannot repair wrong queries that already match the public example. A new generation may remain unchecked when the budget is exhausted. The final test has 45 incorrect tasks; no per-test answer repair is encoded.

A concrete validation failure is the truncated country/roller-coaster question that omits a speed threshold. Its hint assumes positive speed and at least two rides. The query matches the public example, so model rescue is not triggered, but it fails the requirement to pass every grading instance. The other two validation failures concern customer-reference links and exclusion scope.

Questions, schemas, hints and returned strings are untrusted data. Model prompts delimit evidence and instruct SQL-only output; generated text is never evaluated as Python or shell code. SQL is parsed before admission, and the unchanged official executor enforces read-only access, table/function restrictions and execution/output limits. Runtime database access occurs only through official tools. Prompt instructions and SQL validation do not prove semantic correctness.

## Reproduction and provenance

### Run from the unchanged course package root

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-model.txt
python -m mp3_sql.check_data
# Provision the pinned base before offline execution:
python experiments/download_model.py
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MP3_BASE_MODEL="$PWD/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$PWD/adapters/qwen-sql"
export MP3_QUANTIZE=0 MP3_RETRIEVAL=0
export MP3_MODEL_LOGDIR="$PWD/outputs/fresh-model-logs"
python -m mp3_sql.predict --split test --workers 1 \
  --method experiments/sparse_lora_v1_frozen/sparse_lora_agent.py \
  --output outputs/test.json
python -m mp3_sql.score --split test \
  --predictions outputs/test.json --output outputs/test-score.json
```

Use Python 3.11/3.12. On Windows, submission/reproduce_sparse.ps1 sets equivalent environment variables and runs the same frozen method. Install dependencies and provision base weights before going offline. Extract the separately supplied adapter into adapters/qwen-sql. Do not substitute a public third-party adapter or merge the base. Preserve all eight sibling source files listed by frozen.json. Reproducing the complete project requires the unchanged official package, which is not part of the source export.

### Data use and assistance

Weights were fitted only on training tasks; validation was reused for design/configuration selection. Frozen-method self-evaluation used the public test benchmark and aggregate scores. This supports a public-benchmark result, not an unbiased hidden-test generalization claim. Runtime code does not read grading references or extra database instances. Training-ID audits found no validation/test overlap; the deployed adapter hash matches that provenance.

OpenAI Codex assisted implementation, analysis, prompt.

### Submission artifacts and verification limits

Supply the source export, dependency lists, this report, reproduction commands, training logs/provenance, complete original test predictions and unchanged official score JSON. Base weights, data, environments and caches must be excluded. Supply a separate adapter ZIP containing exactly adapter_config.json and adapter_model.safetensors. The local official score is self-reported; instructor reproduction and eligibility review remain outstanding. The source and adapter archives are published in the versioned GitHub Release. Creating the course submission issue and obtaining instructor verification are separate steps.

### References

[Course specification](https://xudongwu-0.github.io/courses/dase7506/mp3/)

[Approved Qwen3 base](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)

[Public CPU-agent review](https://github.com/xudongwu-0/xudongwu-0.github.io/issues/397)

[Public sparse-rescue architecture review](https://github.com/JINWEI-1031/mp3-dase7506/tree/d19ef7f8e14db295e7ddf695353502619d79a51d)
