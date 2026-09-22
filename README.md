# Does small GPT-2 already match instructions to responses?

A reproducible, one-seed experiment comparing pretrained GPT-2 124M, response-only tuning, and instruction tuning. It measures response ranking with easy and hard negatives and separately grades generated answers. Inspired by [Hewitt et al., *Instruction Following without Instruction Tuning*, §4.2](https://arxiv.org/pdf/2409.14254); this is an exploratory adaptation with different data and formatting.

## Five-to-ten-minute proof of concept

Use the checked-in lightweight configuration for an email-sized demonstration. It trains on 24 examples for one epoch, ranks 24 held-out instruction–response pairs, and generates answers for 8 prompts. It uses CPU so it avoids the memory pressure of laptop GPU training.

```sh
.venv/bin/python -u -m probe --config config_poc.json --root artifacts-poc --device cpu run
```

The completed demonstration is already in `artifacts-poc/`. Its note, table, chart, examples, and unsent email draft are written to `artifacts-poc/report/`. The generation grades in this proof of concept are explicitly recorded as LLM judgments. Do not present them as human review.

## Full exploratory study

Python 3.11 is recommended. All commands run from this directory. Downloads, models, and output stay in the workspace. Around 8 GB of free disk space is advisable for the environment, download cache, and resumable training checkpoints.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m probe prepare
.venv/bin/python -m probe audit
```

`prepare` resolves and locks exact Hugging Face revisions, downloads GPT-2 and Dolly, cleans the dataset, builds disjoint duplicate groups, makes approximately 80/10/10 splits, and freezes 200 ranking prompts plus a subset of 100 generation prompts. Sampling cycles through categories until their eligible pools are exhausted; availability of valid-length negatives means category counts can be uneven. The report displays the actual counts. Preparation refuses to overwrite an existing experiment. Use `--root artifacts-new` to create another one. The manifest contains filtering counts, category counts, hashes, source revisions, and all configuration.

Run one short check per objective before the complete study:

```sh
.venv/bin/python -m probe train --mode response --smoke-steps 2
.venv/bin/python -m probe train --mode instruction --smoke-steps 2
.venv/bin/python -m probe train --mode response
.venv/bin/python -m probe train --mode instruction
```

The automatic device is CUDA, then Apple MPS, then CPU. Specify it before the subcommand, e.g. `--device mps train --mode response`. Full tuning on a laptop may take hours; use the smoke timings to estimate the budget. Smoke checkpoints are isolated and cannot be used as scientific results. Completed training commands are idempotent; interrupted full training resumes from the latest completed epoch. Both modes begin with the same pretrained weights, use the same shuffled example order per epoch, and receive exactly the same response tokens and number of updates. Their total input tokens and compute time differ because instruction tuning includes prompts.

To run all compute stages sequentially, use `.venv/bin/python -u -m probe run --allow-unreviewed > artifacts/study.log 2>&1`. This trains both models, evaluates all three conditions, exports blinded answers, and writes a preliminary report. Training comes first so negative review can happen while it runs. `artifacts/study_status.json` records the process and current stage; `study.log` records progress. It never invents human decisions. Rerun the same command to resume interrupted compute. Avoid running another GPU command simultaneously. For an already manually reviewed benchmark, omit `--allow-unreviewed`. To stop a run, send SIGTERM to the `pid` in the status file; the runner also stops its active child and records interruption. The Mac must remain awake for local compute to advance.

## Human review and evaluation

Read `artifacts/review/negative_review.md` and fill in `artifacts/review/negatives.jsonl`. For every example, verify the reference is acceptable, the easy answer is incorrect, and the chosen hard answer is related but incorrect. Enter your name and true/false decisions. Five hard candidates are available where possible; `hard_id` must be one of the frozen options. A same-category answer is not automatically a good hard negative. If none qualifies, leave it rejected and improve the benchmark in a new experiment before publishing; do not approve it just to fill the quota.

For a browser form, run `.venv/bin/python -m probe review --kind negatives` and open `http://127.0.0.1:8765`. It shows one example at a time and saves decisions directly. After generating the blinded sheet, use `review --kind generation` for answer grading. The server binds only to localhost and never serves the private answer key.

```sh
.venv/bin/python -m probe evaluate --condition base
.venv/bin/python -m probe evaluate --condition response
.venv/bin/python -m probe evaluate --condition instruction
.venv/bin/python -m probe blind
```

Exploratory scoring before manual review is possible with `evaluate --condition base --allow-unreviewed`. It remains preliminary. Scoring saves each completed example and resumes after interruption. Changes to chosen negatives invalidate old scoring; move `artifacts/evaluation` aside and rerun all conditions if a review changes `hard_id`. Do not use model scores to choose negative answers.

`blind` produces 300 randomly ordered answers in `artifacts/review/generation.jsonl`, with the rubric in `generation_rubric.md`. The scorer reads prompts, context, references, and anonymous outputs, entering boolean task-adherence and correctness judgments. Keep `artifacts/private/blind_key.jsonl` closed until grading is complete. This is procedural blinding, not access control. The experiment owner may grade the answers, but that should be disclosed; a second human reviewer is preferable when available. AI-generated assessments must be recorded as LLM adjudication and must not be presented as human grading.

```sh
.venv/bin/python -m probe report
.venv/bin/python -m probe report --final
```

A draft report is available at any stage after preparation and shows missing values as pending. Final reporting requires completed evaluation of all three checkpoints, all negative approvals, and all 300 blinded grades. It creates `artifacts/report/note.md`, `results.json`, `ranking.png`, `examples.md`, and an unsent five-sentence email draft. Supply your own name and share links before sending. There is no email-sending integration.

## Fixed method

- **Data:** [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k), human-written English instructions. Context is preserved. Empty required fields, exact duplicate normalized prompts, prompts over 512 tokens, responses over 255 tokens, or complete sequences over 768 tokens are removed rather than truncated.
- **Leakage checks:** connected groups join full prompts at character 3–5-gram TF-IDF cosine similarity ≥0.90, identical contexts of ≥40 normalized characters, and identical responses of ≥80 characters. Entire groups stay within one split. Generic short repeated responses can occur across splits; grouping these would collapse unrelated tasks. These are operational lexical checks, not a guarantee against all semantic overlap or pretraining contamination.
- **Format:** GPT-2 EOS used as BOS, then `Instruction:\n{instruction}\n\nContext:\n{context}\n\nResponse:\n{response}` and EOS; omit the context section when empty. Response tuning uses BOS + `Response:\n{response}` + EOS. Prompt and response are encoded as explicit segments; response tokenization is identical between conditions. No new tokens, demonstrations, or system messages are added.
- **Loss:** prompt and padding labels are masked. Both methods learn response tokens plus EOS. Gradient accumulation is weighted by the number of response tokens across the entire effective batch, including a short final batch.
- **Training:** all parameters, three epochs, AdamW, learning rate 5e-5, weight decay 0.01, 5% warmup and linear decay, gradient norm 1.0, microbatch 2, effective batch 16, seed 42, FP32. Select the lowest validation NLL checkpoint within each condition's own training objective. These loss values are not comparable across objectives. Fixed hyperparameters are an initial small-study choice, not a claim of optimal tuning.
- **Negatives:** different-category random answer versus same-category answer retrieved by instruction/context word TF-IDF similarity ≥0.05. Both have max/min response-token length ratio ≤1.25. All answers are drawn from the held-out test pool, and same-group or identical answers are excluded. Category coverage is limited by eligible negatives; the actual counts define the evaluation population. Human review checks the actual semantic difficulty.
- **Ranking:** strict `log P(gold | prompt) > log P(negative | prompt)`; ties are counted as non-wins and reported separately. Report total response log-likelihood and mean per response token, excluding EOS. EOS-inclusive totals are a sensitivity check. A second control subtracts the same answer's score with only the response prefix, measuring conditioning gain. Also report unconditional ranking, per-category scores, and paired differences between checkpoints.
- **Generation:** the same 100 prompts for all checkpoints, greedy decoding, up to 256 new tokens, no repetition penalty, stop on EOS. Success requires both blinded task-adherence and substantial-correctness flags. This is not AlpacaEval and its rates should not be equated with the paper's win rates.
- **Uncertainty:** 5,000 paired bootstrap resamples of prompt duplicate groups, 95% percentile intervals. Reused negatives and a single training seed limit the interpretation; these are exploratory prompt-sampling intervals, not complete experimental uncertainty.

## Files and interfaces

`python -m probe` provides `prepare`, `audit`, `train`, `evaluate`, `blind`, and `report`. `config.json` holds the fixed experimental choices. Data and reviews use JSONL; manifests and metric summaries use JSON. Scientific results are never filled with smoke-test values or invented scores. A valid null result is a deliverable.

## Attribution

Dolly: Copyright (2023) Databricks, Inc., licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); context may include Wikipedia content under the same license. Preserve attribution and license when sharing derived examples or review sheets. GPT-2: [OpenAI model card](https://huggingface.co/openai-community/gpt2), MIT. Cite the Hewitt et al. paper when describing the motivation and acknowledge the changes in model, dataset, and evaluation.
