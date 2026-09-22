# Results

## Main question

Given an instruction, does the model assign a higher conditional probability to the correct response than to a plausible wrong response?

For each test item, the main comparison is:

\[
P(\text{correct response}\mid\text{instruction}) > P(\text{hard negative}\mid\text{instruction})
\]

## Result table

| Model | Hard, total log P | Hard, per token | Easy, total log P | Easy, per token | Generated success |
|---|---:|---:|---:|---:|---:|
| Pretrained GPT-2 | 17/24 (70.8%) | 16/24 (66.7%) | 21/24 (87.5%) | 21/24 (87.5%) | 0/8 |
| Response-only tuned | 17/24 (70.8%) | 16/24 (66.7%) | 21/24 (87.5%) | 20/24 (83.3%) | 0/8 |
| Instruction-tuned | 17/24 (70.8%) | 16/24 (66.7%) | 22/24 (91.7%) | 20/24 (83.3%) | 0/8 |

![Response ranking using total and per-token log probability](artifacts-poc/report/ranking.png)

The total log probability comparison follows the paper's response-ranking definition. The per-token version checks whether the result depends on response length. A hard negative is a related answer from the same Dolly category; an easy negative comes from a different category.

## Interpretation

GPT-2 small ranked the supplied correct answer above the hard negative on two thirds of this tiny test set. That behavior was already present before either tuning run.

There was no aggregate difference between response-only and instruction tuning on hard negatives: both got 17 of 24 using total log probability and 16 of 24 using per-token log probability. On the per-token measure, each model won one item the other lost. The paired bootstrap interval for their difference was −12.5 to +12.5 percentage points. The sample gives no evidence that one tuning method improved ranking more than the other.

The generation check was stricter. An answer counted as a success only if it addressed the request and was substantially correct. No output passed both checks. **The 0/8 score is sensitive to the generation setup:** all 8 outputs from the base model and all 8 from response-only tuning hit the 48-token cap; 3 of 8 instruction-tuned outputs hit it. Several outputs were repetitive or off task before the cap, while others were incomplete. I reran generation on the same prompts with a 128-token cap and no retraining. All base and response-only outputs still ran to the cap, and none of the [longer texts](artifacts-poc/report/generation_128.jsonl) clearly completed its task. This diagnostic strengthens the observation that greedy output was poor, but it does not test other decoding choices or more training. The [original blinded answers and LLM grades](artifacts-poc/report/graded_generations.jsonl) are available for inspection.

## Scope

This is a 5–10 minute proof of concept: one seed, GPT-2 small, 24 ranking items, 8 generations, and one epoch of tuning. Each tuned model received only six optimizer updates. Training does not guarantee a gain on held-out pairwise accuracy, especially when the pretrained model already prefers many correct answers. Hewitt et al. also report that base models can match or exceed instruction-tuned models on this ranking test, even though their 7B tuning setup and generation results differ substantially from ours. The LLM graded the hard negatives and generated answers; these judgments are recorded in the repository. This is a small demonstration, not a reproduction of their 7B LIMA experiment.

For the full metric table and bootstrap intervals, see [the generated note](artifacts-poc/report/note.md) and [results JSON](artifacts-poc/report/results.json).
