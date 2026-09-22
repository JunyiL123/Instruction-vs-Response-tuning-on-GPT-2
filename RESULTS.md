# Results

## Main question

Given an instruction, does the model assign a higher conditional probability to the correct response than to a plausible wrong response?

For each test item, the main comparison is:

\[
P(\text{correct response}\mid\text{instruction}) > P(\text{hard negative}\mid\text{instruction})
\]

## Result table

| Model | Hard, total log P | Hard, per token | Easy, total log P | Easy, per token |
|---|---:|---:|---:|---:|
| Pretrained GPT-2 | 17/24 (70.8%) | 16/24 (66.7%) | 21/24 (87.5%) | 21/24 (87.5%) |
| Response-only tuned | 17/24 (70.8%) | 16/24 (66.7%) | 21/24 (87.5%) | 20/24 (83.3%) |
| Instruction-tuned | 17/24 (70.8%) | 16/24 (66.7%) | 22/24 (91.7%) | 20/24 (83.3%) |

![Response ranking using total and per-token log probability](artifacts-poc/report/ranking.png)

The total log probability comparison uses the response-ranking definition in [Hewitt et al., §4.2](https://arxiv.org/pdf/2409.14254). The per-token version checks whether the result depends on response length. A hard negative is a related answer from the same Dolly category; an easy negative comes from a different category.

## Interpretation

GPT-2 small ranked the supplied correct answer above the hard negative on two thirds of this tiny test set. That behavior was already present before either tuning run.

There was no aggregate difference between response-only and instruction tuning on hard negatives: both got 17 of 24 using total log probability and 16 of 24 using per-token log probability. On the per-token measure, each model won one item the other lost. The paired bootstrap interval for their difference was −12.5 to +12.5 percentage points. The sample gives no evidence that one tuning method improved ranking more than the other.

## Scope

This run used one seed, GPT-2 small, 24 ranking items, and one epoch of tuning. Each tuned model received six optimizer updates. Training optimizes response likelihood, which need not change the win/loss decision on a particular pair. The sample is too small to establish that the methods have equal ranking performance beyond these items. The LLM judgments for hard negatives are recorded in the repository.

For the full metric table and bootstrap intervals, see [the detailed note](artifacts-poc/report/note.md) and [results JSON](artifacts-poc/report/results.json).
