# Results

## Main question

Given an instruction, does the model assign a higher conditional probability to the correct response than to a plausible wrong response?

For each test item, the main comparison is:

\[
P(\text{correct response}\mid\text{instruction}) > P(\text{hard negative}\mid\text{instruction})
\]

## Result table

| Model / checkpoint | Update | Hard, total log P | Hard, per token | Easy, total log P | Easy, per token |
|---|---:|---:|---:|---:|---:|
| Pretrained GPT-2 | — | 17/24 (70.8%) | 16/24 (66.7%) | 21/24 (87.5%) | 21/24 (87.5%) |
| Response-only, validation choice | 12 | 16/24 (66.7%) | 15/24 (62.5%) | 22/24 (91.7%) | 21/24 (87.5%) |
| Response-only, fixed-update check | 24 | 16/24 (66.7%) | 15/24 (62.5%) | 21/24 (87.5%) | 21/24 (87.5%) |
| Instruction-tuned, validation choice | 24 | 16/24 (66.7%) | 16/24 (66.7%) | 21/24 (87.5%) | 21/24 (87.5%) |

![Response ranking using total and per-token log probability](artifacts-poc/report/ranking.png)

The chart shows the validation-selected checkpoints. The fixed-update row is a separate check using the response-only checkpoint after update 24. The total log probability comparison uses the response-ranking definition in [Hewitt et al., §4.2](https://arxiv.org/pdf/2409.14254). The per-token version checks whether the result depends on response length. A hard negative is a related answer from the same Dolly category; an easy negative comes from a different category.

## Interpretation

GPT-2 small ranked the supplied correct answer above the hard negative on most of this tiny test set before tuning.

After tuning, both models got 16 of 24 hard negatives using total log probability. Instruction tuning got one more item than response-only tuning under per-token scoring (16 versus 15). The paired bootstrap interval for response-only minus instruction tuning on that measure was −12.5 to 0 percentage points. These 24 items do not establish a reliable advantage for either method.

The response-only model also has a saved checkpoint after update 24. Its validation loss was slightly higher than at update 12 (3.1842 versus 3.1776), so the original selection rule chose update 12. As a [fixed-update sensitivity check](artifacts-poc/report/response_fixed_24.json), I scored update 24 against the same test pairs. Hard-negative ranking remained **16/24 by total log probability and 15/24 per token**. This check matches the instruction model's selected update count without changing the primary checkpoint choice after seeing the test results.

## Scope

This run used one seed, GPT-2 small, 24 ranking items, and two epochs of tuning. Both tuned models completed 24 optimizer updates on the same responses. Validation selected the response-only checkpoint after 12 updates and the instruction-tuned checkpoint after 24. Training optimizes response likelihood, which need not improve pairwise accuracy. The LLM judgments for hard negatives are recorded in the repository.

For the full metric table and bootstrap intervals, see [the detailed note](artifacts-poc/report/note.md) and [results JSON](artifacts-poc/report/results.json).
