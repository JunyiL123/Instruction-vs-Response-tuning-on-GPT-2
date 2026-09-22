# GPT-2 instruction–response matching

LLM-adjudicated ranking result

| Checkpoint | Selected update | Easy: total log P | Hard: total log P | Easy: per token | Hard: per token |
|---|---:|---|---|---|---|
| base | — | 87.5% [70.8, 100.0] | 70.8% [50.0, 87.5] | 87.5% [70.8, 100.0] | 66.7% [45.8, 83.3] |
| response | 12 | 91.7% [79.2, 100.0] | 66.7% [45.8, 83.3] | 87.5% [75.0, 100.0] | 62.5% [41.7, 79.2] |
| instruction | 24 | 87.5% [75.0, 100.0] | 66.7% [45.8, 83.3] | 87.5% [75.0, 100.0] | 66.7% [45.8, 83.3] |

**Question.** Does pretrained GPT-2 124M associate instructions with appropriate responses, and does tuning on responses alone change that ranking compared with instruction tuning?

**Setup.** Dolly was cleaned and split by duplicate groups: 24 train, 8 validation, 1360 test. The ranking set contains 24 prompts across these categories: brainstorming 4, classification 4, closed_qa 4, creative_writing 4, general_qa 2, information_extraction 3, open_qa 3. Each prompt has one different-category negative and one related same-category negative. Both negatives have response-token length ratio at most 1.25. Dataset revision: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`; GPT-2 revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Both tuned models saw the same responses for 2 epochs, using AdamW at 0.0001 and effective batch 2. Each run completed 24 updates; the table reports the update count of the checkpoint selected by validation loss under each objective.

**Metrics.** Ranking means the correct response has strictly greater likelihood than the alternative. Main scores exclude EOS; EOS-inclusive totals, unconditional scores, conditioning gains, category results, tie rates, and paired differences are in results.json. Brackets are 95% percentile intervals from 1000 bootstrap resamples of prompt duplicate groups.

**Limits.** This is a small, one-seed ranking experiment with 2 tuning epochs. The response-ranking definition follows [Hewitt et al., §4.2](https://arxiv.org/pdf/2409.14254). Hard negatives were assessed by a recorded LLM adjudicator. Duplicate checks cover the documented lexical threshold and shared-context rules, not every semantic paraphrase or pretraining overlap. Intervals capture sampled-prompt uncertainty, not training-seed variation, adjudicator error, or all dependence from reused negative answers.
