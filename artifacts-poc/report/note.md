# GPT-2 instruction–response matching

LLM-reviewed exploratory result

| Checkpoint | Easy: total log P | Hard: total log P | Easy: per token | Hard: per token | Generated answer success |
|---|---|---|---|---|---|
| base | 87.5% [70.8, 100.0] | 70.8% [50.0, 87.5] | 87.5% [70.8, 100.0] | 66.7% [45.8, 83.3] | 0.0% [0.0, 0.0] |
| response | 87.5% [70.8, 100.0] | 70.8% [50.0, 87.5] | 83.3% [66.7, 95.8] | 66.7% [45.8, 83.3] | 0.0% [0.0, 0.0] |
| instruction | 91.7% [79.2, 100.0] | 70.8% [50.0, 87.5] | 83.3% [66.7, 95.8] | 66.7% [45.8, 83.3] | 0.0% [0.0, 0.0] |

**Question.** Does pretrained GPT-2 124M associate instructions with appropriate responses, and does tuning on responses alone change ranking and generation compared with instruction tuning?

**Setup.** Dolly, cleaned and split by duplicate groups: {'train': 24, 'validation': 8, 'test': 1360}. The frozen ranking set contains 24 prompts; generation uses 8 of those prompts. Sampling cycles through categories, but eligible negative availability limits their representation: {'brainstorming': 4, 'classification': 4, 'closed_qa': 4, 'creative_writing': 4, 'general_qa': 2, 'information_extraction': 3, 'open_qa': 3}. Each prompt has one different-category negative and one same-category, lexically related negative. Both negatives have response-token length ratio at most 1.25. The dataset revision is `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`; GPT-2 revision is `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Full-parameter tuning uses 1 epoch(s), AdamW at 0.0001, effective batch 4, and identical response exposure. Each condition selects its epoch by validation loss under its own objective.

**Metrics.** Ranking means the correct response has strictly greater likelihood than the alternative. Main scores exclude EOS; EOS-inclusive totals, unconditional scores, conditioning gains, category results, tie rates, and paired differences are in results.json. Brackets are 95% percentile intervals from 1000 bootstrap resamples of prompt duplicate groups. Generation succeeds only when the blinded LLM grader marks both task adherence and substantial correctness true. Outputs that hit the 48-token generation cap: {'base': 8, 'response': 8, 'instruction': 3}.

**Limits.** This is deliberately a five-to-ten-minute proof of concept: 24 ranking prompts, 8 generated prompts, one seed, and one short tuning epoch. It is inspired by [Hewitt et al., §4.2](https://arxiv.org/pdf/2409.14254), with a different model scale, dataset, and prompt format. Ranking supplied responses does not establish an ability to generate them. The 48-token cap and greedy decoding limit what the zero generation score means; a bootstrap interval of [0, 0] here is not evidence of certain failure on new prompts. Hard negatives and generated answers were assessed by a recorded LLM adjudicator, so a human check would be required for a research claim. Duplicate checks cover the documented lexical threshold and shared-context rules, not every semantic paraphrase or pretraining overlap. Intervals capture sampled-prompt uncertainty, not training-seed variation, adjudicator error, or all dependence from reused negative answers.
