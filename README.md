# Instruction tuning vs. response-only tuning on GPT-2

A small replication-style exercise motivated by Hewitt et al., [*Instruction Following without Instruction Tuning*](https://arxiv.org/pdf/2409.14254). I asked whether pretrained GPT-2 small already prefers the right answer for an instruction, and whether a tiny response-only fine-tuning run changes that preference differently from instruction tuning.

## Result

See **[RESULTS.md](RESULTS.md)** for the result table, interpretation, and chart.

In this 24-item proof of concept, response-only and instruction tuning tied on the main hard-negative ranking measure: **16/24 (66.7%)**. Neither condition produced a successful answer on the 8-prompt generation check. This is a small null result, not a replication claim.

## What I ran

- Model: GPT-2 small (124M)
- Data: Databricks Dolly 15K
- Conditions: pretrained, response-only fine tuning, instruction fine tuning
- Fine tuning: 24 training examples, 1 epoch, same responses in both tuning conditions
- Evaluation: 24 held-out response-ranking items and 8 held-out generations

Hard negatives and generations were graded by an LLM and labeled as such in the included files. A larger study should use human review and more seeds.

## Reproduce the small run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -u -m probe --config config_poc.json --root artifacts-poc --device cpu run
```

The command was designed to finish in roughly 5–10 minutes on the machine used for this project. It saves the report in `artifacts-poc/report/`.

## Project layout

- `probe/` — preparation, fine tuning, ranking, generation, and reporting code
- `config_poc.json` — settings for the small run
- `artifacts-poc/report/` — checked-in note, chart, examples, and email draft
- `tests/` — data and evaluation checks

## Data and model

Dolly is distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). GPT-2 is used through the [OpenAI Community GPT-2 checkpoint](https://huggingface.co/openai-community/gpt2), released under MIT.
