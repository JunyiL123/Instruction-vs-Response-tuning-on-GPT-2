# Instruction tuning vs. response-only tuning on GPT-2

Hewitt et al. define response ranking as “assigning a higher likelihood to the right response for an instruction than to a desirable response for a random other instruction” ([§4.2](https://arxiv.org/pdf/2409.14254)). This project tests that comparison with GPT-2 small before and after response-only or instruction tuning.

## Result

See **[RESULTS.md](RESULTS.md)** for the result table, interpretation, and chart.

Both tuned models are evaluated after **24 updates**. On hard negatives, both scored **16/24 (66.7%)** using total log probability; per-token scores were **15/24** for response-only tuning and **16/24** for instruction tuning.

## What I ran

- Model: GPT-2 small (124M)
- Data: Databricks Dolly 15K
- Conditions: pretrained, response-only fine tuning, instruction fine tuning
- Fine tuning: 24 training examples, 2 epochs, effective batch 2, same responses in both tuning conditions
- Evaluation: 24 held-out response-ranking items

The final-checkpoint rule was adopted after an earlier validation-selected analysis. The comparison here is exploratory.

An LLM checked the hard negatives for accidental correctness. Its judgments are labeled in the data.

## Reproduce the small run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m probe --config config_poc.json --root artifacts-poc setup-poc
.venv/bin/python -u -m probe --config config_poc.json --root artifacts-poc --device cpu run
```

The checked-in data and negative judgments fix the evaluation set. The setup command downloads the pinned GPT-2 checkpoint. This ranking run took about six minutes locally and writes the report to `artifacts-poc/report/`.

## Project layout

- `probe/` — preparation, fine tuning, ranking, and reporting code
- `config_poc.json` — settings for the small run
- `artifacts-poc/report/` — checked-in note, chart, and metric data
- `tests/` — data and evaluation checks

## Data and model

Dolly is distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). GPT-2 is used through the [OpenAI Community GPT-2 checkpoint](https://huggingface.co/openai-community/gpt2), released under MIT.
