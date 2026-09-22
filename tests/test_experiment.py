import copy

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from probe.common import digest, read_json, read_jsonl, write_json, write_jsonl
from probe.data import balanced_sample, clean_rows, group_duplicates, split_groups, validated_pairs, verify
from probe.modeling import collate, encode_example, prompt_ids, response_loss_sum, score_response
from probe.report import interval, wins


class TinyTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(char) % 29 for char in text]


@pytest.fixture
def row():
    return {"id": "a", "instruction": "Say hello", "context": "Secret context", "response": "Hello.", "category": "qa"}


@pytest.fixture
def model():
    torch.manual_seed(3)
    return GPT2LMHeadModel(GPT2Config(vocab_size=32, n_layer=1, n_head=2, n_embd=16,
                                     n_positions=256, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()


def test_response_training_never_consumes_prompt(row):
    tokenizer = TinyTokenizer()
    changed = {**row, "instruction": "An entirely different request", "context": "DIFFERENT"}
    response = encode_example(tokenizer, row, "response")
    assert response == encode_example(tokenizer, changed, "response")
    instruction = encode_example(tokenizer, row, "instruction")
    assert instruction != encode_example(tokenizer, changed, "instruction")
    assert [x for x in response["labels"] if x != -100] == [x for x in instruction["labels"] if x != -100]
    assert instruction["labels"][-1] == tokenizer.eos_token_id
    assert all(x == -100 for x in instruction["labels"][:len(prompt_ids(tokenizer, row))])


def test_score_matches_transformers_loss_and_ignores_padding(model, row):
    tokenizer = TinyTokenizer()
    encoded = encode_example(tokenizer, row, "instruction")
    batch = collate([encoded], tokenizer.eos_token_id)
    with torch.no_grad():
        manual, count = response_loss_sum(model, batch)
        native = model(**batch).loss
    assert torch.allclose(manual / count, native, atol=1e-6)
    score = score_response(model, tokenizer, row["response"], torch.device("cpu"), row)
    assert score["sum_with_eos"] == pytest.approx(-manual.item(), abs=1e-5)
    longer = encode_example(tokenizer, {**row, "response": row["response"] * 3}, "instruction")
    padded = collate([encoded, longer], tokenizer.eos_token_id)
    with torch.no_grad():
        total, total_count = response_loss_sum(model, padded)
        second, second_count = response_loss_sum(model, collate([longer], tokenizer.eos_token_id))
    assert total.item() == pytest.approx(manual.item()+second.item(), abs=1e-4)
    assert total_count == count + second_count
    fixed = collate([encoded], tokenizer.eos_token_id, len(encoded['input_ids']) + 20)
    with torch.no_grad():
        fixed_loss, fixed_count = response_loss_sum(model, fixed)
    assert fixed_loss.item() == pytest.approx(manual.item(), abs=1e-4)
    assert fixed_count == count
    assert all(x == -100 for x in padded["labels"][0, len(encoded["input_ids"]):])


def test_gradient_accumulation_matches_full_batch(model, row):
    tokenizer = TinyTokenizer()
    rows = [encode_example(tokenizer, row, "instruction"),
            encode_example(tokenizer, {**row, "response": "Longer answer."}, "instruction")]
    accumulated = copy.deepcopy(model)
    batch = collate(rows, tokenizer.eos_token_id)
    loss, count = response_loss_sum(model, batch)
    (loss/count).backward()
    for item in rows:
        partial, _ = response_loss_sum(accumulated, collate([item], tokenizer.eos_token_id))
        (partial/count).backward()
    for left, right in zip(model.parameters(), accumulated.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=2e-6)


def test_cleaning_no_truncation_and_duplicate_removal(row):
    cfg = {"max_prompt_tokens": 120, "max_response_tokens": 20, "max_sequence_tokens": 150}
    rows = [row, row, {**row, "instruction": "Empty", "response": ""},
            {**row, "instruction": "Long", "response": "x"*100}]
    cleaned, removed = clean_rows(rows, TinyTokenizer(), cfg)
    assert len(cleaned) == 1
    assert removed == {"duplicate_prompt": 1, "empty": 1, "overlength": 1}


def test_duplicate_groups_cannot_cross_splits():
    rows = [{"id": str(i), "instruction": f"A unique topic {i*i} elephant {i}", "context": "",
             "response": f"answer {i}", "category": "qa"} for i in range(40)]
    rows[1]["instruction"] = rows[0]["instruction"] + "!"
    rows[2]["context"] = rows[3]["context"] = "Shared document context long enough to enforce source grouping."
    stats = group_duplicates(rows, .99)
    assert stats["groups"] < len(rows)
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert rows[2]["group_id"] == rows[3]["group_id"]
    first = split_groups(rows, 42)
    second = split_groups(rows, 42)
    assert first == second
    sets = [{r["group_id"] for r in v} for v in first.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_balanced_sampling():
    rows = [{"id": f"{c}-{i}", "category": c} for c in ["a", "b", "c"] for i in range(20)]
    chosen = balanced_sample(rows, 12, 42)
    assert chosen == balanced_sample(rows, 12, 42)
    assert [sum(r["category"] == c for r in chosen) for c in ["a", "b", "c"]] == [4, 4, 4]


def test_review_gate_and_selected_candidate_integrity(tmp_path):
    candidates = [{"id": "a", "easy_id": "b", "hard_id": "c", "hard_options": [{"id": "c"}]}]
    reviews = [{"id": "a", "easy_id": "b", "hard_id": "c", "gold_valid": None, "easy_valid": None,
                "hard_valid": None, "reviewer": ""}]
    write_jsonl(tmp_path / "data/candidates.jsonl", candidates)
    write_jsonl(tmp_path / "review/negatives.jsonl", reviews)
    with pytest.raises(ValueError, match="incomplete"):
        validated_pairs(tmp_path)
    assert validated_pairs(tmp_path, True)[1] is False
    reviews[0].update(gold_valid=True, easy_valid=True, hard_valid=True, reviewer="Test human")
    write_jsonl(tmp_path / "review/negatives.jsonl", reviews)
    assert validated_pairs(tmp_path)[1] is True
    reviews[0]["hard_id"] = "injected"
    write_jsonl(tmp_path / "review/negatives.jsonl", reviews)
    with pytest.raises(ValueError, match="hard_id"):
        validated_pairs(tmp_path)


def test_hash_audit_rejects_changed_split(tmp_path):
    row = {"id": "a", "group_id": "a"}
    write_jsonl(tmp_path / "data/train.jsonl", [row])
    write_jsonl(tmp_path / "data/candidates.jsonl", [])
    write_json(tmp_path / "data/generation_ids.json", [])
    write_json(tmp_path / "data/manifest.json", {"split_hashes": {"train": digest([row])},
                                                 "candidates_hash": digest([]), "generation_ids_hash": digest([])})
    verify(tmp_path)
    write_jsonl(tmp_path / "data/train.jsonl", [{**row, "id": "changed"}])
    with pytest.raises(ValueError, match="changed"):
        verify(tmp_path)


def test_paired_bootstrap_and_ties():
    zeros = interval([0]*10, list(range(10)), 100, 42)
    assert zeros["estimate"] == zeros["low"] == zeros["high"] == 0
    paired = interval(np.array([1, 0, 1]) - np.array([1, 0, 1]), ["a", "b", "c"], 100, 42)
    assert paired["low"] == paired["high"] == 0
    row = {"scores": {"gold": {"conditional": {"sum": -2}}, "hard": {"conditional": {"sum": -2}}}}
    assert wins(row, "hard", "conditional.sum") == (0, 1)


def test_reporting_and_blinding_end_to_end_with_synthetic_results(tmp_path):
    """Exercise reporting only; fabricated fixtures never enter real artifacts."""
    from probe.evaluate import blind
    from probe.report import report
    cfg = {"seed": 42, "bootstrap_samples": 50, "generation_examples": 2, "ranking_examples": 2,
           "epochs": 3, "learning_rate": 5e-5, "effective_batch_size": 16, "negative_length_ratio": 1.25,
           "model_id": "test-model", "dataset_id": "test-data", "dataset_file": "test.jsonl",
           "max_sequence_tokens": 768, "max_prompt_tokens": 512, "max_response_tokens": 255,
           "near_duplicate_similarity": .9, "hard_min_similarity": .05}
    rows = [{"id": x, "group_id": x, "category": "qa", "instruction": "Question " + x,
             "context": "", "response": "Answer " + x} for x in ["a", "b", "c", "d"]]
    candidates = [{"id": x, "category": "qa", "easy_id": "c", "hard_id": "d", "hard_options": [{"id": "d"}]} for x in ["a", "b"]]
    pairs = [{k: c[k] for k in ["id", "easy_id", "hard_id"]} for c in candidates]
    write_jsonl(tmp_path / "data/test.jsonl", rows)
    write_jsonl(tmp_path / "data/candidates.jsonl", candidates)
    write_json(tmp_path / "data/generation_ids.json", ["a", "b"])
    sources = {"model_revision": "test-only-model", "dataset_revision": "test-only-dataset"}
    write_json(tmp_path / "raw/sources.json", sources)
    manifest = {"config": cfg, "experiment_id": "fixture", "sources": sources, "split_hashes": {"test": digest(rows)},
                "split_counts": {"train": 0, "validation": 0, "test": 4}, "candidates_hash": digest(candidates),
                "generation_ids_hash": digest(["a", "b"])}
    write_json(tmp_path / "data/manifest.json", manifest)
    reviews = [{**p, "gold_valid": True, "easy_valid": True, "hard_valid": True, "reviewer": "Test fixture", "notes": ""} for p in pairs]
    write_jsonl(tmp_path / "review/negatives.jsonl", reviews)
    for condition in ["base", "response", "instruction"]:
        meta = {"experiment_id": "fixture", "complete": True, "smoke": False}
        write_json(tmp_path / "runs" / condition / "training.json", meta)
        ranking = []
        for x in ["a", "b"]:
            scores = {}
            for candidate, value in [("gold", -1), ("easy", -4), ("hard", -2)]:
                scores[candidate] = {"conditional": {"sum": value, "mean": value, "sum_with_eos": value-1},
                                     "prior": {"sum": value-1, "mean": value-1}, "gain_sum": 1, "gain_mean": 1}
            ranking.append({"id": x, "group_id": x, "category": "qa", "scores": scores})
        generations = [{"id": x, "group_id": x, "category": "qa", "response": "Fixture answer"} for x in ["a", "b"]]
        out = tmp_path / "evaluation" / condition
        write_jsonl(out / "ranking.jsonl", ranking)
        write_jsonl(out / "generations.jsonl", generations)
        write_json(out / "complete.json", {"experiment_id": "fixture", "pairs_hash": digest(pairs),
                                            "checkpoint": "test-only-model" if condition == "base" else digest(meta),
                                            "ranking_hash": digest(ranking), "generation_hash": digest(generations)})
    report(cfg, tmp_path)
    assert read_json(tmp_path / "report/results.json")["ranking_complete"]
    assert not (tmp_path / "report/email_draft.txt").exists()
    report(cfg, tmp_path, final=True)
    results = read_json(tmp_path / "report/results.json")
    assert all("generation" not in result for result in results["models"].values())
    blind(tmp_path, cfg["seed"])
    graded = read_jsonl(tmp_path / "review/generation.jsonl")
    assert len(graded) == 6 and all("condition" not in r for r in graded)
