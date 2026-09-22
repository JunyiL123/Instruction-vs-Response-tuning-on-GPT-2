import random
import time

import torch

from .common import device_for, digest, read_json, read_jsonl, require, same_preparation_config, seed_all, write_json, write_jsonl
from .data import validated_pairs, verify
from .modeling import prompt_ids, score_response
from .train import load_condition


def condition_fingerprint(root, condition):
    if condition == "base":
        return read_json(root / "raw" / "sources.json")["model_revision"]
    return digest(read_json(root / "runs" / condition / "training.json"))


def evaluate(cfg, root, condition, requested_device, allow_unreviewed=False):
    manifest = verify(root)
    require(same_preparation_config(cfg, manifest["config"]), "Preparation config differs from the frozen dataset")
    pairs, reviewed = validated_pairs(root, allow_unreviewed)
    device = device_for(requested_device)
    seed_all(cfg["seed"])
    model, tokenizer = load_condition(root, condition, device)
    model.eval()
    test = {r["id"]: r for r in read_jsonl(root / "data" / "test.jsonl")}
    out = root / "evaluation" / condition
    out.mkdir(parents=True, exist_ok=True)
    identity = {"experiment_id": manifest["experiment_id"], "pairs_hash": digest(pairs),
                "checkpoint": condition_fingerprint(root, condition)}
    if (out / "identity.json").exists():
        require(read_json(out / "identity.json") == identity,
                "Evaluation inputs changed. Move the existing evaluation directory before rerunning.")
    write_json(out / "identity.json", identity)
    scores_path = out / "ranking.jsonl"
    scores = read_jsonl(scores_path) if scores_path.exists() else []
    done = {r["id"] for r in scores}
    prior_cache = {}
    started = time.monotonic()
    for pair in pairs:
        if pair["id"] in done:
            continue
        gold = test[pair["id"]]
        result = {"id": gold["id"], "category": gold["category"], "group_id": gold["group_id"], "scores": {}}
        for key, candidate_id in [("gold", pair["id"]), ("easy", pair["easy_id"]), ("hard", pair["hard_id"])]:
            response = test[candidate_id]["response"]
            conditional = score_response(model, tokenizer, response, device, gold)
            if candidate_id not in prior_cache:
                prior_cache[candidate_id] = score_response(model, tokenizer, response, device)
            prior = prior_cache[candidate_id]
            result["scores"][key] = {"candidate_id": candidate_id, "conditional": conditional, "prior": prior,
                                      "gain_sum": conditional["sum"] - prior["sum"],
                                      "gain_mean": conditional["mean"] - prior["mean"]}
        scores.append(result)
        write_jsonl(scores_path, scores)
        if len(scores) % 10 == 0:
            print(f"{condition}: ranked {len(scores)}/{len(pairs)} in {time.monotonic()-started:.1f}s", flush=True)
    generations_path = out / "generations.jsonl"
    generations = read_jsonl(generations_path) if generations_path.exists() else []
    generated = {r["id"] for r in generations}
    for item_id in ([] if cfg.get("ranking_only", False) else read_json(root / "data" / "generation_ids.json")):
        if item_id in generated:
            continue
        r = test[item_id]
        ids = torch.tensor([prompt_ids(tokenizer, r)], device=device)
        with torch.inference_mode():
            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    do_sample=False, num_beams=1, max_new_tokens=cfg["max_new_tokens"],
                                    repetition_penalty=1.0, eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id, use_cache=True)
        tokens = result[0, ids.shape[1]:].cpu().tolist()
        generations.append({"id": item_id, "category": r["category"], "group_id": r["group_id"],
                            "response": tokenizer.decode(tokens, skip_special_tokens=True),
                            "generated_tokens": len(tokens), "hit_token_limit": len(tokens) == cfg["max_new_tokens"] and tokens[-1] != tokenizer.eos_token_id})
        write_jsonl(generations_path, generations)
        if len(generations) % 10 == 0:
            print(f"{condition}: generated {len(generations)}/{cfg['generation_examples']}", flush=True)
    write_json(out / "complete.json", {**identity, "condition": condition, "manual_negatives_reviewed": reviewed,
                                       "ranking_n": len(scores), "generation_n": len(generations),
                                       "ranking_hash": digest(scores), "generation_hash": digest(generations)})


def blind(root, seed):
    verify(root)
    test = {r["id"]: r for r in read_jsonl(root / "data" / "test.jsonl")}
    rows = []
    identities = {}
    for condition in ["base", "response", "instruction"]:
        path = root / "evaluation" / condition
        require((path / "complete.json").exists(), f"Evaluate {condition} before exporting blinded answers")
        completion = read_json(path / "complete.json")
        generations = read_jsonl(path / "generations.jsonl")
        require(digest(generations) == completion["generation_hash"], "Generation output changed")
        identities[condition] = completion["generation_hash"]
        rows += [(condition, row) for row in generations]
    rng = random.Random(seed + 1000)
    rng.shuffle(rows)
    reviews, key = [], []
    for i, (condition, generated) in enumerate(rows):
        r = test[generated["id"]]
        blind_id = f"answer-{i+1:04d}"
        reviews.append({"blind_id": blind_id, "instruction": r["instruction"], "context": r["context"],
                        "reference": r["response"], "answer": generated["response"],
                        "addresses_task": None, "substantially_correct": None, "reviewer": "", "notes": ""})
        key.append({"blind_id": blind_id, "condition": condition, "id": r["id"], "group_id": r["group_id"]})
    path = root / "review" / "generation.jsonl"
    if path.exists():
        require(read_json(root / "private" / "blind_manifest.json")["generation_hashes"] == identities,
                "Existing blinded review refers to different generations")
        print("Existing blinded review preserved", flush=True)
        return
    write_jsonl(path, reviews)
    write_jsonl(root / "private" / "blind_key.jsonl", key)
    write_json(root / "private" / "blind_manifest.json", {"generation_hashes": identities,
               "content_hash": digest([{k: r[k] for k in ["blind_id", "instruction", "context", "reference", "answer"]} for r in reviews]),
               "key_hash": digest(key)})
    rubric = """# Blinded generation rubric

Read only review/generation.jsonl while grading; keep private/blind_key.jsonl closed.
Enter booleans, your name in reviewer, and a brief reason in notes for every answer.

- addresses_task: true if the answer performs the requested operation and respects explicit essential constraints. A relevant topic alone, repeating the question, or writing another question does not count.
- substantially_correct: true if the answer is coherent and materially correct for the task and supplied context. Minor style differences are acceptable. The Dolly reference is guidance, not an exact-match target; verify factual uncertainty and accept valid alternative answers.
- Success requires both flags to be true. An empty answer fails both. A truncated answer is judged on the text actually present; an unfinished essential answer fails.
- Examples: a correct translation into the wrong requested language fails task adherence; an on-task list with a central false claim fails correctness; a different but valid brainstorming list can pass both.

Use one consistent rubric for all 300 answers. Do not infer model identities or revise the rubric after unblinding. Record unclear references in notes and resolve them before final reporting.
"""
    (root / "review" / "generation_rubric.md").write_text(rubric)
    print(f"Exported {len(reviews)} blinded answers to {path}", flush=True)
