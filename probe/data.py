import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import digest, read_json, read_jsonl, require, same_preparation_config, write_json, write_jsonl
from .modeling import encode_example, prompt_ids, response_ids


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def fetch(cfg, root):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lock = raw / "sources.json"
    if lock.exists():
        sources = read_json(lock)
        require(sources["model_id"] == cfg["model_id"] and sources["dataset_id"] == cfg["dataset_id"],
                "Source IDs differ from existing lock; use a new artifacts directory")
    else:
        api = HfApi()
        sources = {"model_id": cfg["model_id"], "dataset_id": cfg["dataset_id"],
                   "model_revision": api.model_info(cfg["model_id"]).sha,
                   "dataset_revision": api.dataset_info(cfg["dataset_id"]).sha}
        write_json(lock, sources)
    model = snapshot_download(cfg["model_id"], revision=sources["model_revision"],
                              allow_patterns=["config.json", "generation_config.json", "model.safetensors",
                                              "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
    data = hf_hub_download(cfg["dataset_id"], cfg["dataset_file"], repo_type="dataset",
                           revision=sources["dataset_revision"])
    sources.update(model_path=model, dataset_path=data)
    write_json(lock, sources)
    return sources


def clean_rows(raw, tokenizer, cfg):
    rows, seen, removed = [], set(), Counter()
    for index, item in enumerate(raw):
        row = {key: str(item.get(key) or "").strip() for key in ["instruction", "context", "response", "category"]}
        if not all(row[k] for k in ["instruction", "response", "category"]):
            removed["empty"] += 1
            continue
        key = (normalize(row["instruction"]), normalize(row["context"]))
        if key in seen:
            removed["duplicate_prompt"] += 1
            continue
        n_prompt = len(prompt_ids(tokenizer, row))
        n_response = len(response_ids(tokenizer, row["response"], False))
        if n_prompt > cfg["max_prompt_tokens"] or n_response > cfg["max_response_tokens"] or n_prompt + n_response + 1 > cfg["max_sequence_tokens"]:
            removed["overlength"] += 1
            continue
        seen.add(key)
        row.update(id=f"dolly-{index:05d}", response_tokens=n_response)
        rows.append(row)
    return rows, dict(removed)


def group_duplicates(rows, threshold):
    """Connect near-identical full prompts, shared contexts, and long exact responses."""
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    for field, min_length in [("context", 40), ("response", 80)]:
        seen = {}
        for i, r in enumerate(rows):
            key = normalize(r[field])
            if len(key) < min_length:
                continue
            if key in seen:
                union(i, seen[key])
            seen[key] = i
    texts = [normalize(r["instruction"] + " " + r["context"]) for r in rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=100000, dtype=np.float32)
    matrix = vectorizer.fit_transform(texts)
    edges = 0
    for start in range(0, len(rows), 128):
        similarity = (matrix[start:start+128] @ matrix.T).tocoo()
        for a, b, v in zip(similarity.row, similarity.col, similarity.data):
            i = start + int(a)
            if int(b) > i and v >= threshold:
                union(i, int(b))
                edges += 1
    members = defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(i)
    for indices in members.values():
        group_id = min(rows[i]["id"] for i in indices)
        for i in indices:
            rows[i]["group_id"] = group_id
    return {"groups": len(members), "near_duplicate_edges": edges,
            "largest_group": max(map(len, members.values())), "similarity_threshold": threshold}


def split_groups(rows, seed):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    by_category = defaultdict(list)
    for group in groups.values():
        category = Counter(r["category"] for r in group).most_common(1)[0][0]
        by_category[category].append(group)
    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for category in sorted(by_category):
        current = by_category[category]
        rng.shuffle(current)
        total = sum(map(len, current))
        counts = {k: 0 for k in splits}
        for group in current:
            target = max(splits, key=lambda k: {"train": .8, "validation": .1, "test": .1}[k] * total - counts[k])
            splits[target].extend(group)
            counts[target] += len(group)
    return {k: sorted(v, key=lambda r: r["id"]) for k, v in splits.items()}


def balanced_sample(rows, count, seed):
    require(len(rows) >= count, f"Need {count} eligible examples, found {len(rows)}")
    pools = defaultdict(list)
    rng = random.Random(seed)
    for row in rows:
        pools[row["category"]].append(row)
    for values in pools.values():
        rng.shuffle(values)
    result = []
    while len(result) < count:
        for category in sorted(pools):
            if pools[category] and len(result) < count:
                result.append(pools[category].pop())
    return result


def build_candidates(test, cfg):
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(
        [r["instruction"] + " " + r["context"] for r in test])
    similarity = (matrix @ matrix.T).toarray()
    rng = random.Random(cfg["seed"])
    eligible = []
    for i, target in enumerate(test):
        options, easy = [], []
        for j, other in enumerate(test):
            if target["group_id"] == other["group_id"] or normalize(target["response"]) == normalize(other["response"]):
                continue
            ratio = max(target["response_tokens"], other["response_tokens"]) / max(1, min(target["response_tokens"], other["response_tokens"]))
            if ratio > cfg["negative_length_ratio"]:
                continue
            if other["category"] != target["category"]:
                easy.append(other["id"])
            elif similarity[i, j] >= cfg["hard_min_similarity"]:
                options.append({"id": other["id"], "similarity": float(similarity[i, j]), "length_ratio": ratio})
        if easy and options:
            options.sort(key=lambda x: (-x["similarity"], x["id"]))
            eligible.append({"id": target["id"], "category": target["category"],
                             "easy_id": rng.choice(easy), "hard_options": options[:5], "hard_id": options[0]["id"]})
    chosen = balanced_sample(eligible, cfg["ranking_examples"], cfg["seed"])
    return chosen, len(eligible)


def prepare(cfg, root):
    from transformers import AutoTokenizer
    require(not (root / "data" / "manifest.json").exists(), "Prepared data already exist. Use a new root to change the experiment.")
    sources = fetch(cfg, root)
    tokenizer = AutoTokenizer.from_pretrained(sources["model_path"], local_files_only=True)
    raw = read_jsonl(sources["dataset_path"])
    rows, removed = clean_rows(raw, tokenizer, cfg)
    print(f"Cleaned {len(raw)} -> {len(rows)}; grouping near duplicates", flush=True)
    group_info = group_duplicates(rows, cfg["near_duplicate_similarity"])
    splits = split_groups(rows, cfg["seed"])
    for name, records in splits.items():
        write_jsonl(root / "data" / f"{name}.jsonl", records)
    candidates, eligible = build_candidates(splits["test"], cfg)
    write_jsonl(root / "data" / "candidates.jsonl", candidates)
    generation = balanced_sample(candidates, cfg["generation_examples"], cfg["seed"] + 1)
    write_json(root / "data" / "generation_ids.json", [r["id"] for r in generation])
    reviews = [{"id": r["id"], "easy_id": r["easy_id"], "hard_id": r["hard_id"],
                "gold_valid": None, "easy_valid": None, "hard_valid": None,
                "adjudicator": "", "adjudicator_type": "", "notes": ""} for r in candidates]
    write_jsonl(root / "review" / "negatives.jsonl", reviews)
    manifest = {"config": cfg, "sources": sources, "raw_count": len(raw), "removed": removed,
                "grouping": group_info, "split_counts": {k: len(v) for k, v in splits.items()},
                "category_counts": {k: dict(Counter(r["category"] for r in v)) for k, v in splits.items()},
                "split_hashes": {k: digest(v) for k, v in splits.items()},
                "eligible_ranking_examples": eligible, "candidates_hash": digest(candidates),
                "generation_ids_hash": digest([r["id"] for r in generation])}
    manifest["experiment_id"] = digest(manifest)
    write_json(root / "data" / "manifest.json", manifest)
    export_review(root)
    print(manifest["split_counts"], flush=True)


def verify(root):
    manifest = read_json(root / "data" / "manifest.json")
    used = set()
    for name, expected in manifest["split_hashes"].items():
        records = read_jsonl(root / "data" / f"{name}.jsonl")
        require(digest(records) == expected, f"{name} changed after preparation")
        groups = {r["group_id"] for r in records}
        require(not used & groups, "Duplicate group crosses splits")
        used |= groups
    candidates = read_jsonl(root / "data" / "candidates.jsonl")
    require(digest(candidates) == manifest["candidates_hash"], "Frozen candidate set changed")
    require(digest(read_json(root / "data" / "generation_ids.json")) == manifest["generation_ids_hash"], "Generation subset changed")
    return manifest


def validated_pairs(root, allow_unreviewed=False):
    candidates = read_jsonl(root / "data" / "candidates.jsonl")
    reviews = read_jsonl(root / "review" / "negatives.jsonl")
    require(len({r["id"] for r in reviews}) == len(reviews), "Duplicate negative review IDs")
    by_id = {r["id"]: r for r in reviews}
    require(set(by_id) == {r["id"] for r in candidates}, "Negative review IDs differ from frozen set")
    result, complete = [], True
    for candidate in candidates:
        r = by_id[candidate["id"]]
        require(r["easy_id"] == candidate["easy_id"], "Easy negative differs from frozen candidate")
        require(r["hard_id"] in {x["id"] for x in candidate["hard_options"]}, "Choose hard_id from the candidate options")
        adjudicator = r.get("adjudicator", r.get("reviewer", "")).strip()
        adjudicator_type = r.get("adjudicator_type", "human" if adjudicator else "")
        approved = (all(r.get(k) is True for k in ["gold_valid", "easy_valid", "hard_valid"])
                    and bool(adjudicator) and adjudicator_type in {"human", "llm"})
        complete &= approved
        result.append({"id": r["id"], "easy_id": r["easy_id"], "hard_id": r["hard_id"]})
    require(complete or allow_unreviewed, "Manual negative review is incomplete. See review/negative_review.md; draft scoring requires --allow-unreviewed.")
    return result, complete


def export_review(root):
    test = {r["id"]: r for r in read_jsonl(root / "data" / "test.jsonl")}
    candidates = read_jsonl(root / "data" / "candidates.jsonl")
    lines = ["# Negative review", "", "Adjudicator: edit negatives.jsonl. Set adjudicator_type to human or llm and identify the adjudicator. Verify the gold answer, reject accidental correct alternatives, and choose hard_id from the listed options. Set all three *_valid flags to true only after reading the complete item. If no candidate qualifies, leave it unapproved; do not select using model scores.", ""]
    for c in candidates:
        r = test[c["id"]]
        lines += [f"## {r['id']} ({r['category']})", "", r["instruction"], "", r["context"], "", "**Gold:** " + r["response"], "", "**Easy:** " + test[c["easy_id"]]["response"], ""]
        for option in c["hard_options"]:
            other = test[option["id"]]
            lines += [f"**Hard option {option['id']}** (similarity {option['similarity']:.3f}; length ratio {option['length_ratio']:.2f})", "", "Source instruction: " + other["instruction"], "", other["response"], ""]
    (root / "review" / "negative_review.md").write_text("\n".join(lines))
