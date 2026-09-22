from collections import Counter, defaultdict

import numpy as np

from .common import digest, read_json, read_jsonl, require, same_preparation_config, write_json
from .data import validated_pairs, verify
from .evaluate import condition_fingerprint

CONDITIONS = ["base", "response", "instruction"]


def interval(values, groups, repetitions=5000, seed=42):
    """Percentile bootstrap of prompt groups; repeated records stay together."""
    buckets = defaultdict(list)
    for value, group in zip(values, groups):
        buckets[group].append(value)
    require(bool(buckets), "Cannot estimate an empty sample")
    arrays = [np.asarray(x, dtype=float) for x in buckets.values()]
    sums = np.array([x.sum() for x in arrays])
    counts = np.array([len(x) for x in arrays])
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(arrays), size=(repetitions, len(arrays)))
    rates = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)
    low, high = np.quantile(rates, [.025, .975])
    return {"estimate": float(np.mean(values)), "low": float(low), "high": float(high),
            "n": len(values), "groups": len(arrays)}


def wins(row, negative, metric):
    def value(candidate):
        scores = row["scores"][candidate]
        if metric in {"gain_sum", "gain_mean"}:
            return scores[metric]
        kind, measure = metric.split(".")
        return scores[kind][measure]
    positive, other = value("gold"), value(negative)
    return float(positive > other), float(positive == other)


def generation_reviews(root):
    path = root / "review" / "generation.jsonl"
    if not path.exists():
        return {}, False
    rows = read_jsonl(path)
    key = read_jsonl(root / "private" / "blind_key.jsonl")
    meta = read_json(root / "private" / "blind_manifest.json")
    require(digest(key) == meta["key_hash"], "Blind key changed")
    for condition, expected in meta["generation_hashes"].items():
        require(digest(read_jsonl(root / "evaluation" / condition / "generations.jsonl")) == expected,
                "Blinded review is stale")
    require(digest([{k: r[k] for k in ["blind_id", "instruction", "context", "reference", "answer"]} for r in rows]) == meta["content_hash"],
            "Blinded prompt or answer text was changed")
    mapping = {r["blind_id"]: r for r in key}
    require(len(rows) == len(mapping) and len({r["blind_id"] for r in rows}) == len(rows)
            and {r["blind_id"] for r in rows} == set(mapping), "Incomplete or duplicate blinded IDs")
    complete = all(type(r.get("addresses_task")) is bool and type(r.get("substantially_correct")) is bool
                   and bool(r.get("reviewer", "").strip()) for r in rows)
    if not complete:
        return {}, False
    result = defaultdict(list)
    for r in rows:
        item = mapping[r["blind_id"]]
        result[item["condition"]].append({**item, "success": float(r["addresses_task"] and r["substantially_correct"]),
                                         "addresses_task": float(r["addresses_task"]), "correct": float(r["substantially_correct"]),
                                         "notes": r["notes"]})
    return dict(result), True


def formatted(stat):
    if stat is None:
        return "pending"
    return f"{100*stat['estimate']:.1f}% [{100*stat['low']:.1f}, {100*stat['high']:.1f}]"


def report(cfg, root, final=False):
    manifest = verify(root)
    require(same_preparation_config(cfg, manifest["config"]), "Preparation config differs from the frozen dataset")
    pairs, negatives_complete = validated_pairs(root, allow_unreviewed=True)
    generated, generation_complete = generation_reviews(root)
    results, vectors, group_ids = {}, {}, []
    metrics = ["conditional.sum", "conditional.mean", "conditional.sum_with_eos", "prior.sum", "prior.mean", "gain_sum", "gain_mean"]
    for condition in CONDITIONS:
        directory = root / "evaluation" / condition
        if not (directory / "complete.json").exists():
            continue
        done = read_json(directory / "complete.json")
        require(done["experiment_id"] == manifest["experiment_id"] and done["pairs_hash"] == digest(pairs),
                "Evaluation is stale after data or negative review changes")
        require(done["checkpoint"] == condition_fingerprint(root, condition), "Evaluation used a different checkpoint")
        ranking = read_jsonl(directory / "ranking.jsonl")
        outputs = read_jsonl(directory / "generations.jsonl")
        require(digest(ranking) == done["ranking_hash"] and digest(outputs) == done["generation_hash"], "Evaluation outputs changed")
        require(len(ranking) == len(pairs) and {r["id"] for r in ranking} == {r["id"] for r in pairs}, "Ranking example mismatch")
        require(len(outputs) == cfg["generation_examples"] and {r["id"] for r in outputs} == set(read_json(root / "data" / "generation_ids.json")), "Generation example mismatch")
        ranking.sort(key=lambda r: r["id"])
        group_ids = [r["group_id"] for r in ranking]
        results[condition] = {"ranking": {}, "generation": {}, "categories": {}}
        vectors[condition] = {}
        for negative in ["easy", "hard"]:
            for metric in metrics:
                name = negative + ":" + metric
                values = [wins(row, negative, metric)[0] for row in ranking]
                stat = interval(values, group_ids, cfg["bootstrap_samples"], cfg["seed"])
                stat["tie_rate"] = float(np.mean([wins(row, negative, metric)[1] for row in ranking]))
                results[condition]["ranking"][name] = stat
                vectors[condition][name] = np.array(values)
        for category in sorted({r["category"] for r in ranking}):
            subset = [r for r in ranking if r["category"] == category]
            results[condition]["categories"][category] = {n: interval([wins(r, n, "conditional.mean")[0] for r in subset],
                                                                                     [r["group_id"] for r in subset], cfg["bootstrap_samples"], cfg["seed"])
                                                            for n in ["easy", "hard"]}
        if generation_complete:
            reviews = sorted(generated[condition], key=lambda r: r["id"])
            require({r["id"] for r in reviews} == {r["id"] for r in outputs}, "Generation grading mismatch")
            for metric in ["success", "addresses_task", "correct"]:
                results[condition]["generation"][metric] = interval([r[metric] for r in reviews], [r["group_id"] for r in reviews],
                                                                      cfg["bootstrap_samples"], cfg["seed"])
    differences = {}
    for a, b in [("response", "base"), ("instruction", "base"), ("response", "instruction")]:
        if a not in vectors or b not in vectors:
            continue
        differences[a + " minus " + b] = {key: interval(vectors[a][key]-vectors[b][key], group_ids, cfg["bootstrap_samples"], cfg["seed"])
                                           for key in vectors[a]}
        if generation_complete:
            left = sorted(generated[a], key=lambda r: r["id"])
            right = sorted(generated[b], key=lambda r: r["id"])
            differences[a + " minus " + b]["generation_success"] = interval(
                [x["success"]-y["success"] for x, y in zip(left, right)], [x["group_id"] for x in left],
                cfg["bootstrap_samples"], cfg["seed"])
    review_type = "LLM" if negatives_complete and all(r.get("adjudicator_type") == "llm" for r in read_jsonl(root / "review" / "negatives.jsonl")) else "human or mixed"
    ready = len(results) == 3 and negatives_complete and generation_complete
    require(not final or ready, "Final reporting requires all three models, approved negatives, and all blinded generation grades")
    status = "LLM-reviewed exploratory result" if ready else "PRELIMINARY — training, evaluation, or LLM review is incomplete"
    output = root / "report"
    write_json(output / "results.json", {"status": status, "ready_for_email": ready, "experiment_id": manifest["experiment_id"],
                                         "models": results, "paired_differences": differences,
                                         "negative_review_hash": digest(read_jsonl(root / "review" / "negatives.jsonl")),
                                         "negative_adjudication_type": review_type,
                                         "generation_review_hash": digest(read_jsonl(root / "review" / "generation.jsonl")) if generation_complete else None,
                                         "manual_negatives_complete": negatives_complete, "generation_grading_complete": generation_complete})
    header = ["# GPT-2 instruction–response matching", "", status, "",
              "| Checkpoint | Easy: total log P | Hard: total log P | Easy: per token | Hard: per token | Generated answer success |",
              "|---|---|---|---|---|---|"]
    for condition in CONDITIONS:
        r = results.get(condition, {"ranking": {}, "generation": {}})
        values = [formatted(r["ranking"].get(key)) for key in ["easy:conditional.sum", "hard:conditional.sum", "easy:conditional.mean", "hard:conditional.mean"]]
        header.append("| " + " | ".join([condition] + values + [formatted(r["generation"].get("success"))]) + " |")
    capped = {condition: sum(bool(r.get("hit_token_limit", False)) for r in read_jsonl(root / "evaluation" / condition / "generations.jsonl"))
              for condition in results}
    methods = f"""

**Question.** Does pretrained GPT-2 124M associate instructions with appropriate responses, and does tuning on responses alone change ranking and generation compared with instruction tuning?

**Setup.** Dolly, cleaned and split by duplicate groups: {manifest['split_counts']}. The frozen ranking set contains {cfg['ranking_examples']} prompts; generation uses {cfg['generation_examples']} of those prompts. Sampling cycles through categories, but eligible negative availability limits their representation: {dict(Counter(r['category'] for r in read_jsonl(root / 'data' / 'candidates.jsonl')))}. Each prompt has one different-category negative and one same-category, lexically related negative. Both negatives have response-token length ratio at most {cfg['negative_length_ratio']}. The dataset revision is `{manifest['sources']['dataset_revision']}`; GPT-2 revision is `{manifest['sources']['model_revision']}`. Full-parameter tuning uses {cfg['epochs']} epoch(s), AdamW at {cfg['learning_rate']}, effective batch {cfg['effective_batch_size']}, and identical response exposure. Each condition selects its epoch by validation loss under its own objective.

**Metrics.** Ranking means the correct response has strictly greater likelihood than the alternative. Main scores exclude EOS; EOS-inclusive totals, unconditional scores, conditioning gains, category results, tie rates, and paired differences are in results.json. Brackets are 95% percentile intervals from {cfg['bootstrap_samples']} bootstrap resamples of prompt duplicate groups. Generation succeeds only when the blinded LLM grader marks both task adherence and substantial correctness true. Outputs that hit the {cfg.get('max_new_tokens', 'configured')}-token generation cap: {capped}.

**Limits.** This is deliberately a five-to-ten-minute proof of concept: 24 ranking prompts, 8 generated prompts, one seed, and one short tuning epoch. It is inspired by [Hewitt et al., §4.2](https://arxiv.org/pdf/2409.14254), with a different model scale, dataset, and prompt format. Ranking supplied responses does not establish an ability to generate them. The 48-token cap and greedy decoding limit what the zero generation score means; a bootstrap interval of [0, 0] here is not evidence of certain failure on new prompts. Hard negatives and generated answers were assessed by a recorded LLM adjudicator, so a human check would be required for a research claim. Duplicate checks cover the documented lexical threshold and shared-context rules, not every semantic paraphrase or pretraining overlap. Intervals capture sampled-prompt uncertainty, not training-seed variation, adjudicator error, or all dependence from reused negative answers.
"""
    (output / "note.md").write_text("\n".join(header) + methods)
    if results:
        plot_results(results, output)
    if ready:
        response = results["response"]
        base = results["base"]
        instruction = results["instruction"]
        email = f"""Subject: A small GPT-2 experiment inspired by your response-tuning paper

Dear Professor Hewitt,

Your paper made me wonder whether latent instruction–response associations are detectable in GPT-2 124M, and whether response-only tuning changes their expression in generated answers.
I compared the pretrained model with response-only and instruction-tuned versions on the same Dolly training responses, using {cfg['ranking_examples']} held-out prompts with easy and LLM-adjudicated hard negatives plus {cfg['generation_examples']} prompts with blinded generation grading.
On hard negatives, per-token ranking accuracy was {formatted(base['ranking']['hard:conditional.mean'])} for the base model, {formatted(response['ranking']['hard:conditional.mean'])} after response tuning, and {formatted(instruction['ranking']['hard:conditional.mean'])} after instruction tuning; generated-answer success was {formatted(base['generation']['success'])}, {formatted(response['generation']['success'])}, and {formatted(instruction['generation']['success'])}, respectively.
This is a one-seed, small-model study on a different dataset, so I interpret it as an exploratory test rather than a replication of the 7B results.
I have included the short note and reproducible code [INSERT LINKS], and would be interested in whether you think this distinction between response ranking and generation is useful to investigate further.

Best,
[YOUR NAME]
"""
        (output / "email_draft.txt").write_text(email)
        examples(root, generated, output)
    elif (output / "email_draft.txt").exists():
        (output / "email_draft.txt").unlink()
    print(f"{status}\nReport: {output / 'note.md'}", flush=True)


def plot_results(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    names = list(results)
    x = np.arange(len(names))
    for ax, (measure, title) in zip(axes, [("sum", "Total response log P"),
                                           ("mean", "Mean log P per token")]):
        for offset, negative, color in [(-.18, "easy", "#2878a0"), (.18, "hard", "#d38630")]:
            stats = [results[name]["ranking"][f"{negative}:conditional.{measure}"] for name in names]
            y = np.array([r["estimate"] for r in stats]) * 100
            low = np.array([r["low"] for r in stats]) * 100
            high = np.array([r["high"] for r in stats]) * 100
            ax.bar(x+offset, y, width=.34, label=negative.capitalize(), color=color)
            ax.errorbar(x+offset, y, yerr=np.maximum(0, np.vstack([y-low, high-y])),
                        fmt="none", color="black", capsize=3)
        ax.axhline(50, ls="--", color="gray", lw=1)
        ax.set(xticks=x, xticklabels=names, ylim=(0, 105), title=title)
    axes[0].set_ylabel("Correct response preferred (%)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("GPT-2 response ranking on 24 held-out prompts")
    fig.tight_layout()
    fig.savefig(output / "ranking.png", dpi=200)
    plt.close(fig)


def examples(root, reviews, output):
    test = {r["id"]: r for r in read_jsonl(root / "data" / "test.jsonl")}
    lines = ["# Deterministically selected successes and failures", "", "First example by ID in each outcome class, per checkpoint. All outputs remain available in evaluation/.", ""]
    for condition in CONDITIONS:
        generated = {r["id"]: r for r in read_jsonl(root / "evaluation" / condition / "generations.jsonl")}
        for success in [1, 0]:
            candidates = sorted([r for r in reviews[condition] if r["success"] == success], key=lambda r: r["id"])
            if candidates:
                r = candidates[0]
                target = test[r["id"]]
                lines += [f"## {condition}: {'success' if success else 'failure'} — {r['id']}", "", target["instruction"], "", target["context"], "",
                          "**Output:** " + generated[r["id"]]["response"], "", "**Reviewer note:** " + r["notes"], ""]
    (output / "examples.md").write_text("\n".join(lines))
