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


def formatted(stat):
    if stat is None:
        return "pending"
    return f"{100*stat['estimate']:.1f}% [{100*stat['low']:.1f}, {100*stat['high']:.1f}]"


def report(cfg, root, final=False):
    manifest = verify(root)
    require(same_preparation_config(cfg, manifest["config"]), "Preparation config differs from the frozen dataset")
    pairs, negatives_complete = validated_pairs(root, allow_unreviewed=True)
    results, vectors, group_ids = {}, {}, []
    metrics = ["conditional.sum", "conditional.mean", "conditional.sum_with_eos", "prior.sum", "prior.mean", "gain_sum", "gain_mean"]
    for condition in CONDITIONS:
        directory = root / "evaluation" / condition
        if not (directory / "complete.json").exists():
            continue
        done = read_json(directory / "complete.json")
        require(done["experiment_id"] == manifest["experiment_id"] and done["pairs_hash"] == digest(pairs),
                "Evaluation is stale after data or negative review changes")
        require(done["checkpoint"] == condition_fingerprint(root, condition, cfg.get("checkpoint_policy", "validation")),
                "Evaluation used a different checkpoint")
        ranking = read_jsonl(directory / "ranking.jsonl")
        require(digest(ranking) == done["ranking_hash"], "Ranking outputs changed")
        require(len(ranking) == len(pairs) and {r["id"] for r in ranking} == {r["id"] for r in pairs}, "Ranking example mismatch")
        ranking.sort(key=lambda r: r["id"])
        group_ids = [r["group_id"] for r in ranking]
        results[condition] = {"ranking": {}, "categories": {}}
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
    differences = {}
    for a, b in [("response", "base"), ("instruction", "base"), ("response", "instruction")]:
        if a not in vectors or b not in vectors:
            continue
        differences[a + " minus " + b] = {key: interval(vectors[a][key]-vectors[b][key], group_ids, cfg["bootstrap_samples"], cfg["seed"])
                                           for key in vectors[a]}
    review_type = "LLM" if negatives_complete and all(r.get("adjudicator_type") == "llm" for r in read_jsonl(root / "review" / "negatives.jsonl")) else "human or mixed"
    ready = len(results) == 3 and negatives_complete
    require(not final or ready, "Final reporting requires all three models and approved negatives")
    status = "LLM-adjudicated ranking result" if ready else "PRELIMINARY — ranking evaluation is incomplete"
    output = root / "report"
    policy = cfg.get("checkpoint_policy", "validation")
    training = {}
    for condition in ["response", "instruction"]:
        selection = root / "runs" / condition / "best" / "selection.json"
        metadata = root / "runs" / condition / "training.json"
        if metadata.exists() and (policy == "final" or selection.exists()):
            run = read_json(metadata)
            selected = run["history"][-1] if policy == "final" else read_json(selection)
            training[condition] = {"updates_completed": run["updates"],
                                   "reported_update": selected["update"],
                                   "reported_epoch": selected["epoch"],
                                   "validation_response_nll": selected["validation_response_nll"]}
    write_json(output / "results.json", {"status": status, "ranking_complete": ready, "experiment_id": manifest["experiment_id"],
                                         "models": results, "training": training, "checkpoint_policy": policy,
                                         "paired_differences": differences,
                                         "negative_review_hash": digest(read_jsonl(root / "review" / "negatives.jsonl")),
                                         "negative_adjudication_type": review_type,
                                         "negative_adjudication_complete": negatives_complete})
    header = ["# GPT-2 instruction–response matching", "", status, "",
              "| Checkpoint | Update | Easy: total log P | Hard: total log P | Easy: per token | Hard: per token |",
              "|---|---:|---|---|---|---|"]
    for condition in CONDITIONS:
        r = results.get(condition, {"ranking": {}})
        values = [formatted(r["ranking"].get(key)) for key in ["easy:conditional.sum", "hard:conditional.sum", "easy:conditional.mean", "hard:conditional.mean"]]
        reported_update = str(training[condition]["reported_update"]) if condition in training else "—"
        header.append("| " + " | ".join([condition, reported_update] + values) + " |")
    split_counts = manifest["split_counts"]
    splits = f"{split_counts['train']} train, {split_counts['validation']} validation, {split_counts['test']} test"
    categories = Counter(r["category"] for r in read_jsonl(root / "data" / "candidates.jsonl"))
    category_counts = ", ".join(f"{category} {count}" for category, count in sorted(categories.items()))
    epoch_word = "epoch" if cfg["epochs"] == 1 else "epochs"
    checkpoint_text = (f"Both reported checkpoints use final update {training.get('response', {}).get('reported_update', 'pending')}, fixing the number of optimizer steps."
                       if policy == "final" else "Each reported checkpoint was chosen by validation loss under its own objective.")
    selection_limit = ("The fixed-final checkpoint rule was adopted after an initial validation-selected analysis, so this comparison is exploratory. "
                       if policy == "final" else "")
    methods = f"""

**Question.** Does pretrained GPT-2 124M associate instructions with appropriate responses, and does tuning on responses alone change that ranking compared with instruction tuning?

**Setup.** Dolly was cleaned and split by duplicate groups: {splits}. The ranking set contains {cfg['ranking_examples']} prompts across these categories: {category_counts}. Each prompt has one different-category negative and one related same-category negative. Both negatives have response-token length ratio at most {cfg['negative_length_ratio']}. Dataset revision: `{manifest['sources']['dataset_revision']}`; GPT-2 revision: `{manifest['sources']['model_revision']}`. Both tuned models saw the same responses for {cfg['epochs']} {epoch_word}, using AdamW at {cfg['learning_rate']} and effective batch {cfg['effective_batch_size']}. {checkpoint_text}

**Metrics.** Ranking means the correct response has strictly greater likelihood than the alternative. Main scores exclude EOS; EOS-inclusive totals, unconditional scores, conditioning gains, category results, tie rates, and paired differences are in results.json. Brackets are 95% percentile intervals from {cfg['bootstrap_samples']} bootstrap resamples of prompt duplicate groups.

**Limits.** This is a small, one-seed ranking experiment with {cfg['epochs']} tuning {epoch_word}. {selection_limit}The response-ranking definition follows [Hewitt et al., §4.2](https://arxiv.org/pdf/2409.14254). Hard negatives were assessed by a recorded LLM adjudicator. Duplicate checks cover the documented lexical threshold and shared-context rules, not every semantic paraphrase or pretraining overlap. Intervals capture sampled-prompt uncertainty, not training-seed variation, adjudicator error, or all dependence from reused negative answers.
"""
    (output / "note.md").write_text("\n".join(header) + methods)
    if results:
        plot_results(results, output)
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
