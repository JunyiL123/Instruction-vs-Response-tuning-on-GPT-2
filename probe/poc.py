"""Build a small, explicitly scoped proof-of-concept from frozen prepared data."""
from collections import Counter, defaultdict
from pathlib import Path

from .common import ROOT, digest, load_config, read_json, read_jsonl, require, write_json, write_jsonl
from .data import balanced_sample


def build(source, destination, config_path):
    require(not destination.exists(), f"{destination} already exists")
    cfg = load_config(config_path)
    manifest = read_json(source / 'data' / 'manifest.json')
    train = read_jsonl(source / 'data' / 'train.jsonl')
    validation = read_jsonl(source / 'data' / 'validation.jsonl')
    test = read_jsonl(source / 'data' / 'test.jsonl')
    candidates = read_jsonl(source / 'data' / 'candidates.jsonl')
    decisions = {}
    for path in sorted((source / 'review').glob('llm_batch_*.jsonl')):
        decisions.update({x['id']: x for x in read_jsonl(path)})
    accepted = []
    for candidate in candidates:
        decision = decisions.get(candidate['id'])
        if decision and all(decision[k] is True for k in ('gold_valid', 'easy_valid', 'hard_valid')) and decision['confidence'] in {'high', 'medium'}:
            accepted.append({**candidate, 'hard_id': decision.get('hard_id', candidate['hard_id'])})
    require(len(accepted) >= cfg['ranking_examples'], 'Not enough adjudicated valid pairs for POC')
    selected = balanced_sample(accepted, cfg['ranking_examples'], cfg['seed'])
    selected_ids = {x['id'] for x in selected}
    # Make training/validation small while cycling across available task categories.
    tiny_train = balanced_sample(train, 24, cfg['seed'])
    tiny_val = balanced_sample(validation, 8, cfg['seed'] + 1)
    generation = balanced_sample(selected, cfg['generation_examples'], cfg['seed'] + 2)
    destination.mkdir(parents=True)
    write_json(destination / 'raw' / 'sources.json', read_json(source / 'raw' / 'sources.json'))
    write_jsonl(destination / 'data' / 'train.jsonl', tiny_train)
    write_jsonl(destination / 'data' / 'validation.jsonl', tiny_val)
    write_jsonl(destination / 'data' / 'test.jsonl', test)
    write_jsonl(destination / 'data' / 'candidates.jsonl', selected)
    write_json(destination / 'data' / 'generation_ids.json', [x['id'] for x in generation])
    reviews = []
    for candidate in selected:
        decision = decisions[candidate['id']]
        reviews.append({"id": candidate['id'], "easy_id": candidate['easy_id'], "hard_id": candidate['hard_id'],
                        **{k: decision[k] for k in ['gold_valid', 'easy_valid', 'hard_valid', 'confidence', 'notes']},
                        "adjudicator": "Codex (GPT-5), current task", "adjudicator_type": "llm", "disposition": "include"})
    write_jsonl(destination / 'review' / 'negatives.jsonl', reviews)
    output = {"config": cfg, "sources": manifest['sources'], "parent_experiment_id": manifest['experiment_id'],
              "scope": "proof of concept; fixed small subsets, one epoch, no replication claim",
              "split_counts": {"train": len(tiny_train), "validation": len(tiny_val), "test": len(test)},
              "split_hashes": {"train": digest(tiny_train), "validation": digest(tiny_val), "test": digest(test)},
              "candidates_hash": digest(selected), "generation_ids_hash": digest([x['id'] for x in generation]),
              "adjudication": {"type": "llm", "reviewed": len(reviews), "selected_from": len(accepted),
                                "selection_before_model_evaluation": True}}
    output['experiment_id'] = digest(output)
    write_json(destination / 'data' / 'manifest.json', output)
    write_json(destination / 'review' / 'adjudication_manifest.json', {"type": "llm", "reviewed": len(reviews),
               "selected_from": len(accepted), "selection_before_model_evaluation": True, "source_batches": 5})
    print({"train": len(tiny_train), "validation": len(tiny_val), "ranking": len(selected), "generation": len(generation),
           "ranking_categories": dict(Counter(x['category'] for x in selected))})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--destination', type=Path, default=ROOT / 'artifacts-poc')
    parser.add_argument('--config', type=Path, default=ROOT / 'config_poc.json')
    args = parser.parse_args()
    build(args.source, args.destination, args.config)
