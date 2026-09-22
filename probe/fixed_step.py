"""Score the response-only checkpoint after its final optimizer update.

This is a sensitivity check. The primary report continues to use the checkpoint
selected by validation loss.
"""

import argparse
from pathlib import Path

import torch

from .common import digest, load_config, read_json, read_jsonl, require, write_json
from .data import validated_pairs, verify
from .modeling import score_response
from .report import interval
from .train import load_base


def score(root, cfg):
    manifest = verify(root)
    pairs, reviewed = validated_pairs(root)
    require(reviewed, "Negative review must be complete")
    run = read_json(root / "runs" / "response" / "training.json")
    require(run["complete"] and run["updates"] == 24, "Expected the completed 24-update response run")
    checkpoint = torch.load(root / "runs" / "response" / "latest.pt", map_location="cpu",
                            weights_only=True, mmap=True)
    require(checkpoint["experiment_id"] == manifest["experiment_id"] and checkpoint["step"] == 24,
            "Checkpoint does not match this frozen experiment")
    model, tokenizer = load_base(root, torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    del checkpoint
    test = {row["id"]: row for row in read_jsonl(root / "data" / "test.jsonl")}
    wins = {f"{negative}:{metric}": [] for negative in ("easy", "hard")
            for metric in ("sum", "mean")}
    groups = []
    for pair in pairs:
        target = test[pair["id"]]
        groups.append(target["group_id"])
        gold = score_response(model, tokenizer, target["response"], torch.device("cpu"), target)
        for negative in ("easy", "hard"):
            other = score_response(model, tokenizer, test[pair[f"{negative}_id"]]["response"],
                                   torch.device("cpu"), target)
            for metric in ("sum", "mean"):
                wins[f"{negative}:{metric}"].append(float(gold[metric] > other[metric]))
    result = {
        "description": "Fixed-update sensitivity check; primary response checkpoint selected by validation loss at update 12",
        "checkpoint": "response-only after update 24",
        "experiment_id": manifest["experiment_id"],
        "pairs_hash": digest(pairs),
        "selected_for_primary_report": False,
        "update": 24,
        "validation_response_nll": run["history"][-1]["validation_response_nll"],
        "ranking": {key: interval(values, groups, cfg["bootstrap_samples"], cfg["seed"])
                    for key, values in wins.items()},
    }
    write_json(root / "report" / "response_fixed_24.json", result)
    print({key: f"{value['estimate']:.1%}" for key, value in result["ranking"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_poc.json"))
    parser.add_argument("--root", type=Path, default=Path("artifacts-poc"))
    args = parser.parse_args()
    score(args.root, load_config(args.config))
