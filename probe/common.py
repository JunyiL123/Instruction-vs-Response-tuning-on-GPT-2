import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(requested="auto"):
    import torch
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def load_config(path):
    cfg = read_json(path)
    require(cfg.get("checkpoint_policy", "validation") in {"validation", "final"},
            "checkpoint_policy must be validation or final")
    require(cfg["effective_batch_size"] % cfg["micro_batch_size"] == 0,
            "Effective batch size must be divisible by micro batch size")
    require(all(cfg[k] > 0 for k in ["micro_batch_size", "effective_batch_size", "epochs", "learning_rate", "bootstrap_samples"]),
            "Training sizes, learning rate, and bootstrap count must be positive")
    require(cfg["max_prompt_tokens"] + cfg["max_new_tokens"] <= 1024,
            "Generation exceeds GPT-2 context")
    return cfg


PREPARATION_KEYS = {
    "model_id", "dataset_id", "dataset_file", "seed", "max_sequence_tokens",
    "max_prompt_tokens", "max_response_tokens", "near_duplicate_similarity",
    "ranking_examples", "generation_examples", "negative_length_ratio", "hard_min_similarity",
}


def same_preparation_config(left, right):
    return {key: left[key] for key in PREPARATION_KEYS} == {key: right[key] for key in PREPARATION_KEYS}
