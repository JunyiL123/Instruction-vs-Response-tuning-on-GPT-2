"""Download the pinned GPT-2 checkpoint for the checked-in POC data."""

from huggingface_hub import snapshot_download

from .common import read_json, write_json
from .data import verify


def setup(root):
    manifest = verify(root)
    source = manifest["sources"]
    model_path = snapshot_download(
        source["model_id"],
        revision=source["model_revision"],
        allow_patterns=["config.json", "generation_config.json", "model.safetensors",
                        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    )
    write_json(root / "raw" / "sources.json", {
        "model_id": source["model_id"],
        "dataset_id": source["dataset_id"],
        "model_revision": source["model_revision"],
        "dataset_revision": source["dataset_revision"],
        "model_path": model_path,
    })
    print(f"Ready to run with GPT-2 revision {source['model_revision']}")
