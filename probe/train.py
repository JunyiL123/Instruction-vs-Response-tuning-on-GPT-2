import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import device_for, digest, read_json, read_jsonl, require, same_preparation_config, seed_all, write_json
from .data import verify
from .modeling import collate, encode_example, prompt_ids, response_loss_sum


def load_base(root, device):
    sources = read_json(root / "raw" / "sources.json")
    tokenizer = AutoTokenizer.from_pretrained(sources["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(sources["model_path"], local_files_only=True,
                                                attn_implementation="eager").to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def validation_loss(model, encoded, tokenizer, device, batch_size, pad_length=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = move_batch(collate(encoded[start:start+batch_size], tokenizer.eos_token_id, pad_length), device)
            loss, count = response_loss_sum(model, batch)
            total_loss += loss.item()
            total_tokens += count
    return total_loss / total_tokens


def train(cfg, root, mode, requested_device, smoke_steps=0):
    manifest = verify(root)
    require(same_preparation_config(cfg, manifest["config"]), "Preparation config differs from the frozen dataset")
    output = root / "smoke" / f"{mode}-fixed-micro{cfg['micro_batch_size']}-steps{smoke_steps}" if smoke_steps else root / "runs" / mode
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "training.json"
    if meta_path.exists():
        existing = read_json(meta_path)
        require(existing["experiment_id"] == manifest["experiment_id"] and same_preparation_config(existing["config"], cfg),
                "Existing training run belongs to a different experiment")
        if existing.get("complete"):
            print(f"{mode} is already complete: {output}", flush=True)
            return
    device = device_for(requested_device)
    seed_all(cfg["seed"])
    model, tokenizer = load_base(root, device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    rows = read_jsonl(root / "data" / "train.jsonl")
    val_rows = read_jsonl(root / "data" / "validation.jsonl")
    encoded = [encode_example(tokenizer, r, mode) for r in rows]
    validation = [encode_example(tokenizer, r, mode) for r in val_rows]
    # MPS graph caches otherwise grow across hundreds of distinct sequence lengths.
    pad_length = max(len(r['input_ids']) for r in encoded + validation) if device.type == 'mps' else None
    if smoke_steps:
        # A smoke run never replaces a scientific checkpoint.
        encoded = encoded[:cfg["effective_batch_size"] * smoke_steps]
        validation = validation[:cfg["effective_batch_size"]]
    effective = cfg["effective_batch_size"]
    micro = cfg["micro_batch_size"]
    steps_per_epoch = math.ceil(len(encoded) / effective)
    epochs = 1 if smoke_steps else cfg["epochs"]
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * cfg["warmup_fraction"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_loss, start_epoch, global_step = float("inf"), 0, 0
    history = []
    latest = output / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        require(checkpoint["experiment_id"] == manifest["experiment_id"], "Checkpoint belongs to another experiment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        start_epoch, global_step, best_loss = checkpoint["epoch"], checkpoint["step"], checkpoint["best_loss"]
        history = checkpoint["history"]
        del checkpoint
    started = time.monotonic()
    code_hash = digest({p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")})
    meta = {"experiment_id": manifest["experiment_id"], "mode": mode, "smoke": bool(smoke_steps),
            "config": cfg, "device": str(device), "code_hash": code_hash, "torch_version": str(torch.__version__),
            "training_ids_hash": digest([r["id"] for r in rows]), "validation_objective": mode,
            "fixed_padding_length": pad_length, "memory_strategy": "fixed sequence shape on MPS",
            "complete": False, "history": history}
    write_json(meta_path, meta)
    for epoch in range(start_epoch, epochs):
        # Deterministic per-epoch seeding also makes an interrupted epoch replayable.
        seed_all(cfg["seed"] + epoch)
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        order = torch.randperm(len(encoded), generator=generator).tolist()
        model.train()
        running_loss, running_tokens = 0.0, 0
        for start in range(0, len(order), effective):
            indices = order[start:start+effective]
            token_count = sum(sum(t != -100 for t in encoded[i]["labels"]) for i in indices)
            lr_scale = min((global_step + 1) / warmup, max(0.0, (total_steps - global_step) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = cfg["learning_rate"] * lr_scale
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(indices), micro):
                batch = move_batch(collate([encoded[i] for i in indices[offset:offset+micro]], tokenizer.eos_token_id, pad_length), device)
                loss, count = response_loss_sum(model, batch)
                require(bool(torch.isfinite(loss).item()), "Non-finite training loss")
                (loss / token_count).backward()
                running_loss += loss.detach().item()
                running_tokens += count
                del loss, batch
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            if device.type == "mps":
                torch.mps.empty_cache()
            global_step += 1
            if global_step % 10 == 0 or smoke_steps:
                print(f"{mode}: epoch {epoch+1}/{epochs}, update {global_step}/{total_steps}, "
                      f"response NLL {running_loss/running_tokens:.4f}, elapsed {time.monotonic()-started:.1f}s", flush=True)
        val = validation_loss(model, validation, tokenizer, device, micro, pad_length)
        entry = {"epoch": epoch+1, "update": global_step, "train_response_nll": running_loss/running_tokens,
                 "validation_response_nll": val, "elapsed_seconds": time.monotonic()-started}
        history.append(entry)
        if val < best_loss:
            best_loss = val
            model.save_pretrained(output / "best", safe_serialization=True)
            tokenizer.save_pretrained(output / "best")
            write_json(output / "best" / "selection.json", entry)
        # Save epoch boundaries only; interruption replays at most the current epoch.
        if not smoke_steps:
            tmp = output / "latest.tmp"
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "optimizer": optimizer.state_dict(),
                        "experiment_id": manifest["experiment_id"], "epoch": epoch+1, "step": global_step,
                        "best_loss": best_loss, "history": history}, tmp)
            tmp.replace(latest)
        meta.update(history=history, best_validation_response_nll=best_loss,
                    complete=epoch+1 == epochs, updates=global_step, elapsed_seconds=time.monotonic()-started)
        write_json(meta_path, meta)
        print(f"{mode}: epoch {epoch+1}, validation NLL {val:.4f}", flush=True)


def load_condition(root, condition, device):
    if condition == "base":
        return load_base(root, device)
    run = root / "runs" / condition
    meta = read_json(run / "training.json")
    require(meta.get("complete") and not meta.get("smoke"), f"{condition} training is incomplete")
    require(meta["experiment_id"] == read_json(root / "data" / "manifest.json")["experiment_id"], "Checkpoint data mismatch")
    model = AutoModelForCausalLM.from_pretrained(run / "best", local_files_only=True,
                                                attn_implementation="sdpa").to(device)
    tokenizer = AutoTokenizer.from_pretrained(run / "best", local_files_only=True)
    return model, tokenizer
