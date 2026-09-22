"""One tokenization and masking implementation shared by all conditions."""
from .common import require

RESPONSE_PREFIX = "Response:\n"


def prompt_text(row):
    text = "Instruction:\n" + row["instruction"]
    if row.get("context"):
        text += "\n\nContext:\n" + row["context"]
    return text + "\n\n" + RESPONSE_PREFIX


def prompt_ids(tokenizer, row=None):
    text = RESPONSE_PREFIX if row is None else prompt_text(row)
    return [tokenizer.eos_token_id] + tokenizer.encode(text, add_special_tokens=False)


def response_ids(tokenizer, response, include_eos=True):
    ids = tokenizer.encode(response, add_special_tokens=False)
    return ids + ([tokenizer.eos_token_id] if include_eos else [])


def encode_example(tokenizer, row, mode):
    require(mode in {"instruction", "response"}, "Unknown training mode")
    prefix = prompt_ids(tokenizer, row if mode == "instruction" else None)
    target = response_ids(tokenizer, row["response"])
    # Explicit segment tokenization avoids a BPE token crossing the loss boundary.
    return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


def collate(rows, pad_id, pad_to_length=None):
    import torch
    size = max(len(r["input_ids"]) for r in rows)
    if pad_to_length is not None:
        require(size <= pad_to_length, 'Example exceeds fixed batch length')
        size = pad_to_length
    return {
        "input_ids": torch.tensor([r["input_ids"] + [pad_id] * (size - len(r["input_ids"])) for r in rows]),
        "labels": torch.tensor([r["labels"] + [-100] * (size - len(r["labels"])) for r in rows]),
        "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (size - len(r["input_ids"])) for r in rows]),
    }


def token_logps(logits, input_ids):
    import torch
    # logsumexp avoids allocating a second full [batch, length, vocabulary] tensor.
    shifted = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    return shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(shifted, -1)


def response_loss_sum(model, batch):
    scores = token_logps(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              use_cache=False).logits, batch["input_ids"])
    mask = batch["labels"][:, 1:] != -100
    return -(scores * mask).sum(), int(mask.sum().item())


def score_response(model, tokenizer, response, device, row=None):
    import torch
    prefix = prompt_ids(tokenizer, row)
    target = response_ids(tokenizer, response)
    ids = torch.tensor([prefix + target], device=device)
    with torch.inference_mode():
        lp = token_logps(model(input_ids=ids, use_cache=False).logits, ids)[0, len(prefix)-1:]
    values = lp.cpu().tolist()
    return {"sum": sum(values[:-1]), "mean": sum(values[:-1]) / (len(values)-1),
            "sum_with_eos": sum(values), "tokens": len(values)-1}
