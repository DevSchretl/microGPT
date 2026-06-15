"""
HellaSwag evaluation script.

Scores a trained model (see models/) on HellaSwag, the standard
likelihood-based benchmark for base language models of this size (the same one
used in Karpathy's build-nanogpt). No text is generated: each example gives a
context and 4 candidate endings, and the model picks the ending it finds most
likely (lowest average per-token loss). Accuracy is how often that matches the
gold label. Random chance is 25%.

Two metrics are reported:
  * acc_norm -- prediction by lowest *average* per-token loss (length-normalized;
    the headline HellaSwag number)
  * acc      -- prediction by lowest *total* loss over the ending

Usage:
    $ python eval_hellaswag.py
    $ python eval_hellaswag.py --arch gpt3 --limit 300
    $ python eval_hellaswag.py --num-shots 5    # 5-shot (shots from the train split)
"""

import argparse
import json
import os

import requests
import torch
from torch.nn import functional as F
from tqdm import tqdm

from common import build_model, get_device
from models import get_encoding

# Canonical HellaSwag data (rowanz/hellaswag). val = 10,042 examples.
DATA_URLS = {
    "val":   "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "test":  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def download(split, data_dir):
    """Download the HellaSwag jsonl for `split` into data_dir (skip if present)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"hellaswag_{split}.jsonl")
    if os.path.exists(path):
        return path
    url = DATA_URLS[split]
    print(f"downloading {url} -> {path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=f"hellaswag_{split}"
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                bar.update(f.write(chunk))
    return path


def iter_examples(path):
    """Yield parsed HellaSwag examples (one JSON object per line)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_fewshot_prefix(train_path, enc, num_shots):
    """Encode `num_shots` solved train examples (context + correct ending) into a
    single shared in-context prefix to prepend to every evaluated example."""
    shots = []
    for ex in iter_examples(train_path):
        if len(shots) >= num_shots:
            break
        shots.append(ex["ctx"] + " " + ex["endings"][ex["label"]])
    return enc.encode("".join(s + "\n\n" for s in shots))


def render_example(example, enc, prefix_tokens=()):
    """Turn one example into (tokens, mask, label).

    tokens: (4, T) padded token ids for "[few-shot prefix +] context + ending".
    mask:   (4, T) 1.0 on the ending tokens (where the loss is measured), else 0.
    label:  index of the correct ending.
    """
    ctx_tokens = enc.encode(example["ctx"])
    rows, masks = [], []
    for ending in example["endings"]:
        # leading space so the ending joins the context as natural text
        end_tokens = enc.encode(" " + ending)
        rows.append([*prefix_tokens, *ctx_tokens, *end_tokens])
        masks.append([0] * (len(prefix_tokens) + len(ctx_tokens)) + [1] * len(end_tokens))

    max_len = max(len(r) for r in rows)
    tokens = torch.zeros(4, max_len, dtype=torch.long)
    mask = torch.zeros(4, max_len, dtype=torch.long)
    for i, (row, m) in enumerate(zip(rows, masks)):
        tokens[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(m)] = torch.tensor(m, dtype=torch.long)
    return tokens, mask, example["label"]


@torch.no_grad()
def predict(model, tokens, mask):
    """Return (pred_norm, pred_sum): the argmin-loss ending under each metric."""
    logits, _ = model(tokens)  # (4, T, vocab_size)
    # shift so position t predicts token t+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_tokens = tokens[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1),
        reduction="none",
    ).view(shift_tokens.size())  # (4, T-1)

    masked = losses * shift_mask  # zero out context + padding
    sum_loss = masked.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return int(avg_loss.argmin().item()), int(sum_loss.argmin().item())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gpt3", choices=["gpt2", "gpt3"])
    p.add_argument("--checkpoint", default=None, help="weights path (default: <arch>_weights.pth)")
    p.add_argument("--split", default="val", choices=["val", "train", "test"])
    p.add_argument("--limit", type=int, default=0, help="evaluate at most N examples (0 = all)")
    p.add_argument("--data-dir", default="hellaswag", help="where to cache the dataset")
    p.add_argument("--num-shots", type=int, default=0,
                   help="prepend N solved train-split examples as an in-context prefix (0 = zero-shot)")
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = f"{args.arch}_weights.pth"
    return args


def main():
    args = parse_args()

    device = get_device()
    print(f"using device: {device} | arch: {args.arch}")

    model = build_model(args.arch).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state)
    model.eval()

    enc = get_encoding()
    path = download(args.split, args.data_dir)

    # Optional k-shot prefix built from solved train-split examples, shared across
    # every evaluated example (only the candidate ending is ever scored).
    prefix_tokens = ()
    if args.num_shots > 0:
        train_path = download("train", args.data_dir)
        prefix_tokens = build_fewshot_prefix(train_path, enc, args.num_shots)
        print(f"few-shot: {args.num_shots}-shot prefix from train ({len(prefix_tokens)} tokens)")

    total = correct_norm = correct = skipped = 0
    bar = tqdm(iter_examples(path), desc="hellaswag")
    for example in bar:
        if args.limit and total >= args.limit:
            break
        tokens, mask, label = render_example(example, enc, prefix_tokens)
        # HellaSwag rows are short, but skip anything past the context window.
        if tokens.size(1) > model.config.block_size:
            skipped += 1
            continue
        tokens, mask = tokens.to(device), mask.to(device)
        pred_norm, pred_sum = predict(model, tokens, mask)

        total += 1
        correct_norm += int(pred_norm == label)
        correct += int(pred_sum == label)
        bar.set_postfix(acc_norm=f"{correct_norm / total:.4f}", n=total)

    if total == 0:
        print("no examples evaluated")
        return

    print(f"\n{args.arch}  acc_norm {correct_norm / total:.4f}  acc {correct / total:.4f}  (N={total})")
    if skipped:
        print(f"skipped {skipped} example(s) longer than block_size={model.config.block_size}")


if __name__ == "__main__":
    main()
