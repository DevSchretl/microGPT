"""
Text generation script.

Builds the model (see gpt2.py), loads its checkpoint, encodes a prompt, and
samples continuations with top-k sampling.

Usage:
    $ python sample.py
    $ python sample.py --prompt "The pyramids are" --max-new-tokens 50
"""

import argparse

import torch
from torch.nn import functional as F

from chat_tokenizer import get_encoding


def build_model(arch):
    """Instantiate a fresh model for the given architecture name."""
    if arch == "gpt2":
        from gpt2 import GPT, GPTConfig
        return GPT(GPTConfig())
    if arch == "gpt3":
        from gpt3 import GPT3, GPT3Config
        return GPT3(GPT3Config())
    raise ValueError(f"unknown arch: {arch}")


def get_device():
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gpt3", choices=["gpt2", "gpt3"])
    p.add_argument("--checkpoint", default=None, help="weights path (default: <arch>_weights.pth)")
    p.add_argument("--prompt", default="The pyramids are")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--top-k", type=int, default=50)
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

    prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]

    enc = get_encoding()

    for prompt in prompts:
        prompt_tokens = enc.encode(prompt)
        # Repeat the prompt into a batch so we draw num_samples continuations.
        # Rows are identical (same length), so no padding/masking is needed.
        x = (torch.tensor(prompt_tokens, dtype=torch.long, device=device)
             .unsqueeze(0)
             .repeat(args.num_samples, 1))                    # (num_samples, T)

        max_length = len(prompt_tokens) + args.max_new_tokens
        while x.size(1) < max_length:
            with torch.no_grad():
                # Crop to the model's context window in case the prompt is long.
                idx_cond = x[:, -model.config.block_size:]
                logits, _ = model(idx_cond)                   # (B, T, vocab_size)
                logits = logits[:, -1, :]                     # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, args.top_k, dim=-1)
                ix = torch.multinomial(topk_probs, 1)         # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix)     # (B, 1)
                x = torch.cat((x, xcol), dim=1)

        print(f"\n=== {prompt!r} ===")
        for i in range(args.num_samples):
            tokens = x[i, :max_length].tolist()
            print(i+1, ">", enc.decode(tokens))


if __name__ == "__main__":
    main()
