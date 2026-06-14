"""
Chat sampling script for the fine-tuned model.

Loads a model fine-tuned by train.py on the conversation data from
prepare_finetune.py, wraps each prompt in the chat template, and samples the
assistant's reply (stopping at <|assistant_end|>).

Usage:
    $ python sample_chat.py
    $ python sample_chat.py --arch gpt2 --checkpoint gpt2_chat.pth --num-samples 3
"""

import argparse

import torch
from torch.nn import functional as F

from chat_tokenizer import SPECIAL_TOKENS, get_encoding


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
    p.add_argument("--checkpoint", default=None, help="weights path (default: <arch>_chat.pth)")
    p.add_argument("--num-samples", type=int, default=3, help="replies sampled per prompt")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = f"{args.arch}_chat.pth"
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
    bos        = SPECIAL_TOKENS["<|bos|>"]
    user_start = SPECIAL_TOKENS["<|user_start|>"]
    user_end   = SPECIAL_TOKENS["<|user_end|>"]
    asst_start = SPECIAL_TOKENS["<|assistant_start|>"]
    asst_end   = SPECIAL_TOKENS["<|assistant_end|>"]

    prompts = [
        "What is the capital of France?",
        "What is the chemical symbol for gold?",
        "If yesterday was Friday, what day is tomorrow?",
        "What is the opposite of hot?",
        "Can you name the planets of the solar system?",
        "What's a good color to paint a bedroom?",
        "If 5*x + 3 = 13, what is x?",
    ]

    for prompt in prompts:
        # Chat template: a single user turn, then hand the floor to the assistant.
        ids = [bos, user_start, *enc.encode_ordinary(prompt), user_end, asst_start]
        x = (torch.tensor(ids, dtype=torch.long, device=device)
             .unsqueeze(0)
             .repeat(args.num_samples, 1))                     # (num_samples, T)
        prompt_len = x.size(1)
        max_length = prompt_len + args.max_new_tokens

        # Per-sequence stopping: once a row emits <|assistant_end|> we keep
        # padding it with that token and ignore it, finishing early if all stop.
        finished = torch.zeros(args.num_samples, dtype=torch.bool, device=device)
        while x.size(1) < max_length and not finished.all():
            with torch.no_grad():
                idx_cond = x[:, -model.config.block_size:]
                logits, _ = model(idx_cond)                    # (B, T, vocab_size)
                logits = logits[:, -1, :]                      # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, args.top_k, dim=-1)
                ix = torch.multinomial(topk_probs, 1)          # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix)      # (B, 1)
                xcol[finished] = asst_end
                x = torch.cat((x, xcol), dim=1)
                finished |= xcol.squeeze(1) == asst_end

        print(f"\n=== {prompt} ===")
        for i in range(args.num_samples):
            reply = x[i, prompt_len:].tolist()                 # only the assistant's turn
            if asst_end in reply:
                reply = reply[:reply.index(asst_end)]          # trim at the stop token
            print(i + 1, ">", enc.decode(reply))


if __name__ == "__main__":
    main()
