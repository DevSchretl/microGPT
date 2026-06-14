"""
Shared helpers for the entry-point scripts (train.py, sample.py,
sample_chat.py, eval_hellaswag.py) and the web backend.

Holds the two bits of glue every runnable script needs -- mapping an
architecture name to a fresh model, and picking the best available device --
so the logic lives in exactly one place instead of being copied per script.
"""

import torch

from models import GPT, GPT3, GPTConfig, GPT3Config


def build_model(arch):
    """Instantiate a fresh model for the given architecture name."""
    if arch == "gpt2":
        return GPT(GPTConfig())
    if arch == "gpt3":
        return GPT3(GPT3Config())
    raise ValueError(f"unknown arch: {arch}")


def get_device():
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
