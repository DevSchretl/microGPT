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


# -----------------------------------------------------------------------------
# Few-shot prompting
#
# Editable in-context demonstration sets used by the sampling scripts when run
# with --num-shots > 0 (eval_hellaswag.py draws its shots from the dataset
# instead). The model is small, so showing a few worked examples biases it toward
# the demonstrated pattern far more effectively than instructing it in prose.
#
#   * FEWSHOT_TEXT -- completed statements prepended by sample.py (a base model
#     just continues text, so the demos are plain completed sentences).
#   * FEWSHOT_CHAT -- (user, assistant) turns prepended by sample_chat.py.
#
# Edit these freely; the scripts take the first N entries.
# -----------------------------------------------------------------------------

FEWSHOT_TEXT = [
    "The capital of Japan is Tokyo.",
    "The chemical symbol of oxygen is O.",
    "The opposite of up is down.",
    "If yesterday was Monday, then tomorrow will be Wednesday.",
    "If 2*x + 4 = 10, then x is 3.",
]

FEWSHOT_CHAT = [
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the chemical symbol for oxygen?", "The chemical symbol for oxygen is O."),
    ("What is the largest planet in the solar system?", "Jupiter is the largest planet in the solar system."),
    ("Translate 'dog' into Spanish.", "'Dog' is 'perro' in Spanish."),
    ("If yesterday was Monday, what day is tomorrow?", "Tomorrow is Wednesday."),
]


def fewshot_text_prefix(num_shots):
    """Return the first `num_shots` FEWSHOT_TEXT demos as a newline-joined prefix.

    Empty string when num_shots <= 0, so callers can prepend unconditionally.
    """
    if num_shots <= 0:
        return ""
    return "".join(demo + "\n" for demo in FEWSHOT_TEXT[:num_shots])
