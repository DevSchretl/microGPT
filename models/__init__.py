"""
Model architectures and the chat tokenizer.

Re-exports the public surface so callers can write `from models import GPT,
GPT3, get_encoding, ...` without reaching into the individual submodules:

  * gpt2.py     -- GPT-2 (124M): GPT / GPTConfig
  * gpt3.py     -- GPT-3-style (254M): GPT3 / GPT3Config
  * tokenizer.py -- GPT-2 tiktoken encoding extended with chat special tokens
"""

from .gpt2 import GPT, GPTConfig
from .gpt3 import GPT3, GPT3Config
from .tokenizer import SPECIAL_TOKENS, get_encoding

__all__ = [
    "GPT",
    "GPTConfig",
    "GPT3",
    "GPT3Config",
    "SPECIAL_TOKENS",
    "get_encoding",
]
