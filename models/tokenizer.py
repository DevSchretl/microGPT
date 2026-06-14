import tiktoken

SPECIAL_TOKENS = {
    "<|bos|>":             50257,
    "<|user_start|>":      50258,
    "<|user_end|>":        50259,
    "<|assistant_start|>": 50260,
    "<|assistant_end|>":   50261,
    "<|system_start|>":    50262,
    "<|system_end|>":      50263,
}


def get_encoding():
    """Return a gpt2 tiktoken encoding extended with the chat special tokens."""
    base = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **SPECIAL_TOKENS},
    )