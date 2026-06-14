"""
smol-smoltalk conversation preprocessor (supervised fine-tuning data).
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

Algorithmically identical to prepare.py: it tokenizes documents and writes
uint16 .npy shards that train.py / DataLoaderLite consume unchanged. The only
difference is the documents are multi-turn conversations rather than raw text,
so each turn is wrapped in chat special tokens and the model learns to predict
across that structure.

Run simply as:
    $ python prepare_finetune.py
Will save shards to the local directory "smoltalk_chat".

To fine-tune on the result, point train.py at this directory (its DATA_DIR).

Each conversation is rendered as a flat token stream, e.g.:

    <|bos|><|system_start|>You are a helpful assistant.<|system_end|>
    <|user_start|>What is the color of the sky?<|user_end|>
    <|assistant_start|>Red. Wait, possibly blue.<|assistant_end|>
    <|user_start|>lol<|user_end|><|assistant_start|>...<|assistant_end|>

Notes / assumptions:
  - The special tokens carry the structure, so no literal newlines are inserted
    between turns (the line breaks above are only for readability).
  - <|bos|> begins each conversation, playing the same role <|endoftext|> played
    as a document delimiter in pretraining.
  - Loss is computed over every token (same as pretraining); there is no
    assistant-only masking, per "algorithmically identical to pretraining".
  - smol-smoltalk stores each row as a `messages` list of {"role", "content"}
    dicts with "system", "user", and "assistant" roles.
"""

import os
import multiprocessing as mp

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from chat_tokenizer import SPECIAL_TOKENS, get_encoding

local_dir = "smoltalk_chat"
# Smaller than pretraining's 100M: the SFT set is far smaller, so a smaller
# shard guarantees both a val shard (index 0) and at least one train shard.
shard_size = int(1e7)  # 10M tokens per shard

# Module-level so multiprocessing worker processes can access these after import.
# Special-token ids live in chat_tokenizer.py so prep and sampling never drift.
enc = get_encoding()
bos = SPECIAL_TOKENS["<|bos|>"]
ROLE_TOKENS = {
    "system":    (SPECIAL_TOKENS["<|system_start|>"], SPECIAL_TOKENS["<|system_end|>"]),
    "user":      (SPECIAL_TOKENS["<|user_start|>"], SPECIAL_TOKENS["<|user_end|>"]),
    "assistant": (SPECIAL_TOKENS["<|assistant_start|>"], SPECIAL_TOKENS["<|assistant_end|>"]),
}


def tokenize(doc):
    """Tokenize one conversation into a chat-formatted uint16 token array.

    Prepends <|bos|>, then wraps each turn's content between its role's
    start/end tokens. Content is encoded with encode_ordinary so any literal
    "<|...|>" text in the data is treated as plain text, never a special token.
    """
    tokens = [bos]
    for message in doc["messages"]:
        role = message["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unexpected role {role!r}; expected one of {list(ROLE_TOKENS)}")
        start, end = ROLE_TOKENS[role]
        tokens.append(start)
        tokens.extend(enc.encode_ordinary(message["content"]))
        tokens.append(end)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), \
        "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    """Save a tokenized shard to disk as a .npy file."""
    np.save(filename, tokens_np)


if __name__ == "__main__":
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")

    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, ds, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"smoltalk_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"smoltalk_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
