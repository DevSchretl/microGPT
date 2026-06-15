"""
Inference wrapper around the fine-tuned chat model for the web backend.

Reuses the same model construction and chat template as the sample_chat.py CLI
so the web server generates replies exactly the way the CLI does, but keeps the
model loaded across requests and streams tokens out as they are produced.
"""

import threading

import torch
from torch.nn import functional as F

from common import build_model, get_device
from models import SPECIAL_TOKENS, get_encoding

# Real GPT-2 vocab is 0..50256; 50257+ are the chat special tokens (and padding
# up to the model's padded vocab_size), which have no printable byte form.
FIRST_SPECIAL_TOKEN = 50257

# Few-shot priming: a couple of short example turns are prepended to every prompt
# so the (small, 254M) model imitates the concise pattern instead of rambling --
# demonstration works far better than a natural-language "be brief" instruction
# for a model this size. The examples are prompt-only (never shown in the UI), and
# the length of these answers is the knob for how long real replies come out:
# keep them concise but complete (one full sentence) to avoid terse non-answers.
FEWSHOT_EXAMPLES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Why is the sky blue?", "Sunlight scatters off air molecules, and blue scatters the most."),
    ("Any tips for staying focused?", "Work in short blocks and put your phone out of reach."),
]


class ChatModel:
    """A loaded chat model that builds prompts from history and streams replies."""

    def __init__(self, arch, checkpoint, device=None):
        self.device = device or get_device()
        self.model = build_model(arch).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

        self.enc = get_encoding()
        self.bos        = SPECIAL_TOKENS["<|bos|>"]
        self.user_start = SPECIAL_TOKENS["<|user_start|>"]
        self.user_end   = SPECIAL_TOKENS["<|user_end|>"]
        self.asst_start = SPECIAL_TOKENS["<|assistant_start|>"]
        self.asst_end   = SPECIAL_TOKENS["<|assistant_end|>"]
        self.block_size = self.model.config.block_size

        # one model on one device can't run two generations at once
        self._lock = threading.Lock()

        # Encode the few-shot priming turns once; prepended to every prompt below.
        self.fewshot_ids = []
        for q, a in FEWSHOT_EXAMPLES:
            self.fewshot_ids += self._encode_turn("user", q) + self._encode_turn("assistant", a)

    def _encode_turn(self, role, content):
        body = self.enc.encode_ordinary(content)
        if role == "user":
            return [self.user_start, *body, self.user_end]
        return [self.asst_start, *body, self.asst_end]

    def build_prompt(self, messages, max_new_tokens):
        """Tokenize the conversation, dropping oldest whole turns to fit the window."""
        # leave room for <|bos|>, the trailing <|assistant_start|>, and the
        # always-kept few-shot priming turns
        budget = self.block_size - max_new_tokens - 2 - len(self.fewshot_ids)
        turns = [self._encode_turn(m["role"], m["content"]) for m in messages]

        kept, total = [], 0
        for turn in reversed(turns):  # keep the most recent turns that fit
            if total + len(turn) > budget:
                break
            kept.append(turn)
            total += len(turn)
        kept.reverse()

        ids = [self.bos, *self.fewshot_ids]  # prime with the concise examples
        for turn in kept:
            ids.extend(turn)
        ids.append(self.asst_start)  # hand the floor to the assistant
        return ids

    def generate_stream(self, messages, max_new_tokens=200, top_k=50):
        """Yield decoded text chunks of the assistant's reply as tokens are sampled."""
        with self._lock:
            ids = self.build_prompt(messages, max_new_tokens)
            x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

            byte_buf = b""  # holds bytes of an in-progress multibyte character
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    idx_cond = x[:, -self.block_size:]
                    logits, _ = self.model(idx_cond)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                    ix = torch.multinomial(topk_probs, 1)
                    next_id = torch.gather(topk_indices, -1, ix)

                tok = next_id.item()
                if tok >= FIRST_SPECIAL_TOKEN:  # <|assistant_end|> or any special token: stop
                    break
                x = torch.cat((x, next_id), dim=1)

                # decode safely: only emit once we have complete UTF-8 characters
                byte_buf += self.enc.decode_single_token_bytes(tok)
                try:
                    text = byte_buf.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # wait for the rest of a multibyte character
                byte_buf = b""
                if text:
                    yield text

            if byte_buf:  # flush any trailing bytes, ignoring an incomplete tail
                tail = byte_buf.decode("utf-8", errors="ignore")
                if tail:
                    yield tail
