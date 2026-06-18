# GPTs

Train small GPT language models from scratch in PyTorch, then chat with them.

This is a compact, nanoGPT-style codebase for pretraining decoder-only transformers on the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) corpus and, optionally, supervised fine-tuning them into a chat model on [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk). Two architectures are included:

- **GPT-2 (124M)** — the classic architecture: learned position embeddings, LayerNorm, GELU MLP. See [models/gpt2.py](models/gpt2.py).
- **GPT-3-style (254M)** — GPT-2's shape with well-established modern upgrades: RoPE, RMSNorm, SwiGLU, grouped-query attention (GQA), QK-norm, and no biases. Sized at ~40 tokens/parameter for the 10B-token FineWeb-Edu sample. See [models/gpt3.py](models/gpt3.py).

Both models expose the same surface (`forward(idx, targets=None) -> (logits, loss)`, `configure_optimizers`, `config.block_size`), so the same training and sampling scripts drive either one.

For the reasoning behind the architecture, dataset, and training choices, see [DECISIONS.md](DECISIONS.md).

## Performance Progression

Accuracy on [HellaSwag](https://rowanz.github.io/hellaswag/) (`acc_norm`, the
length-normalized headline metric), measured with [eval_hellaswag.py](eval_hellaswag.py).
Random chance is 25%; OpenAI's GPT-2 (124M) scores ~29.7% for reference.

| Model | Setup | HellaSwag |
| --- | --- | ---: |
| **GPT-2 Base** | 124M params, 10B FineWeb-Edu tokens | 31% |
| **GPT-3 Base** | 254M params, 10B FineWeb-Edu tokens | **38%** |
| GPT-3 Base | + 3-shot demos | 36% |
| **GPT-3 Chat** | GPT-3 Base + 500M SmolTalk tokens (SFT) | 34% |
| GPT-3 Chat | + 1-shot demo | 34% |
| GPT-3 Chat | + 3-shot demos | 33% |

The pretrained **GPT-3 Base** scores highest on the benchmark. Few-shot demos and
chat fine-tuning don't help the HellaSwag number, but **GPT-3 Chat** gives by far
the most natural conversational replies, especially with a couple of few-shot demos.
Try it in the [live demo](https://devschretl-microgpt.hf.space/).

## Layout

```
models/    model + tokenizer definitions (importable as `from models import ...`)
data/      dataset preprocessors (run as `python data/<script>.py`)
server/    FastAPI backend that serves the chat model
web/        React + Vite chat frontend
*.py       runnable scripts at the repo root (train / sample / eval)
```

| File | Purpose |
| --- | --- |
| [models/gpt2.py](models/gpt2.py) | GPT-2 model definition (`GPT`, `GPTConfig`) |
| [models/gpt3.py](models/gpt3.py) | GPT-3-style model definition (`GPT3`, `GPT3Config`) |
| [models/tokenizer.py](models/tokenizer.py) | GPT-2 tiktoken encoding extended with chat special tokens |
| [data/prepare.py](data/prepare.py) | Download + tokenize FineWeb-Edu into `.npy` shards (pretraining data) |
| [data/prepare_finetune.py](data/prepare_finetune.py) | Tokenize smol-smoltalk conversations into chat-formatted shards (SFT data) |
| [common.py](common.py) | Shared `build_model()` / `get_device()` helpers plus the editable few-shot demo sets (`FEWSHOT_TEXT`, `FEWSHOT_CHAT`) |
| [train.py](train.py) | Training / fine-tuning loop for either architecture |
| [sample.py](sample.py) | Generate text completions from a pretrained model |
| [sample_chat.py](sample_chat.py) | Generate chat replies from a fine-tuned model |
| [eval_hellaswag.py](eval_hellaswag.py) | Score a model on the HellaSwag benchmark |
| [server/](server/) | FastAPI backend (`/api/chat` SSE stream) for the web app |
| [web/](web/) | React + Vite single-page chat frontend |

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended for training (CPU and Apple `mps` are supported for sampling and small runs). Device selection is automatic: `cuda > mps > cpu`.

Install dependencies:

```bash
pip install torch numpy tiktoken datasets tqdm requests
```

## Quickstart

### 1. Prepare the pretraining data

Downloads the 10B-token FineWeb-Edu sample and writes ~100 tokenized shards to `edu_fineweb10B/` (shard 0 is held out for validation):

```bash
python data/prepare.py
```

### 2. Pretrain a model

```bash
# GPT-3-style model from random init
python train.py --arch gpt3 --scratch

# fill a large GPU and use the tuned compile mode
python train.py --arch gpt3 --scratch --batch-size 256 --compile-mode max-autotune-no-cudagraphs

# GPT-2 from scratch
python train.py --arch gpt2 --scratch
```

Weights are checkpointed every 5000 steps to `<arch>_weights.pth`. Re-running `train.py` without `--scratch` resumes from that checkpoint. Key flags:

- `--batch-size` / `--seq-len` / `--grad-accum` — set the effective tokens/step (`batch x seq_len x grad_accum`).
- `--lr`, `--warmup`, `--weight-decay`, `--steps` — optimization schedule (linear warmup then cosine decay).
- `--no-compile` — disable `torch.compile`.

See `python train.py --help` for the full list.

### 3. Sample completions

```bash
python sample.py --arch gpt3
python sample.py --arch gpt3 --prompt "The pyramids are" --max-new-tokens 50
python sample.py --arch gpt3 --num-shots 3   # prepend few-shot demos from common.FEWSHOT_TEXT
```

### 4. Evaluate on HellaSwag

Reproduce the [Performance Progression](#performance-progression) numbers. The
dataset is downloaded and cached on first run:

```bash
python eval_hellaswag.py --arch gpt3
python eval_hellaswag.py --arch gpt3 --num-shots 3   # 3-shot; demos drawn from the train split
```

## Fine-tuning into a chat model

### 1. Prepare conversation data

Tokenizes smol-smoltalk into chat-formatted shards in `smoltalk_chat/`. Each conversation is rendered as a flat token stream with role markers:

```
<|bos|><|system_start|>...<|system_end|><|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>
```

```bash
python data/prepare_finetune.py
```

### 2. Fine-tune from the pretrained checkpoint

`--init-from` loads a base checkpoint but saves elsewhere, leaving the pretrained weights untouched:

```bash
python train.py --arch gpt3 \
    --init-from gpt3_weights.pth \
    --checkpoint gpt3_chat.pth \
    --data-dir smoltalk_chat
```

### 3. Chat with the fine-tuned model

Wraps each prompt in the chat template and samples the assistant's reply, stopping at `<|assistant_end|>`:

```bash
python sample_chat.py --arch gpt3 --checkpoint gpt3_chat.pth
python sample_chat.py --arch gpt3 --checkpoint gpt3_chat.pth --num-shots 1   # prime with example turns
```

## Web chat app

A FastAPI backend ([server/](server/)) plus a React + Vite frontend ([web/](web/))
let you chat with a fine-tuned model in the browser. The backend discovers every
`*_chat.pth` checkpoint in the repo root, loads it on first use, and streams
replies over Server-Sent Events; the frontend renders the conversation.

```bash
# 1. backend (from the repo root) — serves /api on http://localhost:8000
pip install -r server/requirements.txt
uvicorn server.app:app --reload

# 2. frontend (in another terminal) — dev server proxies /api to the backend
cd web
npm install
npm run dev
```

For a single-origin production build, run `npm run build` in `web/` and the
backend will serve the generated `web/dist/` automatically.

## Notes

- **Tokenizer:** GPT-2 BPE (vocab 50,257, padded to 50,304). Chat fine-tuning adds 7 special tokens (ids 50257–50263) defined in [models/tokenizer.py](models/tokenizer.py).
- **Precision:** training prefers bf16 where available, falling back to fp16 + `GradScaler` on older CUDA GPUs and `mps`.
- **Data and weights** (`edu_fineweb10B/`, `smoltalk_chat/`, `*.pth`) are gitignored — regenerate them with the `prepare` scripts and `train.py`.

## Credits

The model and training code follow Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) / [build-nanogpt](https://github.com/karpathy/build-nanogpt) lineage, with the GPT-3-style architecture modernizing it using techniques common to recent open LLMs (Llama, Qwen, etc.).
