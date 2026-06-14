"""
Language-model training / fine-tuning script.

Builds a model (gpt2 or gpt3), optionally loads a checkpoint, and trains it on
the tokenized shards produced by prepare.py.

Usage:
    $ python train.py                                  # gpt2, resume gpt2_weights.pth
    $ python train.py --arch gpt3 --scratch            # train gpt3 from random init
    $ python train.py --arch gpt3 --scratch --batch-size 256 --compile-mode max-autotune
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from torch.amp import GradScaler

# -----------------------------------------------------------------------------


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


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    """Linear warmup then cosine decay from max_lr down to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


def load_tokens(filename):
    """Load a tokenized shard from a .npy file and return a long tensor."""
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    """Streams pre-tokenized .npy shards from DATA_DIR.

    Iterates over shards whose filename contains `split` in round-robin order,
    yielding (inputs, targets) batches of shape (B, T).
    """

    def __init__(self, B, T, split, data_dir):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        shards = sorted(s for s in os.listdir(data_dir) if split in s)
        assert shards, f"no '{split}' shards found in {data_dir!r} (run prepare.py first)"
        self.shards = [os.path.join(data_dir, s) for s in shards]
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
            print("Next shard: ", self.current_shard)
        return x, y


# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gpt2", choices=["gpt2", "gpt3"])
    p.add_argument("--checkpoint", default=None, help="weights path to save to (default: <arch>_weights.pth)")
    p.add_argument("--init-from", default=None,
                   help="weights to initialize from; trains and saves to --checkpoint, leaving "
                        "these untouched. Use for finetuning (e.g. --init-from gpt2_weights.pth).")
    p.add_argument("--data-dir", default="edu_fineweb10B",
                   help="tokenized shard directory (use 'smoltalk_chat' for finetuning)")
    p.add_argument("--scratch", action="store_true", help="train from random init instead of loading weights")
    p.add_argument("--batch-size", type=int, default=64, help="micro-batch sequences (raise to fill the GPU)")
    p.add_argument("--seq-len", type=int, default=512, help="training context length (shorter = faster)")
    p.add_argument("--grad-accum", type=int, default=16, help="micro-batches per optimizer step")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=None, help="max LR (default: 6e-4 gpt2, 1e-3 gpt3)")
    p.add_argument("--min-lr-frac", type=float, default=0.1, help="cosine floor as a fraction of max LR")
    p.add_argument("--warmup", type=int, default=715, help="linear warmup steps")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"],
                   help="torch.compile mode. Use 'max-autotune-no-cudagraphs' with gradient "
                        "accumulation: the cudagraph modes (max-autotune, reduce-overhead) can "
                        "crash on the tied embedding when forward runs multiple times per step.")
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = f"{args.arch}_weights.pth"
    if args.lr is None:
        args.lr = 6e-4
    return args


def main():
    args = parse_args()

    device = get_device()
    torch.set_float32_matmul_precision('high')

    # Mixed precision: prefer bf16 (fp32 range, no loss scaling). Fall back to
    # fp16 + GradScaler only where bf16 isn't available (older CUDA GPUs, mps).
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    elif device == "mps":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16
    use_scaler = (device == "cuda" and amp_dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print(f"using device: {device} | arch: {args.arch} | precision: {amp_dtype} "
          f"| batch: {args.batch_size}x{args.seq_len} | grad_accum: {args.grad_accum} "
          f"| tokens/step: {tokens_per_step} | max_lr: {args.lr}")

    raw_model = build_model(args.arch).to(device)
    # Init source defaults to the save path (resume), but --init-from lets
    # finetuning load a base checkpoint while saving elsewhere.
    init_path = args.init_from or args.checkpoint
    if args.scratch:
        print("training from scratch")
    elif os.path.exists(init_path):
        checkpoint = torch.load(init_path, map_location=device)
        state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        raw_model.load_state_dict(state)
        print(f"loaded weights from {init_path}")
    elif args.init_from:
        raise FileNotFoundError(f"--init-from {args.init_from!r} not found")
    else:
        print("training from scratch")

    # Configure the optimizer on the raw params, then (optionally) compile for
    # speed. torch.compile shares the same Parameter tensors, so the optimizer
    # updates the right weights and we can save raw_model's clean state dict.
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.lr, device_type=device,
    )
    model = raw_model if args.no_compile else torch.compile(raw_model, mode=args.compile_mode)

    min_lr = args.lr * args.min_lr_frac
    train_loader = DataLoaderLite(B=args.batch_size, T=args.seq_len, split='train', data_dir=args.data_dir)
    val_loader   = DataLoaderLite(B=args.batch_size, T=args.seq_len, split='val',   data_dir=args.data_dir)

    for step in range(args.steps):
        t0 = time.time()

        if True:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device, dtype=amp_dtype):
                        _, loss = model(x, y)
                    val_loss_accum += loss.detach() / val_loss_steps
            print(f"validation loss: {val_loss_accum.item():.4f}")
            model.train()

        # Gradient accumulation: average the loss over grad_accum micro-batches.
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss_accum += loss.detach()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

        lr = get_lr(step, args.lr, min_lr, args.warmup, args.steps)
        for group in optimizer.param_groups:
            group['lr'] = lr

        scaler.step(optimizer)
        scaler.update()

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (args.batch_size * args.seq_len * args.grad_accum) / (t1 - t0)

        if step % 100 == 0:
            print(f"step {step} | loss: {loss_accum.item():.4f} | lr: {lr:.2e} "
                  f"| norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            
        if step % 5000 == 0:
            torch.save(raw_model.state_dict(), args.checkpoint)
            print(f"Model weights saved to {args.checkpoint}!")

    torch.save(raw_model.state_dict(), args.checkpoint)
    print(f"Model weights saved to {args.checkpoint}!")


if __name__ == "__main__":
    main()
