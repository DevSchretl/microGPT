# Design decisions

Notes on the *why* behind this project — the motivation and reasoning for the
architecture, data, and training choices. For *what* the code does and how to run
it, see the [README](README.md).

## Starting from GPT-2

I built [models/gpt2.py](models/gpt2.py) to match the size and architecture of
OpenAI's GPT-2 (124M), following Andrej Karpathy's *Neural Networks: Zero to Hero*
course. The classic-era components (LayerNorm, GELU MLPs, and learned positional
embeddings) were a deliberate choice to stay faithful to the original design.

## Dataset: FineWeb-Edu

I chose the 10B-token [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
sample for its convenient size and high quality. Its focus on educational, factual
content was meant to give the model a more knowledgeable foundation.

## Scaling up to a GPT-3-style model

I wasn't happy with the sampling quality of the GPT-2 model and wanted to see
whether I could do better on a sub-$50 budget. The result is
[models/gpt3.py](models/gpt3.py): a larger model (more layers, a wider embedding
dimension, and more attention heads) with a set of modern upgrades (RoPE, RMSNorm,
SwiGLU, grouped-query attention, QK-norm, and others) adopted in hopes of more
efficient training and better samples.

## Fine-tuning: smol-smoltalk

To turn the base model into a chatbot, I fine-tuned on
[smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
which is purpose-built for training Q&A chat models.

## Few-shot prompting

Left to itself, the model tended to ramble. I added optional few-shot prompting (a short series of concise Q&A demonstrations prepended to the prompt) to steer it
toward brief, coherent answers. OpenAI's 2020 GPT-3 paper showed that models of all
sizes benefit from few-shot prompting, and that held here: it was probably the
simplest and most effective single improvement to the GPT-3 model, making chatbot
responses noticeably more coherent.
