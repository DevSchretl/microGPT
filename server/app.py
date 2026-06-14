"""
FastAPI backend that serves the chat model to the web frontend.

Discovers the available *_chat.pth checkpoints, loads each one lazily (and only
once), and exposes:
  GET  /api/models  -> which chat models are available
  POST /api/chat    -> stream an assistant reply as Server-Sent Events
It also serves the built frontend (web/dist) when that directory exists.
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.inference import ChatModel

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = "_chat.pth"


def discover_checkpoints():
    """Map arch name -> checkpoint path for every *_chat.pth in the project root."""
    found = {}
    for path in sorted(ROOT.glob(f"*{SUFFIX}")):
        arch = path.name[: -len(SUFFIX)]
        if arch in ("gpt2", "gpt3"):
            found[arch] = path
    return found


_checkpoints = discover_checkpoints()
_models = {}  # arch -> ChatModel, loaded on first use


def get_model(arch):
    if arch not in _models:
        _models[arch] = ChatModel(arch, str(_checkpoints[arch]))
    return _models[arch]


app = FastAPI(title="GPT Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    max_new_tokens: int = 200
    top_k: int = 50


@app.get("/api/models")
def list_models():
    return {"models": list(_checkpoints.keys())}


@app.post("/api/chat")
def chat(req: ChatRequest):
    arch = req.model or (next(iter(_checkpoints), None))

    def sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    if arch not in _checkpoints:
        return StreamingResponse(
            iter([sse({"error": f"unknown model: {arch}"})]),
            media_type="text/event-stream",
        )

    model = get_model(arch)
    messages = [m.model_dump() for m in req.messages]
    max_new_tokens = max(1, min(req.max_new_tokens, 1000))
    top_k = max(1, min(req.top_k, 200))

    def event_stream():
        try:
            for delta in model.generate_stream(messages, max_new_tokens, top_k):
                yield sse({"delta": delta})
            yield sse({"done": True})
        except Exception as e:  # surface generation errors to the client
            yield sse({"error": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Serve the built frontend if present (production single-origin deploy).
# Mounted last so it does not shadow the /api routes above.
_dist = ROOT / "web" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
