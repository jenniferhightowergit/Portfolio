# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal, List, Tuple
from enum import Enum

import os

from .wolf_core import (
    chat_with_wolf_openai,
    chat_with_wolf_mistral,
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve audio files
if not os.path.exists("wolf_audio"):
    os.makedirs("wolf_audio")
app.mount("/audio", StaticFiles(directory="wolf_audio"), name="audio")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ModelMode(str, Enum):
  vanilla = "vanilla"
  romance = "romance"


class ChatRequest(BaseModel):
  messages: List[Message]
  mode: ModelMode = ModelMode.vanilla  # default


class ChatResponse(BaseModel):
    reply: str
    audio_url: str | None = None


def messages_to_history_pairs(messages: List[Message]) -> List[Tuple[str, str]]:
    """
    Convert [ {role, content}... ] to [(user, assistant), ...]
    assuming alternating user/assistant pairs.
    """
    pairs: List[Tuple[str, str]] = []
    buffer_user = None

    for m in messages:
        if m.role == "user":
            buffer_user = m.content
        elif m.role == "assistant" and buffer_user is not None:
            pairs.append((buffer_user, m.content))
            buffer_user = None

    return pairs


@app.post("/wolf", response_model=ChatResponse)
def wolf_endpoint(req: ChatRequest):
    user_message = req.messages[-1].content
    history_pairs = messages_to_history_pairs(req.messages[:-1])

    if req.mode == ModelMode.romance:
        reply, audio_path = chat_with_wolf_mistral(user_message, history_pairs)
    else:
        reply, audio_path = chat_with_wolf_openai(user_message, history_pairs)

    audio_url = f"/audio/{os.path.basename(audio_path)}" if audio_path else None
    return ChatResponse(reply=reply, audio_url=audio_url)

