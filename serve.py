"""OpenAI-Compatible FastAPI Serving API for VoidFormer Models."""

from __future__ import annotations

import argparse
import time
import os
import yaml
import torch
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, HTTPException

from voidformer.harness.model_factory import create_model
from voidformer.utils.checkpoint import load_checkpoint

app = FastAPI(title="VoidFormer OpenAI-Compatible API", version="0.1.0")

# Global model state
GLOBAL_MODEL: Optional[torch.nn.Module] = None
GLOBAL_MODEL_NAME: str = "quantum-voidformer"
GLOBAL_MAX_SEQ_LEN: int = 128


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "quantum-voidformer"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: Optional[float] = 1.0


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "voidformer"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


@app.get("/v1/models")
@app.get("/models")
async def list_models() -> ModelListResponse:
    return ModelListResponse(data=[ModelObject(id=GLOBAL_MODEL_NAME)])


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> Dict[str, Any]:
    global GLOBAL_MODEL, GLOBAL_MAX_SEQ_LEN

    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    prompt_text = ""
    for msg in request.messages:
        prompt_text += f"{msg.role}: {msg.content}\n"
    prompt_text += "assistant: "

    prompt_tokens = [ord(c) % 256 for c in prompt_text]
    # Crop to max sequence length to prevent out-of-bounds error
    if len(prompt_tokens) > GLOBAL_MAX_SEQ_LEN:
        prompt_tokens = prompt_tokens[-GLOBAL_MAX_SEQ_LEN:]

    ids = torch.tensor([prompt_tokens], dtype=torch.long)

    with torch.no_grad():
        if hasattr(GLOBAL_MODEL, "generate"):
            gen_ids = GLOBAL_MODEL.generate(ids, max_new_tokens=request.max_tokens, temperature=request.temperature)
        else:
            gen_ids = ids

    new_tokens = gen_ids[0][len(prompt_tokens):]
    assistant_reply = "".join([chr(tok.item() % 128) for tok in new_tokens]).strip()

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_reply,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(new_tokens),
            "total_tokens": len(prompt_tokens) + len(new_tokens),
        },
    }


def load_server_model(
    checkpoint_path: Optional[str] = None,
    config_path: str = "voidformer/configs/tiny.yaml",
    model_type: str = "quantum",
):
    global GLOBAL_MODEL, GLOBAL_MODEL_NAME, GLOBAL_MAX_SEQ_LEN

    if not os.path.exists(config_path):
        config_path = os.path.join("voidformer", config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    GLOBAL_MAX_SEQ_LEN = model_cfg.get("max_seq_len", 128)
    GLOBAL_MODEL_NAME = f"{model_type}-voidformer"

    GLOBAL_MODEL = create_model(
        model_type=model_type,
        vocab_size=model_cfg.get("vocab_size", 256),
        d_model=model_cfg.get("d_model", 128),
        d_void=model_cfg.get("d_void", 128),
        n_layers=model_cfg.get("n_layers", 2),
        n_heads=model_cfg.get("n_heads", 4),
        d_ff=model_cfg.get("d_ff", 256),
        max_seq_len=GLOBAL_MAX_SEQ_LEN,
        use_vqc_layer=model_cfg.get("use_vqc_layer", True),
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading server checkpoint from {checkpoint_path}...")
        load_checkpoint(checkpoint_path, GLOBAL_MODEL)

    GLOBAL_MODEL.eval()
    print("VoidFormer Server Model Loaded Successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run OpenAI-Compatible FastAPI Model Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--config", type=str, default="voidformer/configs/tiny.yaml", help="Path to config")
    parser.add_argument("--model-type", type=str, choices=["quantum", "classical"], default="quantum", help="Model type")
    args = parser.parse_args()

    load_server_model(checkpoint_path=args.checkpoint, config_path=args.config, model_type=args.model_type)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
