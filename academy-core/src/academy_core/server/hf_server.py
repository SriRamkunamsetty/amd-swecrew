"""Minimal OpenAI-compatible server on plain transformers + torch (ROCm).

A fallback for when vLLM is unavailable for the mandated ROCm base image. It is slower than
vLLM but keeps the container on the GPU (the grader requires >= 1 GiB peak VRAM) and supports
the two features our agents rely on:
  * chat completions
  * choice constraints (``structured_outputs.choice`` or ``guided_choice``), implemented
    exactly by scoring each option's log-likelihood, so the answer is always legal.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any

import torch  # type: ignore[import-not-found]
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.0
    structured_outputs: dict[str, Any] | None = None
    guided_choice: list[str] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


def create_app(model_path: str, served_name: str) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    lock = asyncio.Lock()
    app = FastAPI(title="academy hf server")

    def render(req: ChatRequest) -> str:
        kwargs = req.chat_template_kwargs or {}
        try:
            return tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def score_choices(prompt: str, choices: list[str]) -> str:
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        best, best_score = choices[0], float("-inf")
        for choice in choices:
            ids = tokenizer(choice, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            full = torch.cat([prompt_ids, ids], dim=1)
            logits = model(full).logits[0, prompt_ids.shape[1] - 1 : -1]
            logp = torch.log_softmax(logits.float(), dim=-1).gather(1, ids[0].unsqueeze(1)).mean().item()
            if logp > best_score:
                best, best_score = choice, logp
        return best

    @torch.inference_mode()
    def generate(prompt: str, max_tokens: int, temperature: float) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> dict[str, Any]:
        choices = (req.structured_outputs or {}).get("choice") or req.guided_choice
        prompt = render(req)
        async with lock:
            try:
                if choices:
                    text = await asyncio.to_thread(score_choices, prompt, list(choices))
                else:
                    text = await asyncio.to_thread(generate, prompt, req.max_tokens, req.temperature)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-name", default="local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.served_name), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
