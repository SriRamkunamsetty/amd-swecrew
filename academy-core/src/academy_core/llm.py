"""Async client for any OpenAI-compatible chat endpoint (vLLM, SGLang, Ollama, the bundled HF server).

Design rules:
  * The LLM is an *accelerator*, never a single point of failure: every call site has a
    deterministic fallback, so ``LLMError`` is expected and must be handled.
  * Structured output is requested natively when the server supports it
    (JSON schema / choice constraints) and is *always* re-validated client-side.
  * Every call respects a Deadline.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from academy_core.deadline import Deadline
from academy_core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    pass


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    enabled: bool = True
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = ""  # empty -> first model the server reports
    api_key: str = "EMPTY"
    timeout: float = 20.0
    max_retries: int = 1
    # How to send choice constraints: vLLM >= 0.10 uses `structured_outputs`,
    # older vLLM uses `guided_choice`; Ollama/others get prompt-only + client matching.
    guided_style: Literal["structured_outputs", "guided", "none"] = "structured_outputs"
    json_schema_mode: bool = True
    disable_thinking: bool = True  # Qwen3-style chat templates
    extra_body: dict[str, Any] = {}


def strip_reasoning(text: str) -> str:
    text = _THINK.sub("", text)
    if "</think>" in text:  # opening tag was part of the prompt template
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of model text (handles fences, prose, reasoning)."""
    text = strip_reasoning(text)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise LLMError(f"no JSON found in model output: {text[:200]!r}")


def match_choice(text: str, options: list[str]) -> int | None:
    """Map free text back onto one of the allowed options."""
    cleaned = strip_reasoning(text).strip().strip("`'\" .").lower()
    if not cleaned:
        return None
    lowered = [o.lower() for o in options]
    if cleaned in lowered:
        return lowered.index(cleaned)
    number = re.match(r"^\(?(\d+)\)?[.:)]?", cleaned)
    if number:
        idx = int(number.group(1))
        # "1" might be an option label itself; only treat as index if it isn't an option.
        if str(idx) not in lowered and 0 <= idx < len(options):
            return idx
    starts = [i for i, o in enumerate(lowered) if cleaned.startswith(o) or o.startswith(cleaned)]
    if len(starts) == 1:
        return starts[0]
    contains = [i for i, o in enumerate(lowered) if o and o in cleaned]
    if contains:
        return max(contains, key=lambda i: len(lowered[i]))
    return None


class LLMClient:
    def __init__(self, settings: LLMSettings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings or LLMSettings()
        self._client = httpx.AsyncClient(
            base_url=self.settings.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
            timeout=self.settings.timeout,
            transport=transport,
        )
        self._model: str | None = self.settings.model or None
        self._healthy: bool | None = None
        self._health_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ health
    async def available(self, timeout: float = 2.0) -> bool:
        if not self.settings.enabled:
            return False
        if self._healthy is not None:
            return self._healthy
        async with self._health_lock:
            if self._healthy is not None:
                return self._healthy
            try:
                resp = await self._client.get("/models", timeout=timeout)
                resp.raise_for_status()
                models = [m["id"] for m in resp.json().get("data", [])]
                if not self._model:
                    if not models:
                        raise LLMError("server reports no models")
                    self._model = models[0]
                self._healthy = True
                log.info("llm available", extra={"model": self._model, "base_url": self.settings.base_url})
            except Exception as exc:  # noqa: BLE001 - any failure means "not available"
                log.warning("llm unavailable, using fallbacks: %s", exc)
                self._healthy = False
            return self._healthy

    @property
    def model(self) -> str | None:
        return self._model

    # ------------------------------------------------------------------ core call
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        choices: list[str] | None = None,
        deadline: Deadline | None = None,
        logprobs: int | None = None,
        raw: bool = False,
    ) -> Any:
        if not await self.available():
            raise LLMError("llm not available")
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        style = self.settings.guided_style
        if json_schema is not None and self.settings.json_schema_mode:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        if choices:
            if style == "structured_outputs":
                body["structured_outputs"] = {"choice": choices}
            elif style == "guided":
                body["guided_choice"] = choices
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = logprobs
        if self.settings.disable_thinking and style != "none":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body.update(self.settings.extra_body)

        attempts = self.settings.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            timeout = deadline.timeout(self.settings.timeout, reserve=0.2) if deadline else self.settings.timeout
            if deadline and deadline.remaining < 0.5:
                raise LLMError("deadline exhausted before llm call")
            try:
                resp = await self._client.post("/chat/completions", json=body, timeout=timeout)
                if resp.status_code >= 500:
                    raise LLMError(f"server error {resp.status_code}: {resp.text[:200]}")
                if resp.status_code == 400 and ("structured_outputs" in body or "response_format" in body
                                                 or "chat_template_kwargs" in body):
                    # Server does not understand an extension: degrade once, permanently.
                    log.warning("llm rejected request extensions, degrading: %s", resp.text[:200])
                    for key in ("structured_outputs", "guided_choice", "response_format", "chat_template_kwargs"):
                        body.pop(key, None)
                    self.settings.guided_style = "none"
                    self.settings.json_schema_mode = False
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data if raw else (data["choices"][0]["message"].get("content") or "")
            except (httpx.TransportError, httpx.HTTPStatusError, LLMError, KeyError) as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.2)
        raise LLMError(f"llm call failed: {last_exc}")

    # ------------------------------------------------------------------ helpers
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        max_tokens: int = 512,
        deadline: Deadline | None = None,
    ) -> T:
        text = await self.chat(
            messages, max_tokens=max_tokens, json_schema=schema.model_json_schema(), deadline=deadline
        )
        try:
            return schema.model_validate(extract_json(text))
        except ValidationError as exc:
            raise LLMError(f"model output failed validation: {exc}") from exc

    async def choose(
        self,
        messages: list[dict[str, str]],
        options: list[str],
        *,
        deadline: Deadline | None = None,
        max_tokens: int = 32,
    ) -> int:
        """Return the index of the option the model picks. Never returns an illegal option."""
        if not options:
            raise ValueError("options must not be empty")
        text = await self.chat(messages, choices=options, max_tokens=max_tokens, deadline=deadline)
        idx = match_choice(text, options)
        if idx is None:
            raise LLMError(f"model picked no valid option: {text[:120]!r}")
        return idx
