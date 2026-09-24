"""Production REST API.

    uvicorn swecrew.service:app --host 0.0.0.0 --port 8082

Endpoint: POST /v1/fix, GET /healthz, GET /readyz
Auth: set SWECREW_API_KEYS="key1,key2" to require an ``X-API-Key`` header.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from academy_core import setup_logging
from swecrew.config import Settings
from swecrew.orchestrator import SweCrew
from swecrew.task import Task

setup_logging()
_KEYS = {k.strip() for k in os.getenv("SWECREW_API_KEYS", "").split(",") if k.strip()}
_MAX_CONCURRENCY = int(os.getenv("SWECREW_MAX_CONCURRENT_REQUESTS", "2"))  # each fix uses a repo checkout + tests


class FixRequest(BaseModel):
    problem_statement: str = Field(min_length=3, max_length=20_000)
    repo_path: str | None = None
    repo_url: str | None = None
    base_commit: str | None = None
    test_cmd: str | None = None
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    budget_s: float | None = Field(default=None, ge=10, le=1800)
    debug: bool = False


state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    crew = SweCrew(Settings())
    state["crew"] = crew
    state["sem"] = asyncio.Semaphore(_MAX_CONCURRENCY)
    state["started"] = time.time()
    yield
    await crew.aclose()


app = FastAPI(title="SweCrew", version="0.1.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if _KEYS and x_api_key not in _KEYS:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {"status": "ok", "uptime_s": round(time.time() - float(state.get("started", time.time())), 1)}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    crew: SweCrew = state["crew"]  # type: ignore[assignment]
    llm = bool(crew.llm and await crew.llm.available())
    return {"status": "ready", "llm": llm}


@app.post("/v1/fix", dependencies=[Depends(require_key)])
async def fix(req: FixRequest) -> dict[str, object]:
    if not req.repo_path and not req.repo_url:
        raise HTTPException(status_code=400, detail="repo_path or repo_url is required")
    if req.repo_path and not Path(req.repo_path).is_dir():
        raise HTTPException(status_code=400, detail=f"repo_path not found: {req.repo_path}")
    task = Task(problem_statement=req.problem_statement, repo_path=Path(req.repo_path) if req.repo_path else None,
               repo_url=req.repo_url, base_commit=req.base_commit, test_cmd=req.test_cmd,
               fail_to_pass=req.fail_to_pass, pass_to_pass=req.pass_to_pass)
    crew: SweCrew = state["crew"]  # type: ignore[assignment]
    sem: asyncio.Semaphore = state["sem"]  # type: ignore[assignment]
    async with sem:
        result = await crew.fix(task, budget_s=req.budget_s)
    return result.to_output(include_debug=req.debug)
