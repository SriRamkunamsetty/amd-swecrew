"""Container entrypoint: start the model server once, wait until it is healthy, stay alive.

Why this exists: the grading harness runs ``python3 /app/app.py`` as a *fresh process per
test item* inside an already-running container. Loading weights in app.py would blow the
per-item time limit, so the weights live in a long-lived server started here during the
10-minute startup budget, and app.py is a thin client.

Environment:
  LLM_BACKEND            vllm | hf | external | none          (default: vllm)
  LLM_SERVE_MODEL        HF repo id or local path (e.g. /models/qwen)
  LLM_SERVED_NAME        name exposed on the API              (default: "local")
  LLM_PORT               default 8000
  LLM_VRAM_BUDGET_GIB    VRAM the server may reserve          (default: 40)
                         The grader fails anything whose *peak* exceeds 48 GiB, and vLLM's
                         --gpu-memory-utilization is a fraction of the *whole* card, which on a
                         192 GB MI300X would be far above the limit. We convert the budget into
                         a fraction of the detected card size.
  LLM_MAX_MODEL_LEN      optional context cap
  LLM_EXTRA_ARGS         extra CLI args for the backend
  LLM_READY_TIMEOUT      seconds to wait for health          (default: 540)
  APP_SERVE_CMD          optional command to run after the model is ready (e.g. the REST API)
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from academy_core.logging import get_logger, setup_logging

log = get_logger("entrypoint")
READY_MARKER = Path(os.getenv("READY_MARKER", "/tmp/academy_ready"))


def total_vram_gib() -> float | None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return torch.cuda.mem_get_info(0)[1] / 1024**3
    except Exception as exc:  # noqa: BLE001
        log.warning("could not query VRAM via torch: %s", exc)
    return None


def gpu_memory_fraction(budget_gib: float, total_gib: float | None) -> float:
    if not total_gib:
        return 0.80
    return round(max(0.05, min(0.92, budget_gib / total_gib)), 3)


def build_command(backend: str) -> list[str] | None:
    model = os.getenv("LLM_SERVE_MODEL", "")
    port = os.getenv("LLM_PORT", "8000")
    served = os.getenv("LLM_SERVED_NAME", "local")
    extra = shlex.split(os.getenv("LLM_EXTRA_ARGS", ""))
    max_len = os.getenv("LLM_MAX_MODEL_LEN")
    if backend in {"none", "external"}:
        return None
    if not model:
        raise SystemExit("LLM_SERVE_MODEL must be set for backend " + backend)
    if backend == "vllm" and importlib.util.find_spec("vllm") is None:
        log.warning("vLLM is not installed in this image; falling back to the transformers server")
        backend = "hf"
    if backend == "vllm":
        frac = gpu_memory_fraction(float(os.getenv("LLM_VRAM_BUDGET_GIB", "40")), total_vram_gib())
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", served,
            "--host", "127.0.0.1", "--port", port,
            "--gpu-memory-utilization", str(frac),
            "--enable-prefix-caching",
        ]
        if max_len:
            cmd += ["--max-model-len", max_len]
        return cmd + extra
    if backend == "hf":
        cmd = [sys.executable, "-m", "academy_core.server.hf_server", "--model", model,
               "--served-name", served, "--port", port]
        return cmd + extra
    raise SystemExit(f"unknown LLM_BACKEND={backend}")


def wait_ready(base_url: str, proc: subprocess.Popen[bytes] | None, timeout: float) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if proc is not None and proc.poll() is not None:
            log.error("model server exited early with code %s", proc.returncode)
            return False
        try:
            if httpx.get(f"{base_url}/models", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    return False


def warmup(base_url: str) -> None:
    try:
        models = httpx.get(f"{base_url}/models", timeout=5).json()["data"]
        httpx.post(
            f"{base_url}/chat/completions",
            json={"model": models[0]["id"], "messages": [{"role": "user", "content": "ping"}], "max_tokens": 4},
            timeout=120,
        )
        log.info("warmup complete")
    except Exception as exc:  # noqa: BLE001
        log.warning("warmup failed (continuing): %s", exc)


def main() -> int:
    setup_logging()
    backend = os.getenv("LLM_BACKEND", "vllm").lower()
    base_url = os.getenv("LLM_BASE_URL", f"http://127.0.0.1:{os.getenv('LLM_PORT', '8000')}/v1").rstrip("/")
    cmd = build_command(backend)
    children: list[subprocess.Popen[bytes]] = []

    def shutdown(signum: int, _frame: object) -> None:
        log.info("signal %s, stopping children", signum)
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    proc = None
    if cmd:
        log.info("starting model server: %s", " ".join(cmd))
        proc = subprocess.Popen(cmd)
        children.append(proc)
    if backend != "none":
        ok = wait_ready(base_url, proc, float(os.getenv("LLM_READY_TIMEOUT", "540")))
        if ok:
            warmup(base_url)
        else:
            # Keep the container alive: app.py falls back to non-LLM strategies.
            log.error("model server not ready; app will run in degraded mode")
    READY_MARKER.parent.mkdir(parents=True, exist_ok=True)
    READY_MARKER.write_text(str(time.time()))

    app_cmd = os.getenv("APP_SERVE_CMD")
    if app_cmd:
        log.info("starting app server: %s", app_cmd)
        children.append(subprocess.Popen(shlex.split(app_cmd)))

    # Never exit on a child crash: a dead container scores zero, a degraded one still answers.
    while True:
        for child in list(children):
            code = child.poll()
            if code is not None:
                log.error("child %s exited with %s; continuing in degraded mode", child.args, code)
                children.remove(child)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
