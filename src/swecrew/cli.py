"""Harness entrypoint (``python3 /app/app.py ...``).

The official spec for this challenge is not published yet, so this accepts every plausible
input shape (see task.py) and always writes an output file, even on crash or timeout:

    python3 /app/app.py --input-file /app/input/task_01.json    # SWE-bench-style task
    python3 /app/app.py --input-dir /app/input                   # batch mode
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from academy_core import output_path_for, setup_logging, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SweCrew: multi-agent software engineering fixer")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-file", "--input", "--input-task", "--input-json", dest="input_file")
    src.add_argument("--input-dir", dest="input_dir")
    p.add_argument("--output-dir", default=os.getenv("HARNESS_OUTPUT_DIR", "/app/output"))
    p.add_argument("--output-file", default=None)
    p.add_argument("--budget", type=float, default=None, help="seconds for this task")
    p.add_argument("--debug", action="store_true", help="include trace/timing in output")
    p.add_argument("--no-llm", action="store_true")
    return p.parse_args(argv)


def _empty(error: str | None = None) -> dict:
    out = {"patch": "", "model_patch": "", "diff": "", "confidence": 0.0, "files_changed": [], "tests": {},
           "summary": ""}
    if error:
        out["error"] = error[:500]
    return out


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    if args.no_llm:
        os.environ["SWECREW_USE_LLM"] = "false"

    from swecrew.config import Settings
    from swecrew.orchestrator import SweCrew
    from swecrew.task import Task

    settings = Settings()
    budget = args.budget or settings.time_budget_s

    jobs: list[tuple[Path, Path]] = []
    if args.input_dir:
        for path in sorted(Path(args.input_dir).iterdir()):
            if path.is_file() and path.suffix.lower() == ".json":
                jobs.append((path, output_path_for(path, args.output_dir)))
    else:
        out = Path(args.output_file) if args.output_file else output_path_for(args.input_file, args.output_dir)
        jobs.append((Path(args.input_file), out))

    for _, out_path in jobs:
        write_json_atomic(out_path, _empty())  # a valid file exists no matter what happens next

    watchdog = threading.Timer(budget * len(jobs) + 15.0, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()

    async def run() -> None:
        async with SweCrew(settings) as crew:
            for in_path, out_path in jobs:
                try:
                    task = Task.load(in_path)
                    if not task.problem_statement and not task.repo_path:
                        payload = _empty("no problem_statement/repo found in input")
                    else:
                        result = await crew.fix(task, budget)
                        payload = result.to_output(include_debug=args.debug)
                except Exception as exc:  # noqa: BLE001 - never crash the harness
                    payload = _empty(f"{type(exc).__name__}: {exc}")
                write_json_atomic(out_path, payload)
                print(json.dumps({"output": str(out_path), **{k: v for k, v in payload.items() if k != "patch"}},
                                 ensure_ascii=False, default=str), flush=True)

    try:
        asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        print(f"swecrew failed: {exc}", file=sys.stderr)
    watchdog.cancel()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
