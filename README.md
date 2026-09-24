# SweCrew: a test-driven multi-agent software engineering crew

**lablab.ai × AMD AI Academy Challenge, Mini-Challenge 5:** *a multi-agent software engineering
solution.*

Given a repository and a bug/feature description, SweCrew produces a **verified patch**: several
coder agents propose independent fixes in parallel, every fix is actually applied and tested in an
isolated sandbox, and only a fix that provably improves the test outcome (without breaking anything
that was passing) is ever returned. It runs on AMD GPUs (vLLM on ROCm) inside the mandated
challenge container.

## Key idea

**Agents check each other by executing tests, not by agreeing in conversation.** Role-play
multi-agent systems (planner chats with coder chats with reviewer) add tokens without adding
correctness. SweCrew instead uses multiple agents for the two things that actually move the needle
on SWE-bench-style benchmarks: **diverse candidate generation** and **execution-based verification**.

```
task (repo + issue) ─► Workspace (git-tracked sandbox: clone/copy -> baseline commit)
        │
        ▼
repo map (ast-based symbols, no tree-sitter dependency) ──► ranked by relevance to the issue
        │
        ▼
Localizer ── picks real files (never invents a path) ──► shared context (issue + file contents)
        │                                                  reused verbatim across every call below
        ▼                                                  -> vLLM automatic prefix caching means
   ┌─────────────┬─────────────┬─────────────┐               N candidates cost ~1 prefill, not N
   │  Coder #0   │  Coder #1   │  Coder #2   │  (parallel, different temperatures)
   └──────┬──────┴──────┬──────┴──────┬──────┘
          ▼             ▼             ▼
   reset -> apply edits -> run tests (isolated per-task venv, target repo's own deps)
          │             │             │
          └─────────────┴─────────────┘
                         ▼
              Selector: best *executed* outcome, never a regression, ties -> smallest diff
                         │
                         ▼
              Debugger: bounded repair rounds on the most promising attempt
                         │
                         ▼
              final patch = git diff, or "" if nothing verified better than doing nothing
```

## Why this beats "ask a big LLM for a diff"

* **Structured edits, not raw diffs.** The coder emits `{path, search, replace}` blocks (Aider-style),
  applied as literal, all-or-nothing text substitution. Models are unreliable at hunk headers and
  line numbers; they are reliable at "here is the exact text to change".
* **Git owns the diffing.** The sandbox is a git-tracked copy of the repo; `git diff` after edits is
  always a correct unified patch — no hand-rolled diff code to get wrong, and `git reset --hard` between
  candidate attempts means N candidates share one clone instead of N full copies.
* **Never a guess dressed up as a fix.** If no candidate improves on doing nothing (fixes something
  without breaking something else), the output is an empty patch with confidence 0 — not a plausible
  wrong answer.
* **Fits the hardware.** Search/execution is on CPU; the container serves a dedicated code model
  (Qwen2.5-Coder-7B by default) on the GPU, well under 48 GiB.

## Run

```bash
python app.py --input samples/task_01.json --output-dir out --debug   # -> out/task_01_output.json
python app.py --input-dir samples --output-dir out                    # batch
uvicorn swecrew.service:app --port 8082                                # REST API
```

Predicted input (SWE-bench-style; the official contract is not published yet — see below):
```json
{"problem_statement": "add() subtracts instead of adding", "repo": "repo",
 "fail_to_pass": ["tests/test_calc.py::test_add"], "pass_to_pass": ["tests/test_calc.py::test_mul"]}
```
`repo` may be a path already mounted next to the input file, or `repo_url` for a git clone. If no
`fail_to_pass`/`pass_to_pass` node ids are given, SweCrew falls back to pytest's own pass/fail summary.

Output (SWE-bench-compatible field name, plus aliases):
```json
{"patch": "diff --git a/pkg/calc.py b/pkg/calc.py\n...", "model_patch": "...", "confidence": 0.95,
 "files_changed": ["pkg/calc.py"], "tests": {"fail_to_pass_passed": 1, "pass_to_pass_passed": 1}}
```

## Configuration (env `SWECREW_*`)

| Variable | Default | Notes |
|---|---|---|
| `TIME_BUDGET_S` | 240 | set well below the harness per-item limit |
| `NUM_CANDIDATES` | 3 | parallel coder attempts (diversity, not conversation) |
| `MAX_DEBUG_ROUNDS` | 2 | bounded repair loop on the most promising attempt |
| `INSTALL_DEPS` | true | best-effort isolated venv + `pip install -e .`/`-r requirements.txt` for the target repo |
| `DEFAULT_TEST_CMD` | `python -m pytest -q` | used when the task gives no test command and no node ids |
| `USE_LLM` / `LLM_*` | true | any OpenAI-compatible server; vLLM on ROCm in the container |

## Grader-driven container design

Same hard gates as the other mini-challenges in this monorepo (see the root README): ROCm base
image checked by layer identity, ≤ 60 GiB, the model server started once during the 10-minute
startup window (`app.py` is a thin client), peak VRAM converted from a GiB budget into a fraction
of the detected card. This image additionally installs `git` (needed for the sandbox) and defaults
to a **code-specialized** model rather than a general chat model.

```bash
docker build -f Dockerfile -t <registry>/amd-swecrew:v1 .
bash scripts/check_submission.sh <registry>/amd-swecrew:v1 samples 60
docker push <registry>/amd-swecrew:v1   # public registry, but do not publish the image reference in a public repo
```

## Tested, not yet spec-published

The official spec for this mini-challenge was not published when this was built. Everything the
contract depends on lives in `src/swecrew/task.py` (input aliases) and `models.FixResult.to_output`
(output field names), so adapting to the real spec is a small, isolated change. What **is** verified
here, with 34 offline tests (no network, no real LLM — see `tests/test_orchestrator.py` for the
scripted-LLM harness): edit application (unique-match, atomic rollback, path-escape guard), repo
mapping and relevance ranking, the sandbox lifecycle (git baseline, apply/diff/reset, real pytest
execution, regression detection), the full parallel-candidates → debug-repair → selection pipeline,
the harness CLI's always-valid-output guarantee, and the REST API.

> Live-LLM behavior (real model quirks in the structured-edit format) is not yet exercised end to
> end on this machine — only the deterministic pipeline and a scripted fake LLM are. Do a short
> real run against a small model before relying on the coder/debugger prompts as-is.

## Layout

```
src/swecrew/   task.py (input loading) · repo_map.py · sandbox.py (git-tracked workspace) · edits.py
               prompts.py (shared context builder) · agents/ (localizer, coder, debugger, selector)
               orchestrator.py · cli.py (harness) · service.py (REST)
academy-core/  shared runtime: LLM client, harness contract, deadlines, model-server entrypoint
tests/         offline unit + end-to-end tests with a scripted fake LLM and real git/pytest execution
samples/       a tiny synthetic buggy repo + task, for --no-llm smoke testing and check_submission.sh
```

## License

MIT
