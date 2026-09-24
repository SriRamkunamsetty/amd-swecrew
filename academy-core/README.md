# academy-core

Shared runtime for AMD AI Academy challenge containers.

* `llm.LLMClient`: async OpenAI-compatible client. JSON-schema and choice constraints (vLLM `structured_outputs` / `guided_choice`) are always re-validated client-side. Deadline-aware and fail-soft (raises `LLMError`, which callers handle). It degrades permanently on servers that reject extensions (e.g. Ollama).
* `harness`: `grader_normalize` (the official normalization), `output_path_for` (`<stem>_output.json`), atomic JSON writes, tolerant input reader.
* `deadline.Deadline`: wall-clock budgets with child budgets and capped per-operation timeouts.
* `server.entrypoint`: container entrypoint. Starts vLLM (or the bundled transformers server when vLLM is missing), converts `LLM_VRAM_BUDGET_GIB` into `--gpu-memory-utilization` for the detected card, waits for health, warms up, writes `/tmp/academy_ready`, optionally starts the REST app, and never exits on child crashes.
* `server.hf_server`: minimal OpenAI-compatible server on transformers with exact log-likelihood choice scoring.
* `gpu`: AMD VRAM sampling (`amd-smi` / `rocm-smi` / torch) and a background `VramMonitor` that mirrors the grader's peak check.
