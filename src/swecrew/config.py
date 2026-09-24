from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime knobs. Override with SWECREW_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="SWECREW_", extra="ignore")

    # --- budgets ------------------------------------------------------------------------
    time_budget_s: float = 240.0  # keep well under the harness per-item limit
    llm_localize_timeout_s: float = 20.0
    llm_reproduce_timeout_s: float = 25.0
    llm_coder_timeout_s: float = 60.0
    llm_debug_timeout_s: float = 60.0
    test_timeout_s: float = 90.0
    baseline_test_timeout_s: float = 60.0
    env_setup_timeout_s: float = 90.0

    # --- multi-agent shape ----------------------------------------------------------------
    num_candidates: int = 3          # coder agents generating diverse patches in parallel
    max_debug_rounds: int = 2        # bounded repair loop on the best candidate
    max_localized_files: int = 6
    max_candidate_files: int = 40    # repo map candidates shown to the localizer
    coder_temperatures: list[float] = [0.0, 0.4, 0.8]

    # --- repo/test execution ---------------------------------------------------------------
    install_deps: bool = True        # best-effort `pip install` into an isolated per-task venv
    default_test_cmd: str = "python -m pytest -q"
    max_file_read_bytes: int = 60_000
    max_context_chars: int = 24_000

    # --- LLM usage --------------------------------------------------------------------
    use_llm: bool = True

    # --- workspace --------------------------------------------------------------------
    work_dir: str | None = None      # defaults to a fresh temp dir per task
    keep_workspace: bool = False     # for debugging; normally cleaned up
