"""A lightweight repo map: no tree-sitter dependency, so it works in any image.

Python files are parsed with the stdlib ``ast`` module for accurate symbol extraction.
Everything else falls back to regex-based function/class detection, which is good enough to
rank candidate files -- the LLM only ever picks among *files that exist*, never invents paths.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env",
                "dist", "build", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
                "vendor", ".idea", ".vscode", "egg-info"}
CODE_EXT = {".py", ".pyi", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".rb", ".rs", ".c", ".h",
            ".cpp", ".hpp", ".cs", ".php"}
_GENERIC_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function|def|class|public\s+\w+\s+\w+|func)\s+([A-Za-z_][\w]*)",
    re.MULTILINE,
)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "should", "would", "could", "when", "what", "which",
    "from", "into", "have", "has", "not", "are", "was", "were", "will", "does", "did", "raise", "error",
    "raises", "returns", "return", "issue", "bug", "expected", "actual", "test", "tests", "def", "class",
    "self", "none", "true", "false",
}


@dataclass
class FileInfo:
    path: str  # relative, forward-slash
    symbols: list[str] = field(default_factory=list)
    size: int = 0
    score: float = 0.0

    def tokens(self) -> set[str]:
        parts = re.split(r"[/_\-.]", self.path.lower())
        return {p for p in parts if len(p) > 2} | {s.lower() for s in self.symbols}


@dataclass
class RepoMap:
    root: Path
    files: list[FileInfo]

    def by_path(self, path: str) -> FileInfo | None:
        return next((f for f in self.files if f.path == path), None)

    def top(self, n: int) -> list[FileInfo]:
        return sorted(self.files, key=lambda f: f.score, reverse=True)[:n]


def _python_symbols(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names


def _generic_symbols(text: str) -> list[str]:
    return _GENERIC_SYMBOL_RE.findall(text)


def build_repo_map(root: str | Path, max_bytes_per_file: int = 200_000) -> RepoMap:
    root = Path(root)
    files: list[FileInfo] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_EXT:
            continue
        if any(part in IGNORED_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        try:
            size = path.stat().st_size
            if size > max_bytes_per_file:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        symbols = _python_symbols(text) if path.suffix == ".py" else _generic_symbols(text)
        rel = path.relative_to(root).as_posix()
        files.append(FileInfo(rel, symbols, size))
    return RepoMap(root, files)


def _query_tokens(*texts: str) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        tokens |= {w.lower() for w in _WORD_RE.findall(text)} - _STOPWORDS
    return tokens


def rank_by_relevance(repo_map: RepoMap, problem_statement: str, hints: list[str] | None = None) -> RepoMap:
    """Score files by token overlap with the issue text, boosted heavily for exact path mentions."""
    query = _query_tokens(problem_statement)
    hinted = {h.strip().lstrip("./").lower() for h in (hints or []) if h.strip()}
    lowered_statement = problem_statement.lower()
    for info in repo_map.files:
        overlap = len(query & info.tokens())
        score = overlap / max(1, len(query)) if query else 0.0
        name = Path(info.path).name.lower()
        if info.path.lower() in hinted or name in hinted or info.path.lower() in lowered_statement:
            score += 5.0
        elif name in lowered_statement:
            score += 2.0
        if info.path.split("/")[0] in {"test", "tests", "__tests__"} or "test" in Path(info.path).name.lower():
            score *= 0.5  # bias toward implementation files over test files when localizing a fix
        info.score = score
    return repo_map


def extract_traceback_paths(text: str) -> list[str]:
    """Pull file paths out of a Python traceback embedded in the issue text, if any."""
    paths = re.findall(r'File "([^"]+)"', text)
    return [Path(p).name for p in paths]
