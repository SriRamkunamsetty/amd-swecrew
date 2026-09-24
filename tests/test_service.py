import os

os.environ.setdefault("SWECREW_USE_LLM", "false")

from fastapi.testclient import TestClient  # noqa: E402

from swecrew.service import app  # noqa: E402


def test_healthz_and_readyz_and_validation(tmp_path):
    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/readyz").json()["llm"] is False

        resp = client.post("/v1/fix", json={"problem_statement": "fix it"})
        assert resp.status_code == 400  # no repo_path/repo_url

        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
        resp = client.post("/v1/fix", json={"problem_statement": "fix it", "repo_path": str(tmp_path),
                                            "budget_s": 20})
        assert resp.status_code == 200
        data = resp.json()
        assert data["patch"] == "" and data["confidence"] == 0.0


def test_api_key_required_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("SWECREW_API_KEYS", "secret123")
    import importlib

    import swecrew.service as svc
    importlib.reload(svc)
    try:
        with TestClient(svc.app) as client:
            (tmp_path / "a.py").write_text("x = 1\n")
            body = {"problem_statement": "fix it", "repo_path": str(tmp_path)}
            assert client.post("/v1/fix", json=body).status_code == 401
            assert client.post("/v1/fix", json=body, headers={"X-API-Key": "secret123"}).status_code == 200
    finally:
        monkeypatch.delenv("SWECREW_API_KEYS", raising=False)
        importlib.reload(svc)
