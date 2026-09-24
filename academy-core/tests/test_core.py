import json

import httpx
import pytest
from pydantic import BaseModel

from academy_core import Deadline, LLMClient, LLMError, LLMSettings, grader_normalize, output_path_for
from academy_core.harness import read_input_file, write_json_atomic
from academy_core.llm import extract_json, match_choice
from academy_core.server.entrypoint import gpu_memory_fraction


def test_grader_normalize_matches_official_examples():
    for variant in ["7ABC123", "7abc123", "7-ABC-123", "7 ABC 123"]:
        assert grader_normalize(variant) == "7ABC123"
    assert grader_normalize("京A·12345") == "京A12345"
    assert grader_normalize("SPEED LIMIT 65") == "SPEEDLIMIT65"


def test_output_path(tmp_path):
    assert output_path_for("/app/input/image_01.png", tmp_path) == tmp_path / "image_01_output.json"


def test_write_and_read_json(tmp_path):
    p = write_json_atomic(tmp_path / "out" / "a.json", {"text": "ok"})
    assert json.loads(p.read_text()) == {"text": "ok"}
    (tmp_path / "q.txt").write_text("What is X?")
    assert read_input_file(tmp_path / "q.txt") == {"text": "What is X?"}
    (tmp_path / "q.json").write_text('{"question": "Y?"}')
    assert read_input_file(tmp_path / "q.json") == {"question": "Y?"}


def test_deadline_child_never_exceeds_parent():
    d = Deadline(1.0)
    assert d.child(10).seconds <= 1.0
    assert 0.05 <= d.timeout(30, reserve=0.5) <= 0.5


def test_vram_fraction_respects_48gib_budget_on_large_cards():
    assert gpu_memory_fraction(40, 192) == pytest.approx(0.208, abs=1e-3)
    assert gpu_memory_fraction(40, 48) == pytest.approx(0.833, abs=1e-3)
    assert gpu_memory_fraction(40, None) == 0.80


def test_extract_json_variants():
    assert extract_json('<think>hmm</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_match_choice():
    opts = ["up", "down", "left", "right"]
    assert match_choice("Left", opts) == 2
    assert match_choice("I choose: right.", opts) == 3
    assert match_choice("2", opts) == 2
    assert match_choice("banana", opts) is None
    assert match_choice("3", ["1", "2", "3"]) == 2


class Out(BaseModel):
    answer: str


def _transport(content: str, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if status != 200:
            return httpx.Response(status, text="bad")
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler)


async def test_llm_client_json_and_choice():
    async with LLMClient(LLMSettings(base_url="http://x/v1"), transport=_transport('{"answer": "42"}')) as c:
        assert (await c.complete_json([{"role": "user", "content": "q"}], Out)).answer == "42"
    async with LLMClient(LLMSettings(base_url="http://x/v1"), transport=_transport("down")) as c:
        assert await c.choose([{"role": "user", "content": "q"}], ["up", "down"]) == 1


async def test_llm_client_unavailable_raises():
    def handler(request):
        raise httpx.ConnectError("refused")

    async with LLMClient(LLMSettings(base_url="http://x/v1"), transport=httpx.MockTransport(handler)) as c:
        assert not await c.available()
        with pytest.raises(LLMError):
            await c.chat([{"role": "user", "content": "q"}])
