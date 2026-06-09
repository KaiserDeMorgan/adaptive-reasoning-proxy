"""Integration tests for the FastAPI proxy endpoint.

The upstream LLM stream is mocked so these tests run with no network,
no Redis, and no llama-server.
"""

import json
import math

import httpx
import pytest

import main


# --- fake OpenAI streaming objects -----------------------------------------


class _FakeTopLogprob:
    def __init__(self, token: str, logprob: float):
        self.token = token
        self.logprob = logprob


class _FakeLPContent:
    def __init__(self, top_logprobs):
        self.top_logprobs = top_logprobs


class _FakeLogprobs:
    def __init__(self, content):
        self.content = content


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, delta, logprobs):
        self.delta = delta
        self.logprobs = logprobs


class _FakeChunk:
    def __init__(self, token: str, top_lps: dict[str, float]):
        lp_objs = [_FakeLPContent([_FakeTopLogprob(t, lp) for t, lp in top_lps.items()])]
        self.choices = [_FakeChoice(_FakeDelta(token), _FakeLogprobs(lp_objs))]
        self._token = token

    def model_dump_json(self) -> str:
        return json.dumps({"choices": [{"delta": {"content": self._token}}]})


def _peaked() -> dict[str, float]:
    return {"a": math.log(0.97), "b": math.log(0.01), "c": math.log(0.01), "d": math.log(0.01)}


def _uniform() -> dict[str, float]:
    p = math.log(0.25)
    return {"a": p, "b": p, "c": p, "d": p}


def _make_stream(n_tokens: int, top_lps_factory):
    async def _gen():
        for _ in range(n_tokens):
            yield _FakeChunk("x", top_lps_factory())

    async def _create(**kwargs):
        return _gen()

    return _create


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def client_factory(monkeypatch):
    """Return an httpx AsyncClient bound to the ASGI app, with the upstream
    LLM stream and threshold lookup patched."""

    async def _fixed_threshold(task_type, default):
        return 0.35

    monkeypatch.setattr(main, "get_threshold", _fixed_threshold)

    def _build(stream_create):
        monkeypatch.setattr(main.client.chat.completions, "create", stream_create)
        transport = httpx.ASGITransport(app=main.app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    return _build


# --- tests ------------------------------------------------------------------


async def test_health_returns_ok():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_early_stop_on_low_entropy_stream(client_factory):
    # 60 peaked (low-entropy) tokens should trip the entropy stop.
    ac = client_factory(_make_stream(60, _peaked))
    async with ac:
        resp = await ac.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "What is 2+2?"}]},
        )
    body = resp.text
    assert resp.status_code == 200
    assert "entropy_stop" in body
    assert '"early_stop": true' in body
    assert body.rstrip().endswith("data: [DONE]\n\n".rstrip())


async def test_no_early_stop_on_high_entropy_stream(client_factory):
    # 30 uniform (high-entropy) tokens: should finish naturally, no edrm_meta.
    ac = client_factory(_make_stream(30, _uniform))
    async with ac:
        resp = await ac.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "Tell me a story"}]},
        )
    body = resp.text
    assert resp.status_code == 200
    assert "entropy_stop" not in body
    assert "[DONE]" in body


async def test_forces_logprobs_in_upstream_request(client_factory, monkeypatch):
    captured = {}

    async def _create(**kwargs):
        captured.update(kwargs)

        async def _gen():
            for _ in range(3):
                yield _FakeChunk("x", _uniform())

        return _gen()

    ac = client_factory(_create)
    async with ac:
        await ac.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 5
    assert captured["stream"] is True
