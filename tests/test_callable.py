"""Tests for CallableAdapter (v0.2)."""

import pytest
from a2a.types import Part, TextPart

from a2a_adapter.integrations.callable import CallableAdapter


# ── Invoke ───────────────────────────────────────────────────


class TestCallableAdapterInvoke:
    async def test_string_response(self):
        async def echo(inputs):
            return f"echo: {inputs['message']}"

        adapter = CallableAdapter(func=echo)
        result = await adapter.invoke("hello")
        assert result == "echo: hello"

    async def test_dict_response_with_known_key(self):
        async def agent(inputs):
            return {"response": "answer"}

        adapter = CallableAdapter(func=agent)
        result = await adapter.invoke("q")
        assert result == "answer"

    async def test_dict_response_fallback_to_json(self):
        async def agent(inputs):
            return {"custom_key": 42}

        adapter = CallableAdapter(func=agent)
        result = await adapter.invoke("q")
        assert "custom_key" in result
        assert "42" in result

    async def test_non_string_cast(self):
        async def agent(inputs):
            return 42

        adapter = CallableAdapter(func=agent)
        result = await adapter.invoke("q")
        assert result == "42"

    async def test_context_id_passed_through(self):
        captured = {}

        async def agent(inputs):
            captured.update(inputs)
            return "ok"

        adapter = CallableAdapter(func=agent)
        await adapter.invoke("hi", context_id="ctx-123")
        assert captured["context_id"] == "ctx-123"


# ── Streaming ────────────────────────────────────────────────


class TestCallableAdapterStream:
    async def test_streaming_yields_chunks(self):
        async def gen(inputs):
            for word in inputs["message"].split():
                yield word

        adapter = CallableAdapter(func=gen, streaming=True)
        chunks = []
        async for chunk in adapter.stream("hello world"):
            chunks.append(chunk)
        assert chunks == ["hello", "world"]

    async def test_non_string_chunks_cast(self):
        async def gen(inputs):
            yield 1
            yield 2

        adapter = CallableAdapter(func=gen, streaming=True)
        chunks = []
        async for chunk in adapter.stream("go"):
            chunks.append(chunk)
        assert chunks == ["1", "2"]


# ── supports_streaming ───────────────────────────────────────


class TestCallableAdapterStreaming:
    def test_not_streaming_by_default(self):
        async def f(inputs):
            return "ok"

        adapter = CallableAdapter(func=f)
        assert adapter.supports_streaming() is False

    def test_streaming_when_enabled(self):
        async def f(inputs):
            yield "ok"

        adapter = CallableAdapter(func=f, streaming=True)
        assert adapter.supports_streaming() is True


# ── Metadata ─────────────────────────────────────────────────


class TestCallableAdapterMetadata:
    def test_default_metadata(self):
        async def my_func(inputs):
            return "ok"

        adapter = CallableAdapter(func=my_func)
        meta = adapter.get_metadata()
        assert meta.name == "my_func"
        assert meta.streaming is False

    def test_custom_metadata(self):
        async def f(inputs):
            return "ok"

        adapter = CallableAdapter(
            func=f,
            name="TestAgent",
            description="A test agent",
            streaming=True,
            skills=[{"id": "s1", "name": "skill1", "description": "d"}],
            provider={"organization": "Acme"},
            documentation_url="https://example.com/docs",
            icon_url="https://example.com/icon.png",
        )
        meta = adapter.get_metadata()
        assert meta.name == "TestAgent"
        assert meta.description == "A test agent"
        assert meta.streaming is True
        assert len(meta.skills) == 1
        assert meta.provider == {"organization": "Acme"}
        assert meta.documentation_url == "https://example.com/docs"
        assert meta.icon_url == "https://example.com/icon.png"
