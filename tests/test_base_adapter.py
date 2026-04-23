"""Tests for BaseA2AAdapter contract and AdapterMetadata."""

from typing import AsyncIterator

import pytest
from a2a.types import Part

from a2a_adapter.base_adapter import AdapterMetadata, BaseA2AAdapter


# ── AdapterMetadata defaults ─────────────────────────────────


class TestAdapterMetadata:
    def test_defaults(self):
        m = AdapterMetadata()
        assert m.name == ""
        assert m.description == ""
        assert m.version == "1.0.0"
        assert m.skills == []
        assert m.input_modes == ["text"]
        assert m.output_modes == ["text"]
        assert m.streaming is False
        assert m.provider is None
        assert m.documentation_url is None
        assert m.icon_url is None

    def test_custom_values(self):
        m = AdapterMetadata(
            name="Test",
            description="desc",
            version="2.0.0",
            streaming=True,
            provider={"organization": "Acme"},
        )
        assert m.name == "Test"
        assert m.streaming is True
        assert m.provider == {"organization": "Acme"}


# ── BaseA2AAdapter contract ──────────────────────────────────


class ConcreteAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return f"echo: {user_input}"


class StreamingConcreteAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return "full"

    async def stream(self, user_input, context_id=None, **kwargs):
        for word in user_input.split():
            yield word


class TestBaseA2AAdapterContract:
    async def test_invoke_returns_string(self):
        adapter = ConcreteAdapter()
        result = await adapter.invoke("hello")
        assert result == "echo: hello"

    async def test_stream_raises_not_implemented(self):
        adapter = ConcreteAdapter()
        with pytest.raises(NotImplementedError):
            async for _ in adapter.stream("hi"):
                pass

    def test_supports_streaming_false_by_default(self):
        adapter = ConcreteAdapter()
        assert adapter.supports_streaming() is False

    def test_supports_streaming_true_when_overridden(self):
        adapter = StreamingConcreteAdapter()
        assert adapter.supports_streaming() is True

    async def test_cancel_is_noop_by_default(self):
        adapter = ConcreteAdapter()
        await adapter.cancel()

    async def test_close_is_noop_by_default(self):
        adapter = ConcreteAdapter()
        await adapter.close()

    def test_get_metadata_returns_empty(self):
        adapter = ConcreteAdapter()
        m = adapter.get_metadata()
        assert isinstance(m, AdapterMetadata)
        assert m.name == ""

    async def test_async_context_manager(self):
        async with ConcreteAdapter() as adapter:
            result = await adapter.invoke("test")
            assert result == "echo: test"

    async def test_streaming_adapter_yields_chunks(self):
        adapter = StreamingConcreteAdapter()
        chunks = []
        async for chunk in adapter.stream("hello world"):
            chunks.append(chunk)
        assert chunks == ["hello", "world"]


# ── Multimodal invoke ────────────────────────────────────────


class MultimodalAdapter(BaseA2AAdapter):
    async def invoke(self, user_input, context_id=None, **kwargs):
        return [Part(text="multi")]


class TestMultimodalInvoke:
    async def test_invoke_returns_parts_list(self):
        adapter = MultimodalAdapter()
        result = await adapter.invoke("hi")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == "multi"
