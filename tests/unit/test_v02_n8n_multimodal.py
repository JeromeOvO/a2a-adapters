"""Tests for N8nAdapter V0.2 multimodal methods after V1.0 proto migration."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import Part

from a2a_adapter.integrations.n8n import N8nAdapter


class TestBuildMultimodalPayload:
    """Test _build_multimodal_payload with V1.0 proto Part fields."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test",
            multimodal_mode=True,
        )

    async def test_text_only_parts_ignored(self, adapter):
        """Parts with only text set should not appear in files/images."""
        ctx = MagicMock()
        text_part = Part(text="hello")
        ctx.message.parts = [text_part]

        payload = await adapter._build_multimodal_payload("hello", "ctx-1", ctx)
        assert "files" not in payload
        assert "images" not in payload

    async def test_url_part_detected_as_file(self, adapter):
        """A Part with url set should be fetched and added to files."""
        ctx = MagicMock()
        file_part = Part(url="http://example.com/doc.pdf", filename="doc.pdf", media_type="application/pdf")
        ctx.message.parts = [file_part]

        mock_resp = MagicMock()
        mock_resp.content = b"pdf-bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        adapter._client = mock_client

        payload = await adapter._build_multimodal_payload("check this", "ctx-1", ctx)
        assert "files" in payload
        assert len(payload["files"]) == 1
        assert payload["files"][0]["name"] == "doc.pdf"
        assert payload["files"][0]["mime_type"] == "application/pdf"
        assert payload["files"][0]["data"] == base64.b64encode(b"pdf-bytes").decode("utf-8")
        assert payload["files"][0]["uri"] == "http://example.com/doc.pdf"

    async def test_image_categorized_separately(self, adapter):
        """A Part with image/* media_type should go to images, not files."""
        ctx = MagicMock()
        img_part = Part(url="http://example.com/photo.png", filename="photo.png", media_type="image/png")
        ctx.message.parts = [img_part]

        mock_resp = MagicMock()
        mock_resp.content = b"png-bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        adapter._client = mock_client

        payload = await adapter._build_multimodal_payload("look at this", "ctx-1", ctx)
        assert "images" in payload
        assert "files" not in payload
        assert payload["images"][0]["name"] == "photo.png"

    async def test_raw_part_no_fetch(self, adapter):
        """A Part with raw bytes should use data directly, not HTTP fetch."""
        ctx = MagicMock()
        raw_part = Part(raw=b"raw-content", filename="data.bin", media_type="application/octet-stream")
        ctx.message.parts = [raw_part]

        payload = await adapter._build_multimodal_payload("process", "ctx-1", ctx)
        assert "files" in payload
        assert payload["files"][0]["data"] == base64.b64encode(b"raw-content").decode("utf-8")


class TestFetchFileContent:
    """Test _fetch_file_content with V1.0 proto Part fields."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(webhook_url="http://localhost:5678/webhook/test")

    async def test_url_part_fetches(self, adapter):
        part = Part(url="http://example.com/file.bin")
        mock_resp = MagicMock()
        mock_resp.content = b"fetched"
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.is_closed = False
        adapter._client = mock_client

        result = await adapter._fetch_file_content(part)
        assert result == b"fetched"

    async def test_raw_part_returns_directly(self, adapter):
        part = Part(raw=b"inline-data")
        result = await adapter._fetch_file_content(part)
        assert result == b"inline-data"

    async def test_empty_part_raises(self, adapter):
        part = Part(text="just text")
        with pytest.raises(ValueError, match="no url or raw"):
            await adapter._fetch_file_content(part)


class TestExtractResponse:
    """Test _extract_response with V1.0 Part construction."""

    @pytest.fixture
    def adapter(self):
        return N8nAdapter(
            webhook_url="http://localhost:5678/webhook/test",
            multimodal_mode=True,
        )

    def test_text_only_response(self, adapter):
        output = {"output": "hello world"}
        parts = adapter._extract_response(output)
        assert len(parts) == 1
        assert parts[0].text == "hello world"

    def test_with_files(self, adapter):
        output = {
            "output": "here are results",
            "files": [{"url": "http://example.com/report.pdf", "name": "report.pdf", "mime_type": "application/pdf"}],
        }
        parts = adapter._extract_response(output)
        assert len(parts) == 2
        assert parts[0].text == "here are results"
        assert parts[1].url == "http://example.com/report.pdf"
        assert parts[1].filename == "report.pdf"
        assert parts[1].media_type == "application/pdf"

    def test_with_images(self, adapter):
        output = {
            "output": "chart generated",
            "images": [{"url": "http://example.com/chart.png", "name": "chart.png", "mimeType": "image/png"}],
        }
        parts = adapter._extract_response(output)
        assert len(parts) == 2
        assert parts[1].url == "http://example.com/chart.png"
        assert parts[1].media_type == "image/png"

    def test_empty_response(self, adapter):
        parts = adapter._extract_response({})
        assert len(parts) == 1
        assert parts[0].text == "[Empty response]"

    def test_non_dict_response(self, adapter):
        parts = adapter._extract_response([{"output": "from list"}])
        assert len(parts) == 1
        assert "from list" in parts[0].text
