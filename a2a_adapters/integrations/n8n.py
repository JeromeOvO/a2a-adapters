"""
n8n adapter for A2A Protocol.

This adapter enables n8n workflows to be exposed as A2A-compliant agents
by forwarding A2A messages to n8n webhooks.
"""

import json
import asyncio
import time
import uuid
from typing import Any, Dict

from httpx import HTTPStatusError, ConnectError, ReadTimeout
from a2a.types import Message, MessageSendParams, Task, TextPart
from ..adapter import BaseAgentAdapter


class N8nAgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating n8n workflows as A2A agents.
    
    This adapter forwards A2A message requests to an n8n webhook URL and
    translates the response back to A2A format.
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: int = 30,
        headers: Dict[str, str] | None = None,
        max_retries: int = 2,  
        backoff: float = 0.25,
    ):
        """
        Initialize the n8n adapter.
        
        Args:
            webhook_url: The n8n webhook URL to send requests to
            timeout: HTTP request timeout in seconds (default: 30)
            headers: Optional additional HTTP headers to include in requests
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.headers = dict(headers) if headers else {}
        self.max_retries = max(0, int(max_retries))
        self.backoff = float(backoff)  
        self._client: httpx.AsyncClient | None = None           
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def handle(self, params: MessageSendParams) -> Message | Task:
        """Handle a non-streaming A2A message request."""
        framework_input = await self.to_framework(params)
        framework_output = await self.call_framework(framework_input, params)
        return await self.from_framework(framework_output, params)

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """
        Convert A2A message parameters to n8n webhook payload format.
        
        Extracts the user's message text and constructs a JSON payload
        suitable for posting to an n8n webhook.
        
        Args:
            params: A2A message parameters
            
        Returns:
            Dictionary with 'message' and optional 'metadata' keys
        """
        # Extract text from the last user message
        def _extract_text(msg) -> str:
            if not hasattr(msg, "content"):
                return ""
            c = msg.content
            if isinstance(c, str):
                 return c.strip()
            if isinstance(c, list):
                 parts: list[str] = []
                 for it in c:
                    t = getattr(it, "text", "")
                    t = (t or "").strip()
                    if t:
                        parts.append(t)
                 return " ".join(parts)

             return ""

        user_message = ""
        if params.messages:
            user_message = _extract_text(params.messages[-1])

        return {
            "message": user_message,
            "metadata": {
                "session_id": getattr(params, "session_id", None),
                "context": getattr(params, "context", None),
            },
        }

    async def call_framework(
         self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Dict[str, Any]:
        client = await self._get_client()
        req_id = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": req_id,           
            **self.headers,
        }

     for attempt in range(self.max_retries + 1):
        start = time.monotonic()
        try:
            resp = await client.post(self.webhook_url, json=framework_input, headers=headers)
            dur_ms = int((time.monotonic() - start) * 1000)

            # 4xx 
            if 400 <= resp.status_code < 500:
                text = (await resp.aread()).decode(errors="ignore")
                raise ValueError(f"n8n webhook returned {resp.status_code} (req_id={req_id}, {dur_ms}ms): {text[:512]}")

            # 5xx raise_for_status
            resp.raise_for_status()
            return resp.json()

        except HTTPStatusError as e:
            # only 5xx
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"n8n upstream 5xx after retries (req_id={req_id}): {e}") from e

        except (ConnectError, ReadTimeout) as e:
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff * (2 ** attempt))
                continue

    async def from_framework(
        self, framework_output: Dict[str, Any], params: MessageSendParams
    ) -> Message | Task:
        """
        Convert n8n webhook response to A2A Message.
        
        Args:
            framework_output: JSON response from n8n
            params: Original A2A parameters
            
        Returns:
            A2A Message with the n8n response
        """
        # Extract response text from n8n output
        # Support common n8n response formats
        if "output" in framework_output:
            response_text = str(framework_output["output"])
        elif "result" in framework_output:
            response_text = str(framework_output["result"])
        elif "message" in framework_output:
            response_text = str(framework_output["message"])
        else:
            # Fallback: serialize entire response as JSON
            response_text = json.dumps(framework_output, indent=2)

        return Message(
            role="assistant",
            content=[TextPart(type="text", text=response_text)],
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def supports_streaming(self) -> bool:
        """Check if this adapter supports streaming responses."""
        return False

