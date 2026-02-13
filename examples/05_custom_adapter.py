"""
Example: Custom Adapter as A2A Agent

Subclass BaseA2AAdapter — implement invoke(), get a full A2A server.

Usage:
    python examples/05_custom_adapter.py
"""

from a2a_adapter import BaseA2AAdapter, serve_agent


class EchoAdapter(BaseA2AAdapter):
    async def invoke(self, user_input: str, context_id: str | None = None) -> str:
        return f"You said: {user_input}"


serve_agent(EchoAdapter(), port=8003)
