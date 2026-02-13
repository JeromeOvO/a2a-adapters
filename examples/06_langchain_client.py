"""
Example: Use a2a-adapter to call a remote A2A agent from LangChain

This wraps a remote A2A agent as a LangChain tool, so you can use it
inside any LangChain chain or agent.

Prerequisites:
- An A2A agent running on port 9000 (e.g., python examples/01_single_n8n_agent.py)
- pip install langchain-openai httpx

Usage:
    python examples/06_langchain_client.py
"""

import asyncio
import json

import httpx
from langchain_core.tools import tool


@tool
def ask_a2a_agent(question: str) -> str:
    """Ask a remote A2A agent a question and return its answer."""
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "msg-1",
                "parts": [{"kind": "text", "text": question}],
            }
        },
    }
    resp = httpx.post("http://localhost:9000", json=payload, timeout=60)
    data = resp.json()
    msg = data.get("result", {}).get("status", {}).get("message", {})
    parts = msg.get("parts", [])
    return " ".join(p["text"] for p in parts if p.get("kind") == "text") or json.dumps(data)


async def main():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([ask_a2a_agent])

    print("Asking LLM (with A2A agent tool)...")
    resp = await llm.ainvoke([HumanMessage(content="What is 25 * 37 + 18? Use the tool.")])
    print(f"Response: {resp}")

    # If the LLM made a tool call, execute it
    for tc in resp.tool_calls:
        result = ask_a2a_agent.invoke(tc["args"])
        print(f"Tool result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
