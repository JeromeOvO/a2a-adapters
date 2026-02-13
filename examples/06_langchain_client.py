"""
Example: LangChain Agent that calls a remote A2A agent via a2a-adapter

Uses a2a-adapter's to_a2a() to host a LangChain ReAct agent as an A2A
server. The LangChain agent has a tool that calls another remote A2A agent,
showing how A2A agents can compose with each other.

Prerequisites:
- pip install a2a-adapter[langchain]
- OPENAI_API_KEY set in environment
- Another A2A agent running on port 9000 (e.g., python examples/01_single_n8n_agent.py)

Usage:
    python examples/06_langchain_client.py
"""

import json

import httpx
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from a2a_adapter import LangChainAdapter, serve_agent


# ── Tool: call a remote A2A agent ──

@tool
def ask_remote_agent(question: str) -> str:
    """Ask a remote A2A agent (running on port 9000) a question."""
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


# ── LangChain agent with the tool ──

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([ask_remote_agent])

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the ask_remote_agent tool to delegate questions to a specialist agent when needed."),
        ("user", "{input}"),
    ])
    | llm
)

# ── A2A: 3 lines ──

adapter = LangChainAdapter(
    runnable=chain,
    input_key="input",
    name="Orchestrator Agent",
    description="LangChain agent that can delegate to remote A2A agents",
)

serve_agent(adapter, port=8006)
