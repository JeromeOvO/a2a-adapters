"""
Example: OpenAI Codex as A2A Agent

Expose the OpenAI Codex CLI as an A2A-compatible agent.

Prerequisites:
- Codex CLI installed: npm install -g @openai/codex
- OPENAI_API_KEY set in environment

Usage:
    python examples/codex_agent.py
    python examples/codex_agent.py /path/to/project
    # Agent card at http://localhost:9011/.well-known/agent.json
"""

import os
import sys

from a2a_adapter import CodexAdapter, serve_agent

working_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

adapter = CodexAdapter(
    working_dir=working_dir,
    timeout=600,
    name="Codex Agent",
    description="OpenAI Codex CLI coding agent",
)

print(f"Starting Codex A2A agent...")
print(f"  Working directory: {working_dir}")
print(f"  Agent card: http://localhost:9011/.well-known/agent.json")

serve_agent(adapter, port=9011)
