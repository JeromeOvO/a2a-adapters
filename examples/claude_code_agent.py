"""
Example: Claude Code as A2A Agent

Expose the Claude Code CLI as an A2A-compatible agent.

Prerequisites:
- Claude Code CLI installed: npm install -g @anthropic-ai/claude-code
- ANTHROPIC_API_KEY set in environment

Usage:
    python examples/claude_code_agent.py
    python examples/claude_code_agent.py /path/to/project
    # Agent card at http://localhost:9010/.well-known/agent.json
"""

import os
import sys

from a2a_adapter import ClaudeCodeAdapter, serve_agent

working_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

adapter = ClaudeCodeAdapter(
    working_dir=working_dir,
    timeout=600,
    name="Claude Code Agent",
    description="Claude Code AI coding agent — code generation, debugging, refactoring",
)

print(f"Starting Claude Code A2A agent...")
print(f"  Working directory: {working_dir}")
print(f"  Agent card: http://localhost:9010/.well-known/agent.json")

serve_agent(adapter, port=9010)
