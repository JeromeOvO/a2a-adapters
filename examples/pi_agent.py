"""Example: expose one persistent Pi coding-agent session over A2A.

Prerequisites:
    - Install Pi so ``pi`` is on PATH, or set ``A2A_PI_COMMAND`` to a
      source command such as ``npx tsx /path/to/pi/packages/coding-agent/src/cli.ts``.
    - Configure Pi's model credentials before starting this server.

Usage:
    python examples/pi_agent.py
    python examples/pi_agent.py /path/to/project
"""

import os
import shlex
import sys

from a2a_adapter import PiAdapter, serve_agent


working_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
pi_command = shlex.split(os.getenv("A2A_PI_COMMAND", "pi"))

adapter = PiAdapter(
    working_dir=working_dir,
    pi_command=pi_command,
    session_id=os.getenv("A2A_PI_SESSION_ID", "a2a-pi-agent"),
    name="Pi Agent",
    description="Persistent Pi coding-agent session",
)

print("Starting Pi A2A agent...")
print(f"  Working directory: {working_dir}")
print(f"  Command: {pi_command}")
print("  Agent card: http://localhost:9012/.well-known/agent-card.json")

serve_agent(adapter, port=9012)
