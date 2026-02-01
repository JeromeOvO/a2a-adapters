#!/usr/bin/env python3
"""
Example: OpenClaw Agent via A2A Protocol

This example demonstrates how to expose an OpenClaw agent as an A2A-compliant
agent using the OpenClawAgentAdapter.

Requirements:
- OpenClaw CLI installed and in PATH (`npm install -g openclaw`)
- OpenClaw configured with API keys (ANTHROPIC_API_KEY, etc.)
- Valid OpenClaw configuration at ~/.openclaw/config.yaml

Usage:
    # Start the A2A server
    python examples/08_openclaw_agent.py

    # In another terminal, test with curl:
    curl -X POST http://localhost:9008/message/send \
        -H "Content-Type: application/json" \
        -d '{"message": {"role": "user", "parts": [{"text": "Write a hello world function in Python"}]}}'

    # Poll for task completion:
    curl http://localhost:9008/task/{task_id}
"""

import asyncio
import logging

from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from a2a_adapter import load_a2a_agent, serve_agent

# Configure logging - use DEBUG to see OpenClaw adapter internals
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Reduce noise from other loggers
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Server configuration
AGENT_PORT = 9008


async def setup_agent():
    """Set up and return the agent configuration."""
    
    # Load the OpenClaw adapter
    # async_mode=False: blocks until command completes, returns Message directly
    # async_mode=True: returns Task immediately and processes in background (requires polling)
    adapter = await load_a2a_agent({
        "adapter": "openclaw",
        "session_id": "a2a-demo-session",  # Optional: auto-generated if not provided
        "agent_id": None,  # Optional: use default agent
        "thinking": "low",  # Thinking level: off|minimal|low|medium|high|xhigh
        "timeout": 300,  # Command timeout in seconds
        "async_mode": False,  # Block until command completes, return Message directly
        # "openclaw_path": "openclaw",  # Path to openclaw binary
        # "working_directory": None,  # Working directory for subprocess
        # "env_vars": {},  # Additional environment variables
    })
    
    # Define the agent card (A2A metadata)
    agent_card = AgentCard(
        name="OpenClaw Agent",
        description="Personal AI super agent powered by OpenClaw. Can help with a wide variety "
                    "of tasks including coding, research, automation, and more.",
        url=f"http://localhost:{AGENT_PORT}",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(
            streaming=False,  # OpenClaw CLI doesn't support streaming
            pushNotifications=False,
        ),
        skills=[
            AgentSkill(
                id="general",
                name="General Assistant",
                description="Help with a wide variety of tasks, answer questions, and provide guidance",
                tags=["general", "assistant"],
            ),
            AgentSkill(
                id="coding",
                name="Coding Assistant",
                description="Help with coding tasks, code review, debugging, and software development",
                tags=["coding", "development"],
            ),
            AgentSkill(
                id="research",
                name="Research & Analysis",
                description="Research topics, analyze information, and provide insights",
                tags=["research", "analysis"],
            ),
        ],
    )
    
    return adapter, agent_card


def main():
    """Main entry point - setup agent and start server."""
    
    # Run async setup
    adapter, agent_card = asyncio.run(setup_agent())
    
    logger.info("Starting OpenClaw A2A agent server on http://localhost:%d", AGENT_PORT)
    logger.info("Agent card: %s", agent_card.name)
    logger.info("Async mode: %s", adapter.supports_async_tasks())
    
    # Start the A2A server (this will block)
    # This will handle:
    # - POST /message/send - Send a message to the agent (returns Task in async mode)
    # - GET /task/{task_id} - Poll for task status
    # - POST /task/{task_id}/cancel - Cancel a running task
    # - DELETE /task/{task_id} - Delete a completed task
    serve_agent(
        agent_card=agent_card,
        adapter=adapter,
        port=AGENT_PORT,
    )


if __name__ == "__main__":
    main()
