"""
Example: Single n8n Agent Server

This example demonstrates how to expose an n8n workflow as an A2A-compliant agent.
The n8n workflow receives A2A messages via webhook and returns responses.

Prerequisites:
- A running n8n instance with a webhook configured
- Configure the adapter to match your n8n workflow's expected payload format

Usage:
    python examples/01_single_n8n_agent.py
"""


import asyncio

from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a_adapter import load_a2a_agent, serve_agent


async def setup_agent():
    """Set up and return the agent configuration."""
    # Configuration for the n8n adapter
    webhook_url = "http://localhost:5678/webhook/my-webhook"

    # Configuration for the agent card
    agent_port = 9000
    agent_name = "N8n Math Agent"
    agent_description = "Math operations agent powered by an n8n workflow. Can " \
        "perform calculations, solve equations, and provide mathematical insights."
    agent_url = f"http://localhost:{agent_port}"
    agent_version = "1.0.0"
    agent_default_input_modes = ["text"]
    agent_default_output_modes = ["text"]
    agent_capabilities = AgentCapabilities(streaming=False)
    agent_skills = [
        AgentSkill(id="calculate",
        name="calculate",
        description="Perform mathematical calculations",
        tags=["math", "calculation"]),
        AgentSkill(id="solve_equation",
        name="solve_equation",
        description="Solve mathematical equations",
        tags=["math", "equation"]),
    ]
    agent_card = AgentCard(
        name=agent_name,
        description=agent_description,
        url=agent_url,
        version=agent_version,
        default_input_modes=agent_default_input_modes,
        default_output_modes=agent_default_output_modes,
        capabilities=agent_capabilities,
        skills=agent_skills,
    )

    # Load the adapter with custom payload mapping
    adapter = await load_a2a_agent({
        "adapter": "n8n",
        "webhook_url": webhook_url,
        "timeout": 30,
        # Custom payload template - add static fields your n8n workflow expects
        "payload_template": {
            "name": "A2A Agent",
        },
        # Custom message field name - use "event" instead of default "message"
        "message_field": "event",
        "headers": {
            # Optional: Add custom headers if your n8n webhook requires authentication
            # "Authorization": "Bearer YOUR_TOKEN"
        }
    })

    return adapter, agent_card, agent_port, webhook_url


def main():
    """Main entry point - setup agent and start server."""

    # Run async setup
    adapter, agent_card, agent_port, webhook_url = asyncio.run(setup_agent())

    # Start serving the agent (this will block)
    print(f"Starting N8n Math Agent on port {agent_port}...")
    print(f"Webhook URL: {webhook_url}")
    serve_agent(agent_card=agent_card, adapter=adapter, port=agent_port)


if __name__ == "__main__":
    main()

