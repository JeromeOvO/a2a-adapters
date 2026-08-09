"""Thin command-line entry point for serving coding-agent adapters."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from .integrations.claude_code import ClaudeCodeAdapter
from .integrations.codex import CodexAdapter
from .integrations.openclaw import OpenClawAdapter, VALID_THINKING_LEVELS
from .integrations.pi import PiAdapter
from .server import serve_agent


_PI_MANAGED_OPTIONS = {
    "--mode",
    "--print",
    "-p",
    "--continue",
    "-c",
    "--resume",
    "-r",
    "--session",
    "--session-id",
    "--session-dir",
    "--fork",
    "--no-session",
}
_CODEX_MANAGED_OPTIONS = {
    "--json",
    "--output-last-message",
    "-o",
    "--ephemeral",
    "--cd",
    "-C",
}
_CLAUDE_CODE_MANAGED_OPTIONS = {
    "--print",
    "-p",
    "--output-format",
    "--resume",
    "--continue",
    "-c",
    "--session-id",
    "--disallowedTools",
}
_OPENCLAW_MANAGED_OPTIONS = {
    "--local",
    "--message",
    "--json",
    "--session-id",
    "--thinking",
    "--agent",
}
_MANAGED_OPTIONS = {
    "pi": _PI_MANAGED_OPTIONS,
    "codex": _CODEX_MANAGED_OPTIONS,
    "claude": _CLAUDE_CODE_MANAGED_OPTIONS,
    "openclaw": _OPENCLAW_MANAGED_OPTIONS,
}


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _add_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-h", "--help", "-help", action="help", help="show this help message")


def _add_server_options(
    parser: argparse.ArgumentParser,
) -> None:
    _add_help(parser)
    parser.add_argument(
        "--cwd", default=".", help="Agent working directory (default: current directory)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="A2A server bind host")
    parser.add_argument("--port", type=_port, default=9000, help="A2A server port")
    parser.add_argument(
        "--timeout", type=_positive_float, default=600.0, help="Agent timeout in seconds"
    )
    parser.add_argument("--name", default="", help="Agent Card display name")
    parser.add_argument("--description", default="", help="Agent Card description")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    parser.add_argument("--executable", help="Path to the underlying agent executable")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a2a-adapter",
        description="Expose a supported local agent as an A2A server.",
        epilog="For CLI-backed agents, pass native options after '--'.",
        add_help=False,
    )
    _add_help(parser)
    subparsers = parser.add_subparsers(dest="agent", required=True)

    pi_parser = subparsers.add_parser("pi", help="Serve Pi over A2A", add_help=False)
    _add_server_options(pi_parser)
    pi_parser.add_argument("--session-id", default="a2a-pi-agent")
    pi_parser.add_argument("--session-dir")

    codex_parser = subparsers.add_parser("codex", help="Serve Codex over A2A", add_help=False)
    _add_server_options(codex_parser)

    claude_parser = subparsers.add_parser("claude", help="Serve Claude over A2A", add_help=False)
    _add_server_options(claude_parser)

    openclaw_parser = subparsers.add_parser(
        "openclaw", help="Serve OpenClaw over A2A", add_help=False
    )
    _add_server_options(openclaw_parser)
    openclaw_parser.add_argument("--session-id")
    openclaw_parser.add_argument("--agent-id")
    openclaw_parser.add_argument("--thinking", choices=sorted(VALID_THINKING_LEVELS), default="low")

    return parser


def _split_agent_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    try:
        separator = values.index("--")
    except ValueError:
        return values, []
    return values[:separator], values[separator + 1 :]


def _option_name(value: str) -> str:
    return value.split("=", 1)[0]


def _validate_agent_args(
    parser: argparse.ArgumentParser,
    agent: str,
    agent_args: Sequence[str],
) -> None:
    managed = _MANAGED_OPTIONS[agent]
    for value in agent_args:
        option = _option_name(value)
        if option in managed:
            parser.error(
                f"{agent} option {option!r} is managed by a2a-adapter and cannot be passed after '--'"
            )
    if agent == "codex" and any(value in {"resume", "review"} for value in agent_args):
        parser.error("Codex subcommands are managed by a2a-adapter")


def _working_directory(parser: argparse.ArgumentParser, value: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        parser.error(f"working directory does not exist or is not a directory: {path}")
    return str(path)


def _pi_command(parser: argparse.ArgumentParser, executable: str | None) -> list[str]:
    if executable:
        return [executable]
    command = shlex.split(os.getenv("A2A_PI_COMMAND", "pi"))
    if not command:
        parser.error("A2A_PI_COMMAND must not be empty")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, construct an existing adapter, and serve it."""
    parser = _build_parser()
    wrapper_args, agent_args = _split_agent_args(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(wrapper_args)
    _validate_agent_args(parser, args.agent, agent_args)
    working_dir = _working_directory(parser, args.cwd)

    if args.agent == "pi":
        adapter = PiAdapter(
            working_dir=working_dir,
            pi_command=_pi_command(parser, args.executable),
            cli_args=agent_args,
            session_id=args.session_id,
            session_dir=args.session_dir,
            timeout=args.timeout,
            name=args.name,
            description=args.description,
        )
    elif args.agent == "codex":
        adapter = CodexAdapter(
            working_dir=working_dir,
            codex_path=args.executable or "codex",
            cli_args=agent_args,
            timeout=args.timeout,
            name=args.name,
            description=args.description,
        )
    elif args.agent == "claude":
        adapter = ClaudeCodeAdapter(
            working_dir=working_dir,
            claude_path=args.executable or "claude",
            cli_args=agent_args,
            timeout=args.timeout,
            name=args.name,
            description=args.description,
        )
    elif args.agent == "openclaw":
        adapter = OpenClawAdapter(
            session_id=args.session_id,
            agent_id=args.agent_id,
            thinking=args.thinking,
            timeout=args.timeout,
            openclaw_path=args.executable or "openclaw",
            working_directory=working_dir,
            cli_args=agent_args,
            name=args.name,
            description=args.description,
        )
    else:  # pragma: no cover - argparse limits this to registered subcommands
        raise AssertionError(f"unsupported agent: {args.agent}")

    display_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Starting {args.agent} as an A2A agent")
    print(f"Working directory: {working_dir}")
    print(f"Agent Card: http://{display_host}:{args.port}/.well-known/agent-card.json")
    serve_agent(
        adapter,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
