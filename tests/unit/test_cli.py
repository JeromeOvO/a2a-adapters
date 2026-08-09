"""Tests for the thin a2a-adapter command-line entry point."""

from unittest.mock import patch

import pytest

from a2a_adapter.cli import main
from a2a_adapter.integrations.claude_code import ClaudeCodeAdapter
from a2a_adapter.integrations.codex import CodexAdapter
from a2a_adapter.integrations.openclaw import OpenClawAdapter
from a2a_adapter.integrations.pi import PiAdapter


def test_pi_cli_uses_current_contract_and_forwards_native_args(tmp_path, monkeypatch):
    monkeypatch.setenv("A2A_PI_COMMAND", "npx tsx /opt/pi/src/cli.ts")

    with patch("a2a_adapter.cli.serve_agent") as serve:
        result = main(
            [
                "pi",
                "--cwd",
                str(tmp_path),
                "--port",
                "9012",
                "--",
                "--model",
                "anthropic/example",
                "--thinking",
                "high",
            ]
        )

    assert result == 0
    adapter = serve.call_args.args[0]
    assert isinstance(adapter, PiAdapter)
    assert adapter.working_dir == str(tmp_path.resolve())
    assert adapter.pi_command == ["npx", "tsx", "/opt/pi/src/cli.ts"]
    assert adapter.cli_args == ["--model", "anthropic/example", "--thinking", "high"]
    serve.assert_called_once_with(adapter, host="127.0.0.1", port=9012, log_level="info")


def test_codex_cli_forwards_native_args_before_adapter_protocol_args(tmp_path):
    with patch("a2a_adapter.cli.serve_agent") as serve:
        main(
            [
                "codex",
                "--cwd",
                str(tmp_path),
                "--executable",
                "/opt/codex",
                "--",
                "--model",
                "example-model",
                "--sandbox",
                "workspace-write",
            ]
        )

    adapter = serve.call_args.args[0]
    assert isinstance(adapter, CodexAdapter)
    assert adapter.codex_path == "/opt/codex"
    assert adapter.cli_args == ["--model", "example-model", "--sandbox", "workspace-write"]
    assert adapter._build_command("hello", "ctx").args == [
        "/opt/codex",
        "exec",
        "--model",
        "example-model",
        "--sandbox",
        "workspace-write",
        "--json",
        "hello",
    ]

    adapter._sessions["ctx"] = "thread-1"
    assert adapter._build_command("continue", "ctx").args == [
        "/opt/codex",
        "exec",
        "--model",
        "example-model",
        "--sandbox",
        "workspace-write",
        "resume",
        "thread-1",
        "--json",
        "continue",
    ]


def test_claude_cli_forwards_native_args(tmp_path):
    with patch("a2a_adapter.cli.serve_agent") as serve:
        main(
            [
                "claude",
                "--cwd",
                str(tmp_path),
                "--",
                "--model",
                "example-model",
            ]
        )

    adapter = serve.call_args.args[0]
    assert isinstance(adapter, ClaudeCodeAdapter)
    command = adapter._build_command("hello", "ctx").args
    assert command[:4] == ["claude", "--model", "example-model", "-p"]


def test_openclaw_cli_maps_owned_options_and_forwards_native_args(tmp_path):
    with patch("a2a_adapter.cli.serve_agent") as serve:
        main(
            [
                "openclaw",
                "--cwd",
                str(tmp_path),
                "--agent-id",
                "main",
                "--thinking",
                "high",
                "--",
                "--native-option",
                "value",
            ]
        )

    adapter = serve.call_args.args[0]
    assert isinstance(adapter, OpenClawAdapter)
    assert adapter.working_directory == str(tmp_path.resolve())
    assert adapter._build_command("hello", "session") == [
        "openclaw",
        "agent",
        "--native-option",
        "value",
        "--local",
        "--message",
        "hello",
        "--json",
        "--session-id",
        "session",
        "--thinking",
        "high",
        "--agent",
        "main",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["-help"],
        ["pi", "-help"],
        ["codex", "-help"],
        ["claude", "-help"],
        ["openclaw", "-help"],
    ],
)
def test_help_aliases_include_usage(argv, capsys):
    with pytest.raises(SystemExit, match="0"):
        main(argv)
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("agent", "managed_option"),
    [
        ("pi", "--mode"),
        ("pi", "--session-id=other"),
        ("pi", "-pignored"),
        ("codex", "--json"),
        ("codex", "-C/tmp"),
        ("codex", "-opath"),
        ("claude", "--output-format"),
        ("claude", "-csession-id"),
        ("openclaw", "--message"),
    ],
)
def test_cli_rejects_adapter_managed_native_options(tmp_path, agent, managed_option):
    with pytest.raises(SystemExit, match="2"):
        main([agent, "--cwd", str(tmp_path), "--", managed_option])


def test_cli_rejects_missing_working_directory(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        main(["pi", "--cwd", str(tmp_path / "missing")])


def test_cli_does_not_expose_hermes_in_v1():
    with pytest.raises(SystemExit, match="2"):
        main(["hermes"])


def test_cli_uses_official_claude_command_name_only():
    with pytest.raises(SystemExit, match="2"):
        main(["claude-code"])
