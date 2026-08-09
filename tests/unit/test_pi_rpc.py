"""Protocol tests for the persistent Pi JSONL RPC process."""

import asyncio
import json
from collections.abc import Callable
from unittest.mock import patch

import pytest

from a2a_adapter.integrations._pi_rpc import (
    PiProcessLostError,
    PiRpcProcess,
    PiTurnAbortedError,
)


class QueueReader:
    def __init__(self) -> None:
        self._lines: asyncio.Queue[bytes] = asyncio.Queue()

    async def readline(self) -> bytes:
        return await self._lines.get()

    async def read(self, _size: int = -1) -> bytes:
        return await self._lines.get()

    def feed_json(self, value: dict) -> None:
        self._lines.put_nowait((json.dumps(value) + "\n").encode())

    def feed_eof(self) -> None:
        self._lines.put_nowait(b"")


class CommandWriter:
    def __init__(self, on_command: Callable[[dict], None]) -> None:
        self._on_command = on_command

    def write(self, data: bytes) -> None:
        self._on_command(json.loads(data.decode()))

    async def drain(self) -> None:
        return None


class FakeProcess:
    def __init__(self, on_command: Callable[[dict], None]) -> None:
        self.stdout = QueueReader()
        self.stderr = QueueReader()
        self.stdin = CommandWriter(on_command)
        self.returncode: int | None = None
        self._exited = asyncio.Event()
        self.pid = 1234

    async def wait(self) -> int:
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.exit(-15)

    def kill(self) -> None:
        self.exit(-9)

    def exit(self, code: int) -> None:
        if self.returncode is not None:
            return
        self.returncode = code
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._exited.set()


def _rpc(tmp_path, *, timeout: float = 1) -> PiRpcProcess:
    return PiRpcProcess(
        command=["pi"],
        working_dir=str(tmp_path),
        session_id="session-1",
        session_dir=str(tmp_path / "sessions"),
        timeout=timeout,
        startup_timeout=timeout,
        cancel_grace_timeout=timeout,
    )


@pytest.mark.asyncio
async def test_prompt_streams_text_until_agent_settled(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "hello"},
                }
            )
            process.stdout.feed_json({"type": "agent_end", "willRetry": True})
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": " world"},
                }
            )
            process.stdout.feed_json({"type": "agent_settled"})

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations._pi_rpc.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        chunks = [chunk async for chunk in rpc.prompt("hello")]

    assert chunks == ["hello", " world"]
    await rpc.close()


@pytest.mark.asyncio
async def test_process_loss_after_prompt_acceptance_is_not_retryable(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
            process.exit(1)

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations._pi_rpc.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        with pytest.raises(PiProcessLostError) as exc_info:
            _ = [chunk async for chunk in rpc.prompt("do work")]

    assert exc_info.value.accepted is True


@pytest.mark.asyncio
async def test_abort_uses_rpc_and_waits_for_settled(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
        elif command["type"] == "abort":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "abort",
                    "success": True,
                }
            )
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "error", "reason": "aborted"},
                }
            )
            process.stdout.feed_json({"type": "agent_settled"})

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations._pi_rpc.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        turn = asyncio.create_task(_collect(rpc.prompt("work")))
        await asyncio.sleep(0)
        await rpc.abort()
        with pytest.raises(PiTurnAbortedError, match="aborted"):
            await turn

    await rpc.close()


@pytest.mark.asyncio
async def test_blocking_extension_ui_is_cancelled_instead_of_hanging(tmp_path):
    process: FakeProcess
    extension_responses: list[dict] = []

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
            process.stdout.feed_json(
                {
                    "type": "extension_ui_request",
                    "id": "ui-1",
                    "method": "confirm",
                    "message": "Continue?",
                }
            )
        elif command["type"] == "extension_ui_response":
            extension_responses.append(command)
            process.stdout.feed_json(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {
                        "type": "text_delta",
                        "delta": "cancelled safely",
                    },
                }
            )
            process.stdout.feed_json({"type": "agent_settled"})

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations._pi_rpc.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        chunks = [chunk async for chunk in rpc.prompt("work")]

    assert chunks == ["cancelled safely"]
    assert extension_responses == [
        {"type": "extension_ui_response", "id": "ui-1", "cancelled": True}
    ]
    await rpc.close()


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_leave_pi_turn_running(tmp_path):
    process: FakeProcess

    def on_command(command: dict) -> None:
        if command["type"] == "get_state":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "session-1"},
                }
            )
        elif command["type"] == "prompt":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "prompt",
                    "success": True,
                }
            )
        elif command["type"] == "abort":
            process.stdout.feed_json(
                {
                    "id": command["id"],
                    "type": "response",
                    "command": "abort",
                    "success": True,
                }
            )

    process = FakeProcess(on_command)
    rpc = _rpc(tmp_path)

    with patch(
        "a2a_adapter.integrations._pi_rpc.create_subprocess_exec",
        return_value=process,
    ):
        await rpc.start()
        turn = asyncio.create_task(_collect(rpc.prompt("work")))
        await asyncio.sleep(0)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    assert process.returncode == -15
    assert rpc.is_running is False


async def _collect(iterator):
    return [item async for item in iterator]
