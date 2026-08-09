"""Pi coding agent adapter for the A2A Protocol."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from ..base_adapter import AdapterMetadata, BaseA2AAdapter
from ..exceptions import CancelledByAdapterError
from ._pi_rpc import PiProcessLostError, PiRpcProcess, PiTurnAbortedError


class PiAdapter(BaseA2AAdapter):
    """Expose one persistent Pi RPC process and session as an A2A agent.

    The first request starts Pi lazily. Every later request reuses the same
    process and durable session. Turns are globally serialized because Pi's
    asynchronous agent events are session-scoped rather than prompt-correlated.
    """

    def __init__(
        self,
        working_dir: str,
        *,
        pi_command: Sequence[str] | None = None,
        session_id: str = "a2a-pi-agent",
        session_dir: str | None = None,
        timeout: float = 600,
        startup_timeout: float = 30,
        cancel_grace_timeout: float = 10,
        env_vars: dict[str, str] | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        enforce_single_context: bool = True,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
        _rpc_factory: Callable[[], PiRpcProcess] | None = None,
    ) -> None:
        if isinstance(pi_command, str):
            raise TypeError("pi_command must be a sequence of arguments, not a string")
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", session_id):
            raise ValueError(
                "session_id must contain only alphanumeric characters, '-', '_', "
                "and '.', and must start and end with an alphanumeric character"
            )

        self.working_dir = working_dir
        self.pi_command = list(pi_command or ["pi"])
        self.session_id = session_id
        self.session_dir = session_dir or os.path.join(
            working_dir, ".a2a-adapter", "pi", "sessions"
        )
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.cancel_grace_timeout = cancel_grace_timeout
        self.env_vars = dict(env_vars) if env_vars else {}
        self.model = model
        self.model_provider = model_provider
        self.enforce_single_context = enforce_single_context

        self._name = name
        self._description = description
        self._skills = skills or []
        self._provider = provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

        self._rpc: PiRpcProcess | None = None
        self._rpc_factory = _rpc_factory
        self._startup_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._active_task_id: str | None = None
        self._cancelled_tasks: set[str] = set()
        self._bound_context_id: str | None = None
        self._closed = False

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs: Any) -> str:
        chunks = [chunk async for chunk in self.stream(user_input, context_id=context_id, **kwargs)]
        return "".join(chunks)

    async def stream(
        self, user_input: str, context_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        context = kwargs.get("context")
        task_id = context.task_id if context else (context_id or "_direct")

        self._raise_if_cancelled(task_id)
        async with self._turn_lock:
            self._raise_if_cancelled(task_id)
            self._bind_context(context_id)
            rpc = await self._ensure_started()
            self._active_task_id = task_id
            try:
                async for chunk in rpc.prompt(user_input):
                    yield chunk
            except PiTurnAbortedError as error:
                if task_id in self._cancelled_tasks:
                    raise CancelledByAdapterError(f"Task {task_id} was cancelled") from error
                raise
            except PiProcessLostError as error:
                if self._rpc is rpc:
                    self._rpc = None
                if task_id in self._cancelled_tasks:
                    raise CancelledByAdapterError(f"Task {task_id} was cancelled") from error
                raise
            finally:
                self._active_task_id = None
                self._cancelled_tasks.discard(task_id)
                if not rpc.is_running and self._rpc is rpc:
                    self._rpc = None

    async def cancel(self, context_id: str | None = None, **kwargs: Any) -> None:
        context = kwargs.get("context")
        if context is None:
            return
        task_id = context.task_id
        self._cancelled_tasks.add(task_id)

        if task_id == self._active_task_id and self._rpc is not None:
            await self._rpc.abort()

    async def close(self) -> None:
        async with self._startup_lock:
            if self._closed:
                return
            self._closed = True
            rpc = self._rpc
            self._rpc = None
        if rpc is not None:
            await rpc.close()

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "PiAdapter",
            description=self._description or "Pi coding agent",
            streaming=True,
            skills=self._skills,
            provider=self._provider,
            documentation_url=self._documentation_url,
            icon_url=self._icon_url,
        )

    async def _ensure_started(self) -> PiRpcProcess:
        if self._closed:
            raise RuntimeError("PiAdapter is closed")
        if self._rpc is not None and self._rpc.is_running:
            return self._rpc

        async with self._startup_lock:
            if self._closed:
                raise RuntimeError("PiAdapter is closed")
            if self._rpc is not None and self._rpc.is_running:
                return self._rpc

            stale_rpc = self._rpc
            self._rpc = None
            if stale_rpc is not None:
                await stale_rpc.close()

            rpc = self._create_rpc()
            try:
                await rpc.start()
            except Exception:
                await rpc.close()
                raise
            self._rpc = rpc
            return rpc

    def _create_rpc(self) -> PiRpcProcess:
        if self._rpc_factory is not None:
            return self._rpc_factory()
        return PiRpcProcess(
            command=self.pi_command,
            working_dir=self.working_dir,
            session_id=self.session_id,
            session_dir=self.session_dir,
            env_vars=self.env_vars,
            provider=self.model_provider,
            model=self.model,
            timeout=self.timeout,
            startup_timeout=self.startup_timeout,
            cancel_grace_timeout=self.cancel_grace_timeout,
        )

    def _bind_context(self, context_id: str | None) -> None:
        if not self.enforce_single_context or context_id is None:
            return
        if self._bound_context_id is None:
            self._bound_context_id = context_id
            return
        if self._bound_context_id != context_id:
            raise RuntimeError(
                "This PiAdapter exposes one session and is already bound to "
                f"A2A context {self._bound_context_id!r}; received {context_id!r}"
            )

    def _raise_if_cancelled(self, task_id: str) -> None:
        if task_id in self._cancelled_tasks:
            self._cancelled_tasks.discard(task_id)
            raise CancelledByAdapterError(f"Task {task_id} was cancelled before execution started")
