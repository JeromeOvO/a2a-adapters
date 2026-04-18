# Plan: Hermes Agent A2A Integration (Gateway Pattern)

**Status:** Draft
**Date:** 2026-04-16
**Repo:** `a2a-adapter`
**Depends on:** `a2a-adapter` (v0.2+, see DESIGN_V0.2.md), `a2a-sdk` (a2a-python), `hermes-agent` (on PYTHONPATH at runtime)

---

## Problem

Hermes Agent is a full-featured AI agent with tool use, persistent memory, subagent
delegation, and multiple interaction modes. It has no A2A Protocol support.

| # | Gap | Impact |
|---|-----|--------|
| 1 | Hermes Agent is **not A2A-compatible**. | Cannot participate in the Hybro marketplace or multi-agent workflows. |
| 2 | `AIAgent.run_conversation()` is **synchronous** and expects the caller to manage conversation history. | Need async bridge + history management strategy. |
| 3 | No **streaming** via A2A SSE for sync agents with callbacks. | Poor UX for long-running tasks. |

---

## Design Decision: Gateway Pattern

The adapter mirrors how Hermes's own production gateway (`gateway/run.py`) handles
multi-turn conversations:

1. **Load** conversation history from `SessionDB` (SQLite) using `context_id` as `session_id`.
2. **Create** a fresh `AIAgent` instance per turn (same as `gateway/run.py:5663`).
3. **Call** `agent.run_conversation(user_message=..., conversation_history=history, task_id=session_id)`.
4. **Let** `AIAgent` persist the updated history back to `SessionDB` via its internal `_flush_messages_to_session_db()`.

This approach was chosen over two alternatives:

| Approach | Pros | Cons |
|----------|------|------|
| **Gateway pattern** (chosen) | Mirrors proven Hermes gateway code path. Durable persistence via SQLite. No new abstractions. Structured output. Streaming support. | Requires `hermes-agent` importable on `sys.path`. |
| StatefulAdapter (in-memory sessions) | Generic reusable base class. | New abstraction with session management, locking, TTL — complexity that Hermes's own `SessionDB` already solves. In-memory history lost on restart. |
| CLI subprocess | Zero Python coupling. | No `--json` output flag. Parsing human-readable terminal output. No streaming. High per-turn latency (Python startup). |

### Why not StatefulAdapter?

Hermes already has a battle-tested session persistence layer (`SessionDB` in
`hermes_state.py`) used by its gateway across Telegram, Discord, Slack, and other
platforms. The gateway pattern reuses this directly, giving us:

- **Durable persistence** — history survives adapter restarts.
- **No in-memory session management** — no locks, no TTL cleanup, no session eviction.
- **Proven code path** — identical to how ~6 Hermes messaging platforms handle multi-turn.
- **Lower complexity** — the adapter is a thin `BaseA2AAdapter` subclass, not a new framework.

### Compatibility with a2a-adapter v0.2 architecture

`HermesAdapter` is a standard `BaseA2AAdapter` subclass at **Layer 1** (Framework
Driver). No changes to the bridge (Layer 3), SDK (Layer 4), or public API (Layer 5):

| v0.2 Layer | Component | Change Required |
|------------|-----------|----------------|
| Layer 5: Public API | `serve_agent()`, `to_a2a()`, `build_agent_card()` | None — works as-is |
| Layer 4: A2A SDK | `DefaultRequestHandler`, `TaskManager`, etc. | None — delegated |
| Layer 3: Bridge | `AdapterAgentExecutor` | None — `HermesAdapter` implements `invoke()`/`stream()`/`cancel()` from `BaseA2AAdapter` |
| Layer 1: Framework Driver | **`HermesAdapter`** (NEW) | New integration |

`HermesAdapter` upholds the v0.2 design principles:
- **SDK-First (§2.1):** Delegates all protocol handling to the SDK. No task management, SSE, or push notification logic.
- **Minimal Surface (§2.2):** Just `invoke()`, `stream()`, `cancel()`, `close()`, and `get_metadata()`.
- **Layered Escape Hatch (§2.3):** Level 1 = `invoke()` (text in/out), Level 2 = `stream()` (callback-based streaming). Level 3 = bypass adapter and use `AgentExecutor` directly.
- **Open-Closed (§2.4):** Adding HermesAdapter doesn't modify any core file beyond registration.

### Class hierarchy

```
BaseA2AAdapter                          (stateless, invoke-only)
  ├── OpenClawAdapter                   (stateless, subprocess)
  ├── N8nAdapter                        (stateless, HTTP webhook)
  ├── LangGraphAdapter                  (stateless, graph.invoke())
  └── HermesAdapter                     (gateway pattern: SessionDB + AIAgent per turn)
```

---

## Hermes Agent Interface Summary

The adapter wraps `AIAgent` from `run_agent.py`. Key interface points:

```python
class AIAgent:
    def __init__(
        self,
        model: str = "anthropic/claude-opus-4.6",
        session_id: str = None,
        session_db: SessionDB = None,      # <-- shared SessionDB instance
        platform: str = None,
        enabled_toolsets: list = None,
        quiet_mode: bool = False,
        # ... plus provider, api_mode, callbacks, etc.
    ): ...

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "final_response": str,
                "messages": List[Dict],   # updated conversation history
                "api_calls": int,
                "completed": bool,
                "interrupted": bool,       # if interrupt() was called
                "partial": bool,           # if response was truncated
                "error": str | None,
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
            }
        """

    def interrupt(self, message: str = None) -> None:
        """Request graceful interruption from another thread."""
```

**Critical constraint:** `run_conversation()` is **synchronous** — it blocks the
calling thread. The adapter handles this via `ThreadPoolExecutor` +
`loop.run_in_executor()`.

**History management:** `run_conversation()` expects `conversation_history` to be
passed in by the caller. When `session_db` is provided and `session_id` is set,
`AIAgent` internally calls `_persist_session()` on every exit path (normal
completion, error, and interrupt). `_persist_session()` does two things:
1. `_save_session_log(messages)` — writes the full session to a JSONL file at
   `~/.hermes/sessions/{session_id}.jsonl`.
2. `_flush_messages_to_session_db(messages)` — persists to SQLite.

This is how Hermes's own gateway works — load history, pass it in, let the agent
persist updates.

**Note on JSONL accumulation:** Because `_persist_session()` always writes JSONL
log files, each A2A turn produces a file under `~/.hermes/sessions/`. For
high-traffic deployments, this directory should be monitored and periodically
cleaned. The JSONL files are useful for debugging but not required by the adapter
(history is loaded from SQLite via `SessionDB`).

**Note on history loading:** The Hermes gateway uses `session_store.load_transcript()`
which tries SQLite first and falls back to JSONL files. The adapter uses
`session_db.get_messages_as_conversation()` which reads SQLite directly. Since the
adapter always writes via `SessionDB`, the JSONL fallback is unnecessary — SQLite
is the authoritative store.

---

## Architecture

**Mapping to v0.2 layers** (see DESIGN_V0.2.md §3.1):

```
┌──────────────────────────────────────────────────────┐
│  Layer 5: Public API                                  │
│    serve_agent(adapter, port=9010)                    │
│    to_a2a(adapter)                                    │
│    build_agent_card(adapter)                          │
├──────────────────────────────────────────────────────┤
│  Layer 4: A2A SDK (unchanged, delegated)              │
│    DefaultRequestHandler, TaskManager, TaskUpdater,   │
│    A2AStarletteApplication, EventQueue, TaskStore     │
├──────────────────────────────────────────────────────┤
│  Layer 3: Bridge (unchanged)                          │
│    AdapterAgentExecutor                               │
│      execute() → adapter.invoke() / adapter.stream()  │
│      cancel()  → adapter.cancel()                     │
├──────────────────────────────────────────────────────┤
│  Layer 1: Framework Driver                            │
│    HermesAdapter(BaseA2AAdapter)                      │
│      invoke():                                        │
│        1. session_id = context_id or generate          │
│        2. history = session_db.get_messages(session_id)│
│        3. agent = AIAgent(session_id, session_db, ...) │
│        4. result = run_in_executor(                    │
│             agent.run_conversation(msg, history))      │
│        5. return result["final_response"]              │
│      stream():                                        │
│        Same as invoke but with stream_callback → Queue │
│      cancel():                                        │
│        _running_agents[task_id].interrupt()             │
│                                                        │
│    Hermes internals (not modified):                    │
│      SessionDB (hermes_state.py) — SQLite persistence │
│      AIAgent._flush_messages_to_session_db() — auto   │
└──────────────────────────────────────────────────────┘
```

### Request flow (invoke)

```
A2A client sends message/send with context_id="conv-123"
    │
    ▼
AdapterAgentExecutor.execute()
    │  user_input = context.get_user_input()
    │  context_id = context.context_id
    ▼
HermesAdapter.invoke(user_input, context_id="conv-123")
    │
    │  1. session_id = "conv-123"
    │  2. history = session_db.get_messages_as_conversation("conv-123")
    │     → returns [] (first turn) or [msg1, msg2, ...] (subsequent)
    │  3. agent = AIAgent(session_id="conv-123", session_db=session_db, ...)
    │  4. result = await run_in_executor(
    │       agent.run_conversation(
    │           user_message=user_input,
    │           conversation_history=history,
    │           task_id="conv-123",
    │       )
    │     )
    │  5. AIAgent internally persists updated history to SessionDB
    │  6. return result["final_response"]
    │
    ▼
AdapterAgentExecutor
    │  TaskUpdater.add_artifact(response_text)
    │  TaskUpdater.complete()
    ▼
Client receives completed task with response
```

### Threading model

```
┌─────────────────────────────────────────────────────┐
│                 asyncio event loop                    │
│            (uvicorn / Starlette ASGI)                │
│                                                       │
│  HTTP request → JSON-RPC dispatch → invoke()/stream() │
│    │                                                  │
│    │ loop.run_in_executor(ThreadPool, ...)            │
│    │      │                                           │
│    │      │ (blocks in thread)                        │
│    │      ▼                                           │
│    │ ┌─────────────────────────────────┐              │
│    │ │   ThreadPoolExecutor            │              │
│    │ │   agent.run_conversation()      │              │
│    │ │   ↳ emit(chunk) ───────────────┼─► queue      │
│    │ └─────────────────────────────────┘              │
│    │                                                  │
│    │ SSE response ◄── queue.get() ◄── chunks         │
└─────────────────────────────────────────────────────┘
```

---

## HermesAdapter Implementation

### Constructor

```python
class HermesAdapter(BaseA2AAdapter):

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        enabled_toolsets: list[str] | None = None,
        max_workers: int = 4,
        name: str = "",
        description: str = "",
        skills: list[dict] | None = None,
        agent_provider: dict | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        self._model = model
        self._provider_name = provider
        self._enabled_toolsets = enabled_toolsets
        self._name = name
        self._description = description
        self._skills = skills or []
        self._agent_provider = agent_provider
        self._documentation_url = documentation_url
        self._icon_url = icon_url

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._session_db: SessionDB | None = None
        self._running_agents: dict[str, AIAgent] = {}  # task_id → agent
```

### _ensure_session_db() — lazy initialization

```python
def _ensure_session_db(self) -> SessionDB:
    if self._session_db is None:
        from hermes_state import SessionDB
        self._session_db = SessionDB()
    return self._session_db
```

Lazy so the adapter class always loads for registry/loader purposes. If
`hermes-agent` is not on `sys.path`, this raises `ImportError` with a clear
message on first use.

### _make_agent() — fresh AIAgent per turn

```python
def _make_agent(self, session_id: str) -> AIAgent:
    from run_agent import AIAgent
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config = load_config()
    model_cfg = config.get("model")
    default_model = ""
    config_provider = None
    if isinstance(model_cfg, dict):
        default_model = str(model_cfg.get("default") or "")
        config_provider = model_cfg.get("provider")
    elif isinstance(model_cfg, str) and model_cfg.strip():
        default_model = model_cfg.strip()

    kwargs = {
        "platform": "a2a",
        "enabled_toolsets": self._enabled_toolsets,
        "quiet_mode": True,
        "session_id": session_id,
        "session_db": self._ensure_session_db(),
        "model": self._model or default_model,
    }

    try:
        runtime = resolve_runtime_provider(
            requested=self._provider_name or config_provider,
        )
        kwargs.update({
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "base_url": runtime.get("base_url"),
            "api_key": runtime.get("api_key"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
        })
    except Exception:
        pass

    return AIAgent(**kwargs)
```

A fresh `AIAgent` is created per turn, matching the gateway pattern at
`gateway/run.py:5663`. The shared `SessionDB` instance is passed via the
`session_db` parameter so the agent persists history automatically.

### invoke() — synchronous execution via thread pool

```python
async def invoke(
    self, user_input: str, context_id: str | None = None, **kwargs,
) -> str:
    session_id = context_id or f"a2a-{uuid.uuid4().hex[:12]}"
    db = self._ensure_session_db()
    history = db.get_messages_as_conversation(session_id) or []

    agent = self._make_agent(session_id)
    task_id = self._extract_task_id(kwargs)
    self._running_agents[task_id] = agent

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            self._executor,
            lambda: agent.run_conversation(
                user_message=user_input,
                conversation_history=history,
                task_id=session_id,
            ),
        )
    finally:
        self._running_agents.pop(task_id, None)

    return result.get("final_response", "")
```

`_extract_task_id` is a small helper for pulling the task identifier out of the
bridge-provided `context` kwarg:

```python
@staticmethod
def _extract_task_id(kwargs) -> str:
    context = kwargs.get("context")
    if context and hasattr(context, "task_id"):
        return context.task_id
    return uuid.uuid4().hex
```

No explicit history save needed — `AIAgent` persists to `SessionDB` internally
via `_persist_session()` on every exit path (normal completion, error, and
interrupt). Note that `_persist_session()` also writes JSONL session logs to
`~/.hermes/sessions/` — see "Note on JSONL accumulation" above.

### stream() — callback-based streaming via Queue bridge

```python
async def stream(
    self, user_input: str, context_id: str | None = None, **kwargs,
) -> AsyncIterator[str]:
    session_id = context_id or f"a2a-{uuid.uuid4().hex[:12]}"
    db = self._ensure_session_db()
    history = db.get_messages_as_conversation(session_id) or []

    agent = self._make_agent(session_id)
    task_id = self._extract_task_id(kwargs)
    self._running_agents[task_id] = agent

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def emit(chunk: str):
        loop.call_soon_threadsafe(queue.put_nowait, chunk)

    def _run():
        try:
            return agent.run_conversation(
                user_message=user_input,
                conversation_history=history,
                task_id=session_id,
                stream_callback=emit,
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = loop.run_in_executor(self._executor, _run)

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

        await task
    finally:
        self._running_agents.pop(task_id, None)
```

The `try/finally` on `_run()` guarantees the sentinel `None` is always pushed,
preventing hangs even if `run_conversation` raises. The `await task` after the
loop re-raises any exception, which the `AdapterAgentExecutor` bridge catches
and converts to a `failed` task state.

### cancel()

```python
async def cancel(self, context_id: str | None = None, **kwargs) -> None:
    task_id = self._extract_task_id(kwargs)
    agent = self._running_agents.get(task_id)
    if agent:
        agent.interrupt()
```

`interrupt()` is thread-safe — it sets an atomic flag that the agent checks
between tool-use iterations. The agent will finish its current LLM call and
then return with `interrupted=True`.

**Note on `**kwargs`:** The `AdapterAgentExecutor` bridge passes `context=context`
as a keyword argument when calling `cancel()`, `invoke()`, and `stream()`. Using
`**kwargs` on all three methods ensures forward compatibility with the bridge
layer without requiring HermesAdapter to depend on `RequestContext`.

**Concurrency:** `_running_agents` is a `dict[str, AIAgent]` keyed by `task_id`
(extracted from `kwargs["context"].task_id`). This ensures `cancel()` interrupts
the correct agent when multiple requests are in-flight.

### close()

```python
async def close(self) -> None:
    self._executor.shutdown(wait=False, cancel_futures=True)
```

### get_metadata()

```python
def get_metadata(self) -> AdapterMetadata:
    return AdapterMetadata(
        name=self._name or "HermesAdapter",
        description=self._description or (
            "Hermes AI Agent — multi-purpose assistant with tool use, "
            "persistent memory, and subagent delegation."
        ),
        streaming=True,
        skills=self._skills,
        provider=self._agent_provider,
        documentation_url=self._documentation_url,
        icon_url=self._icon_url,
    )
```

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| First turn (`context_id` is new) | `session_db.get_messages_as_conversation()` returns `[]`. Agent starts fresh, persists first turn to DB. |
| Subsequent turns (same `context_id`) | History loaded from SQLite. Agent continues conversation with full context. |
| `context_id=None` | Ephemeral session with generated UUID. History persisted in DB but unlikely to be reused. Over time, these accumulate in `SessionDB` — acceptable since `SessionDB` already handles old session cleanup in the gateway. |
| `run_conversation` raises | Bridge layer catches, sets task `state=failed`. `stream()` sentinel is still sent via `finally`. |
| Concurrent requests, same `context_id` | Both create separate `AIAgent` instances but read/write the same `session_id` in `SessionDB`. SQLite handles concurrent access. However, history may diverge — the second request may not see the first request's updates if they overlap. A2A callers should serialize per-context requests. |
| Concurrent requests, different `context_id` | Fully parallel — no contention. |
| Thread pool exhausted | New requests queue in the executor. |
| `cancel()` for no running agent | No-op — `_running_agents.get(task_id)` returns `None`. |
| Adapter restarts | History survives in SQLite. Next request with the same `context_id` picks up where it left off. |

---

## File Layout

```
a2a-adapter/
└── a2a_adapter/
    └── integrations/
        └── hermes.py                # HermesAdapter (BaseA2AAdapter subclass)
```

No changes to `hermes-agent`.

**Registration** (changes to existing `a2a-adapter` files):

1. `a2a_adapter/__init__.py` — add `"HermesAdapter"` to `_ADAPTER_LAZY_MAP` and `__all__`.
2. `a2a_adapter/integrations/__init__.py` — add `"HermesAdapter"` to `__all__`
   and a lazy import entry.
3. `a2a_adapter/loader.py` — add `"hermes"` to `_BUILTIN_MAP`:
   ```python
   "hermes": ("a2a_adapter.integrations.hermes", "HermesAdapter"),
   ```
4. `ARCHITECTURE.md` — add Hermes to the Framework Adapters table.

---

## Entry Point

**Option 1: Config-driven** (recommended for production — works with the loader):

```python
from a2a_adapter import load_adapter, serve_agent

adapter = load_adapter({
    "adapter": "hermes",
    "model": "anthropic/claude-sonnet-4-20250514",
    "provider": "openrouter",
    "enabled_toolsets": ["hermes-cli"],
})
serve_agent(adapter, port=9010)
```

**Option 2: Direct import** (recommended for development):

```python
from a2a_adapter import HermesAdapter, serve_agent

adapter = HermesAdapter(
    model="anthropic/claude-sonnet-4-20250514",
    provider="openrouter",
    enabled_toolsets=["hermes-cli"],
)
serve_agent(adapter, port=9010)
```

```bash
# Verify AgentCard
curl http://localhost:9010/.well-known/agent-card.json
```

---

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| None` | `None` (agent default) | LLM model identifier |
| `provider` | `str \| None` | `None` (auto-detect) | Provider (openrouter, anthropic, openai, etc.) |
| `enabled_toolsets` | `list[str] \| None` | `None` (uses Hermes default for the platform) | Tool surface for A2A sessions. Valid toolsets: `hermes-cli`, `hermes-acp`, `hermes-telegram`, etc. See `toolsets.py`. |
| `max_workers` | `int` | `4` | Thread pool size for concurrent requests |
| `name` | `str` | `"HermesAdapter"` | Agent name in AgentCard |
| `description` | `str` | (built-in) | Agent description in AgentCard |
| `skills` | `list[dict] \| None` | `[]` | Skill dicts for AgentCard |
| `agent_provider` | `dict \| None` | `None` | Dict with 'organization' and 'url' keys |
| `documentation_url` | `str \| None` | `None` | URL to the agent's documentation |
| `icon_url` | `str \| None` | `None` | URL to an icon for the agent |

---

## Dependencies

`HermesAdapter` has no additional pip dependencies beyond the existing `a2a-adapter`
SDK requirements (`a2a-sdk`, `starlette`, `uvicorn`). The `ThreadPoolExecutor` and
`asyncio.Queue` are stdlib.

`HermesAdapter` imports from `hermes-agent`'s existing modules at runtime:
- `run_agent.AIAgent`
- `hermes_state.SessionDB`
- `hermes_cli.config.load_config`
- `hermes_cli.runtime_provider.resolve_runtime_provider`

Since hermes-agent is not a pip package, it must be available on `sys.path` (e.g.,
`PYTHONPATH=/path/to/hermes-agent`).

All Hermes imports are lazy — inside `_make_agent()` and `_ensure_session_db()` — so
the adapter class always loads for registry/loader purposes. If `hermes-agent` is
not on the path, the first `invoke()` call raises `ImportError` with a clear message.

---

## Testing Plan

| # | Test | Method |
|---|------|--------|
| 1 | `_make_agent()` loads config + resolves provider | pytest with mocked `hermes_cli` modules |
| 2 | `invoke()` loads history from SessionDB, calls `run_conversation`, returns response | pytest with mocked `AIAgent` and `SessionDB` |
| 3 | `invoke()` passes `session_db` to `AIAgent` constructor | pytest: verify kwargs |
| 4 | `invoke()` / `stream()` / `cancel()` accept `context=...` kwarg without error | pytest: call with keyword arg |
| 5 | `stream()` yields chunks via `stream_callback` → Queue bridge | pytest with mocked `AIAgent` |
| 6 | `stream()` sentinel sent even when `run_conversation` raises | pytest: verify no hang |
| 7 | `cancel()` calls `agent.interrupt()` | pytest with mock |
| 8 | Multi-turn: second `invoke()` with same `context_id` receives prior history | pytest: mock SessionDB to return history on second call |
| 9 | Ephemeral: `context_id=None` generates UUID session_id | pytest |
| 10 | AgentCard served at `/.well-known/agent-card.json` | `curl` / pytest |
| 11 | `close()` shuts down thread pool | pytest |
| 12 | End-to-end with Hybro Hub | `hybro-hub` discovery + task execution |
| 13 | A2A Inspector compatibility | Test with `a2a-agent-inspector` |

---

## Out of Scope (Future)

- **Multimodal output.** `HermesAdapter` currently returns `str` only. Surfacing
  images/files from Hermes tool-use as `Part` objects is a follow-up.
- **Push notifications.** Handled by the A2A SDK's `DefaultRequestHandler` if the
  caller sends `pushNotificationConfig`. No adapter-level code needed.
- **A2UI surfaces.** Hermes's tool-use events could be mapped to A2UI `DataPart`
  payloads. Requires A2UI renderer support in the frontend.
- **Concurrent same-context_id serialization.** Currently, concurrent requests to the
  same `context_id` may produce divergent history. If this becomes a real issue, add
  per-`context_id` `asyncio.Lock` in the adapter (simple dict of locks).
- **Authentication.** The adapter runs unauthenticated. Auth can be added via
  middleware or reverse proxy.

---

## Implementation Order

All work is in the `a2a-adapter` repo. No changes to `hermes-agent`.

1. **Create** `a2a_adapter/integrations/hermes.py` with `HermesAdapter`.
2. **Implement** `__init__`, `_ensure_session_db`, `_make_agent`, `invoke`, `stream`, `cancel`, `close`, `get_metadata`.
3. **Register** in `__init__.py`, `integrations/__init__.py`, and `loader.py`.
4. **Test** with mocked `AIAgent` and `SessionDB` — verify all 11 unit tests pass.
5. **Verify** non-streaming + streaming with `curl` and `a2a-agent-inspector`.
6. **Update** `ARCHITECTURE.md`.
7. **Test** with Hybro Hub end-to-end.
