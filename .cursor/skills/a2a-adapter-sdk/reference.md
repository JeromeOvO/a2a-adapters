# a2a-adapter SDK — API Reference

## BaseA2AAdapter (abstract base class)

```python
from a2a_adapter import BaseA2AAdapter
```

| Method | Required | Signature | Description |
|---|---|---|---|
| `invoke` | **Yes** | `async (str, str\|None) -> str` | Execute agent, return text |
| `stream` | No | `async (str, str\|None) -> AsyncIterator[str]` | Yield text chunks |
| `cancel` | No | `async () -> None` | Cancel current execution |
| `close` | No | `async () -> None` | Release resources |
| `get_metadata` | No | `() -> AdapterMetadata` | Metadata for AgentCard |
| `supports_streaming` | No | `() -> bool` | Auto-detects from stream() override |

## AdapterMetadata (dataclass)

```python
from a2a_adapter import AdapterMetadata

AdapterMetadata(
    name="",                          # Agent name (defaults to class name in card)
    description="",                   # What the agent does
    version="1.0.0",                  # Semantic version
    skills=[],                        # List of skill dicts: [{"id", "name", "description", "tags"}]
    input_modes=["text"],             # Supported input MIME types
    output_modes=["text"],            # Supported output MIME types
    streaming=False,                  # Whether adapter supports streaming
)
```

## Server Functions

### serve_agent

```python
serve_agent(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,   # Override auto-generated card
    host: str = "0.0.0.0",
    port: int = 9000,
    log_level: str = "info",
    **kwargs,                              # Passed to uvicorn.run()
)
```

### to_a2a

```python
to_a2a(
    adapter: BaseA2AAdapter,
    agent_card: AgentCard | None = None,
    task_store: TaskStore | None = None,   # Default: InMemoryTaskStore
    **card_overrides,                      # name=, description=, url=, version=, streaming=
) -> Starlette ASGI app
```

### build_agent_card

```python
build_agent_card(
    adapter: BaseA2AAdapter,
    **overrides,     # name=, description=, url=, version=, streaming=
) -> AgentCard
```

Default url is `http://localhost:9000`. Override with `url="http://prod:8080"`.

## Loader Functions

### load_adapter

```python
load_adapter(config: dict) -> BaseA2AAdapter
```

Config dict must have `"adapter"` key. Remaining keys passed as constructor kwargs.

Valid adapter values: `"n8n"`, `"langchain"`, `"langgraph"`, `"crewai"`, `"openclaw"`, `"callable"`, or any registered name.

### register_adapter

```python
@register_adapter("my_name")
class MyAdapter(BaseA2AAdapter): ...
```

Registered adapters take priority over built-ins with same name.

## N8nAdapter Constructor

```python
N8nAdapter(
    webhook_url: str,                  # REQUIRED
    timeout: int = 30,                 # HTTP timeout seconds
    headers: dict | None = None,       # Extra HTTP headers
    message_field: str = "message",    # Payload field name
    parse_json_input: bool = True,     # Auto-parse JSON strings
    input_mapper: Callable | None = None,  # (raw_input, context_id) -> dict
    default_inputs: dict | None = None,    # Merge into every request
    max_retries: int = 3,
    retry_delay: float = 1.0,
    name: str = "",                    # For AgentCard
    description: str = "",             # For AgentCard
)
```

## LangChainAdapter Constructor

```python
LangChainAdapter(
    runnable: Any,                     # REQUIRED: LangChain Runnable
    input_key: str = "input",          # Input dict key
    output_key: str | None = None,     # Extract specific output key
    parse_json_input: bool = True,
    input_mapper: Callable | None = None,
    default_inputs: dict | None = None,
    name: str = "",
    description: str = "",
)
```

Streaming auto-detected: `hasattr(runnable, "astream")`.

## LangGraphAdapter Constructor

```python
LangGraphAdapter(
    graph: Any,                        # REQUIRED: CompiledGraph
    input_key: str = "messages",       # Input dict key
    output_key: str | None = None,
    parse_json_input: bool = True,
    input_mapper: Callable | None = None,
    default_inputs: dict | None = None,
    name: str = "",
    description: str = "",
)
```

When `input_key="messages"`, text is auto-wrapped in `HumanMessage` if langchain_core is installed.

## CrewAIAdapter Constructor

```python
CrewAIAdapter(
    crew: Any,                         # REQUIRED: CrewAI Crew instance
    timeout: int = 300,                # Execution timeout seconds
    input_key: str = "inputs",         # Crew input key
    parse_json_input: bool = True,
    input_mapper: Callable | None = None,
    default_inputs: dict | None = None,
    name: str = "",
    description: str = "",
)
```

Uses `kickoff_async()` with sync `kickoff()` fallback for older CrewAI versions.

## OpenClawAdapter Constructor

```python
OpenClawAdapter(
    thinking: str = "low",             # "none", "low", "medium", "high"
    agent_id: str | None = None,
    session_id: str | None = None,     # Auto-generated if None
    timeout: int = 600,                # Subprocess timeout seconds
    openclaw_path: str = "openclaw",   # Binary path
    working_directory: str | None = None,
    env_vars: dict | None = None,
    name: str = "",
    description: str = "",
)
```

Supports `cancel()` (kills subprocess) and `close()` (kills any running process).

## CallableAdapter Constructor

```python
CallableAdapter(
    func: Callable,                    # REQUIRED: async/sync callable(dict) -> str
    streaming: bool = False,           # True if func is async generator
    name: str = "",
    description: str = "",
)
```

When `streaming=True`, `func` must be an async generator yielding str chunks.
