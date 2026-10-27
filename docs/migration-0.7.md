# Upgrading from 0.6.x to 0.7.0

0.7.0 adds native Responses tool declarations and custom-call streaming/replay.
It is an opt-in surface: the default wire, tool layout and legacy `ToolSpec`
constructors are unchanged. Use `mutil-ai = "0.7"` (or `cargo add
mutil-ai@0.7`).

The release commit is `9b33d95379a9e69907c8f55381d05a560b458c27` (tag `v0.7.0`).

## Source compatibility

- `Part` adds `NativeToolCall(NativeToolCall)`. Downstream exhaustive matches
  must add an arm.
- `StreamEvent` adds `NativeToolProgress { index, item_id, input_so_far }` and
  `NativeToolCompleted { index, call }`. Downstream exhaustive matches must add
  arms. `NativeToolProgress` is progress only; `NativeToolCompleted` records item
  completion, not permission to dispatch a write.
- `ChatRequest` adds `tool_declarations: Vec<ToolDeclaration>`. Struct literals
  without `..Default::default()` must add the field; serialized requests that
  omit it deserialize as an empty list.

## Declarations are Responses-only

`ToolDeclaration` represents a flat client tool or a one-level namespace of
function/custom tools. `CustomToolFormat` is `Text` or `Grammar { syntax,
definition }`. The caller supplies the complete grammar; the SDK does not bundle
the Codex grammar.

`ChatRequest::tool_declarations` is accepted only for `Protocol::OpenAiResponses`.
Chat, Anthropic Messages and Gemini fail closed with
`unsupported_tool_representation`. Declarations are encoded into the Responses
body directly; they cannot be smuggled through `extra_body.tools` or
`extra_body.input`, which remain reserved fields.

Duplicate names are rejected per scope, regardless of kind. A namespace also
occupies a name in the top-level scope. Provider-specific naming and capability
rules remain the caller's responsibility.

```rust
use mutil_ai::{ChatRequest, CustomToolFormat, ScopedToolDeclaration, ToolDeclaration};

let request = ChatRequest::user("apply the patch").tool_declarations([
    ToolDeclaration::Tool {
        tool: ScopedToolDeclaration::Custom {
            name: "apply_patch".into(),
            description: "Apply a raw patch".into(),
            format: CustomToolFormat::Text,
        },
    },
]);
```

For a scoped layout, use `ToolDeclaration::Namespace { name, description,
tools }` with `ScopedToolDeclaration::Function` / `Custom` entries. Nested
namespaces cannot be represented.

## Native calls and replay

`NativeToolCall` keeps `kind`, `namespace`, `name`, the real `call_id`, optional
`item_id`, raw `payload`, `completion` and optional `provider_state`. Call
`validate_completed()` before admitting a completed call. A custom call keeps
raw text; it is not parsed as JSON arguments.

Completed native calls register normalizer pending by the real `call_id`.
Results replay as `custom_tool_call_output` or `function_call_output` according
to the original kind. Missing or orphan native results are explicit errors, and
opaque Responses client-tool items are rejected rather than double-encoded.
Persist the complete native call; a provider item is not a substitute.

## Tests and boundaries

The release is covered by declaration, custom-stream, mixed-history replay and
loopback HTTP tests. Run:

```bash
cargo test --offline -j 1 --test tool_declarations
cargo test --offline -j 1 --test native_tool_stream
cargo test --offline -j 1 --test native_tool_replay
cargo test --offline -j 1 --test endpoint_dispatch native_custom_call_replays_result_in_next_http_body
```

This release does not claim real-provider/OAuth capability, and does not
complete every truncation/reconnect or terminal-reconciliation edge case.
Applications must still enforce cancellation, permission and write-scheduling
policy before executing native calls.
