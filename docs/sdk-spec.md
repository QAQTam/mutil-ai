# mutil-ai SDK Spec / Handoff

> 状态：Draft v0.14 / API freeze candidate
> 更新时间：2026-09-25
> crate：`mutil-ai`
> lib：`mutil_ai`
> 当前版本：`0.2.1`
> Rust edition：2024
> API freeze：见 [`api-freeze-0.2.md`](api-freeze-0.2.md)
> 实现交接：见 [`handoff-0.2.md`](handoff-0.2.md)

本文同时承担两个职责：

1. 定义这个 SDK 长期要遵守的契约；
2. 给后续开发者或 Agent 一份当前实现与下一步工作的交接。

---

## 1. 一句话定义

`mutil-ai` 是一个 provider-neutral 的 Rust 基础库：

> 把不同模型厂商混乱的 role、tool、reasoning、streaming、header、error 和 retry 差异，收敛成一套稳定的中立语义；provider-specific 行为留在 profile/adapter，不进入 SDK 公共 API 或下游 runtime。

它不是：

- Agent 框架
- prompt 编排框架
- memory / RAG 框架
- 模型路由器
- 自动发现 provider 的网络探测工具
- 试图让所有 provider 变成“字段完全相同”的兼容层

---

## 2. 核心设计原则

### 2.1 入口宽松，出口严格

入口允许：

- `Developer`
- `Custom(String)`
- 不完整 tool call
- orphan tool result
- provider-specific reasoning alias
- 未知 provider 字段

出口必须严格：

- 只发送目标协议认识的 role
- tool call 与 tool result 必须配对
- provider-specific state 只能回到同一个 provider/protocol
- 不支持的能力必须 error/drop/downgrade，不能猜测

### 2.2 统一语义，不统一 wire fields

统一的是：

```text
Message
Part
Reasoning
ToolCall
ToolResult
Usage
StreamEvent
Error
ResponseMetadata
```

不是：

```text
reasoning_content
reasoning
reasoning_details
thinking
encrypted_content
```

字段差异由 `ProviderProfile` 和 adapter 处理。

### 2.3 显式端点，拒绝魔法推断

Base URL 不能唯一决定协议行为。

例如同一厂商可能有：

```text
/chat/completions
/responses
/anthropic/v1/messages
```

因此下游至少应明确：

- protocol surface
- base URL / path
- auth style
- profile
- model

SDK 可以提供 registry 查找，但不能在生产热路径中靠探测猜协议。

### 2.4 不丢 opaque state，也不跨 provider 偷传

以下内容必须视为不透明：

- Anthropic `signature`
- OpenAI Responses `encrypted_content`
- Gemini `thoughtSignature`
- provider reasoning item id

规则：

- 同协议可原样回放
- 跨协议默认丢弃
- 非 assistant reasoning 默认丢弃
- 不在日志中默认展开 encrypted/raw state

### 2.5 SDK 不承载 Agent runtime

SDK 的公开调用边界只有：

```rust
ChatRequest
ChatResponse
StreamEvent
Reasoning
ToolCall
ToolResult
```

下游 runtime 负责：

```text
会话历史
工具注册与执行
权限审批
循环步数
终止条件
Agent 调度
```

示例 [`../examples/minimal_agent.rs`](../examples/minimal_agent.rs) 展示如何只用公开
API 实现一个最小 runtime；该示例不是库导出，也不承诺 API 稳定性。

下游 runtime 不处理 provider quirks：

```text
thinking.type
enable_thinking
reasoning_format
thinking.keep
clear_thinking
input_json_delta
signature_delta
```

这些由 adapter/profile 在边界处完成。

---

## 3. 当前实现状态

### 3.1 已实现模块

```text
src/
  types.rs             中立 message / reasoning / provider state
  normalize.rs         role 降级、tool 配对、跨协议 reasoning 清洗
  adapter/
    openai_chat.rs     OpenAI Chat Completions
    endpoint.rs        显式 EndpointSpec 的四协议 dispatch
    openai_responses.rs OpenAI Responses
    anthropic.rs       Anthropic Messages
    gemini.rs          Gemini generateContent
  retry.rs             Retry-After / reset header / backoff
  sse.rs               手写增量 SSE parser
  stream.rs            统一 StreamEvent 和流式事件组装
  headers.rs           UA / 请求头 / query / HeaderInjector / Idempotency-Key
  profile.rs           EndpointSpec / ProviderProfile / ModelProfile 基础类型
  cancel.rs            可克隆协作取消令牌
  report.rs            transform / normalization / wire degradation 报告
  transform.rs         请求 transform hook
  blocking.rs          feature-gated 同步 wrapper

examples/
  hello.rs             直接调用 ModelAdapter 的最小示例
  minimal_agent.rs     下游 runtime 参考实现，不属于公开库 API
```

### 3.2 当前已支持的协议面

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages
- Gemini generateContent / streamGenerateContent

### 3.3 当前已支持的核心能力

- 内部 role 到 provider role 的归一化
- tool call / tool result 配对修复
- Anthropic / Gemini 相邻消息合并
- OpenAI Chat / Responses / Anthropic / Gemini 非流式
- OpenAI Chat / Responses / Anthropic / Gemini 流式
- reasoning summary/text/encrypted/redacted
- provider state 原样回放
- 跨 provider reasoning state 清洗
- ResponseMetadata
- Retry-After 与 provider reset header
- HeaderInjector
- HeaderInjector per-retry refresh
- Idempotency-Key
- `extra_body` / request-level extra headers / `extra_query`
- 显式 `EndpointSpec` 的 `EndpointAdapter`
- `EndpointAdapter` dispatch Chat / Responses / Anthropic / Gemini
- `OpenAICompatible` 保留为 `EndpointAdapter` 兼容别名
- `ProviderProfile` / `ModelProfile` 基础类型与 request defaults
- ReasoningAliases 驱动非流式与流式 OpenAI Chat 响应解析
- ThinkingRequestProfile 驱动 thinking/effort/budget 请求字段
- ReasoningReplayPolicy 驱动历史 reasoning 回放
- Capability gating（reasoning/tool/stream/multimodal）
- DeepSeek / Qwen / Kimi / GLM / Doubao profile presets
- explicit built-in `ProfileRegistry`
- typed ErrorKind / status / provider / request id
- opt-in stream reconnect / Last-Event-ID / event dedup
- tool_choice / response_format / stop / seed 中立请求参数
- context-free structured `AuditSink` / attempt / retry / reconnect events
- 手写 SSE parser
- cooperative cancellation / `ErrorKind::Cancelled`
- retry、response body 读取与 reconnect 取消感知
- stream idle timeout
- 完整 usage typed fields 与 raw passthrough
- `ToolCallProgress` / `ServerToolStatus` / `Retrying` / `Error` stream events
- 结构化 ToolResult 与 URL/base64/file-ref 图片输入
- OpenAI Responses server tool 声明、状态和 opaque item round-trip
- `RequestTransform` 与 `CompletionReport`
- typed provider request options
- feature-gated `BlockingAdapter`
- 本地 HTTP/SSE 集成测试

### 3.4 当前明确未完成

- Kimi K2.6 `thinking.keep` 以外的更多模型级矩阵
- Qwen effort/budget 冲突 fixture
- 真实 provider fixture / ignored live probe
- tracing span/context 传播与第三方 metrics exporter
- provider probe 工具
- fixture 生成器和更多真实 provider fixture

---

## 4. 公共中立模型

### 4.1 Message 与 Role

内部 role：

```rust
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Custom(String),
}
```

出口 role：

```rust
pub enum ExternalRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}
```

默认降级（`SystemPlacement::MergeIntoTop`）：

```text
Developer -> System
Custom(_) -> User
```

`SystemPlacement::FirstToTopRestInPlace`（仅 OpenAI Chat/Responses 生效）：
首条 System 提顶，其余 System/Developer 原位保留；Anthropic/Gemini 忽略该
策略并维持上述降级表。

### 4.2 Part

```rust
pub enum Part {
    Text {
        text: String,
        provider_state: Option<ProviderState>,
    },
    Reasoning(Reasoning),
    ImageUrl { image_url: ImageUrl },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}
```

`Part::Text.provider_state` 用于 Gemini 等可能把签名挂在可见文本上的协议。

### 4.3 Reasoning

```rust
pub enum ReasoningKind {
    Summary,
    Text,
    Encrypted,
    Redacted,
}

pub struct Reasoning {
    pub kind: ReasoningKind,
    pub summary: Option<String>,
    pub text: Option<String>,
    pub state: Option<ProviderState>,
}
```

`state` 是不透明 provider state，例如：

- OpenAI Responses reasoning item
- Anthropic thinking signature
- Gemini thoughtSignature

### 4.4 ProviderState

```rust
pub struct ProviderState {
    pub format: ProviderStateFormat,
    pub data: serde_json::Value,
}
```

规则：

- `data` 不解释、不修改
- 只在相同 `format` 的目标 adapter 中回放
- 跨格式默认丢弃
- UI 不应默认展示 `data`

---

## 5. 归一化契约

归一化集中在 `normalize.rs`。

### 5.1 Role

- SystemPlacement::MergeIntoTop（默认）：System 全部提取为 system
  instructions，Developer 降级为 System
- SystemPlacement::FirstToTopRestInPlace：首条 System 提顶，其后
  System/Developer 原位保留（`ExternalRole::System` /
  `ExternalRole::Developer`）
- Custom 默认降级为 User
- Anthropic/Gemini 合并相邻同 role 消息

### 5.2 Tool pairing

默认行为：

1. tool call 没有 id：合成 `call_N`
2. tool result 没有 id：按 name 配对并补 id
3. tool call 没有 result：合成 error result
4. orphan tool result：默认丢弃
5. Anthropic/Gemini tool result 放入 user turn

### 5.3 Reasoning

- assistant reasoning 可保留
- 非 assistant reasoning 默认丢弃并记录 report
- foreign provider state 默认丢弃并记录 report
- `preserve_foreign_reasoning` 只用于调试/高级场景，adapter 仍不得把 foreign state 发给错误 provider

---

## 6. 请求与响应元数据

### 6.1 ChatRequest

当前：

```rust
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: Option<ToolChoice>,
    pub response_format: Option<ResponseFormat>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub reasoning: Option<ReasoningConfig>,
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}
```

`extra_body` 的合并规则：

1. profile request defaults 先合并；
2. 单次 `ChatRequest.extra_body` 后合并，因此可覆盖 profile 同名扩展；
3. canonical wire fields 永远不能被覆盖；冲突返回
   `ReservedExtraBodyField`，而不是静默让请求变形；
4. request-level headers 使用 `RequestOptions::header` /
   `RequestOptions::extra_headers`；
5. query 使用 `RequestOptions::extra_query`，协议保留参数会在发请求前拒绝。

中立请求参数映射：

```text
                       tool_choice response_format stop seed
OpenAI Chat                yes          yes        yes  yes
OpenAI Responses           yes          yes        no   no
Anthropic Messages         yes          no         yes  no
Gemini generateContent     yes          yes        yes  yes
```

不支持的组合返回 `Unsupported`，不会静默丢弃。`ToolChoice::Tool(name)` 在工具
不存在时返回 `InvalidRequest`。

### 6.2 ResponseMetadata

```rust
pub struct ResponseMetadata {
    pub status: u16,
    pub request_id: Option<String>,
    pub headers: HeaderMap,
}
```

已挂到：

- `ChatResponse.metadata`
- `StreamEvent::Start.metadata`
- `StreamEvent::Done.response.metadata`

---

## 7. Streaming 契约

### 7.1 StreamEvent

```rust
pub enum StreamEvent {
    Start {
        provider: &'static str,
        model: String,
        metadata: ResponseMetadata,
    },
    TextDelta {
        text: String,
    },
    ReasoningDelta {
        kind: ReasoningKind,
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    Usage {
        usage: Usage,
    },
    Retry {
        delay: Duration,
    },
    Done {
        response: Box<ChatResponse>,
        finish_reason: Option<String>,
    },
}
```

### 7.2 最重要的规则

下游应持久化：

```rust
Done.response.message
```

不应自己拼接 delta 作为最终历史。

### 7.3 Terminal marker

不同协议：

```text
OpenAI Chat         [DONE] 或 EOF
OpenAI Responses    response.completed / failed / incomplete
Anthropic Messages  message_stop
Gemini              EOF / finishReason
```

SDK 负责将这些统一成：

```rust
StreamEvent::Done
```

### 7.4 Retry 与 reconnect

开始接收 stream 前的 HTTP 错误继续按 `RetryPolicy` 重试。

流开始后的 reconnect 默认关闭。调用方显式提供：

```rust
RequestOptions::new().stream_reconnect(
    StreamReconnectPolicy::new()
        .max_attempts(2)
        .delay(Duration::from_millis(250))
        .require_event_id(true),
)
```

实现规则：

- 只有 transport error，或非 EOF-terminal 协议遇到异常 EOF，才 reconnect；
- `require_event_id=true` 时没有看到 SSE `id` 就拒绝 reconnect；
- reconnect 携带 `Last-Event-ID`；
- 复用同一个 mapper，不能重新从空字符串拼接；
- `(id, data)` 最近 4096 条用于去重；
- reconnect 时保留 session、idempotency key、body 和静态 headers；
- 已收到 `Done` 后不再 reconnect。

---

## 8. Header、Session 与 Idempotency

### 8.1 Header 优先级

```text
SDK 默认 UA
    ↓
TransportConfig
    ↓
RequestOptions
    ↓
HeaderInjector
```

`HeaderInjector` 是受信任层，可以设置 `Authorization`。

### 8.2 Session

当前不强制 sessionID。

`RequestContext.session_id` 不会自动上 wire。
如果需要 gateway affinity，调用方必须显式：

```rust
RequestOptions::new().header("x-sessionid", session_id)?
```

或通过 `HeaderInjector` 注入。

Retry 不会自动换 session。

同一个逻辑请求的重试复用同一份 body、静态 headers、session 和 idempotency
key；`HeaderInjector` 会在每次 HTTP attempt 前重新执行，因此可以刷新 token 或
临时 routing header。

### 8.3 Idempotency-Key

```rust
RequestOptions::new().idempotency_key("request-123")
```

会发送：

```http
Idempotency-Key: request-123
```

### 8.4 Audit

审计通过 `AuditSink` 接收结构化事件，默认关闭。事件不包含：

- messages / prompt；
- reasoning；
- tool arguments / result；
- request body / response body；
- API key。

事件可以包含：

```text
provider / protocol / model
attempt
status
duration
response-headers elapsed
stream first-token elapsed / kind
request-body bytes
cumulative decoded response-body bytes
outcome
ErrorKind
retryable
usage
provider request id
normalization repair counts
profile id / model-profile hit / capability snapshot (opt-in)
```

时序语义：

- `ResponseHeaders.elapsed` 是逻辑请求开始到 HTTP response headers 到达；
- `RequestFinished.timing.time_to_headers` 是首个成功 HTTP response；
- `RequestFinished.timing.time_to_first_token` 是首个 text / reasoning /
  tool-call stream delta；
- reqwest 不暴露独立 DNS / TCP / TLS 阶段，因此审计不会声称能区分这些阶段。

`Normalization` 只发送聚合计数，不发送自定义 role 名、tool 名、call id 或消息。
流在 terminal event 前被 drop 时，`RequestFinished.outcome` 为 `Cancelled`；可用
`record_cancellation(false)` 关闭。

关联 ID 的隐私开关：

```rust
AuditConfig::enabled()
    .include_sdk_request_id(false)
    .include_provider_request_id(true)
    .include_session_id(false)
    .include_usage(true)
    .include_profile(false)
    .record_timing(true)
    .record_normalization(true)
    .record_cancellation(true)
```

`session_id` 默认关闭，因为它是上下文标识，可能包含租户或用户信息。自定义
profile id 也可能包含租户信息，所以同样默认关闭。

生产投递使用非阻塞有界队列：

```rust
let (sink, receiver, stats) = bounded_audit_channel(4096);
let config = AuditConfig::enabled().sample_every(10);
```

- `try_send`，不阻塞模型请求；
- 队列满：丢弃并增加 `dropped_full`；
- receiver 关闭：丢弃并增加 `dropped_disconnected`；
- `sample_every` 按整个逻辑请求采样，保持 start/attempt/finish 一致。

---

## 9. Endpoint 与 Profile

本节描述当前已经落地的 profile 抽象。`EndpointSpec`、`AuthStyle`、
`ProviderProfile`、`ModelProfile`、reasoning alias、thinking request、
replay policy 和 capability gating 已接入 `EndpointAdapter`。具体协议的
wire mapping 继续由四个协议模块负责。

### 9.1 ProtocolSurface

```rust
pub enum ProtocolSurface {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
    GeminiGenerateContent,
}
```

### 9.2 EndpointSpec

```rust
pub struct EndpointSpec {
    pub protocol: ProtocolSurface,
    pub base_url: String,
    pub path: String,
    pub stream_path: Option<String>,
    pub auth: AuthStyle,
    pub profile: ProfileSelector,
}
```

`protocol` 和 `path` 必须显式提供。`stream_path` 用于 Gemini 这类流式路径不同的
协议；`{model}` 可由 `EndpointSpec::url_for(model, streaming)` 替换。SDK 不会
根据 URL 推断 protocol。

### 9.3 AuthStyle

```rust
pub enum AuthStyle {
    Bearer,
    XApiKey { header: HeaderName },
    QueryKey { parameter: String },
    Custom {
        header: HeaderName,
        value_prefix: Option<String>,
    },
    None,
}
```

### 9.4 ProfileSelector

```rust
pub enum ProfileSelector {
    Generic,
    Custom(Arc<ProviderProfile>),
    Builtin(ProfileId),
}
```

`Builtin` 通过显式 `ProfileRegistry` 解析，不根据 base URL 或 path 猜测。
未知 ProfileId 会返回 `Unsupported`。

```rust
let registry = ProfileRegistry::new();
let profile = registry.resolve(&ProfileId::from("qwen"));
```

已注册 ID/alias 包括：

```text
deepseek
qwen / bailian / dashscope
kimi / moonshot
glm / zhipu / bigmodel
doubao / ark / volcengine
openai
anthropic
gemini / google
openai-compatible
```

### 9.5 ProviderProfile

```rust
pub struct ProviderProfile {
    pub id: ProfileId,
    pub request: RequestProfile,
    pub reasoning: ReasoningProfile,
    pub tools: ToolProfile,
    pub stream: StreamProfile,
    pub usage: UsageProfile,
    pub capabilities: Capabilities,
    pub max_tokens_semantics: MaxTokensSemantics,
}
```

endpoint 负责 protocol/path/auth；provider profile 负责请求默认字段、
reasoning、tool、stream、usage 和 capability 行为，避免同一信息有两个 owner。

当前提供以下文档级 preset，它们只描述行为，不绑定 URL：

```rust
ProviderProfile::deepseek_compatible()
ProviderProfile::qwen_compatible()
ProviderProfile::kimi_compatible()
ProviderProfile::glm_compatible()
ProviderProfile::doubao_compatible()
```

preset 中没有完全确认的模型级差异必须通过 `ModelProfile` 显式覆盖。

### 9.6 ModelProfile

```rust
pub struct ModelProfile {
    pub matcher: ModelMatcher,
    pub thinking: ThinkingRequestProfile,
    pub reasoning_aliases: ReasoningAliases,
    pub replay: Option<ReasoningReplayPolicy>,
    pub max_tokens_semantics: MaxTokensSemantics,
    pub stream_function_arguments: bool,
    pub capabilities: Option<Capabilities>,
}
```

### 9.7 EndpointAdapter

```rust
let endpoint = EndpointSpec::new(
    ProtocolSurface::OpenAiChat,
    "https://gateway.example/v1",
    "/chat/completions",
    AuthStyle::Bearer,
);

let model = EndpointAdapter::new("qwen3", endpoint)
    .api_key_from_env("GATEWAY_API_KEY");
```

`EndpointAdapter` 根据显式 `protocol` dispatch：

```text
OpenAiChat
OpenAiResponses
AnthropicMessages
GeminiGenerateContent
```

`OpenAICompatible` 保留为兼容别名。所有协议的 profile request extras 都会参与
合并，`extra_body` 仍不能覆盖各协议的 canonical 字段。Gemini streaming 使用
`stream_path`，并在 query 中追加 `alt=sse`。

---

## 10. Reasoning Profile

### 10.1 Alias

```rust
pub struct ReasoningAliases {
    pub response_text: Vec<String>,
    pub response_summary: Vec<String>,
    pub request_replay: Vec<String>,
    pub encrypted: Vec<String>,
    pub signature: Vec<String>,
}
```

例如：

```text
Kimi          -> reasoning_content
GLM           -> reasoning_content
Qwen          -> reasoning_content
MiniMax       -> reasoning_content + reasoning_details
StepFun       -> reasoning 或 reasoning_content
Doubao Ark    -> reasoning_content + encrypted_content
```

### 10.2 Replay policy

```rust
pub enum ReasoningReplayPolicy {
    Never,
    SameProvider,
    PreserveInHistory,
    RequiredForToolCalls,
    EncryptedOnly,
    ModelDefined,
}
```

### 10.3 Thinking request

```rust
pub enum ThinkingRequestProfile {
    None,
    EnabledFlag(String),
    ThinkingObject,
    Effort(String),
    MappedEffort {
        field: String,
        mapping: EffortMapping,
    },
    BudgetTokens(String),
    Field {
        path: String,
        value: Value,
    },
    Composite(Vec<ThinkingRequestProfile>),
}
```

已经接入 `EndpointAdapter` 的 OpenAI Chat 行为：

- `EnabledFlag` -> `enable_thinking: bool`
- `ThinkingObject` -> `thinking: { type, budget_tokens? }`
- `Effort` -> 直接写 effort 字段
- `MappedEffort` -> 按 provider 映射后写 effort 字段
- `BudgetTokens` -> 仅在 reasoning 未关闭时写预算
- `Field` -> 设置静态或嵌套字段，例如 `thinking.keep`
- `Composite` -> 按顺序应用多个字段

Kimi 模型 preset：

```rust
ModelProfile::kimi_k3()    // reasoning_effort
ModelProfile::kimi_k2_6()  // thinking.type + thinking.keep = all
```

目前没有直接表达的模型级字段包括：

```text
reasoning_format
reasoning_split
preserve_thinking
clear_thinking
```

这些字段在 fixture 和语义确认前继续通过显式 `extra_body` 或未来的
`ModelProfile` variant 接入，不做跨 provider 猜测。`clear_thinking` /
`preserve_thinking` 现在也可以通过静态 `Field` 显式配置。

### 10.4 Capability gating

`Capabilities` 包含：

```rust
pub struct Capabilities {
    pub reasoning: ReasoningCapabilities,
    pub tools: ToolCapabilities,
    pub streaming: StreamCapabilities,
    pub tool_choice: bool,
    pub structured_output: bool,
    pub seed: bool,
    pub stop: bool,
    pub multimodal: bool,
    pub server_side_state: bool,
}
```

`EndpointAdapter` 在发请求前检查：

- reasoning 请求是否允许；
- reasoning summary/text/encrypted 是否允许；
- 是否允许 replay 已存在的 provider state；
- tool call 与 streaming tool call 是否允许；
- streaming 是否允许；
- 图片输入是否允许。

`ModelProfile.capabilities` 非空时覆盖 `ProviderProfile.capabilities`。
- UI summary：显式 Downgrade
- provider-specific unknown field：禁止猜测

---

## 11. Tool Profile

需要描述：

- tool call id 是否必有
- id 是否可合成
- tool result 是否必须用 `tool_call_id`
- 是否允许并行工具调用
- 是否支持 tool streaming
- tool arguments 是否分片
- 工具参数是否可能不是合法 JSON
- strict schema 是否支持

当前实现已经支持：

- 合成 id
- 按 name 配对
- 缺失 result 合成 error
- orphan result 丢弃
- OpenAI Chat / Responses / Anthropic / Gemini streaming tool args 累积

未来增加：

```rust
pub enum ToolCallIdPolicy {
    Preserve,
    Synthesize,
    NamePair,
}
```

---

## 12. Stream Profile

```rust
pub enum StreamTerminal {
    DataDone,
    ResponseCompleted,
    MessageStop,
    Eof,
}

pub enum UsagePlacement {
    EveryChunk,
    LastChunk,
    UsageEvent,
    None,
}
```

需要处理的特殊行为：

- Responses 不发 `[DONE]`
- Chat 通常发 `[DONE]`
- Anthropic 发 `message_stop`
- Gemini 通常 EOF + finishReason
- usage 可能在每个 chunk、最后 chunk 或 completed response 中
- reasoning delta 可能是增量，也可能在完成时给完整 state
- encrypted content 可能只在最后出现
- tool arguments 可能跨多个 chunk

---

## 13. Error 与 Retry Spec

当前稳定分类：

```rust
pub enum ErrorKind {
    Authentication,
    PermissionDenied,
    NotFound,
    InvalidRequest,
    RateLimited,
    Overloaded,
    Timeout,
    Connection,
    Decode,
    StreamProtocol,
    ProviderInternal,
    Unsupported,
    Configuration,
    Normalization,
    Tool,
    MaxSteps,
    Unknown,
}
```

`Error` 提供：

```rust
error.kind() -> ErrorKind
error.status() -> Option<u16>
error.provider() -> Option<&str>
error.request_id() -> Option<&str>
error.retry_after() -> Option<Duration>
error.retry_source() -> Option<RetrySource>
error.is_retryable() -> bool
error.body_bytes() -> Option<usize>
error.provider_error() -> Option<ProviderErrorInfo>
error.raw_body() -> Option<&str>
```

`Error::Api` 的 `Display` 和 `Debug` 默认脱敏 provider body，只展示 provider、
status 和 body 字节数。`provider_error()` 对常见 OpenAI、Anthropic、Gemini 和
兼容网关 JSON 形状做结构化摘要，仅保留清洗后的 code/type/status、message
字节数和 details 数量。原文仅能通过显式 `raw_body()` 或字段访问取得；普通日志
不应调用它们。

状态映射：

```text
401                 Authentication
403                 PermissionDenied
404                 NotFound
400/409/413/415/416/422 InvalidRequest
408                 Timeout
425                 Overloaded
429                 RateLimited
500..=599           ProviderInternal
```

retryable 默认包括：

```text
RateLimited
Overloaded
Timeout
Connection
ProviderInternal
```

`RetryPolicy` 继续复用 `Error::is_retryable()` 和 `retry_after()`，避免两套判断
逻辑漂移。

默认 `require_idempotency_key = true`：

- 有 `Idempotency-Key`：按 `max_attempts` 重试；
- 没有 key：只发送第一次，不自动重放；
- 显式 `.require_idempotency_key(false)` 才允许无 key 重放。

这样可以避免一次普通 POST 在超时/429 后被静默重复执行或重复计费。

仍待补充：

- provider body 的结构化错误码解析；
- content-filter 与 context-length 的更精确识别；
- cancellation；
- request id 进入非 API 错误链。

---

## 14. 测试规范

### 14.1 当前测试层

- 单元测试
- normalize tests
- headers tests
- retry tests
- SSE boundary tests
- 手写 parser vs sse-stream 性能对照
- 手写 parser vs qaqh data-only decoder 性能对照（隔离 worktree）
- 10,000 session smoke + ignored stress test
- 本地 HTTP/SSE 集成测试
- retry 同 session/idempotency 测试

### 14.2 必须新增

- provider fixture：non-stream / stream
- arbitrary chunk split fixture
- reasoning alias fixture
- reasoning replay fixture
- tool id repair fixture
- no `[DONE]` Responses fixture
- usage placement fixture
- CJK / emoji chunk fixture
- retry header rebuild test
- HeaderInjector per-retry test
- cancellation test
- stream resume test

### 14.3 Probe

开发阶段提供：

```bash
cargo run --example provider_probe -- kimi
cargo run --example provider_probe -- qwen
```

Probe 只用于开发/生成 fixture，不进入正常请求路径。

---

## 15. 国内兼容矩阵

详细来源见：

- [`docs/provider-compatibility.md`](provider-compatibility.md)

当前已确认：

- DeepSeek：OpenAI Chat + Responses + Anthropic Messages
- Qwen：OpenAI Chat + Responses + Anthropic Messages + DashScope
- Kimi：OpenAI Chat + Responses + Anthropic Messages
- GLM：OpenAI Chat + Responses + Anthropic Messages
- Doubao Ark：OpenAI Chat + Responses + Anthropic Messages
- Hunyuan：OpenAI Chat + Anthropic Messages
- MiniMax：OpenAI Chat + Anthropic Messages
- StepFun：OpenAI Chat + Anthropic Messages
- 百度 Qianfan：OpenAI-compatible Chat
- SiliconFlow：OpenAI-compatible Chat
- ModelScope：待进一步核对

---

## 16. Roadmap

### Phase 0：已完成

- role normalization
- tool pairing
- four provider adapters
- retry
- SSE parser
- reasoning normalization
- streaming event model
- response metadata
- HeaderInjector
- Idempotency-Key

### Phase 1：基础库 1.0

- [x] extra_body / extra_headers / extra_query
- [x] EndpointSpec
- [x] ProviderProfile 基础类型
- [x] ModelProfile 基础类型
- [x] Capabilities 与 EndpointAdapter capability gating
- [x] EndpointAdapter 四协议 generic dispatch
- [x] OpenAICompatible 兼容别名
- [x] ReasoningAliases / ThinkingRequest / ReasoningReplayPolicy 驱动 Chat profile
- [x] ToolCallIdPolicy 类型（尚未驱动 normalize）
- [x] typed ErrorKind / status / provider / request id
- [ ] provider probe
- [ ] domestic provider fixtures

### Phase 2：生产级

- [x] stream reconnect / resume / Last-Event-ID / dedup
- [x] per-retry header refresh
- cancellation
- [x] context-free audit hooks
- tracing spans / metrics exporters
- [x] request id propagation
- [x] idempotent retry policy
- [x] structured output
- cache/control
- [x] provider registry
- fallback / routing as separate layer

### Phase 3：扩展能力

- embeddings
- rerank
- batch
- files
- realtime / websocket
- audio
- image generation
- server-side tools
- MCP integration

---

## 17. 不可破坏的 invariants

1. `response.text()` 不包含 reasoning。
2. tool call 与 tool result 在出口前必须配对。
3. 跨 provider opaque state 不得误传。
4. `Done.response` 是流式历史的唯一权威结果。
5. 同一个逻辑请求的 retry 必须复用同一 session 和 idempotency key。
6. 未知 provider 字段不得自动跨协议重放。
7. 下游 runtime 不直接处理 provider wire fields。
8. Provider 特判不得进入 SDK 公共 API 或下游主循环。
9. 不支持的能力必须显式 error/drop/downgrade。
10. 不进行隐式网络探测。

---

## 18. 当前交接注意事项

### 18.1 优先阅读

- `README.md`
- `docs/provider-compatibility.md`
- `src/types.rs`
- `src/normalize.rs`
- `src/stream.rs`
- `src/headers.rs`
- `src/profile.rs`
- `src/adapter/*.rs`

### 18.2 当前已知风险

- Qwen effort/budget 冲突通过模型级覆盖解决，但还没有真实模型矩阵 fixture。
- profile preset 来自官方文档，但还没有带日期的真实 provider fixture。
- `EndpointAdapter` 已支持四协议，但 provider preset 的 reasoning 映射主要针对
  OpenAI Chat surface；非 Chat 协议遇到 `Never`、`RequiredForToolCalls`、
  `EncryptedOnly` 等 replay policy 时会显式报错，避免静默违背策略。
- Responses `finish_reason` 目前没有统一语义，complete 时通常为 None。
- OpenAI Chat 的 `reasoning_details` 流式合并还需要按 provider fixture 验证。
- ModelScope 文档字段尚未提取完整。
- 国内 provider 真实模型版本变化快，fixture 必须带日期。

### 18.3 下一次最应该做什么

```text
1. Qwen 模型级 effort/budget 冲突矩阵
2. provider_probe（仅开发期）
3. Kimi/Qwen/DeepSeek/GLM/Doubao 带日期 fixtures
4. cancellation
5. tracing / metrics
```

### 18.4 不应做什么

- 不要给每个厂商写一个独立 provider adapter。
- 不要把 reasoning alias 写进 SDK 或下游 runtime 主循环。
- 不要把 Agent loop、工具执行或权限策略重新加回库公共 API。
- 不要根据 base URL 自动猜协议。
- 不要把所有 reasoning 都强制转成普通 text。
- 不要在没有 fixture 的情况下“兼容”新 provider。
- 不要把 provider 私有字段硬编码进通用 ChatRequest。

---

## 19. 验收标准

一个“基本完成”的 SDK 应满足：

1. 切换 provider 不需要改下游 runtime 或请求构造代码。
2. non-stream 和 stream 使用同一中立语义。
3. reasoning/tool/usage/error 都有稳定出口。
4. 未知 provider 必须显式配置 profile。
5. 所有 provider quirks 都在 profile/adapter 内。
6. 跨 provider state 默认不会误传。
7. retry 不串 session，不改变 idempotency key。
8. 有 fixture、mock server、probe 和边界测试。
9. 新字段可通过 `extra_*` 接入，无需改核心模型。
10. API 有版本策略，不因 provider 新字段频繁破坏下游。

---

## 20. 结论

这个 SDK 的“最大公约数”不是一组相同 JSON 字段，而是：

```text
统一语义
+ 显式 protocol surface
+ endpoint profile
+ model profile
+ capability gating
+ opaque provider state
+ strict egress
+ lenient ingress
```

只要守住这条边界，provider 再多、字段再乱，也不会把复杂度泄漏给下游 runtime。
