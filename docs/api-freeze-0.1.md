# mutil-ai API Freeze 0.1

> 状态：冻结候选，供团队评审是否引入。
> crate：`mutil-ai` / `mutil_ai`
> version：`0.1.0`
> edition：`2024`
> freeze date：`2026-09-24`

本文档不是 1.0 发布声明。它的作用是冻结当前公开 API 的名称和语义，避免团队
评审期间继续漂移。冻结窗口内只接受：

- 明确 bugfix；
- 不改变现有调用方式的兼容性修复；
- 文档、测试和诊断修正。

不接受：

- 修改现有公开类型/方法语义；
- 重命名公开类型或 builder；
- 改变 normalization、stream、retry、audit 的默认行为；
- 将 provider-specific 字段提升进 Agent 主循环。

## 1. 冻结范围

冻结的是 SDK 的中立语义和公开 API，不是所有 provider 的 wire 行为。

```text
stable candidate
  neutral Message / Part / Reasoning / ToolCall / ToolResult
  normalization egress contract
  Protocol / EndpointSpec / ProviderProfile / ModelProfile
  ModelAdapter / complete / stream
  StreamEvent / Done.response
  RetryPolicy / ErrorKind / Error helpers
  RequestContext / HeaderInjector / RequestOptions
  AuditConfig / AuditEvent / AuditSink
  SSE parser public limits

not guaranteed
  exact provider response fields
  exact provider error-code vocabulary
  provider model version behavior
  benchmark numbers
  future embeddings/rerank/batch/realtime APIs
```

## 2. 稳定入口

### 2.1 Agent 使用的主入口

```rust
use mutil_ai::{
    Agent, AgentBuilder, ChatRequest, ChatResponse, Message, ModelAdapter,
    OpenAIChat, OpenAIResponses, AnthropicMessages, GeminiGenerateContent,
    EndpointAdapter, EndpointSpec,
};
```

冻结语义：

- Agent 只构造 `ChatRequest`，不处理 provider wire fields。
- `ModelAdapter::complete_with` 是普通请求入口。
- `ModelAdapter::stream_with` 是流式请求入口。
- `EndpointAdapter` 必须显式选择协议、URL、path、auth 和 profile。
- SDK 不从 base URL 猜协议。

### 2.2 中立消息模型

```rust
use mutil_ai::{
    Message, Part, Role, Reasoning, ReasoningKind,
    ToolCall, ToolResult, ProviderState, ProviderStateFormat,
};
```

冻结语义：

- 应用历史可以包含 `System / Developer / User / Assistant / Tool / Custom`。
- `Reasoning` 只表达 `summary / text / encrypted / redacted` 等中立类别。
- opaque provider state 只表示“原样回放”，SDK 不解释内部内容。
- `ToolCall` 和 `ToolResult` 是不同 part，不要求上游 Agent 自己理解 provider
  的 pairing 规则。

### 2.3 Normalization

```rust
use mutil_ai::{
    normalize, normalize_with_options, Protocol,
    NormalizeOptions, NormalizeReport, NormalizeAction,
};
```

冻结的默认出口：

```text
System / User / Assistant / Tool
```

冻结的默认转换：

| 输入 | 默认输出 |
|---|---|
| `Developer` | `System` |
| `Custom(_)` | `User` |
| assistant tool call 无 result | 合成 error tool result |
| orphan tool result | 丢弃 |
| foreign reasoning state | 丢弃 |
| reasoning 出现在非 assistant | 丢弃 |
| Anthropic / Gemini 相邻同 role | 合并 |

重要不变量：

1. Adapter 只能看到归一化后的对话。
2. 出口不得出现 developer/custom role。
3. 出口 tool call/result 必须配对。
4. 跨 provider opaque state 默认不得误传。
5. 所有修复必须进入 `NormalizeReport`。

### 2.4 Endpoint 与 Profile

```rust
use mutil_ai::{
    AuthStyle, Capabilities, EndpointSpec, ProtocolSurface,
    ProviderProfile, ModelProfile, ProfileSelector, ProfileRegistry,
};
```

冻结语义：

- `ProtocolSurface` 明确区分：
  - OpenAI Chat Completions
  - OpenAI Responses
  - Anthropic Messages
  - Gemini generateContent
- `EndpointSpec` 负责 protocol、base URL、path、auth。
- `ProviderProfile` 负责请求默认字段、reasoning alias/replay/thinking、
  capability gate 等行为。
- `ModelProfile` 只覆盖匹配模型的差异，不改变 protocol surface。
- capability 不支持时，必须显式 error/drop/downgrade，不得猜测。

### 2.5 Streaming

```rust
use mutil_ai::{
    ModelStream, StreamEvent, StreamReconnectPolicy,
    collect_stream, next_event,
};
```

冻结语义：

- `StreamEvent::Start` 表示 HTTP response 已接受。
- text、reasoning、tool-call 使用独立 delta。
- `Done.response` 是最终历史的唯一权威结果。
- 调用方不得通过拼接 delta 代替 `Done.response` 持久化历史。
- retry/reconnect 状态由 SDK 管理，不泄漏为 provider wire 字段。
- 消费者提前 drop stream 产生 `AuditOutcome::Cancelled`。
- EOF 没有 provider terminal event 时返回 `StreamProtocol` 错误。

### 2.6 Retry 与 Session

```rust
use mutil_ai::{
    RetryPolicy, RetrySource, RetryDirective,
    RequestOptions, RequestContext, HeaderInjector,
};
```

冻结语义：

- 同一个逻辑请求的自动重试复用：
  - session id；
  - idempotency key；
  - request body；
  - request context。
- `HeaderInjector` 可以在每次 attempt 前刷新认证/temporary header。
- 已产生 stream delta 后，SDK 不自动重放整个模型请求。
- POST 默认要求 `Idempotency-Key` 才允许自动重放。
- session id 不会替代 idempotency key，也不会被 SDK 自动生成。

### 2.7 Error

```rust
use mutil_ai::{Error, ErrorKind, ProviderErrorInfo};
```

冻结语义：

- `ErrorKind` 是稳定高层分类，调用方不应匹配错误字符串。
- `status()`、`provider()`、`request_id()`、`retry_after()`、`retry_source()`、
  `is_retryable()` 保持现有含义。
- `Error::Api` 的 `Display` / `Debug` 默认脱敏 provider body。
- `body_bytes()` 和 `provider_error()` 可安全用于日志/metrics。
- `raw_body()` 是显式敏感访问，不得默认进入日志。
- provider error code/type/status 只是清洗后的诊断数据，不承诺每个 provider
  都有值。

### 2.8 Audit

```rust
use mutil_ai::{
    AuditConfig, AuditEvent, AuditOutcome, AuditSink,
    bounded_audit_channel,
};
```

冻结语义：

- 审计默认关闭。
- 审计永不包含 prompt、message、reasoning、tool 内容、request/response body。
- session id、SDK request id、profile id 默认不记录。
- provider request id 默认可用，可关闭。
- normalization 只记录聚合计数。
- timing 明确区分 response headers 和 first token；不声称能区分 DNS/TCP/TLS。
- byte counts 只记录大小。
- `Success / Failure / Cancelled` 是完整最终 outcome。

## 3. 稳定性等级

### 3.1 Frozen candidate

这些 API 在团队评审期间冻结：

- `Message`、`Part`、`Reasoning`、`ToolCall`、`ToolResult`
- `ChatRequest`、`ChatResponse`
- `Protocol`、`EndpointSpec`、`ProviderProfile`、`ModelProfile`
- `ModelAdapter`、四协议 adapter
- `StreamEvent`、`StreamReconnectPolicy`
- `RetryPolicy`、`ErrorKind`
- `RequestContext`、`RequestOptions`、`HeaderInjector`
- `AuditConfig`、`AuditEvent`、`AuditSink`

### 3.2 可向后兼容扩展

以下内容可以在不改变现有字段语义的前提下增加：

- 新 provider preset；
- 新 reasoning alias；
- 新 capability 字段；
- 新 `ErrorKind`；
- 新 `StreamEvent`；
- 新 `AuditEvent`；
- 新 provider error shape。

新增 enum variant 对下游 exhaustive match 仍可能造成编译影响，因此引入前必须
在 changelog 中明确说明。

### 3.3 不承诺

- provider 文档未确认的字段猜测；
- provider 模型随时变化的默认值；
- 对任意 OpenAI-compatible 网关的自动识别；
- 未带 fixture 的新厂商兼容；
- 未指定 provider/model/date 的长期 wire 稳定性。

## 4. 团队接入建议

推荐路径：

```text
qaqh Agent / gate
    -> bridge layer
    -> mutil-ai neutral types
    -> EndpointAdapter / built-in adapter
```

不推荐：

```text
直接替换 qaqh-gate stateful proxy
```

原因：

- `qaqh-gate` 已有 stateful incremental history、web search、OpenCode headers、
  DSML/XML tool parser、skill envelope 和完整 ToolResult 语义。
- `mutil-ai` 当前是 provider-neutral SDK core，不是完整 Agent gateway。
- 更合理的做法是 feature-gated bridge 或双后端，先验证边界，再决定是否替换。

## 5. 评审必须确认的问题

1. `Message / Part / Reasoning` 是否足够承载现有 Agent 历史？
2. strict egress 默认是否接受 developer→system、custom→user？
3. 是否接受显式 `ProtocolSurface`，拒绝 URL 自动推断？
4. `EndpointSpec + ProviderProfile + ModelProfile` 的分层是否适合团队维护？
5. `Done.response` 作为流式历史唯一权威结果是否与现有持久化一致？
6. audit 默认不记录 session/profile/body 是否符合生产要求？
7. cancellation 作为 `RequestFinished { outcome: Cancelled }` 是否满足 metrics？
8. 团队选择 bridge、双后端，还是最终替换 `qaqh-gate`？

## 6. 冻结后的验收命令

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets --quiet
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --quiet
```

冻结候选的当前结果：

```text
151 passed, 3 ignored
```

## 7. 变更控制

在团队给出评审结论前：

- 不提升 crate 版本；
- 不改公开 API 名称；
- 不改 normalization 默认策略；
- 不改 retry 安全默认值；
- 不改 audit 隐私默认值；
- 不改 `Done.response` 语义；
- 不增加新的必选依赖。

如果必须修复，记录：

```text
issue:
affected API:
semantic change: yes/no
migration:
tests:
```

## 8. 结论

`mutil-ai 0.1` 的冻结语义可以概括为：

```text
入口宽松
+ 出口严格
+ 协议显式
+ provider quirks 留在 profile/adapter
+ retry 不串 session
+ stream 历史以 Done.response 为准
+ audit 无上下文留痕
```

这份冻结用于团队研判，不等同于 1.0 API 承诺。
