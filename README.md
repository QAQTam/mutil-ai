# mutil-ai（教学版）

一个适合刚开始写 Agent 的 Rust 小 SDK。

它的目标不是一次支持所有高级功能，而是先回答一个关键问题：

> 内部 agent 历史可以很乱，但发到 provider 之前，如何变成干净、严格的 role 和工具配对？

这个库不照搬现有 SDK 的 API 设计。它自己定义一套很小的中立消息模型，然后让每个 provider adapter 只负责：

1. 调用归一化层；
2. 把干净的消息序列化成 provider 的 HTTP 请求；
3. 把 provider 响应转回中立模型。

适配器仍然必须遵守各家的 HTTP 协议，因为不遵守就无法调用；但应用层不需要直接面对这些差异。

国内主要模型的协议面和字段差异汇总见：

- [`docs/provider-compatibility.md`](docs/provider-compatibility.md)

SDK 的长期契约、当前实现和交接清单见：

- [`docs/sdk-spec.md`](docs/sdk-spec.md)

供团队研判是否引入的冻结候选 API 语义见：

- [`docs/api-freeze-0.1.md`](docs/api-freeze-0.1.md)

下游 gateway / agent 的通用扩展能力登记见：

- [`docs/consumer-capability-requirements.md`](docs/consumer-capability-requirements.md)

## 1. 核心思想

### 内部 role

应用内部允许出现这些角色：

```text
System / Developer / User / Assistant / Tool / Custom(String)
```

这样旧 agent、插件、工作流可以保留自己的语义，不必为了某个 provider 立刻重写历史。

### 出口 role

到 provider 之前，只会留下安全角色：

```text
System / User / Assistant / Tool
```

默认降级规则：

| 内部 role | 出口处理 |
|---|---|
| `System` | 提取为 system instructions |
| `Developer` | 降级为 `System` |
| `Custom("...")` | 降级为 `User` |
| `User` | `User` |
| `Assistant` | `Assistant` / Gemini 的 `model` |
| `Tool` | provider 对应的 tool result |

## 2. 工具配对清洗

严格 provider 最怕的是“有 tool call，没有 tool result”或者“有 tool result，但没有对应 tool call”。

本库默认规则：

1. assistant 的 `ToolCall` 没有 id：自动补 `call_1`、`call_2`。
2. tool result 没有 id：按工具名匹配，补上对应 id。
3. tool call 后面没有 result：自动合成一个错误 result，保证配对完整。
4. 孤儿 tool result：默认丢弃，避免污染上游。
5. Anthropic / Gemini：合并相邻 user/assistant 消息，并把 tool result 放进 user turn。

可以直接运行示例查看清洗过程：

```bash
cargo run --example normalize
```

输出会显示：

```text
DowngradedDeveloper
DowngradedCustomRole { role: "reviewer" }
AssignedToolCallId { name: "get_weather", id: "call_1" }
SynthesizedMissingToolResult { call_id: "call_1", name: "get_weather" }
```

## 3. 最小用法

```rust
use mutil_ai::{Agent, OpenAI};

#[tokio::main]
async fn main() -> mutil_ai::Result<()> {
    let model = OpenAI::responses("gpt-5.2");

    let mut agent = Agent::builder()
        .model(model)
        .system("你是一个简洁、耐心的 Rust 老师。")
        .build()?;

    let answer = agent.ask("用三句话解释什么是 agent。").await?;
    println!("{answer}");

    Ok(())
}
```

`OPENAI_API_KEY` 从环境变量读取。

切换 provider 不需要改 Agent 代码：

```rust
let model = OpenAI::chat("gpt-4.1");
// let model = OpenAI::responses("gpt-5.2");
// let model = mutil_ai::Anthropic::messages("claude-sonnet-4-5");
// let model = mutil_ai::Gemini::generate_content("gemini-3-flash");
```

## 4. 给 Agent 加工具

```rust
use mutil_ai::{Agent, OpenAI, tool_fn};
use serde_json::json;

let weather = tool_fn(
    "get_weather",
    "查询一个城市的天气",
    json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" }
        },
        "required": ["city"]
    }),
    |arguments| async move {
        let city = arguments["city"].as_str().unwrap_or("unknown");
        Ok::<_, mutil_ai::Error>(json!({
            "city": city,
            "weather": "sunny"
        }))
    },
);

let mut agent = Agent::builder()
    .model(OpenAI::responses("gpt-5.2"))
    .system("需要天气时调用工具。")
    .tool(weather)
    .max_steps(4)
    .build()?;

let answer = agent.ask("广州天气怎么样？").await?;
println!("{answer}");
```

完整示例见：

```bash
cargo run --example tool_agent
```

## 5. 统一思考链：message 与 reasoning

各家返回思考内容时，字段名和容器都不一样：

| 协议 | 原始字段 |
|---|---|
| OpenAI Chat 兼容接口 | `message.reasoning_content`、`message.reasoning`、`message.reasoning_details` |
| OpenAI Responses | output item `type = "reasoning"`，其中可有 `summary`、`content`、`encrypted_content` |
| Anthropic Messages | content block `thinking` / `redacted_thinking`，其中 `signature` / `data` 必须原样回传 |
| Gemini generateContent | part `thought: true`、`thoughtSignature`；签名也可能挂在 `functionCall` part 上 |

应用层不需要判断这些字段。响应统一变成：

```rust
Part::Text { text, provider_state }        // 给用户看的最终消息；state 可携带 Gemini 签名
Part::Reasoning(Reasoning {                 // 思考上下文或摘要
    kind: ReasoningKind,
    summary: Option<String>,
    text: Option<String>,
    state: Option<ProviderState>,
})
Part::ToolCall(call)                        // 工具调用
Part::ToolResult(result)                    // 工具结果
```

`ReasoningKind` 只有四种：

```text
Summary    // 摘要，例如 OpenAI Responses summary
Text       // 可读思考文本，例如 Anthropic thinking / Gemini thought
Encrypted  // 加密推理状态，用户不可读
Redacted   // 安全策略抹除的块
```

`state` 是 provider 专用的不透明数据，里面可能包含 OpenAI reasoning item、
Anthropic `signature`、Gemini `thoughtSignature` 等。应用不要解析或修改它。

下游可以只做这种匹配：

```rust
for part in &response.message.parts {
    match part {
        Part::Text { text, .. } => println!("message: {text}"),
        Part::Reasoning(reasoning) if reasoning.kind == ReasoningKind::Summary => {
            println!("summary: {}", reasoning.summary.as_deref().unwrap_or(""));
        }
        Part::Reasoning(reasoning) => {
            println!("reasoning: {}", reasoning.text.as_deref().unwrap_or(""));
        }
        Part::ToolCall(call) => println!("tool: {}", call.name),
        Part::ToolResult(result) => println!("tool result: {}", result.content),
        Part::ImageUrl { .. } => {}
    }
}
```

`response.text()` 和 `message.text_content()` 永远不包含 reasoning，避免把思考过程
误当最终答案。可运行：

```bash
cargo run --example reasoning
```

### 请求配置也统一

不同 provider 的开关同样由 adapter 映射：

```rust
use mutil_ai::{
    ChatRequest, ReasoningConfig, ReasoningEffort, ReasoningSummary,
};

let request = ChatRequest::user("分析这个问题")
    .reasoning(
        ReasoningConfig::new()
            .effort(ReasoningEffort::High)
            .summary(ReasoningSummary::Detailed)
            .include_text(true)
            .include_encrypted(true),
    );
```

映射规则：

- OpenAI Chat：`effort` -> `reasoning_effort`
- OpenAI Responses：`effort/summary/include_encrypted` -> `reasoning` 与 `include`
- Anthropic：`mode/budget_tokens/include_text` -> `thinking`
- Gemini：`effort/budget_tokens/include_text` -> `generationConfig.thinkingConfig`

不支持的目标字段会忽略，不会猜一个近似字段塞进去。

常用请求控制也有中立表达：

```rust
use mutil_ai::{ResponseFormat, ToolChoice};

let request = ChatRequest::user("返回结构化天气结果")
    .tools(tools)
    .tool_choice(ToolChoice::Required)
    .response_format(ResponseFormat::JsonSchema {
        name: "weather".to_string(),
        schema: serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temperature": {"type": "number"}
            },
            "required": ["city", "temperature"]
        }),
        strict: Some(true),
    })
    .stop(["END"])
    .seed(42);
```

各协议只映射自己支持的字段；例如 Anthropic 不接受中立
`response_format`，会返回 `Unsupported`，不会静默忽略。

### 回放边界

- 同一协议的 reasoning `state` 会原样回放。
- Anthropic 的 `thinking` / `redacted_thinking` 按原顺序放回 assistant content。
- OpenAI Responses 的 reasoning item 按原对象放回 `input`。
- Gemini 的 `thoughtSignature` 保留在对应 reasoning part、tool call 或可见 text part 上。
- 跨协议的 opaque state 默认在归一化层丢弃，绝不把 Claude/Gemini 签名发给 OpenAI。
- 非 assistant 消息里的 reasoning 默认丢弃并写进 `NormalizeReport`，不会悄悄提升成用户文本。

这样出口仍然只认识一套简单内容类型，provider 特例被限制在 adapter 内。

参考入口：

- OpenAI Reasoning: <https://developers.openai.com/api/docs/guides/reasoning>
- Anthropic Extended Thinking: <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>
- Gemini Thinking: <https://ai.google.dev/gemini-api/docs/thinking>
- Gemini Thought Signatures: <https://ai.google.dev/gemini-api/docs/thought-signatures>
- OpenRouter Reasoning Tokens: <https://openrouter.ai/docs/use-cases/reasoning-tokens>

## 6. 统一 streaming

四个内置 adapter 都实现了：

```rust
async fn stream_with(
    &self,
    request: &ChatRequest,
    options: &RequestOptions,
) -> Result<ModelStream>;
```

下游只看统一的 `StreamEvent`：

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

调用示例：

```rust
use mutil_ai::{ChatRequest, ModelAdapter, OpenAI, StreamEvent, next_event};

let model = OpenAI::chat("gpt-5.2");
let request = ChatRequest::user("你好");

let mut stream = model.stream(&request).await?;
while let Some(event) = next_event(&mut stream).await {
    match event? {
        StreamEvent::TextDelta { text } => print!("{text}"),
        StreamEvent::ReasoningDelta { text, .. } => print!("[thinking] {text}"),
        StreamEvent::ToolCallDelta { arguments_delta, .. } => {
            print!("[tool args] {arguments_delta}");
        }
        StreamEvent::Done { response, .. } => {
            // 持久化时使用组装好的完整 message，不要自己拼 delta。
            println!("\n{}", response.text());
        }
        _ => {}
    }
}
```

已经处理的关键问题：

- OpenAI Chat：`delta.content`、`reasoning_content`、`reasoning_details`
- OpenAI Responses：text / reasoning summary / reasoning text / function arguments 事件
- Anthropic：`content_block_delta`、`thinking_delta`、`signature_delta`、`input_json_delta`
- Gemini：`:streamGenerateContent?alt=sse`、thought、text、functionCall、thoughtSignature
- 工具参数按 index 累积，并在 `Done.response` 中解析成 JSON
- reasoning 文本与最终签名/encrypted state 在 `Done.response` 中保持完整
- usage 单独发送，并同时保存在最终 response
- `Start.metadata` 和 `Done.response.metadata` 包含 status、request id 和 response headers
- SSE 的 `retry:` 字段映射成 `StreamEvent::Retry`

当前边界：

- 开始接收流之前的 HTTP 错误会按 `RetryPolicy` 重试。
- 流开始后的 transport 断线可以显式启用 reconnect。
- reconnect 必须看到 SSE `id`，会携带 `Last-Event-ID`，并复用同一个 mapper。
- 已收到重复的 `(id, data)` 会被去重，避免重连后重复文本或工具参数。
- 只有 Chat/Gemini 允许把 EOF 当正常结束；Responses/Anthropic 未看到终止事件时
  会把它视为可恢复断线。
- `collect_stream` 可以忽略增量渲染，直接收集最终 `ChatResponse`。

启用断线重连：

```rust
use std::time::Duration;
use mutil_ai::{RequestOptions, StreamReconnectPolicy};

let options = RequestOptions::new().stream_reconnect(
    StreamReconnectPolicy::new()
        .max_attempts(2)
        .delay(Duration::from_millis(250))
        .require_event_id(true),
);

let mut stream = model.stream_with(&request, &options).await?;
```

重连时会发送：

```http
Last-Event-ID: <last-seen-id>
```

session、idempotency key 和原始 request body 保持不变。

可运行示例：

```bash
cargo run --example streaming
```

## 7. Retry 头怎么识别

`Retry-After` 是 HTTP response header，不是 body。常见于：

- `429 Too Many Requests`
- `503 Service Unavailable`

它有两种格式：

```http
Retry-After: 120
```

表示等 120 秒。

```http
Retry-After: Wed, 21 Oct 2026 07:28:00 GMT
```

表示等到这个 HTTP 日期。日期已经过去时，等待时间按 0 处理。

本库会：

1. 检查 response 的 `retry-after` header；
2. 先尝试按秒数解析；
3. 失败后按 HTTP date 解析；
4. 保存到 `Error::Api.retry_after`；
5. 通过 `error.retry_after()` 和 `error.is_retryable()` 提供给重试逻辑。

```rust
use std::time::Duration;

match model.complete(&request).await {
    Ok(response) => println!("{}", response.text()),
    Err(error) if error.is_retryable() => {
        let delay = error
            .retry_after()
            .unwrap_or(Duration::from_secs(1));
        eprintln!("retry after {delay:?}: {error}");
    }
    Err(error) => return Err(error),
}
```

错误本身有稳定分类，不需要匹配错误字符串：

```rust
use mutil_ai::ErrorKind;

match model.complete(&request).await {
    Ok(response) => println!("{}", response.text()),
    Err(error) if error.kind() == ErrorKind::RateLimited => {
        eprintln!("rate limited: {:?}", error.request_id());
    }
    Err(error) if error.kind() == ErrorKind::Authentication => {
        eprintln!("check credentials");
    }
    Err(error) => return Err(error),
}
```

`Error` 还提供：

- `kind()`：稳定的 `ErrorKind`；
- `status()`：HTTP status；
- `provider()`：provider 名称；
- `request_id()`：响应中可用的 request id；
- `retry_after()` / `retry_source()`；
- `is_retryable()`；
- `body_bytes()`：provider 错误 body 的字节数；
- `provider_error()`：安全的结构化错误摘要；
- `raw_body()`：显式读取 provider 原始错误 body。

`provider_error()` 兼容常见 OpenAI/Anthropic/Gemini 及兼容网关结构，返回经过
长度和字符白名单清洗的 `code`、`error_type`、`status`、message 字节数和
details 数量；不返回 provider message 原文。

`Error::Api` 的 `Display` 和 `Debug` 默认不会输出 provider body，避免 `eprintln!`
或日志框架意外记录被回显的 prompt、账号信息或工具内容。只有显式调用
`raw_body()` 或直接访问字段才会取得原文。

### RetryPolicy 设计

重试逻辑分三层：

```text
Retry-After / 状态码
        ↓
RetryPolicy::delay_for()
        ↓
retry_async / send_json_retry
        ↓
adapter
```

adapter 不自己决定 sleep 多久，只负责重建 HTTP request。

```rust
use mutil_ai::{OpenAI, RetryPolicy};
use std::time::Duration;

let model = OpenAI::responses("gpt-5.2").retry_policy(
    RetryPolicy::default()
        .max_attempts(5)
        .base_delay(Duration::from_millis(300))
        .max_delay(Duration::from_secs(10))
        .jitter_ratio(0.2)
        .max_retry_after(Duration::from_secs(60)),
);
```

规则：

1. 只有 `error.is_retryable()` 为 true 才重试。
2. 有 `Retry-After`：优先遵守服务端时间。
3. `Retry-After` 超过 `max_retry_after`：停止，避免无限等待。
4. 没有 `Retry-After`：指数退避，并加 jitter。
5. `max_attempts` 包含第一次请求。
6. 默认 `require_idempotency_key = true`：没有 `Idempotency-Key` 时只发送一次，
   不自动重放 POST。

显式接受无幂等键重试：

```rust
let policy = RetryPolicy::default()
    .max_attempts(3)
    .require_idempotency_key(false);
```

只有明确知道请求可安全重放时才关闭这个保护。

### Header 优先级

解析顺序固定为：

```text
1. retry-after-ms
2. retry-after
3. provider 专属 reset header
```

`retry-after-ms` 虽然不是标准头，但 OpenAI、Anthropic、Google 的官方 SDK
都支持，而且精度比整数秒高。

### OpenAI

官方 OpenAPI 中，`429` 和 `503` 响应会返回：

```http
Retry-After: 3
```

官方 Python SDK 还会优先识别：

```http
retry-after-ms: 1500
```

作为兜底，本库支持：

```http
x-ratelimit-reset-requests: 6m0s
x-ratelimit-reset-tokens: 1h
```

支持 `ns / us / µs / ms / s / m / h` 以及组合形式，例如 `1h30m`。

### Anthropic

官方 SDK 同样优先识别 `retry-after-ms`，然后是 `retry-after`。

Anthropic 的 reset header 是 RFC3339 UTC 时间：

```http
anthropic-ratelimit-requests-reset: 2026-02-25T20:02:32Z
anthropic-ratelimit-tokens-reset: 2026-02-25T20:02:36Z
anthropic-ratelimit-input-tokens-reset: 2026-02-25T20:02:37Z
anthropic-ratelimit-output-tokens-reset: 2026-02-25T20:02:36Z
```

### Google Gemini

官方 `python-genai` SDK 支持：

```http
retry-after-ms: 1
```

Google API 的错误 body 也可能包含 `RetryInfo.retryDelay`，但它不是 header，
因此当前版本先不把它混进 header 解析层。

参考入口：

- OpenAI Rate Limits: <https://developers.openai.com/api/docs/guides/rate-limits>
- Anthropic Rate Limits: <https://platform.claude.com/docs/en/api/rate-limits>
- Gemini Rate Limits: <https://ai.google.dev/gemini-api/docs/rate-limits>

## 8. 手写 SSE parser

`src/sse.rs` 是一个增量式、零 I/O 的 SSE parser：

```rust
let mut parser = SseParser::new();
for chunk in response.bytes_stream() {
    for event in parser.push(&chunk?)? {
        match event {
            SseEvent::Message(message) => println!("{}", message.data),
            SseEvent::Retry(delay) => println!("retry={delay:?}"),
        }
    }
}
```

它不依赖 `sse-stream`、`tokio-util` 或 `http-body`，只处理字节状态。

支持：

- chunk 任意切分
- `\n`、`\r\n`、单独 `\r`
- UTF-8 BOM
- 注释和未知字段
- 多行 `data`
- `event`、`id`、`retry`
- `id` 跨事件继承
- `event` 在 dispatch 后重置
- line/event 大小限制
- EOF 丢弃未完成事件

UTF-8 默认遵循 WHATWG EventSource：非法字节替换为 U+FFFD。需要
`sse-stream` 风格的严格报错时使用：

```rust
let parser = SseParser::strict();
```

正确性测试覆盖：

- 每个字节边界切分
- CRLF 跨 chunk
- BOM 跨 chunk
- 多行 data
- ID 继承
- event 重置
- retry 非法值和溢出
- NUL ID
- colonless data
- UTF-8 replace / strict
- CJK 在 codepoint 中间被切开
- emoji 在 codepoint 中间被切开
- CJK event/id/data
- line limit 按 UTF-8 字节数计算
- BOM 后接 CJK
- EOF 未完成事件

和 `sse-stream 0.3.0` 的对照测试：

```bash
cargo test --test sse_compare -- --ignored --nocapture
cargo test --test sse_concurrency -- --ignored --nocapture
```

第一个测试模拟 200,000 个 token，也就是 10,000 tok/s 持续 20 秒的流，
分别测量 1 / 10 / 100 个事件一个 chunk。

第二个测试按机器逻辑 CPU 数动态选择 1 / 4 / 8 / 12 个并发客户端，
每个客户端解析 300,000 个 token，测量：

- worker CPU 时间
- 每 token CPU 时间
- 轮次间 CPU 变异系数
- 客户端间 CPU 变异系数
- 相对 10,000 tok/s 的 CPU 余量

当前机器结果：

```text
1-8 clients:
  手写 parser 的每 token CPU 更低，通常低 15%-35%
  两种实现的轮次波动都很小

12/12 logical CPUs, 1 event/chunk:
  手写 parser 仍然更低

12/12 logical CPUs, 10 events/chunk:
  sse-stream 的每 token CPU 略低，客户端间波动也更小

所有场景:
  两种 parser 相对 10,000 tok/s 都有约 85x-230x CPU 余量
```

这说明 10,000 tok/s 本身远远达不到 SSE parser 的饱和点。真正让 CPU
产生波动的主要会是 JSON、网络、模型服务、日志和上层 agent 调度，而不是
SSE 字节解析。并发饱和时，内存分配器、SMT 和调度器的影响会开始超过 parser
本身。

### 与 qaqh-gate `SseDecoder` 的隔离对比

在独立 worktree 中直接比较了 qaqh 的 data-only decoder 与本库通用 parser：

```text
release, 500,000 events, 5 samples
```

优化后：

```text
ASCII:
  qaqh 约 83-95 ns/event
  mutil-ai 约 121-125 ns/event
  差距约 1.3x-1.47x

CJK + emoji:
  qaqh 约 107-109 ns/event
  mutil-ai 约 144-154 ns/event
  差距约 1.34x-1.42x
```

多客户端：

```text
1-8 clients:
  每事件 CPU 差距约 1.14x-1.47x

12 logical CPUs:
  差距扩大，约 2.0x-2.4x
  主要来自分配器、SMT 和调度竞争
```

qaqh 的 decoder 只处理 `data:`，不保留 event/id/retry，不执行严格 UTF-8
策略，因此它更快是预期结果；本库保留了完整 SSE 语义和 reconnect 所需状态。

### 10,000 session 压测

新增：

```bash
cargo test --test sse_sessions
cargo test --test sse_sessions -- --ignored --nocapture
```

测试模型：

- 10,000 个 parser 状态同时存活；
- 12 个 worker 轮转推进；
- 每个 session 独立解析；
- 校验事件总数，任何丢失都会失败；
- 常规 smoke test 每次提交运行；
- 100 events/session 的 heavy 版本标记为 ignored。

当前机器 release 结果：

```text
qaqh:    约 46M-60M events/s
mutil-ai: 约 17M-18M events/s
```

按网站常见 10,000 并发 session、每 session 平均 10-100 events/s 估算：

```text
总事件率约 100k-1M events/s
mutil-ai 仍有约 17x-170x 的解析余量
```

因此万 session 的目标不是 SSE parser 瓶颈，而是：

- 每 session 的内存占用；
- async task 数量和调度；
- 网络连接/TLS；
- JSON 与业务状态机；
- 日志和审计管线。


## 9. Header 与 User-Agent

SDK 提供默认 UA，但允许客户端覆盖：

```rust
let transport = TransportConfig::new()
    .user_agent("my-app/1.2.3")?
    .header("x-route", "prod")?
    .client_info(
        ClientInfo::new("my-app", "1.2.3")
            .instance_id("node-7"),
    );

let model = OpenAI::responses("gpt-5.2").transport(transport);
```

请求级 header：

```rust
let options = RequestOptions::new()
    .header("x-sessionid", session_id)?
    .user_agent("my-app/1.2.3 (gateway)")?
    .context(RequestContext::new().session_id(session_id));
```

`x-sessionid`、`x-tenant-id` 这类业务字段不写死进 adapter，由客户端透传。

动态认证或 session header 通过 `HeaderInjector`：

```rust
use std::sync::Arc;
use async_trait::async_trait;
use mutil_ai::{HeaderInjector, RequestContext, Result, TransportConfig};

struct GatewayInjector;

#[async_trait]
impl HeaderInjector for GatewayInjector {
    async fn inject(&self, context: &RequestContext) -> Result<reqwest::header::HeaderMap> {
        let mut headers = reqwest::header::HeaderMap::new();
        // 这里可以读取缓存 token，或从安全 token provider 获取。
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_static("Bearer refreshed-token"),
        );
        if let Some(session_id) = &context.session_id {
            headers.insert(
                reqwest::header::HeaderName::from_static("x-sessionid"),
                reqwest::header::HeaderValue::from_str(session_id)?,
            );
        }
        Ok(headers)
    }
}

let transport = TransportConfig::new()
    .header_injector(Arc::new(GatewayInjector));
```

Header 优先级：

```text
SDK 默认 UA
    ↓
ClientConfig / TransportConfig
    ↓
RequestOptions
    ↓
HeaderInjector
```

`Authorization`、`Content-Type`、`Content-Length`、`Host`、`Accept`
默认是 protected header，普通配置层不能静默覆盖。`HeaderInjector` 属于受信任的
transport 配置，可以设置 `Authorization`，适合 OAuth/token refresh 场景。

幂等请求可以传：

```rust
let options = RequestOptions::new().idempotency_key("request-123");
```

SDK 会发送 `Idempotency-Key: request-123`，同一个逻辑请求重试时沿用同一个 key。

自定义 header 默认要求 ASCII。CJK 等非 ASCII 值需要由调用方自行编码，
例如 base64 或 percent encoding；否则返回 `NonAsciiHeaderValue`。

`HeaderInjector` 会在每次 HTTP attempt 前重新执行。自动重试可以刷新 token 或
临时 header，但 session、idempotency key、静态 body 和静态 headers 仍保持同一
逻辑请求的稳定性。已经产生 stream delta 后仍不会自动重放。

### 审计与数据点（无上下文留痕）

审计默认关闭；启用后只记录结构化传输事件，不记录 prompt、message、
reasoning、tool arguments/result、request body 或 response body：

```rust
use std::sync::Arc;
use mutil_ai::{AuditConfig, AuditEvent, AuditSink, TransportConfig};

struct Metrics;

impl AuditSink for Metrics {
    fn record(&self, event: AuditEvent) {
        // 发送到 metrics / audit pipeline。
        eprintln!("{event:?}");
    }
}

let transport = TransportConfig::new()
    .audit_sink(Arc::new(Metrics))
    .audit_config(AuditConfig::enabled());
```

可审计数据点包括：

- `RequestStarted`
- `AttemptStarted`
- `ResponseHeaders`
- `AttemptFinished`
- `FirstToken`
- `Normalization`
- `RetryScheduled`
- `StreamReconnectScheduled`
- `RequestFinished`（`Success` / `Failure` / `Cancelled`）

每条事件可包含：

- provider、protocol、model；
- attempt 编号；
- status、耗时、outcome；
- response headers 到达耗时与流式 TTFT；
- request body 字节数与累计 decoded response body 字节数；
- `ErrorKind`、是否 retryable；
- usage；
- provider request id；
- normalization 修复类型的聚合数量；
- 显式开启后包含 profile id / model profile 命中状态 / capability 快照。

`ResponseHeaders.elapsed` 和 `RequestFinished.timing.time_to_headers` 从逻辑请求
开始计时。它们包含 DNS、连接、TLS、请求发送和服务端处理到响应头到达的总时间；
reqwest 没有向本 SDK 暴露这些阶段的独立时刻，所以不会伪造更细的 connect/TLS
拆分。`time_to_first_token` 仅在流式响应收到首个 text、reasoning 或 tool-call
delta 时产生。

`Normalization` 只包含每类修复的计数，例如 developer 降级、tool call/result
补对。它不会记录自定义 role 名、tool 名、call id 或消息内容。

`RequestFinished.bytes.request_body` 是一次 attempt 的序列化 body 大小；
`bytes.response_body` 是本次逻辑请求跨重试/reconnect 收到的累计 decoded 字节数，
包含 provider error body，但不包含正文内容。

流在收到 terminal event 前被消费者 drop 时，会发出
`RequestFinished { outcome: Cancelled }`。可用 `record_cancellation(false)`
关闭，或在 metrics 中区分客户端主动取消与 provider 失败。

默认不包含：

- session id；
- SDK request id；
- profile id 和 capability 快照；
- prompt / message / reasoning / tool 内容；
- request/response body。

需要关联 ID 时必须显式打开：

```rust
let config = AuditConfig::enabled()
    .include_provider_request_id(true)
    .include_sdk_request_id(false)
    .include_session_id(false)
    .include_usage(true)
    .include_profile(false)
    .record_timing(true)
    .record_normalization(true);
```

session id 默认关闭，因为它可能包含租户或用户上下文。自定义 profile id 也可能
编码租户信息，因此默认不进入审计。

生产环境推荐使用非阻塞有界队列：

```rust
use mutil_ai::{AuditConfig, TransportConfig, bounded_audit_channel};

let (sink, receiver, stats) = bounded_audit_channel(4096);

let transport = TransportConfig::new()
    .audit_sink(sink)
    .audit_config(AuditConfig::enabled().sample_every(10));

// receiver 应交给后台线程/任务处理。
let snapshot = stats.snapshot();
println!(
    "accepted={}, dropped_full={}, disconnected={}",
    snapshot.accepted, snapshot.dropped_full, snapshot.dropped_disconnected
);
```

队列满时直接丢弃并计数，不会阻塞模型请求。`sample_every(N)` 按整个逻辑请求
采样，避免只采到 attempt、漏掉最终结果。

## 10. 显式端点与扩展字段

同一个厂商可能同时提供 Chat Completions、Responses 和 Anthropic 三种协议。
base URL 不能唯一决定行为，因此通用 adapter 要求显式传入 `EndpointSpec`：

```rust
use mutil_ai::{
    AuthStyle, EndpointAdapter, EndpointSpec, ProtocolSurface, ProviderProfile,
};

let profile = ProviderProfile::qwen_compatible();

let endpoint = EndpointSpec::new(
    ProtocolSurface::OpenAiChat,
    "https://gateway.example/v1",
    "/chat/completions",
    AuthStyle::Bearer,
)
.profile(profile);

let model = EndpointAdapter::new("qwen3", endpoint)
    .api_key_from_env("GATEWAY_API_KEY");
```

`OpenAICompatible` 保留为 `EndpointAdapter` 的兼容别名；新代码优先使用
`EndpointAdapter`，因为它同时 dispatch：

```text
OpenAiChat
OpenAiResponses
AnthropicMessages
GeminiGenerateContent
```

协议仍然必须由调用方显式选择，不能靠 URL 猜测。Gemini 的普通与流式路径也由
`EndpointSpec` 分别描述。

内置 preset 只描述行为，不绑定 URL：

```rust
ProviderProfile::deepseek_compatible();
ProviderProfile::qwen_compatible();
ProviderProfile::kimi_compatible();
ProviderProfile::glm_compatible();
ProviderProfile::doubao_compatible();
```

也可以通过显式 `ProfileId` 查找，仍然不猜 URL：

```rust
use mutil_ai::{EndpointSpec, ProfileId, ProfileSelector};

let endpoint = EndpointSpec::openai_chat("https://gateway.example/v1")
    .profile_selector(ProfileSelector::Builtin(ProfileId::from("qwen")));
```

未知 ID 会返回 `Unsupported`，不会回退到 URL 猜测。

例如 Qwen preset 会把中立配置转换成：

```text
ReasoningMode::Disabled       -> enable_thinking = false
ReasoningConfig::budget_tokens -> thinking_budget
```

DeepSeek preset 会转换 `thinking.type`，并使用它文档要求的 effort 映射：

```text
minimal -> low
medium  -> high
xhigh   -> high
```

模型级差异用 `ModelProfile` 覆盖。SDK 已经提供 Kimi 的两个常见 preset：

```rust
use mutil_ai::{EndpointAdapter, ModelProfile};

let model = EndpointAdapter::new("kimi-k3", endpoint)
    .api_key_from_env("MOONSHOT_API_KEY")
    .model_profile(ModelProfile::kimi_k3());

let model = EndpointAdapter::new("kimi-k2.6", endpoint)
    .api_key_from_env("MOONSHOT_API_KEY")
    .model_profile(ModelProfile::kimi_k2_6());
```

`kimi_k2_6()` 会生成：

```json
{
  "thinking": {
    "type": "enabled",
    "keep": "all"
  }
}
```

capability 在发请求前检查：

```rust
let mut profile = ProviderProfile::qwen_compatible();
profile.capabilities.multimodal = false;
```

此时请求中如果有图片，会返回 `Unsupported`，不会先发一个必然失败的请求。


单次请求的 provider 扩展字段：

```rust
let request = ChatRequest::user("你好")
    .extra_body("enable_thinking", true)
    .extra_body("reasoning_format", "parsed");

let options = RequestOptions::new()
    .header("x-sessionid", "session-1")?
    .extra_query("tenant", "team-a");
```

合并规则是：profile 默认值先写入，请求级 `extra_body` 后写入，所以同名扩展
字段由单次请求覆盖。`model`、`messages`、`input`、`stream` 等 canonical
字段不可覆盖；冲突会在发请求前返回 `ReservedExtraBodyField`，不会静默生成
一个错误请求。

`EndpointAdapter` 根据 `EndpointSpec.protocol` dispatch 到 Chat、Responses、
Anthropic 或 Gemini；普通路径和 Gemini 流式路径均显式配置，不会根据 URL 自动
猜测。

## 11. 代码结构

```text
src/
  types.rs             中立 message / reasoning / provider state
  normalize.rs         内部 role -> 出口 role，工具配对和跨协议清洗
  profile.rs           EndpointSpec / ProviderProfile / ModelProfile
  adapter/
    openai_chat.rs     OpenAI Chat Completions
    endpoint.rs        显式 EndpointSpec 的四协议 dispatch
    openai_responses.rs OpenAI Responses
    anthropic.rs       Anthropic Messages
    gemini.rs          Gemini generateContent
  agent.rs             适合新手的 Agent 循环
  tool.rs              closure 风格工具
  retry.rs             Retry-After / reset header / backoff
  sse.rs               手写增量 SSE parser
  stream.rs            统一 StreamEvent 和流式事件组装
  headers.rs           UA / 请求头 / query / 注入接口
```

关键点：**role 转换、工具修复和跨 provider reasoning 清洗只在 `normalize.rs`
做一次**，adapter 不各写一套补丁。

## 12. 当前版本刻意不做的内容

这是教学版 v0.1，目前优先把 role、工具配对、retry、SSE、reasoning 和统一
streaming 边界讲清楚，因此暂时不做：

- streaming 断线自动重连、`Last-Event-ID` 去重
- WebSocket / Realtime
- OpenAI Responses 的 `previous_response_id` / `store`
- Anthropic `cache_control`
- Gemini Interactions API
- embedding / rerank / speech
- 自动把不同 provider 的 reasoning 互译；当前选择安全丢弃，而不是猜测

这些以后都可以加，但应该加在明确的位置，而不是把 provider 特例塞进 Agent 主循环。
