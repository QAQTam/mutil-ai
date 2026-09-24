# 国内模型 API 兼容矩阵（第一轮）

> 状态日期：2026-09-24
> 目的：先确认“协议 surface”，再决定 adapter/profile，而不是按公司写散乱特判。
> 标记：
> - ✅ 官方文档已确认
> - 🟡 官方文档已找到，但字段细节仍需逐项核对
> - ⚪ 尚未找到/尚未确认
> - ⚠️ 兼容接口存在明显语义差异

## 1. 协议面总览

很多厂商不是一个协议，而是同时暴露多个协议面。SDK 应该按协议面适配：

| 厂商 | OpenAI Chat | OpenAI Responses | Anthropic Messages | 原生协议 |
|---|---:|---:|---:|---:|
| DeepSeek | ✅ | ✅ | ✅ | - |
| 阿里 Qwen / 百炼 | ✅ | ✅ | ✅ | DashScope ✅ |
| Moonshot Kimi | ✅ | ✅ | ✅ | - |
| 智谱 GLM | ✅ | ✅ | ✅ | 智谱 HTTP/SDK ✅ |
| 火山方舟 Doubao | ✅ | ✅ | ✅ | Ark Runtime ✅ |
| 腾讯 Hunyuan | ✅ | ⚪ | ✅ | 腾讯云 SDK |
| MiniMax | ✅ | ⚪ | ✅ | MiniMax API |
| 百度 Qianfan / Ernie | ✅ | ⚪ | ⚪ | 千帆原生 API |
| StepFun | ✅ | ⚪ | ✅ | Step API |
| SiliconFlow | ✅ | ⚪ | ⚪ | - |
| ModelScope API-Inference | 🟡 | ⚪ | ⚪ | - |

### 结论

不能把“OpenAI-compatible”理解成完全相同的协议。
同一个 provider 也可能同时需要：

```text
ProviderProfile
  + ProtocolSurface::OpenAiChat
  + ProtocolSurface::OpenAiResponses
  + ProtocolSurface::AnthropicMessages
```

## 2. 各厂商官方文档入口

### DeepSeek

- OpenAI Chat：<https://api-docs.deepseek.com/api/create-chat-completion>
- Responses：<https://api-docs.deepseek.com/guides/responses_api>
- Anthropic Messages：<https://api-docs.deepseek.com/guides/anthropic_api>
- Thinking：<https://api-docs.deepseek.com/guides/thinking_mode>

已确认：

- OpenAI base：`https://api.deepseek.com`
- Anthropic base：`https://api.deepseek.com/anthropic`
- 请求侧出现：
  - `thinking: { "type": "enabled" | "disabled" }`
  - `reasoning_effort: none | low | high | max`
- 兼容映射：
  - `minimal` -> `low`
  - `medium` / `xhigh` -> `high`
- Chat Completions 的 thinking mode 支持工具调用。
- Chat Prefix Completion 使用 Beta base URL，并支持输入历史 `reasoning_content`。
- 需要 probe 确认：当前模型的非流式/流式 reasoning 响应字段是否始终为 `reasoning_content`。

### 阿里 Qwen / 百炼

- OpenAI Chat：<https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions>
- Responses：<https://help.aliyun.com/zh/model-studio/openai-compatible-responses>
- Anthropic Messages：<https://help.aliyun.com/zh/model-studio/anthropic-api-messages>
- DashScope：<https://help.aliyun.com/zh/model-studio/qwen-api-via-dashscope>

已确认：

- 百炼同时提供 OpenAI Chat、OpenAI Responses、Anthropic Messages 和 DashScope 原生接口。
- OpenAI-compatible base 使用业务空间域名：
  - `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`
- Anthropic-compatible base：
  - `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/apps/anthropic`
- Anthropic-compatible 鉴权支持：
  - `x-api-key`
  - `Authorization: Bearer`
- OpenAI-compatible thinking 请求字段：
  - `enable_thinking: bool`
  - `thinking_budget: int`
  - `reasoning_effort: none | minimal | low | medium | high | xhigh | max`
- 开启 thinking 后，思考内容通过 `reasoning_content` 返回。
- 部分模型禁止同时传 `thinking_budget` 和 `reasoning_effort`。
- 历史 thinking 是否参与上下文：
  - `preserve_thinking`
  - `clear_thinking`
- Anthropic-compatible 的 thinking block 会返回 `signature`，但百炼文档注明部分模型当前固定为空字符串。
- Anthropic-compatible 的 `max_tokens` 语义会随模型变化，可能包含或不包含 thinking tokens。

### Moonshot Kimi

- OpenAI Chat：<https://platform.moonshot.cn/docs/api/chat>
- Responses：<https://platform.moonshot.cn/docs/api/responses>
- Messages：<https://platform.moonshot.cn/docs/api/messages>

已确认：

- OpenAI base：`https://api.moonshot.cn/v1`
- OpenAI Chat 使用 Bearer 鉴权。
- 非流式响应：
  - `content`
  - `reasoning_content`
- Chat streaming：
  - SSE
  - `stream_options.include_usage = true`
  - usage 在最后一个 chunk
  - 最终发送 `data: [DONE]`
- `kimi-k3`：
  - 始终推理
  - 顶层 `reasoning_effort`
  - 支持 `low | high | max`
- `kimi-k2.6` / `kimi-k2.7-code`：
  - `thinking.type`
  - `thinking.keep`
- Preserved Thinking：
  - K2.6 默认不保留历史 `reasoning_content`
  - K2.7-code 固定 `keep = all`
  - 保留模式要求历史 assistant reasoning 原样、按序回传
- Messages API：
  - `POST /anthropic/v1/messages`
  - 支持 `thinking`、`text`、`tool_use`
  - streaming 支持 `thinking_delta`、`signature_delta`、`input_json_delta`

### 智谱 GLM

- OpenAI Chat：<https://docs.bigmodel.cn/cn/guide/develop/openai/introduction>
- Claude Messages：<https://docs.bigmodel.cn/cn/guide/develop/claude/introduction>
- Responses：<https://docs.bigmodel.cn/cn/guide/develop/responses/introduction>
- Thinking：<https://docs.bigmodel.cn/cn/guide/capabilities/thinking>
- Tool streaming：<https://docs.bigmodel.cn/cn/guide/capabilities/stream-tool>

已确认：

- OpenAI Chat base：`https://open.bigmodel.cn/api/paas/v4/`
- Anthropic base：`https://open.bigmodel.cn/api/anthropic`
- Responses base：`https://open.bigmodel.cn/api/v1`
- 请求侧：
  - `thinking: { type: enabled | disabled }`
  - `reasoning_effort`
  - `clear_thinking`
  - `tool_stream`
- 响应侧：
  - `reasoning_content`
- GLM-5.3 / 5.3-FLASH 不再允许 `thinking.type = disabled`。
- Responses 流式结束依赖：
  - `response.completed`
  - `response.failed`
  - `response.incomplete`
  - `error`
  - 官方文档明确不发送 `data: [DONE]`
- Responses 支持 `store` 和 `previous_response_id`。

### 火山方舟 Doubao

- Chat API：<https://www.volcengine.com/docs/82379/1494384>
- Responses API：<https://www.volcengine.com/docs/82379/1585135>
- Messages API：<https://www.volcengine.com/docs/82379/2655179>
- Thinking：<https://www.volcengine.com/docs/82379/1449737>
- Streaming：<https://www.volcengine.com/docs/82379/2123275>

已确认：

- Chat endpoint：
  - `POST https://ark.cn-beijing.volces.com/api/v3/chat/completions`
- Anthropic-compatible Messages endpoint：
  - `POST https://ark.cn-beijing.volces.com/api/compatible/v1/messages`
- Responses endpoint：
  - `POST https://ark.cn-beijing.volces.com/api/v3/responses`
- Chat 请求：
  - `thinking.type`
  - `reasoning_effort`
- Chat 响应：
  - `reasoning_content`
  - `encrypted_content`
- 流式：
  - Chat 使用 `[DONE]`
  - reasoning 完成后、正式文本前可能输出完整 `encrypted_content`
  - `encrypted_content` 优先级高于 `reasoning_content`
- Responses：
  - `response.reasoning_summary_text.delta`
  - `response.output_text.delta`
  - `response.function_call_arguments.delta`
  - `response.completed`
- Messages：
  - `thinking` block 带 `signature`
  - 必须完整回传以保持工具调用推理连续性

### 腾讯 Hunyuan

- OpenAI 兼容：<https://cloud.tencent.com/document/product/1729/111007>
- Anthropic 兼容：<https://cloud.tencent.com/document/product/1729/127293>
- 原 ChatCompletions：<https://cloud.tencent.com/document/product/1729/105701>

已确认：

- OpenAI base：`https://api.hunyuan.cloud.tencent.com/v1`
- Anthropic base：`https://api.hunyuan.cloud.tencent.com/anthropic`
- OpenAI-compatible 使用 `Authorization: Bearer`。
- Anthropic-compatible 使用 `x-api-key`。
- Anthropic-compatible 支持：
  - `text`
  - `thinking`
  - `tool_use`
  - `tool_result`
- Anthropic-compatible thinking block 包含：
  - `thinking`
  - `signature`
- 原生 ChatCompletions 使用 `EnableThinking`，且目前明确只对 `hunyuan-a13b` 生效。
- OpenAI-compatible 的 thinking 字段需要 probe 确认，不应直接假设为 `enable_thinking` 或 `thinking.type`。

### MiniMax

- OpenAI Chat：<https://platform.minimaxi.com/docs/api-reference/text-chat-openai>
- Anthropic Messages：<https://platform.minimaxi.com/docs/api-reference/text-chat-anthropic>
- Anthropic SDK：<https://platform.minimaxi.com/docs/api-reference/text-anthropic-api>

已确认：

- OpenAI base：`https://api.minimax.cn/v1`
- Anthropic base：`https://api.minimax.cn/anthropic`
- OpenAI-compatible：
  - `thinking.type`
  - `reasoning_split`
- `reasoning_split` 开启后：
  - `reasoning_content`
  - `reasoning_details`
- MiniMax M3 支持：
  - `thinking: adaptive`
  - 图片
  - 视频
  - 工具调用
- Anthropic-compatible：
  - `thinking`
  - `signature`
  - `tool_use`
  - `tool_result`
- 多轮 Function Call 时必须完整回传 assistant 的全部 content block。

### 百度 Qianfan / Ernie

- 文本生成 API：<https://cloud.baidu.com/doc/qianfan-api/s/3m7of64lb>

已确认：

- OpenAI-compatible endpoint：
  - `POST https://qianfan.baidubce.com/v2/chat/completions`
- 鉴权：
  - `Authorization: Bearer ...`
- 请求侧：
  - `thinking.type`
  - `enable_thinking`
  - `thinking_budget`
  - `thinking_strategy`
  - `reasoning_effort`
- 响应侧：
  - `reasoning_content`
- `finish_reason` 除标准值外可能出现：
  - `content_filter`
- 目前未确认是否提供 Responses 或 Anthropic Messages 兼容面。

### StepFun

- Chat Completions：<https://platform.stepfun.com/docs/zh/api-reference/chat/chat-completion-create>
- Messages：<https://platform.stepfun.com/docs/zh/api-reference/chat/messages-create>
- Reasoning：<https://platform.stepfun.com/docs/zh/guides/developer/reasoning>

已确认：

- OpenAI base：`https://api.stepfun.com/v1`
- Anthropic-compatible Messages：
  - `https://api.stepfun.com/v1/messages`
- Step Plan Messages：
  - `https://api.stepfun.com/step_plan/v1/messages`
- 请求侧：
  - `reasoning_format`
  - `reasoning_effort`
- `reasoning_format = general`：
  - 响应字段为 `reasoning`
- `reasoning_format = deepseek-style`：
  - 响应字段为 `reasoning_content`
- 非流式响应可能同时返回 `reasoning` 和 `reasoning_content`。
- Chat streaming 以 `data: [DONE]` 结束。
- Messages API 使用 Anthropic block/event 形式，包括 `thinking_delta`、`signature_delta`、`input_json_delta`。

### SiliconFlow

- Chat Completions：<https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions>

已确认：

- OpenAI-compatible base：`https://api.siliconflow.cn/v1`
- 请求侧：
  - `enable_thinking`
  - `thinking_budget`
  - `reasoning_effort`
- 响应侧：
  - `reasoning_content`
- 流式通常以 `data: [DONE]` 结束。
- SiliconFlow 是聚合平台，实际 reasoning 行为必须按底层模型再细分。

### ModelScope API-Inference

- 入口：<https://modelscope.cn/docs/model-service/API-Inference/intro>

当前状态：🟡

- 页面内容依赖前端加载，第一轮未提取到完整字段。
- 已知生态里通常提供 OpenAI-compatible endpoint，但需要后续用官方页面和 probe 确认。
- 不应在没有确认前写死 base URL、thinking 字段或 reasoning alias。

## 3. 国内厂商最大公约数

### 最稳定的公共部分

```text
POST /chat/completions 或等价兼容路径
Authorization: Bearer <key>
model
messages[]
stream
temperature
max_tokens / max_completion_tokens
tools[]
tool_choice
response_format
```

消息角色：

```text
system
user
assistant
tool
```

工具调用：

```text
assistant.tool_calls[].id
assistant.tool_calls[].type = function
assistant.tool_calls[].function.name
assistant.tool_calls[].function.arguments

role = tool
tool_call_id
content
```

流式：

```text
choices[0].delta.content
choices[0].delta.tool_calls
choices[0].finish_reason
usage
```

### 不存在统一标准的部分

这些字段不能硬编码成一个实现：

- thinking 开关字段
- reasoning 响应字段
- reasoning replay policy
- signature / encrypted content
- `[DONE]` 是否存在
- usage 是否在每个 chunk 返回
- `max_tokens` 是否包含 thinking tokens
- reasoning effort 的合法值和映射规则
- tool stream 开关
- 历史 reasoning 是否保留
- system role 的限制
- 模型是否允许关闭 thinking

## 4. 值得优先实现的特殊能力

### 4.1 `ReasoningAlias`

需要支持：

```text
reasoning
reasoning_content
reasoning_details
thinking
encrypted_content
```

并且支持按模型覆盖，而不是只按 provider。

### 4.2 `ThinkingRequestProfile`

需要支持：

```text
enable_thinking: bool
thinking: { type: enabled | disabled | adaptive }
thinking_budget: int
reasoning_effort: enum
reasoning_format: general | deepseek-style
reasoning_split: bool
clear_thinking / preserve_thinking / thinking.keep
```

### 4.3 `ReasoningReplayPolicy`

建议：

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

例子：

- Kimi K2.x：可选保留，K2.7-code 强制保留。
- GLM：`clear_thinking` 控制。
- Qwen：`preserve_thinking` / `clear_thinking` 控制。
- Doubao：工具调用场景要求回传 `encrypted_content`。
- MiniMax Anthropic：必须完整回传 content blocks。
- StepFun：输出字段由 `reasoning_format` 决定。

### 4.4 `StreamProfile`

需要描述：

```text
DoneMarker::DataDone
DoneMarker::ResponseCompleted
DoneMarker::MessageStop
DoneMarker::Eof
UsagePlacement::EveryChunk
UsagePlacement::LastChunk
UsagePlacement::Event
```

### 4.5 `ProtocolSurface`

不要只写 `OpenAI`：

```rust
pub enum ProtocolSurface {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
    GeminiGenerateContent,
}
```

同一厂商：

```text
DeepSeek + OpenAiChat
DeepSeek + OpenAiResponses
DeepSeek + AnthropicMessages
```

## 5. 下一步

当前已完成：

1. `extra_body`、request-level extra headers、`extra_query`；
2. `EndpointSpec`、`AuthStyle`、`ProviderProfile`、`ModelProfile` 基础类型；
3. 显式端点的 `EndpointAdapter`，支持 Chat / Responses / Anthropic / Gemini
   dispatch；`OpenAICompatible` 保留为兼容别名；
4. reasoning alias、thinking request、replay policy 驱动 Chat profile；
5. capability gating；
6. DeepSeek / Qwen / Kimi / GLM / Doubao 的文档级 behavior preset；
7. Kimi K3 与 K2.6 的 model preset；
8. HeaderInjector per-retry token refresh，同时保持 session/idempotency 稳定；
9. opt-in SSE reconnect、Last-Event-ID 和 `(id, data)` 去重；
10. `tool_choice` / `response_format` / `stop` / `seed` 中立请求参数及协议映射；
11. 默认要求 `Idempotency-Key` 才自动重放 POST；
12. 显式 `ProfileRegistry`，支持文档级 provider ID 和 alias 查找；
13. context-free 审计事件，记录传输结果、response-header 时延、TTFT、request/response 字节数、cancellation、profile/capability 快照和 normalization 聚合计数，但不记录 prompt/reasoning/tool/body；
14. 非阻塞有界审计队列、丢弃统计和整请求采样；`Error::Api` 默认展示脱敏 provider body，并提供安全结构化错误摘要。

下一步：

1. 为 Qwen effort/budget 冲突、Doubao `encrypted_content` 工具续传建立
   带日期的 fixture；
2. 增加 ignored live probe：
   - 普通文本
   - thinking
   - tool call
   - tool result 第二轮
   - streaming
3. 继续补查：
   - 01.AI
   - Baichuan
   - iFlytek Spark
   - SenseTime
   - 360
   - Tencent TokenHub
   - ModelScope 具体字段

## 6. 第一轮最重要的结论

最值得抽象的不是“公司适配器”，而是：

```text
ProtocolSurface
  + AuthProfile
  + RequestProfile
  + ReasoningProfile
  + ToolProfile
  + StreamProfile
```

这样才能同时覆盖：

- 一个公司支持三种协议；
- 一个模型支持多种 reasoning 字段；
- 同一兼容协议下字段语义仍然不同；
- 聚合平台底层模型行为不同；
- 未来厂商新增字段时不需要改核心消息模型。
