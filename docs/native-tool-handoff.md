# Codex 工具协议对齐：SDK 工作交接

日期：2026-10-27。约束：后续工作禁用子代理。跨仓设计真源为 `../qaqh-backend/docs/codex-tools对齐/spec.md`，实施选择与同步单为同目录 `contract.md`，任务和验收分别见 `tasks.md`、`checklist.md`。本文件是未完成工作记录，不替代上述契约。

## 交接结论

**0.7.0 已发布 native 声明/回放表面，但真实 provider/OAuth 与完整截断/重连策略仍未验收。** `normalize` 只对 Responses 放行 `tool_declarations`，其他协议继续显式拒绝；`to_responses_body` 会编码 flat custom 与 namespace function/custom。custom 解码、回放和 namespaced function 终态回填已有测试，真实 provider 探针和端点能力仍不在本版本承诺内。

当前发布版本：`0.7.0`（tag `v0.7.0`）。本文描述该版本的边界；后续工作请以 `git log`/tag 和实际 API 为准。

## 代码位置及当前边界

- `src/types.rs`：`ToolDeclaration`、`ScopedToolDeclaration`、`CustomToolFormat`、`NativeToolCall`、`Part::NativeToolCall`，以及 `ChatRequest::tool_declarations`。旧 `ToolSpec::new` 保留。声明校验同作用域重名，但当前不是完整工具选择/能力 API。
- `src/normalize.rs`：native 调用只接受 Responses，完成性和身份通过校验后按真实 `call_id` 登记 pending；native 缺结果或不能配对时报错。`tool_declarations` 只在 Responses 放行，其他协议仍返回 `unsupported_tool_representation`；旧 function 历史规则仍需做混合历史覆盖。
- `src/adapter/openai_responses.rs`：声明已接入 `to_responses_body`（flat custom 与一层 namespace），并保留独立 wire 形状测试。已有 custom 完整响应解析、部分 SSE 累计/终态对账、native 结果类型回放、namespaced function 终态回填；截断、重连和所有异常边界仍需继续补齐。
- `src/native_tool_stream.rs`：单个 custom 输入累计器，delta 和完整快照分离；调用者仍须按 response/index 路由并处理重连重复帧，不能把重复片段简单按内容消重。
- `src/stream.rs`：`NativeToolProgress` 仅进度，`NativeToolCompleted` 仅证明 item 完成；两者都不替代用户取消、权限、response 失败和调度策略。
- `docs/native-tool-progress.md`：现有英文进度说明；与本交接同属未发布记录。

## 验证证据

在当前工作区执行过：

- `cargo test --offline -j 1`：exit 0；默认测试及示例编译通过，明确标记 ignored 的性能/压力测试未运行。最近一次在添加 HTTP 测试和 normalizer 安全校验后执行。
- `cargo test --offline -j 1 --test endpoint_dispatch native_custom_call_replays_result_in_next_http_body`：1 passed。loopback 服务合成 custom 响应，再检查错误结果的第二轮实际 HTTP body；**因为首轮声明被门禁拒绝，仅证明回放半环**。
- 新增声明形状、流快照冲突、原生身份、normalizer 配对等针对性测试，以及 `git diff --check` 均执行通过；全仓格式此前存在与本工作无关的差异，未声称全仓 `cargo fmt --check` 通过。
- tests 为合成数据，不是线上抓包；未运行真实 OAuth、模型能力、DeepSeek namespace/grammar 或 SIWC 端点探针。

## 优先接续步骤

1. **先审查安全与兼容性**：复核全新公开 `Part`/`StreamEvent` 变体与 `ChatRequest` 新字段的源码兼容影响；老消费者穷尽 match 和无 `..Default` 的 struct literal 需要迁移。核对 native/legacy 混合批次的 orphan、缺失结果、重复 ID、错误结果和 raw provider item 双登记/双编码。
2. **完成 T02**：将声明、工具选择 `(kind, namespace, name)`、作用域唯一性和 capability 拒绝贯穿整个请求路径；真实 Responses 请求输出 flat/namespace 声明。确保非 Responses 明确拒绝。复核 provider/profile/transform/extra_body 不绕过限制；只在整个调用及结果闭环验证充分后移除 Responses 的失败关闭门禁。
3. **完成 T03**：依据契约 fixtures 测试并修正 custom added/delta/done/item/terminal 状态机；覆盖交错调用、response/index 作用域、item_id/call_id 冲突、重复帧、终态缺项、正文共存、截断、失败、取消和重连。明确已完成 item 事实与下游是否可调度的区别。审查当前 namespaced function 的流式处理和终态补全，不得回退为错误身份。
4. **完成 T04**：采用真实 call_id 登记和配对，保持原 kind 编码结果；覆盖旧 function 与新 custom 混合历史、名称/namespace 碰撞、missing/orphan 结果和无法编码历史。不得用 item_id 补造 call_id，也不得将 opaque ProviderItem 当成第二个可执行调用。
5. **验收与同步**：把 `../qaqh-backend/docs/codex-tools对齐/fixtures/tool-protocol.json` 的目标接入 SDK 生产编解码与 mock server，验证声明 → 完整/流式调用 → 结果 → 第二轮实际 HTTP body，以及旧 API 回归。不通过 `extra_body.tools/input` 绕过。取得完整 Codex grammar 的来源和 Apache-2.0 归属证据，不能使用示例语法替代。记录测试与未验证边界后，才准备实际版本/revision、迁移说明和跨仓 pin；发布及真实账号探针须遵循相应授权与门禁。

## 特别注意

- 目前 T01 在后端契约中也未全部通过，公开来源、SDK API 与持久化位置仍有待核对；不得把草稿形状当作已经验证的线上支持。
- 默认生产配置不得改变，不更改 OAuth 凭据头、origin，也不在普通正文中提取补丁。
- API 类型和错误分类目前仍是进行中设计；本次使用 `Error::InvalidRequest`/`StreamProtocol` 中的标识字符串，尚非冻结的结构化错误枚举。
- 公开 `NativeToolCompleted` 可能先于 terminal response；调用方不可据此绕过终态冲突、权限或取消检查。后续必须依据契约明确生产调度策略。
