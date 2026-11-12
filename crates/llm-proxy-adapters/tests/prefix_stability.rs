//! Prefix-stability tests (plan 06 step 6, audit conflict 5): no enabled
//! feature changes the prompt prefix — the conversation content providers
//! cache on — unless the operator opted into prompt-cache hygiene. The
//! property is pinned for the Anthropic same-format route (reasoning
//! translation and foreign-signature stripping included), for every
//! cross-format encode between the four dialects, and for the Gemini,
//! Responses and Chat same-format passthroughs: encoding a conversation and
//! the same conversation extended by one turn must produce identical bytes
//! for every turn the earlier request already carried.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;

use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_adapters::protocols;
use llm_proxy_adapters::{
    AnthropicAdapter, OpenAiChatAdapter, ProviderAdapter, ProviderAdapterRegistry,
    ProviderAdapterTarget, ProxyRequest,
};
use llm_proxy_core::{AuthStyle, PromptCacheConfig};
use llm_proxy_protocol::core::CoreRequest;
use secrecy::SecretString;
use serde_json::{Value, json};

fn target(prompt_cache: Option<PromptCacheConfig>) -> ProviderAdapterTarget {
    ProviderAdapterTarget {
        provider_name: "anthropic".to_owned(),
        adapter_name: "messages".to_owned(),
        protocol: llm_proxy_adapters::ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
        endpoint: "https://upstream.example/v1/messages".to_owned(),
        auth_style: AuthStyle::XApiKey,
        api_key: SecretString::from("upstream-key"),
        requested_model: "claude-sonnet-4-6".to_owned(),
        upstream_model: "claude-sonnet-4-20250514".to_owned(),
        headers: Arc::new(HashMap::new()),
        lossy: false,
        forward_inbound_headers: Arc::new(Vec::new()),
        prompt_cache: prompt_cache.map(Arc::new),
    }
}

/// A client body with cache markers on the stable prefix: the system block
/// and every message except the newest carry `cache_control`.
fn marked_client_body() -> Value {
    json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "metadata": {"user_id": "turn-7-correlation"},
        "system": [
            {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "First turn.", "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "First reply."}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "Newest turn — changes every request."}
            ]}
        ]
    })
}

fn core_request() -> CoreRequest {
    // The real client decode of the same body, as the server produces it.
    let typed: llm_proxy_protocol::anthropic::MessageRequest =
        serde_json::from_value(marked_client_body()).expect("wire request parses");
    llm_proxy_protocol::client::anthropic::decode_request(typed).expect("core request decodes")
}

fn wire(body: &Value) -> ClientWireRequest {
    ClientWireRequest::new(ClientWireFormat::AnthropicMessages, body.clone())
}

/// The same, for client dialects other than Anthropic Messages.
fn wire_as(format: ClientWireFormat, body: &Value) -> ClientWireRequest {
    ClientWireRequest::new(format, body.clone())
}

fn encode(body: &Value, target: &ProviderAdapterTarget) -> Value {
    let request = AnthropicAdapter
        .encode_request(&core_request(), target, &wire(body))
        .expect("encode");
    serde_json::from_slice(&request.body).expect("encoded body is JSON")
}

/// The prefix the provider's prompt cache keys on: tools, system, and every
/// message up to and including the block carrying the last cache marker.
fn prefix_of(body: &Value) -> Value {
    let mut prefix = json!({});
    if let Some(map) = prefix.as_object_mut() {
        if let Some(tools) = body.get("tools") {
            map.insert("tools".to_owned(), tools.clone());
        }
        if let Some(system) = body.get("system") {
            map.insert("system".to_owned(), system.clone());
        }
        let messages = body.get("messages").and_then(Value::as_array);
        let mut cut = messages.map_or(0, Vec::len);
        if let Some(messages) = messages {
            for (message_index, message) in messages.iter().enumerate() {
                let blocks = message.get("content").and_then(Value::as_array);
                if blocks.is_some_and(|blocks| {
                    blocks
                        .iter()
                        .any(|block| block.get("cache_control").is_some_and(|v| !v.is_null()))
                }) {
                    cut = message_index + 1;
                }
            }
        }
        if let Some(messages) = body.get("messages").and_then(Value::as_array) {
            map.insert(
                "messages".to_owned(),
                Value::Array(messages.iter().take(cut).cloned().collect()),
            );
        }
    }
    prefix
}

#[test]
fn default_features_keep_the_marked_prefix_byte_identical() {
    // Every phase-1/phase-2 feature is on: model aliasing (the target maps
    // to another upstream model). Still, the prompt prefix before the last
    // cache breakpoint must be exactly the client's.
    let client = marked_client_body();
    let encoded = encode(&client, &target(None));

    assert_eq!(
        prefix_of(&client),
        prefix_of(&encoded),
        "the prefix before the last breakpoint changed: {encoded}"
    );
    // The route edit (phase 1) does apply — to the routing field only.
    assert_eq!(encoded["model"], json!("claude-sonnet-4-20250514"));
}

#[test]
fn hygiene_off_by_default_leaves_volatile_bytes_in_place() {
    let client = marked_client_body();
    let encoded = encode(&client, &target(None));
    assert_eq!(
        encoded["metadata"]["user_id"],
        json!("turn-7-correlation"),
        "without the opt-in nothing is stripped"
    );
    assert!(
        encoded["messages"][2]["content"][0]["cache_control"].is_null(),
        "without the opt-in no markers are injected"
    );
}

#[test]
fn opted_in_hygiene_strips_volatile_bytes_only() {
    let client = marked_client_body();
    let config = PromptCacheConfig {
        strip_volatile: vec!["/metadata/user_id".to_owned()],
        inject_markers: false,
    };
    let encoded = encode(&client, &target(Some(config)));

    // The volatile pointer is gone; the prefix is otherwise untouched.
    assert!(
        encoded["metadata"].is_object() && encoded["metadata"].get("user_id").is_none(),
        "the configured volatile pointer must be stripped: {}",
        encoded["metadata"]
    );
    assert_eq!(
        prefix_of(&client),
        prefix_of(&encoded),
        "stripping volatile bytes must not disturb the prefix"
    );
    // Marker injection was not opted into.
    assert!(encoded["messages"][2]["content"][0]["cache_control"].is_null());
}

#[test]
fn opted_in_marker_injection_marks_only_unmarked_stable_blocks() {
    let client = marked_client_body();
    let config = PromptCacheConfig {
        strip_volatile: Vec::new(),
        inject_markers: true,
    };
    let encoded = encode(&client, &target(Some(config)));

    // The client's own markers are kept, verbatim and in place.
    assert_eq!(
        encoded["system"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert_eq!(
        encoded["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    // The newest message stays unmarked.
    assert!(encoded["messages"][2]["content"][0]["cache_control"].is_null());
    // Two client markers spent two of four breakpoints: the remaining
    // budget marks the assistant message's last block (the newest-but-one
    // turn), not the newest.
    assert!(
        encoded["messages"][1]["content"][0]["cache_control"].is_object(),
        "the leftover breakpoint lands on the stable prefix: {encoded}"
    );
}

#[test]
fn chat_adapter_strips_volatile_bytes_but_injects_no_markers() {
    // The Chat Completions wire format has no cache markers; the hygiene
    // opt-in still strips volatile bytes at the provider edge.
    let client = json!({
        "model": "gpt-4o",
        "user": "correlation-token-9",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let core: CoreRequest = llm_proxy_protocol::client::openai_chat::decode_request(
        serde_json::from_value::<llm_proxy_protocol::openai::ChatCompletionRequest>(client.clone())
            .expect("wire request parses"),
    )
    .expect("core decodes");
    let mut chat_target = target(Some(PromptCacheConfig {
        strip_volatile: vec!["/user".to_owned()],
        inject_markers: true,
    }));
    chat_target.protocol =
        llm_proxy_adapters::ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS);
    chat_target.upstream_model = "gpt-4o-2024-11-20".to_owned();

    let request = OpenAiChatAdapter
        .encode_request(
            &core,
            &chat_target,
            &ClientWireRequest::new(ClientWireFormat::OpenAiChatCompletions, client),
        )
        .expect("encode");
    let encoded: Value = serde_json::from_slice(&request.body).expect("JSON");
    assert!(encoded.get("user").is_none(), "volatile field stripped");
    assert_eq!(
        encoded["messages"],
        json!([{"role": "user", "content": "hello"}]),
        "no marker fields are injected on chat"
    );
}

// ---------------------------------------------------------------------------
// Prefix stability across reasoning translation, cross-format encodes and the
// other same-format passthroughs (plan 06 item: "no enabled feature changes
// bytes before the last breakpoint unless opted in", demonstrated for every
// adapter, not only the Anthropic default path).
// ---------------------------------------------------------------------------

/// A target for `protocol` whose upstream model is `upstream_model`, built the
/// way `chat_adapter_strips_volatile_bytes_but_injects_no_markers` builds its
/// Chat target: same credentials shape, per-protocol endpoint.
fn target_for(
    protocol: &str,
    upstream_model: &str,
    prompt_cache: Option<PromptCacheConfig>,
) -> ProviderAdapterTarget {
    let mut adapter_target = target(prompt_cache);
    adapter_target.protocol = llm_proxy_adapters::ProtocolName::new(protocol);
    adapter_target.upstream_model = upstream_model.to_owned();
    adapter_target.endpoint = endpoint_for(protocol).to_owned();
    adapter_target
}

fn endpoint_for(protocol: &str) -> &'static str {
    match protocol {
        protocols::ANTHROPIC_MESSAGES => "https://upstream.example/v1/messages",
        protocols::OPENAI_CHAT_COMPLETIONS => "https://upstream.example/v1/chat/completions",
        protocols::OPENAI_RESPONSES => "https://upstream.example/v1/responses",
        _ => "https://upstream.example/v1beta/models/{model}:generateContent",
    }
}

fn adapter_for(protocol: &str) -> Arc<dyn ProviderAdapter> {
    ProviderAdapterRegistry::builtin()
        .get(protocol)
        .expect("built-in adapter is registered")
}

/// Encode and return the parsed upstream body plus the request the transport
/// would send.
fn encode_full(
    core: &CoreRequest,
    adapter_target: &ProviderAdapterTarget,
    client: &ClientWireRequest,
) -> (Value, ProxyRequest) {
    let request = adapter_for(adapter_target.protocol.as_str())
        .encode_request(core, adapter_target, client)
        .expect("encode");
    let body = serde_json::from_slice(&request.body).expect("encoded body is JSON");
    (body, request)
}

fn encode_via(
    core: &CoreRequest,
    adapter_target: &ProviderAdapterTarget,
    client: &ClientWireRequest,
) -> Value {
    encode_full(core, adapter_target, client).0
}

fn anthropic_core_of(body: &Value) -> CoreRequest {
    let typed: llm_proxy_protocol::anthropic::MessageRequest =
        serde_json::from_value(body.clone()).expect("wire request parses");
    llm_proxy_protocol::client::anthropic::decode_request(typed).expect("core request decodes")
}

fn chat_core_of(body: &Value) -> CoreRequest {
    llm_proxy_protocol::client::openai_chat::decode_request(
        serde_json::from_value::<llm_proxy_protocol::openai::ChatCompletionRequest>(body.clone())
            .expect("wire request parses"),
    )
    .expect("core request decodes")
}

fn responses_core_of(body: &Value) -> CoreRequest {
    llm_proxy_protocol::client::responses::decode_request(body).expect("core request decodes")
}

fn gemini_core_of(body: &Value, url_model: &str) -> CoreRequest {
    llm_proxy_protocol::client::gemini::decode_request(body, url_model, false)
        .expect("core request decodes")
}

/// Serialize the first `k` conversation items: the bytes a provider's prompt
/// cache keys on when the same conversation grows by one turn.
fn prefix_bytes(items: &[Value], k: usize) -> Vec<u8> {
    let head: Vec<Value> = items.iter().take(k).cloned().collect();
    serde_json::to_vec(&head).expect("prefix items serialize")
}

/// Append one newest user text turn to an Anthropic/Chat body.
fn push_user_message(body: &mut Value, text: &str) {
    body.get_mut("messages")
        .and_then(Value::as_array_mut)
        .expect("messages array")
        .push(json!({"role": "user", "content": text}));
}

/// Append one newest user message item to a Responses `input` array.
fn push_responses_user_turn(body: &mut Value, text: &str) {
    body.get_mut("input")
        .and_then(Value::as_array_mut)
        .expect("input array")
        .push(json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}]
        }));
}

/// A Chat Completions conversation with a system prompt and a full tool turn
/// (assistant `tool_calls`, then a `tool` result message).
fn chat_tool_conversation(extra_turns: usize) -> Value {
    let mut body = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is in src/main.rs?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "read_file", "arguments": "{\"path\":\"src/main.rs\"}"
                }}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "fn main() {}"}
        ]
    });
    for turn in 0..extra_turns {
        push_user_message(&mut body, &format!("Follow-up {turn}."));
    }
    body
}

/// An Anthropic conversation with a system prompt, tools, and a full tool
/// turn (assistant `tool_use`, then a `tool_result`).
fn anthropic_tool_conversation(extra_turns: usize) -> Value {
    let mut body = json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "system": [{"type": "text", "text": "You are terse."}],
        "tools": [{
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is in src/main.rs?"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Checking src/main.rs."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                 "input": {"path": "src/main.rs"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "fn main() {}"}
            ]}
        ]
    });
    for turn in 0..extra_turns {
        push_user_message(&mut body, &format!("Follow-up {turn}."));
    }
    body
}

/// A Responses conversation with `instructions`, a function tool, and a full
/// tool turn (`function_call`, then `function_call_output`).
fn responses_tool_conversation(extra_turns: usize) -> Value {
    let mut body = json!({
        "model": "gpt-5.1",
        "instructions": "You are terse.",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }],
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "What is in src/main.rs?"}
            ]},
            {"type": "function_call", "call_id": "call_1", "name": "read_file",
             "arguments": "{\"path\":\"src/main.rs\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "fn main() {}"}
        ]
    });
    for turn in 0..extra_turns {
        push_responses_user_turn(&mut body, &format!("Follow-up {turn}."));
    }
    body
}

/// Encodes `base` and the same conversation extended by one turn through the
/// same adapter/target and returns both encoded bodies.
fn encode_base_and_extended(
    client_format: ClientWireFormat,
    core_of: &dyn Fn(&Value) -> CoreRequest,
    base: &Value,
    extended: &Value,
    adapter_target: &ProviderAdapterTarget,
) -> (Value, Value) {
    (
        encode_via(
            &core_of(base),
            adapter_target,
            &wire_as(client_format, base),
        ),
        encode_via(
            &core_of(extended),
            adapter_target,
            &wire_as(client_format, extended),
        ),
    )
}

/// The cross-format property: the encoded conversation items of the short
/// request are byte-identical to the first items of the extended request, so
/// the provider's cache survives the extra turn.
fn assert_conversation_prefix_stable(short: &Value, long: &Value, key: &str, context: &str) {
    let short_items = short
        .get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{context}: encoded body carries a `{key}` array: {short}"));
    let long_items = long
        .get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{context}: encoded body carries a `{key}` array: {long}"));
    assert_eq!(
        long_items.len(),
        short_items.len() + 1,
        "{context}: extending the conversation adds exactly one item"
    );
    assert_eq!(
        prefix_bytes(short_items, short_items.len()),
        prefix_bytes(long_items, short_items.len()),
        "{context}: the first {} encoded `{key}` items changed when the \
         conversation grew by one turn: short={short} long={long}",
        short_items.len()
    );
}

#[test]
fn reasoning_translation_never_changes_the_prompt_prefix() {
    // The client opts into reasoning and carries its own cache markers on the
    // stable prefix (the shape a replaying Claude Code client sends).
    let thinking_body = json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64000,
        "thinking": {"type": "enabled", "budget_tokens": 200000},
        "system": [
            {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}
        ],
        "tools": [{
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": {"type": "object", "properties": {}}
        }],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Find the entrypoint.", "cache_control": {"type": "ephemeral"}}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is src/main.rs."}]},
            {"role": "user", "content": [{"type": "text", "text": "Newest turn."}]}
        ]
    });
    let mut effort_body = thinking_body.clone();
    let effort_object = effort_body.as_object_mut().expect("body is an object");
    effort_object.remove("thinking");
    effort_object.insert("reasoning_effort".to_owned(), json!("high"));

    // Two target models with different documented thinking limits: the
    // translated `thinking` differs between them, nothing else may.
    let adapter_targets = [
        target_for(
            protocols::ANTHROPIC_MESSAGES,
            "claude-sonnet-4-20250514",
            None,
        ),
        target_for(protocols::ANTHROPIC_MESSAGES, "gemini-2.5-pro", None),
    ];

    let mut budgets = Vec::new();
    for adapter_target in &adapter_targets {
        let encoded = encode_via(
            &anthropic_core_of(&thinking_body),
            adapter_target,
            &wire(&thinking_body),
        );
        assert_eq!(
            prefix_of(&thinking_body),
            prefix_of(&encoded),
            "reasoning translation changed the marked prefix: {encoded}"
        );
        assert_eq!(encoded["max_tokens"], thinking_body["max_tokens"]);
        assert_eq!(encoded["system"], thinking_body["system"]);
        assert_eq!(encoded["tools"], thinking_body["tools"]);
        assert_eq!(encoded["messages"], thinking_body["messages"]);
        let thinking = encoded
            .get("thinking")
            .unwrap_or_else(|| panic!("the thinking object is translated, not dropped: {encoded}"));
        assert_eq!(thinking["type"], json!("enabled"));
        let budget = thinking
            .get("budget_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_else(|| panic!("integer budget_tokens: {thinking}"));
        assert!(
            budget < 200000,
            "the raw client budget must never reach the wire unclamped: {thinking}"
        );
        budgets.push(budget);
        assert!(
            encoded.get("reasoning_effort").is_none(),
            "reasoning_effort is not an Anthropic field: {encoded}"
        );

        let encoded = encode_via(
            &anthropic_core_of(&effort_body),
            adapter_target,
            &wire(&effort_body),
        );
        assert_eq!(
            prefix_of(&effort_body),
            prefix_of(&encoded),
            "effort translation changed the marked prefix: {encoded}"
        );
        assert_eq!(encoded["system"], effort_body["system"]);
        assert_eq!(encoded["tools"], effort_body["tools"]);
        assert_eq!(encoded["messages"], effort_body["messages"]);
        assert!(
            encoded.get("reasoning_effort").is_none(),
            "the foreign field is translated away, not forwarded: {encoded}"
        );
        let thinking = encoded
            .get("thinking")
            .unwrap_or_else(|| panic!("an effort level becomes a thinking budget: {encoded}"));
        assert_eq!(
            thinking["budget_tokens"],
            json!(16384),
            "`high` maps to its documented budget on both models: {thinking}"
        );
    }
    assert_ne!(
        budgets[0], budgets[1],
        "the two target models must clamp differently, or the per-model limits are not exercised"
    );
}

#[test]
fn foreign_signature_stripping_is_deterministic_across_turns() {
    // The one same-format content edit (Edit 3): a foreign `signature` field
    // on client-facing `tool_use` blocks (a Gemini thought signature replayed
    // through an Anthropic client) is dropped on every turn alike, so the
    // turn's encoded bytes do not drift as the conversation grows.
    let signed_conversation = |extra_turns: usize| {
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Find the entrypoint."}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Checking src/main.rs."},
                    {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                     "input": {"path": "src/main.rs"}, "signature": "c2lnLXR1cm4tMQ"}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "fn main() {}"}
                ]}
            ]
        });
        for turn in 0..extra_turns {
            push_user_message(&mut body, &format!("Follow-up {turn}."));
        }
        body
    };

    let turn_2 = signed_conversation(1);
    let turn_3 = signed_conversation(2);
    let adapter_target = target_for(
        protocols::ANTHROPIC_MESSAGES,
        "claude-sonnet-4-20250514",
        None,
    );
    let encoded_turn_2 = encode_via(
        &anthropic_core_of(&turn_2),
        &adapter_target,
        &wire_as(ClientWireFormat::AnthropicMessages, &turn_2),
    );
    let encoded_turn_3 = encode_via(
        &anthropic_core_of(&turn_3),
        &adapter_target,
        &wire_as(ClientWireFormat::AnthropicMessages, &turn_3),
    );

    // The client bodies really carry the signature, so the stripping below is
    // observable rather than vacuous.
    assert!(
        turn_2["messages"][1]["content"][1]
            .get("signature")
            .is_some(),
        "the fixture must send a signed tool_use block: {}",
        turn_2["messages"][1]
    );

    let messages_2 = encoded_turn_2["messages"].as_array().expect("messages");
    let messages_3 = encoded_turn_3["messages"].as_array().expect("messages");
    assert_eq!(messages_2.len() + 1, messages_3.len());
    assert_eq!(
        prefix_bytes(messages_2, messages_2.len()),
        prefix_bytes(messages_3, messages_2.len()),
        "the signed turn encoded differently in the turn-2 and turn-3 bodies"
    );
    let tool_use = &encoded_turn_2["messages"][1]["content"][1];
    assert_eq!(tool_use["type"], json!("tool_use"));
    assert!(
        tool_use.get("signature").is_none(),
        "a foreign signature must be stripped, not forwarded: {tool_use}"
    );
}

#[test]
fn extending_a_conversation_keeps_the_encoded_prefix_cross_format_chat_to_anthropic() {
    let base = chat_tool_conversation(0);
    let extended = chat_tool_conversation(1);
    let adapter_target = target_for(
        protocols::ANTHROPIC_MESSAGES,
        "claude-sonnet-4-20250514",
        None,
    );
    let (short, long) = encode_base_and_extended(
        ClientWireFormat::OpenAiChatCompletions,
        &chat_core_of,
        &base,
        &extended,
        &adapter_target,
    );

    assert_conversation_prefix_stable(&short, &long, "messages", "Chat client -> Anthropic target");
    assert_eq!(
        short["system"], long["system"],
        "the hoisted system prompt must not drift"
    );
    assert_eq!(
        short["tools"], long["tools"],
        "encoded tools must not drift"
    );
    // The tool turn really crossed formats (the comparison is not vacuous).
    assert_eq!(
        short["messages"][1]["content"][0]["type"],
        json!("tool_use")
    );
    assert_eq!(
        short["messages"][2]["content"][0]["type"],
        json!("tool_result")
    );
}

#[test]
fn extending_a_conversation_keeps_the_encoded_prefix_cross_format_anthropic_to_chat() {
    let base = anthropic_tool_conversation(0);
    let extended = anthropic_tool_conversation(1);
    let adapter_target = target_for(
        protocols::OPENAI_CHAT_COMPLETIONS,
        "gpt-4o-2024-11-20",
        None,
    );
    let (short, long) = encode_base_and_extended(
        ClientWireFormat::AnthropicMessages,
        &anthropic_core_of,
        &base,
        &extended,
        &adapter_target,
    );

    assert_conversation_prefix_stable(&short, &long, "messages", "Anthropic client -> Chat target");
    assert_eq!(
        short["tools"], long["tools"],
        "encoded tools must not drift"
    );
    assert_eq!(
        short["messages"][0]["role"],
        json!("system"),
        "the system prompt leads the encoded messages"
    );
    assert_eq!(
        short["messages"][2]["tool_calls"][0]["id"],
        json!("toolu_1")
    );
    assert_eq!(short["messages"][3]["tool_call_id"], json!("toolu_1"));
    assert_eq!(
        short["messages"][3]["content"],
        json!("fn main() {}"),
        "the tool result text must ride the encoded tool message"
    );
}

#[test]
fn extending_a_conversation_keeps_the_encoded_prefix_cross_format_anthropic_to_gemini() {
    let base = anthropic_tool_conversation(0);
    let extended = anthropic_tool_conversation(1);
    let adapter_target = target_for(
        protocols::GEMINI_GENERATE_CONTENT,
        "gemini-2.5-pro-002",
        None,
    );
    let (short, long) = encode_base_and_extended(
        ClientWireFormat::AnthropicMessages,
        &anthropic_core_of,
        &base,
        &extended,
        &adapter_target,
    );

    assert_conversation_prefix_stable(
        &short,
        &long,
        "contents",
        "Anthropic client -> Gemini target",
    );
    assert_eq!(
        short["tools"], long["tools"],
        "encoded tools must not drift"
    );
    // The system prompt is folded into contents, and the tool turn crossed.
    assert_eq!(short["contents"][0]["role"], json!("user"));
    let model_turn = &short["contents"][3];
    assert_eq!(
        model_turn["parts"][1]["functionCall"]["name"],
        json!("read_file")
    );
    assert_eq!(
        short["contents"][4]["parts"][0]["functionResponse"]["name"],
        json!("read_file")
    );
}

#[test]
fn extending_a_conversation_keeps_the_encoded_prefix_cross_format_chat_to_responses() {
    let base = chat_tool_conversation(0);
    let extended = chat_tool_conversation(1);
    let adapter_target = target_for(protocols::OPENAI_RESPONSES, "gpt-5.1-2026-01", None);
    let (short, long) = encode_base_and_extended(
        ClientWireFormat::OpenAiChatCompletions,
        &chat_core_of,
        &base,
        &extended,
        &adapter_target,
    );

    assert_conversation_prefix_stable(&short, &long, "input", "Chat client -> Responses target");
    assert_eq!(
        short["tools"], long["tools"],
        "encoded tools must not drift"
    );
    assert_eq!(
        short["input"][0]["role"],
        json!("developer"),
        "the system prompt leads the encoded input"
    );
    assert_eq!(short["input"][2]["type"], json!("function_call"));
    assert_eq!(short["input"][2]["call_id"], json!("call_1"));
    assert_eq!(short["input"][3]["type"], json!("function_call_output"));
}

#[test]
fn extending_a_conversation_keeps_the_encoded_prefix_cross_format_responses_to_anthropic() {
    let base = responses_tool_conversation(0);
    let extended = responses_tool_conversation(1);
    let adapter_target = target_for(
        protocols::ANTHROPIC_MESSAGES,
        "claude-sonnet-4-20250514",
        None,
    );
    let (short, long) = encode_base_and_extended(
        ClientWireFormat::OpenAiResponses,
        &responses_core_of,
        &base,
        &extended,
        &adapter_target,
    );

    assert_conversation_prefix_stable(
        &short,
        &long,
        "messages",
        "Responses client -> Anthropic target",
    );
    assert_eq!(
        short["system"], long["system"],
        "the instructions must not drift"
    );
    assert_eq!(
        short["tools"], long["tools"],
        "encoded tools must not drift"
    );
    assert_eq!(
        short["messages"][1]["content"][0]["type"],
        json!("tool_use")
    );
    assert_eq!(
        short["messages"][2]["content"][0]["type"],
        json!("tool_result")
    );
}

#[test]
fn gemini_same_format_passthrough_keeps_contents_byte_identical() {
    let body = json!({
        "systemInstruction": {"parts": [{"text": "You are terse."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Weather in SF?"}]},
            {"role": "model", "parts": [
                {"functionCall": {"name": "get_weather", "args": {"city": "SF"}},
                 "thoughtSignature": "c2ln"}
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"name": "get_weather", "response": {"temp": 72}}}
            ]}
        ],
        "generationConfig": {"temperature": 0.3, "thinkingConfig": {"thinkingBudget": 1024}},
        "tools": [{"functionDeclarations": [
            {"name": "get_weather", "description": "weather", "parameters": {"type": "object"}}
        ]}],
        "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT"}]
    });
    let adapter_target = target_for(
        protocols::GEMINI_GENERATE_CONTENT,
        "gemini-2.5-pro-002",
        None,
    );
    let core = gemini_core_of(&body, "gemini-2.5-flash");
    let (encoded, request) = encode_full(
        &core,
        &adapter_target,
        &wire_as(ClientWireFormat::GeminiGenerateContent, &body),
    );

    assert_eq!(
        encoded["contents"], body["contents"],
        "same-format passthrough must forward contents untouched: {encoded}"
    );
    assert_eq!(encoded["systemInstruction"], body["systemInstruction"]);
    assert_eq!(encoded["tools"], body["tools"]);
    assert_eq!(encoded["generationConfig"], body["generationConfig"]);
    assert_eq!(encoded["safetySettings"], body["safetySettings"]);
    // The model alias is applied where Gemini carries it: the URL, not the
    // body.
    assert!(
        request.url.contains("gemini-2.5-pro-002"),
        "the upstream model must be interpolated into the URL: {}",
        request.url
    );
}

#[test]
fn responses_same_format_passthrough_keeps_input_and_instructions_byte_identical() {
    let body = json!({
        "model": "gpt-5.1",
        "instructions": "You are terse.",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "List the files."}
            ]},
            {"type": "function_call", "call_id": "call_1", "name": "shell",
             "arguments": "{\"cmd\":\"ls\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "a.txt\nb.txt"}
        ],
        "tools": [{"type": "function", "name": "shell", "description": "run",
                   "parameters": {"type": "object"}}],
        "max_output_tokens": 512,
        "store": false,
        "include": ["reasoning.encrypted_content"]
    });
    let adapter_target = target_for(protocols::OPENAI_RESPONSES, "gpt-5.1-2026-01", None);
    let core = responses_core_of(&body);
    let encoded = encode_via(
        &core,
        &adapter_target,
        &wire_as(ClientWireFormat::OpenAiResponses, &body),
    );

    assert_eq!(
        encoded["input"], body["input"],
        "same-format passthrough must forward input untouched: {encoded}"
    );
    assert_eq!(encoded["instructions"], body["instructions"]);
    assert_eq!(encoded["tools"], body["tools"]);
    assert_eq!(encoded["max_output_tokens"], body["max_output_tokens"]);
    assert_eq!(encoded["store"], body["store"]);
    assert_eq!(encoded["include"], body["include"]);
    // The one surgical same-format edit: the model alias.
    assert_eq!(encoded["model"], json!("gpt-5.1-2026-01"));
}

#[test]
fn chat_same_format_passthrough_keeps_messages_byte_identical() {
    let body = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "List the files."},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "shell", "arguments": "{\"cmd\":\"ls\"}"
                }}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"}
        ],
        "temperature": 0.2,
        "top_p": 0.9
    });
    let adapter_target = target_for(
        protocols::OPENAI_CHAT_COMPLETIONS,
        "gpt-4o-2024-11-20",
        None,
    );
    let core = chat_core_of(&body);
    let encoded = encode_via(
        &core,
        &adapter_target,
        &wire_as(ClientWireFormat::OpenAiChatCompletions, &body),
    );

    assert_eq!(
        encoded["messages"], body["messages"],
        "same-format passthrough must forward messages untouched: {encoded}"
    );
    assert_eq!(encoded["temperature"], body["temperature"]);
    assert_eq!(encoded["top_p"], body["top_p"]);
    // The one surgical same-format edit: the model alias.
    assert_eq!(encoded["model"], json!("gpt-4o-2024-11-20"));
}
