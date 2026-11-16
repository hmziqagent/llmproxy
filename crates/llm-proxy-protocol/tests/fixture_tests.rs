//! Fixture-driven integration tests for the client protocol adapters: decode
//! `input.json` to core and compare with `core.json`, then encode a
//! `CoreResponse` and compare key fields with `output.json`.

#[cfg(test)]
mod tests {

    use llm_proxy_protocol::anthropic::{MessageEvent, MessageRequest as AnthropicMessageRequest};
    use llm_proxy_protocol::client::{
        anthropic as anthropic_adapter, openai_chat as openai_adapter,
    };
    use llm_proxy_protocol::core::{CoreEvent, CoreResponse};
    use llm_proxy_protocol::openai::{ChatCompletionChunk, ChatCompletionRequest};

    use std::fs;
    use std::path::Path;

    const FIXTURE_ROOT: &str = "tests/fixtures";

    fn read_fixture(dir: &Path, name: &str) -> serde_json::Value {
        let path = dir.join(name);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("failed to parse {} as JSON: {e}", path.display()))
    }

    fn read_fixture_raw(dir: &Path, name: &str) -> String {
        let path = dir.join(name);
        fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
    }

    fn assert_decode_matches_core(
        adapter: &str,
        input_json: &serde_json::Value,
        core_json: &serde_json::Value,
    ) {
        let core_request = match adapter {
            "anthropic" => {
                let req: AnthropicMessageRequest = serde_json::from_value(input_json.clone())
                    .unwrap_or_else(|e| panic!("failed to parse Anthropic input: {e}"));
                anthropic_adapter::decode_request(req).unwrap()
            }
            "openai_chat" => {
                let req: ChatCompletionRequest = serde_json::from_value(input_json.clone())
                    .unwrap_or_else(|e| panic!("failed to parse OpenAI input: {e}"));
                openai_adapter::decode_request(req).unwrap()
            }
            _ => panic!("unknown adapter: {adapter}"),
        };
        let serialized =
            serde_json::to_value(&core_request).expect("core request should serialize");
        assert_eq!(
            serialized, *core_json,
            "decoded core does not match core.json for adapter {adapter}"
        );
    }

    fn assert_malformed_returns_error(adapter: &str, input_json: &serde_json::Value) {
        let result = match adapter {
            "anthropic" => {
                let req: AnthropicMessageRequest = match serde_json::from_value(input_json.clone())
                {
                    Ok(r) => r,
                    Err(_) => return,
                };
                anthropic_adapter::decode_request(req)
            }
            "openai_chat" => {
                let req: ChatCompletionRequest = match serde_json::from_value(input_json.clone()) {
                    Ok(r) => r,
                    Err(_) => return,
                };
                openai_adapter::decode_request(req)
            }
            _ => panic!("unknown adapter: {adapter}"),
        };
        assert!(
            result.is_err(),
            "expected error for malformed input in adapter {adapter}, got success"
        );
    }

    fn assert_tool_calls_match(out_m: &serde_json::Value, exp_m: &serde_json::Value) {
        if exp_m.get("tool_calls").is_none() {
            return;
        }
        let out_tcs = out_m.get("tool_calls").and_then(|v| v.as_array());
        let exp_tcs = exp_m.get("tool_calls").and_then(|v| v.as_array());
        assert_eq!(
            out_tcs.map(|a| a.len()),
            exp_tcs.map(|a| a.len()),
            "tool_calls count mismatch"
        );
        let (Some(out_tc_arr), Some(exp_tc_arr)) = (out_tcs, exp_tcs) else {
            return;
        };
        for (i, (otc, etc)) in out_tc_arr.iter().zip(exp_tc_arr.iter()).enumerate() {
            if etc.get("id").is_some() {
                assert_eq!(otc.get("id"), etc.get("id"), "tool_calls[{i}].id mismatch");
            }
            assert_eq!(
                otc.get("function").and_then(|f| f.get("name")),
                etc.get("function").and_then(|f| f.get("name")),
                "tool_calls[{i}].function.name mismatch"
            );
        }
    }

    fn assert_encode_matches_output(
        adapter: &str,
        response: CoreResponse,
        output_json: &serde_json::Value,
    ) {
        match adapter {
            "anthropic" => {
                let out = anthropic_adapter::encode_response(response)
                    .expect("anthropic encode_response should succeed");
                let serialized = serde_json::to_value(&out)
                    .expect("serialized Anthropic response should be valid JSON");
                assert_eq!(
                    serialized.get("type").and_then(|v| v.as_str()),
                    output_json.get("type").and_then(|v| v.as_str()),
                    "type mismatch"
                );
                assert_eq!(
                    serialized.get("stop_reason"),
                    output_json.get("stop_reason"),
                    "stop_reason mismatch"
                );
                assert_eq!(
                    serialized.get("content"),
                    output_json.get("content"),
                    "content mismatch"
                );
            }
            "openai_chat" => {
                let out = openai_adapter::encode_response(response)
                    .expect("openai encode_response should succeed");
                let serialized = serde_json::to_value(&out)
                    .expect("serialized OpenAI response should be valid JSON");
                assert_eq!(
                    serialized.get("object").and_then(|v| v.as_str()),
                    output_json.get("object").and_then(|v| v.as_str()),
                    "object mismatch"
                );
                let out_choices = serialized
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|a| a.first());
                let exp_choices = output_json
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|a| a.first());
                assert_eq!(
                    out_choices.and_then(|c| c.get("finish_reason")),
                    exp_choices.and_then(|c| c.get("finish_reason")),
                    "finish_reason mismatch"
                );
                let out_msg = out_choices.and_then(|c| c.get("message"));
                let exp_msg = exp_choices.and_then(|c| c.get("message"));
                if let (Some(out_m), Some(exp_m)) = (out_msg, exp_msg) {
                    assert_eq!(
                        out_m.get("content"),
                        exp_m.get("content"),
                        "message content mismatch"
                    );
                    assert_tool_calls_match(out_m, exp_m);
                }
                if output_json.get("model").is_some() {
                    assert_eq!(
                        serialized.get("model"),
                        output_json.get("model"),
                        "model mismatch"
                    );
                }
                if let Some(exp_usage) = output_json.get("usage") {
                    let out_usage = serialized.get("usage");
                    assert!(
                        out_usage.is_some(),
                        "expected usage in encoded output but got none"
                    );
                    if let Some(ou) = out_usage {
                        assert_eq!(
                            ou.get("prompt_tokens"),
                            exp_usage.get("prompt_tokens"),
                            "usage.prompt_tokens mismatch"
                        );
                        assert_eq!(
                            ou.get("completion_tokens"),
                            exp_usage.get("completion_tokens"),
                            "usage.completion_tokens mismatch"
                        );
                    }
                }
            }
            _ => panic!("unknown adapter: {adapter}"),
        };
    }

    fn run_non_stream_fixture(adapter: &str, case: &str) {
        let dir = Path::new(FIXTURE_ROOT).join(adapter).join(case);
        let input = read_fixture(&dir, "input.json");

        if case == "malformed" {
            assert_malformed_returns_error(adapter, &input);
            return;
        }

        let core = read_fixture(&dir, "core.json");

        assert_decode_matches_core(adapter, &input, &core);

        if dir.join("output.json").exists() {
            let output = read_fixture(&dir, "output.json");
            let core_response_raw = read_fixture_raw(&dir, "core-response.json");
            let response: CoreResponse = serde_json::from_str(&core_response_raw)
                .unwrap_or_else(|e| panic!("failed to parse core-response.json: {e}"));
            assert_encode_matches_output(adapter, response, &output);
        }
    }

    #[test]
    fn anthropic_plain_text_request_fixture() {
        run_non_stream_fixture("anthropic", "plain-text-request");
    }

    #[test]
    fn anthropic_system_prompt_fixture() {
        run_non_stream_fixture("anthropic", "system-prompt");
    }

    #[test]
    fn anthropic_tool_call_fixture() {
        run_non_stream_fixture("anthropic", "tool-call");
    }

    #[test]
    fn anthropic_tool_result_fixture() {
        run_non_stream_fixture("anthropic", "tool-result");
    }

    #[test]
    fn anthropic_thinking_fixture() {
        run_non_stream_fixture("anthropic", "thinking");
    }

    #[test]
    fn anthropic_cache_control_fixture() {
        run_non_stream_fixture("anthropic", "cache-control");
    }

    #[test]
    fn anthropic_tool_choice_fixture() {
        run_non_stream_fixture("anthropic", "tool-choice");
    }

    #[test]
    fn anthropic_stop_reason_fixture() {
        run_non_stream_fixture("anthropic", "stop-reason");
    }

    #[test]
    fn anthropic_usage_fixture() {
        run_non_stream_fixture("anthropic", "usage");
    }

    #[test]
    fn anthropic_malformed_fixture() {
        run_non_stream_fixture("anthropic", "malformed");
    }

    #[test]
    fn openai_plain_text_request_fixture() {
        run_non_stream_fixture("openai_chat", "plain-text-request");
    }

    #[test]
    fn openai_system_prompt_fixture() {
        run_non_stream_fixture("openai_chat", "system-prompt");
    }

    #[test]
    fn openai_tool_call_fixture() {
        run_non_stream_fixture("openai_chat", "tool-call");
    }

    #[test]
    fn openai_tool_result_fixture() {
        run_non_stream_fixture("openai_chat", "tool-result");
    }

    #[test]
    fn openai_cache_control_fixture() {
        run_non_stream_fixture("openai_chat", "cache-control");
    }

    #[test]
    fn openai_thinking_fixture() {
        run_non_stream_fixture("openai_chat", "thinking");
    }

    #[test]
    fn openai_tool_choice_fixture() {
        run_non_stream_fixture("openai_chat", "tool-choice");
    }

    #[test]
    fn openai_stop_reason_fixture() {
        run_non_stream_fixture("openai_chat", "stop-reason");
    }

    #[test]
    fn openai_usage_fixture() {
        run_non_stream_fixture("openai_chat", "usage");
    }

    #[test]
    fn openai_malformed_fixture() {
        run_non_stream_fixture("openai_chat", "malformed");
    }

    #[test]
    fn all_anthropic_non_stream_fixtures_have_required_files() {
        let cases = [
            "plain-text-request",
            "system-prompt",
            "tool-call",
            "tool-result",
            "thinking",
            "cache-control",
            "tool-choice",
            "stop-reason",
            "usage",
            "malformed",
        ];
        for case in &cases {
            let dir = Path::new(FIXTURE_ROOT).join("anthropic").join(case);
            assert!(
                dir.join("input.json").exists(),
                "anthropic/{case}/input.json is missing"
            );
            if *case != "malformed" {
                assert!(
                    dir.join("core.json").exists(),
                    "anthropic/{case}/core.json is missing"
                );
                assert!(
                    dir.join("output.json").exists(),
                    "anthropic/{case}/output.json is missing"
                );
                assert!(
                    dir.join("core-response.json").exists(),
                    "anthropic/{case}/core-response.json is missing"
                );
            }
        }
    }

    #[test]
    fn all_openai_non_stream_fixtures_have_required_files() {
        let cases = [
            "plain-text-request",
            "system-prompt",
            "tool-call",
            "tool-result",
            "cache-control",
            "thinking",
            "tool-choice",
            "stop-reason",
            "usage",
            "malformed",
        ];
        for case in &cases {
            let dir = Path::new(FIXTURE_ROOT).join("openai_chat").join(case);
            assert!(
                dir.join("input.json").exists(),
                "openai_chat/{case}/input.json is missing"
            );
            if *case != "malformed" {
                assert!(
                    dir.join("core.json").exists(),
                    "openai_chat/{case}/core.json is missing"
                );
                assert!(
                    dir.join("output.json").exists(),
                    "openai_chat/{case}/output.json is missing"
                );
                assert!(
                    dir.join("core-response.json").exists(),
                    "openai_chat/{case}/core-response.json is missing"
                );
            }
        }
    }

    fn all_streaming_cases() -> Vec<(&'static str, &'static str)> {
        vec![
            ("anthropic", "streaming-text"),
            ("anthropic", "streaming-tool"),
            ("anthropic", "streaming-usage"),
            ("anthropic", "streaming-error"),
            ("anthropic", "streaming-ping"),
            ("anthropic", "streaming-thinking"),
            ("openai_chat", "streaming-text"),
            ("openai_chat", "streaming-tool"),
            ("openai_chat", "streaming-usage"),
            ("openai_chat", "streaming-error"),
            ("openai_chat", "streaming-ping"),
            ("openai_chat", "streaming-thinking"),
        ]
    }

    #[test]
    fn all_streaming_fixtures_have_required_files() {
        for (adapter, case) in &all_streaming_cases() {
            let dir = Path::new(FIXTURE_ROOT).join(adapter).join(case);
            assert!(
                dir.join("input.sse").exists(),
                "{adapter}/{case}/input.sse is missing"
            );
            assert!(
                dir.join("core-events.json").exists(),
                "{adapter}/{case}/core-events.json is missing"
            );
            assert!(
                dir.join("output.sse").exists(),
                "{adapter}/{case}/output.sse is missing"
            );
        }
    }

    fn required_client_non_stream_cases() -> Vec<(&'static str, Vec<&'static str>)> {
        vec![
            (
                "plain-text-request",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            (
                "system-prompt",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            (
                "tool-call",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            (
                "tool-result",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            (
                "thinking",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            (
                "cache-control",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            (
                "tool-choice",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            (
                "stop-reason",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            (
                "usage",
                vec![
                    "input.json",
                    "core.json",
                    "output.json",
                    "core-response.json",
                ],
            ),
            ("malformed", vec!["input.json"]),
        ]
    }

    fn required_client_streaming_cases() -> Vec<(&'static str, Vec<&'static str>)> {
        vec![
            (
                "streaming-text",
                vec!["input.sse", "core-events.json", "output.sse"],
            ),
            (
                "streaming-tool",
                vec!["input.sse", "core-events.json", "output.sse"],
            ),
            (
                "streaming-usage",
                vec!["input.sse", "core-events.json", "output.sse"],
            ),
            (
                "streaming-error",
                vec!["input.sse", "core-events.json", "output.sse"],
            ),
            (
                "streaming-ping",
                vec!["input.sse", "core-events.json", "output.sse"],
            ),
            (
                "streaming-thinking",
                vec!["input.sse", "core-events.json", "output.sse"],
            ),
        ]
    }

    fn collect_missing_fixtures(
        adapter: &str,
        cases: &[(&str, Vec<&str>)],
        missing: &mut Vec<String>,
    ) {
        for (case, files) in cases {
            let dir = Path::new(FIXTURE_ROOT).join(adapter).join(case);
            missing.extend(
                files
                    .iter()
                    .filter(|file| !dir.join(file).exists())
                    .map(|file| format!("{adapter}/{case}/{file}")),
            );
        }
    }

    #[test]
    fn coverage_matrix_all_required_client_fixtures_exist() {
        let adapters = ["anthropic", "openai_chat"];
        let mut missing: Vec<String> = Vec::new();

        for adapter in &adapters {
            collect_missing_fixtures(adapter, &required_client_non_stream_cases(), &mut missing);
            collect_missing_fixtures(adapter, &required_client_streaming_cases(), &mut missing);
        }

        assert!(
            missing.is_empty(),
            "missing required client fixture files:\n  {}",
            missing.join("\n  ")
        );
    }

    #[ignore = "client StreamDecoder (input.sse -> CoreEvents) not yet implemented"]
    #[test]
    fn streaming_decode_round_trip() {
        panic!(
            "client streaming decode path is unimplemented: parse input.sse through \
         a client StreamDecoder and compare CoreEvents against core-events.json \
         (mirror the adapters crate's stream-decode fixture runner), plus a \
         malformed/truncation case"
        );
    }

    #[test]
    fn streaming_core_events_json_is_valid() {
        for (adapter, case) in &all_streaming_cases() {
            let dir = Path::new(FIXTURE_ROOT).join(adapter).join(case);
            let raw = read_fixture_raw(&dir, "core-events.json");
            let events: Vec<CoreEvent> = serde_json::from_str(&raw).unwrap_or_else(|e| {
                panic!("failed to parse {adapter}/{case}/core-events.json: {e}")
            });
            assert!(
                !events.is_empty(),
                "{adapter}/{case}/core-events.json should contain at least one event"
            );
        }
    }

    #[test]
    fn streaming_sse_fixtures_are_well_formed() {
        for (adapter, case) in &all_streaming_cases() {
            let dir = Path::new(FIXTURE_ROOT).join(adapter).join(case);

            let input = read_fixture_raw(&dir, "input.sse");
            assert!(
                !input.is_empty(),
                "{adapter}/{case}/input.sse should not be empty"
            );
            let input_has_sse = input
                .lines()
                .any(|line| line.starts_with("data:") || line.starts_with("event:"));
            assert!(
                input_has_sse,
                "{adapter}/{case}/input.sse should contain SSE-formatted lines"
            );

            let output = read_fixture_raw(&dir, "output.sse");
            let output_trimmed = output.trim();
            if !output_trimmed.is_empty() {
                let output_has_sse = output_trimmed
                    .lines()
                    .any(|line| line.starts_with("data:") || line.starts_with("event:"));
                assert!(
                    output_has_sse,
                    "{adapter}/{case}/output.sse should contain SSE-formatted lines (or be empty/whitespace-only)"
                );
            }
        }
    }

    #[test]
    fn streaming_encode_round_trip() {
        for (adapter, case) in &all_streaming_cases() {
            let dir = Path::new(FIXTURE_ROOT).join(adapter).join(case);

            let core_events_raw = read_fixture_raw(&dir, "core-events.json");
            let events: Vec<CoreEvent> =
                serde_json::from_str(&core_events_raw).unwrap_or_else(|e| {
                    panic!("failed to parse {adapter}/{case}/core-events.json: {e}")
                });

            let default_id = if *adapter == "anthropic" {
                "msg_default".to_owned()
            } else {
                "chatcmpl-default".to_owned()
            };
            let msg_id = events
                .iter()
                .find_map(|e| match e {
                    CoreEvent::MessageStart { id, .. } => id.clone(),
                    _ => None,
                })
                .unwrap_or(default_id);
            let model = events
                .iter()
                .find_map(|e| match e {
                    CoreEvent::MessageStart { model, .. } => Some(model.requested.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "unknown".to_owned());

            match *adapter {
                "anthropic" => encode_anthropic_stream(adapter, case, &events, msg_id, model),
                "openai_chat" => encode_openai_stream(adapter, case, &events, msg_id, model),
                _ => panic!("unknown adapter: {adapter}"),
            }
        }
    }

    fn encode_anthropic_stream(
        adapter: &str,
        case: &str,
        events: &[CoreEvent],
        msg_id: String,
        model: String,
    ) {
        let mut encoder = anthropic_adapter::StreamEncoder::new(msg_id, model);
        let mut total_data_lines = 0;

        for event in events {
            let message_events = encoder
                .encode_event(event.clone())
                .unwrap_or_else(|e| panic!("encode_event failed for {adapter}/{case}: {e}"));
            for me in &message_events {
                verify_anthropic_event_json(adapter, case, me);
                total_data_lines += 1;
            }
        }

        let final_events = encoder
            .finish()
            .unwrap_or_else(|e| panic!("finish failed for {adapter}/{case}: {e}"));
        for me in &final_events {
            verify_anthropic_event_json(adapter, case, me);
            total_data_lines += 1;
        }

        assert!(
            total_data_lines > 0,
            "{adapter}/{case}: encoder should produce at least one SSE event"
        );
    }

    fn encode_openai_stream(
        adapter: &str,
        case: &str,
        events: &[CoreEvent],
        msg_id: String,
        model: String,
    ) {
        let mut encoder = openai_adapter::StreamEncoder::new(msg_id, model, 1000, false);
        let mut total_data_lines = 0;

        for event in events {
            let Ok(chunks) = encoder.encode_event(event.clone()) else {
                assert!(
                    matches!(event, CoreEvent::Error { .. }),
                    "{adapter}/{case}: encode_event failed for non-error event"
                );
                continue;
            };
            for chunk in &chunks {
                verify_openai_chunk_json(adapter, case, chunk);
                total_data_lines += 1;
            }
        }

        let final_chunks = encoder
            .finish()
            .unwrap_or_else(|e| panic!("finish failed for {adapter}/{case}: {e}"));
        for chunk in &final_chunks {
            verify_openai_chunk_json(adapter, case, chunk);
            total_data_lines += 1;
        }

        let is_error_only = events
            .iter()
            .all(|e| matches!(e, CoreEvent::Error { .. } | CoreEvent::Ping));
        if !is_error_only {
            assert!(
                total_data_lines > 0,
                "{adapter}/{case}: encoder should produce at least one SSE chunk"
            );
        }
    }

    fn verify_anthropic_event_json(adapter: &str, case: &str, me: &MessageEvent) {
        let json = serde_json::to_string(me).unwrap_or_else(|e| {
            panic!("failed to serialize MessageEvent for {adapter}/{case}: {e}")
        });
        let known_types = [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
            "ping",
            "error",
        ];
        assert!(
            known_types.contains(&me.r#type.as_str()),
            "{adapter}/{case}: unknown event type '{}'",
            me.r#type
        );
        assert!(
            json.starts_with('{') && json.ends_with('}'),
            "{adapter}/{case}: MessageEvent JSON should be an object, got: {json}"
        );
    }

    fn verify_openai_chunk_json(adapter: &str, case: &str, chunk: &ChatCompletionChunk) {
        let json = serde_json::to_string(chunk).unwrap_or_else(|e| {
            panic!("failed to serialize ChatCompletionChunk for {adapter}/{case}: {e}")
        });
        assert!(
            json.starts_with('{') && json.ends_with('}'),
            "{adapter}/{case}: ChatCompletionChunk JSON should be an object, got: {json}"
        );
    }
}
