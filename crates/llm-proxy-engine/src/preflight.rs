//! Context-window preflight (plan 05 step 8, backlog 4, conflict 15) and
//! the input token estimate it and admission share.
//!
//! The estimate counts the text of the system blocks and messages with the
//! engine's [`Counter`]: BPE for model ids tiktoken knows, chars/4 for
//! everything else. Tools, tool results, images and other non-text blocks
//! are not counted, so the estimate is a lower bound that can still
//! overshoot on text that tokenizes densely. That is why the default is
//! warn: an over-window estimate is logged and the request goes upstream,
//! which answers with the authoritative count. `enforce` rejects only when
//! the estimate clears the window by the configured margin, and only when
//! the catalog knows the window at all. A provider token count is not used.

use llm_proxy_core::{Counter, MessageContent, PreflightConfig, PreflightMode};
use llm_proxy_protocol::core::{CoreContent, CoreRequest, CoreRole};

use crate::error::EngineError;

/// The engine's input token estimate for a request (see the module docs
/// for what it does and does not count).
#[must_use]
pub fn estimate_input_tokens(counter: &Counter, core: &CoreRequest) -> u64 {
    let system = text_of(&core.system);
    let messages: Vec<MessageContent> = core
        .messages
        .iter()
        .map(|message| MessageContent::new(role_name(message.role), text_of(&message.content)))
        .collect();
    u64::try_from(counter.count_messages(&core.model.requested, &system, &messages))
        .unwrap_or(u64::MAX)
}

fn role_name(role: CoreRole) -> &'static str {
    match role {
        CoreRole::System => "system",
        CoreRole::User => "user",
        CoreRole::Assistant => "assistant",
        CoreRole::Tool => "tool",
        _ => "user",
    }
}

fn text_of(blocks: &[CoreContent]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let CoreContent::Text { text, .. } = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

/// What preflight knows about one request.
#[derive(Debug, Clone, Copy)]
pub struct PreflightInput<'a> {
    /// The upstream model the request encodes for.
    pub model: &'a str,
    /// The estimated input tokens.
    pub estimated_input_tokens: u64,
    /// The request's `max_tokens`, when set.
    pub max_output_tokens: Option<u64>,
    /// The model's context window from the catalog, when known.
    pub context_length: Option<u32>,
}

/// Runs preflight: `Ok(())` admits (possibly after a warning), `Err` is
/// the enforce-mode rejection. With no known window there is nothing to
/// judge and the request is admitted silently.
pub fn preflight(config: PreflightConfig, input: PreflightInput<'_>) -> Result<(), EngineError> {
    if config.mode == PreflightMode::Off {
        return Ok(());
    }
    let Some(window) = input.context_length else {
        return Ok(());
    };
    let window = u64::from(window);
    let projected = input
        .estimated_input_tokens
        .saturating_add(input.max_output_tokens.unwrap_or(0));
    if projected <= window {
        return Ok(());
    }
    let margin = u64::from(config.margin_tokens);
    let certainly_over = input.estimated_input_tokens >= window.saturating_add(margin);
    if config.mode == PreflightMode::Enforce && certainly_over {
        return Err(EngineError::InvalidRequest(format!(
            "prompt is too long: an estimated {} input tokens against the {window}-token context window of model \"{}\" (rejection margin {margin} tokens)",
            input.estimated_input_tokens, input.model
        )));
    }
    tracing::warn!(
        model = %input.model,
        estimated_input_tokens = input.estimated_input_tokens,
        max_output_tokens = input.max_output_tokens,
        context_length = window,
        mode = ?config.mode,
        "preflight: the request may not fit the model's context window (estimate ignores tools and images)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(
        estimate: u64,
        max_output: Option<u64>,
        window: Option<u32>,
    ) -> PreflightInput<'static> {
        PreflightInput {
            model: "m",
            estimated_input_tokens: estimate,
            max_output_tokens: max_output,
            context_length: window,
        }
    }

    fn config(mode: PreflightMode, margin: u32) -> PreflightConfig {
        PreflightConfig {
            mode,
            margin_tokens: margin,
        }
    }

    #[test]
    fn warn_mode_never_rejects() {
        preflight(
            config(PreflightMode::Warn, 0),
            input(10_000, Some(10), Some(100)),
        )
        .expect("warn admits even a hopeless request");
    }

    #[test]
    fn enforce_rejects_only_past_the_margin() {
        let enforce = config(PreflightMode::Enforce, 100);
        // Over the window but inside the margin: warn, admit.
        preflight(enforce, input(150, None, Some(100))).expect("inside the margin");
        preflight(enforce, input(199, None, Some(100))).expect("one short of the margin");
        // At the margin: reject.
        let refused = preflight(enforce, input(200, None, Some(100))).expect_err("past the margin");
        assert!(matches!(refused, EngineError::InvalidRequest(_)));
        assert_eq!(refused.status_code(), 400);
    }

    #[test]
    fn max_tokens_count_toward_the_warning_but_not_the_rejection() {
        let enforce = config(PreflightMode::Enforce, 0);
        // Input fits; only the requested output overflows: the provider
        // decides (it truncates or errors), so preflight admits.
        preflight(enforce, input(90, Some(50), Some(100)))
            .expect("output overflow is not rejected");
    }

    #[test]
    fn unknown_window_and_off_mode_admit_silently() {
        preflight(
            config(PreflightMode::Enforce, 0),
            input(u64::MAX, None, None),
        )
        .expect("no window, nothing to judge");
        preflight(
            config(PreflightMode::Off, 0),
            input(u64::MAX, None, Some(1)),
        )
        .expect("off never judges");
    }

    #[test]
    fn the_estimate_counts_text_blocks_only() {
        use llm_proxy_protocol::core::{CoreMessage, ModelRef, SamplingOptions};
        let counter = Counter::new();
        let text_only = CoreRequest {
            model: ModelRef {
                requested: "some-model".to_owned(),
                upstream: None,
            },
            messages: vec![CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::Text {
                    text: "a".repeat(400),
                    cache: None,
                }],
            }],
            system: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: Default::default(),
            provider_hints: Default::default(),
        };
        let with_image = CoreRequest {
            messages: vec![CoreMessage {
                role: CoreRole::User,
                content: vec![
                    CoreContent::Text {
                        text: "a".repeat(400),
                        cache: None,
                    },
                    CoreContent::Image {
                        source: serde_json::json!({"type": "base64", "data": "x".repeat(10_000)}),
                    },
                ],
            }],
            ..text_only.clone()
        };
        let base = estimate_input_tokens(&counter, &text_only);
        assert!(base >= 100, "400 chars is about 100 tokens, got {base}");
        assert_eq!(
            estimate_input_tokens(&counter, &with_image),
            base,
            "images are not counted: the estimate is a documented lower bound"
        );
    }
}
