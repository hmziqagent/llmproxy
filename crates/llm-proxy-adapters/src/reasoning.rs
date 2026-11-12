//! Reasoning-parameter translation at the provider edge (BUG-20, backlog 13):
//! converts between the OpenAI `reasoning_effort` levels, the Anthropic
//! `thinking.budget_tokens` object and Gemini's `thinkingBudget`, with
//! per-model clamps. Nothing is forwarded unfiltered: an effort the target
//! dialect cannot name, or a thinking object with an unrecognized shape, is an
//! error naming the problem.

use llm_proxy_protocol::core::SamplingOptions;
use serde_json::Value;

use crate::error::ProviderError;

/// Effort levels accepted on the OpenAI-family wire.
pub const EFFORT_LEVELS: [&str; 4] = ["minimal", "low", "medium", "high"];

/// Documented effort → budget mapping used when translating an effort level
/// into a token budget (Anthropic/Gemini). Budgets are advisory thinking
/// ceilings, not exact spend.
const EFFORT_BUDGETS: [(&str, i64); 4] = [
    ("minimal", 1024),
    ("low", 2048),
    ("medium", 8192),
    ("high", 16384),
];

/// Derives an effort level from a token budget (the inverse of
/// [`EFFORT_BUDGETS`], by threshold).
#[must_use]
pub fn effort_from_budget(budget: i64) -> &'static str {
    if budget >= 12288 {
        "high"
    } else if budget >= 4096 {
        "medium"
    } else {
        "low"
    }
}

/// Validates a client-supplied effort level; unknown values are errors naming
/// the value, never silently forwarded.
pub fn validate_effort(effort: &str) -> Result<(), ProviderError> {
    if EFFORT_LEVELS.contains(&effort) {
        return Ok(());
    }
    Err(ProviderError::InvalidRequest(format!(
        "reasoning_effort {effort:?} is not one of {}",
        EFFORT_LEVELS.join(", ")
    )))
}

/// Per-model thinking budget limits used to clamp translated budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingLimits {
    /// Smallest budget the model accepts.
    pub min: i64,
    /// Largest budget the model accepts.
    pub max: i64,
}

/// Per-model thinking limits. Known families carry their documented ranges;
/// unknown models fall back to a conservative per-protocol default.
#[must_use]
pub fn thinking_limits(model: &str) -> ThinkingLimits {
    let lower = model.to_ascii_lowercase();
    if lower.starts_with("claude") {
        // Claude 3.7+ extended thinking: budgets of 1024..~64k tokens, and the
        // budget must stay below max_tokens (enforced by the Anthropic edge).
        ThinkingLimits {
            min: 1024,
            max: 64_000,
        }
    } else if lower.starts_with("gemini-2.5-pro") {
        ThinkingLimits {
            min: 128,
            max: 32_768,
        }
    } else if lower.starts_with("gemini") {
        ThinkingLimits {
            min: 0,
            max: 24_576,
        }
    } else {
        ThinkingLimits {
            min: 1024,
            max: 64_000,
        }
    }
}

/// Clamps a budget into the model's limits.
#[must_use]
pub fn clamp_budget(budget: i64, limits: ThinkingLimits) -> i64 {
    budget.clamp(limits.min, limits.max)
}

/// The reasoning intent expressed by a request's sampling fields: an effort
/// level, a token budget, both, or neither. Foreign combinations surface as
/// errors rather than being forwarded.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReasoningIntent {
    /// OpenAI-style effort level, if the client sent one.
    pub effort: Option<String>,
    /// Token budget, if the client sent a thinking object with one.
    pub budget_tokens: Option<i64>,
}

/// Extracts the [`ReasoningIntent`] from core sampling fields: validates the
/// effort level and only accepts thinking objects of the documented shape
/// `{"type": "enabled", "budget_tokens": N}` (`"disabled"` clears thinking).
pub fn intent_from_sampling(sampling: &SamplingOptions) -> Result<ReasoningIntent, ProviderError> {
    let mut intent = ReasoningIntent::default();
    if let Some(effort) = &sampling.reasoning_effort {
        validate_effort(effort)?;
        intent.effort = Some(effort.clone());
    }
    if let Some(thinking) = &sampling.thinking {
        // `thinking: null` means absent (some clients serialize None as null).
        if thinking.is_null() {
            return Ok(intent);
        }
        let obj = thinking.as_object().ok_or_else(|| {
            ProviderError::InvalidRequest(
                "thinking must be an object like {\"type\": \"enabled\", \"budget_tokens\": N}"
                    .to_owned(),
            )
        })?;
        match obj.get("type").and_then(Value::as_str) {
            Some("enabled") => {
                let budget = obj
                    .get("budget_tokens")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| {
                        ProviderError::InvalidRequest(
                            "thinking.type \"enabled\" requires an integer budget_tokens"
                                .to_owned(),
                        )
                    })?;
                if budget < 0 {
                    return Err(ProviderError::InvalidRequest(format!(
                        "thinking.budget_tokens must be >= 0, got {budget}"
                    )));
                }
                intent.budget_tokens = Some(budget);
            }
            Some("disabled") => {}
            other => {
                return Err(ProviderError::InvalidRequest(format!(
                    "thinking.type must be \"enabled\" or \"disabled\", got {other:?}"
                )));
            }
        }
    }
    Ok(intent)
}

impl ReasoningIntent {
    /// The effective token budget (an explicit budget wins over the effort
    /// level's documented budget), clamped to the model's limits.
    #[must_use]
    pub fn budget_for(&self, model: &str) -> Option<i64> {
        let limits = thinking_limits(model);
        if let Some(budget) = self.budget_tokens {
            return Some(clamp_budget(budget, limits));
        }
        self.effort
            .as_deref()
            .and_then(|effort| {
                EFFORT_BUDGETS
                    .iter()
                    .find(|(level, _)| *level == effort)
                    .map(|(_, budget)| *budget)
            })
            .map(|budget| clamp_budget(budget, limits))
    }

    /// The effective effort level (an explicit level wins; otherwise derived
    /// from the budget).
    #[must_use]
    pub fn effort_for(&self) -> Option<&str> {
        if let Some(effort) = self.effort.as_deref() {
            return Some(effort);
        }
        self.budget_tokens.map(effort_from_budget)
    }
}

/// Builds the Anthropic `thinking` object for a request: translated and
/// clamped, never forwarded raw. Returns `None` when the request carries no
/// reasoning intent, or — on a `lossy` route — when the requested budget
/// cannot fit under `max_tokens`, dropped with a warning. On a strict route a
/// fit failure is a [`ProviderError::InvalidRequest`] naming the limit, so a
/// client that asked for reasoning never silently receives a non-thinking
/// answer.
pub fn anthropic_thinking(
    sampling: &SamplingOptions,
    upstream_model: &str,
    max_tokens: Option<i32>,
    lossy: bool,
) -> Result<Option<Value>, ProviderError> {
    let intent = intent_from_sampling(sampling)?;
    let Some(budget) = intent.budget_for(upstream_model) else {
        return Ok(None);
    };
    let limits = thinking_limits(upstream_model);
    let mut budget = budget.max(limits.min);
    if let Some(max_tokens) = max_tokens {
        // Anthropic requires budget_tokens < max_tokens.
        let headroom = i64::from(max_tokens) - 1;
        if headroom < limits.min {
            if lossy {
                tracing::warn!(
                    max_tokens,
                    upstream_model,
                    "thinking budget cannot fit under max_tokens; dropping thinking"
                );
                return Ok(None);
            }
            let min = limits.min;
            let min_plus_one = limits.min + 1;
            return Err(ProviderError::InvalidRequest(format!(
                "thinking budget {budget} cannot fit under max_tokens {max_tokens} \
                 (minimum thinking budget for {upstream_model} is {min}); raise max_tokens \
                 above {min_plus_one} or remove thinking/reasoning_effort"
            )));
        }
        if budget > headroom {
            tracing::warn!(budget, headroom, "thinking budget clamped below max_tokens");
            budget = headroom;
        }
    }
    let mut thinking = serde_json::json!({
        "type": "enabled",
        "budget_tokens": budget,
    });
    // Auxiliary subfields of the client's thinking object that the proxy has
    // no opinion about (e.g. Claude Code's `display` mode) survive the
    // rewrite; only the reasoning knobs `type` and `budget_tokens` are
    // filtered and clamped.
    if let Some(raw) = sampling.thinking.as_ref().and_then(Value::as_object)
        && let Some(out) = thinking.as_object_mut()
    {
        for (key, value) in raw {
            if key != "type" && key != "budget_tokens" && !out.contains_key(key) {
                out.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(Some(thinking))
}

/// Resolves the OpenAI-family `reasoning_effort` level for a request:
/// validated when the client sent one, derived from a thinking budget when
/// they did not. Never forwards an Anthropic `thinking` object to a chat API.
pub fn chat_effort(sampling: &SamplingOptions) -> Result<Option<String>, ProviderError> {
    let intent = intent_from_sampling(sampling)?;
    Ok(intent.effort_for().map(str::to_owned))
}

/// Resolves the Gemini `thinkingBudget` for a request, clamped to the model's
/// limits.
pub fn gemini_budget(
    sampling: &SamplingOptions,
    upstream_model: &str,
) -> Result<Option<i64>, ProviderError> {
    let intent = intent_from_sampling(sampling)?;
    Ok(intent.budget_for(upstream_model))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling(effort: Option<&str>, thinking: Option<Value>) -> SamplingOptions {
        SamplingOptions {
            reasoning_effort: effort.map(str::to_owned),
            thinking,
            ..SamplingOptions::default()
        }
    }

    #[test]
    fn unknown_effort_is_an_error_naming_the_value() {
        let err = validate_effort("maximum").unwrap_err();
        assert!(err.to_string().contains("maximum"), "{err}");
    }

    #[test]
    fn effort_maps_to_a_budget_and_back() {
        let intent = intent_from_sampling(&sampling(Some("high"), None)).unwrap();
        assert_eq!(intent.budget_for("claude-sonnet-4-5"), Some(16_384));
        assert_eq!(effort_from_budget(16_384), "high");
        assert_eq!(effort_from_budget(5_000), "medium");
        assert_eq!(effort_from_budget(1_500), "low");
    }

    #[test]
    fn anthropic_thinking_from_budget_is_clamped_below_max_tokens() {
        let sampling = sampling(
            None,
            Some(serde_json::json!({"type": "enabled", "budget_tokens": 50_000})),
        );
        let thinking = anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(8_192), false)
            .unwrap()
            .expect("thinking must be representable");
        assert_eq!(thinking["budget_tokens"], serde_json::json!(8_191));
        assert_eq!(thinking["type"], serde_json::json!("enabled"));
    }

    #[test]
    fn anthropic_thinking_from_effort_honours_the_model_minimum() {
        let sampling = sampling(Some("minimal"), None);
        let thinking = anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(100_000), false)
            .unwrap()
            .expect("thinking must be representable");
        assert_eq!(thinking["budget_tokens"], serde_json::json!(1_024));
    }

    #[test]
    fn anthropic_thinking_dropped_when_max_tokens_leaves_no_room() {
        let sampling = sampling(Some("high"), None);
        let thinking =
            anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(1_024), true).unwrap();
        assert!(thinking.is_none());
    }

    #[test]
    fn thinking_that_cannot_fit_is_an_error_naming_the_limit() {
        let sampling = sampling(Some("high"), None);
        let err = anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(1_000), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("max_tokens"), "{err}");
        assert!(err.contains("1024"), "{err}");
    }

    #[test]
    fn thinking_that_cannot_fit_is_dropped_when_lossy() {
        let sampling = sampling(Some("high"), None);
        let thinking =
            anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(1_000), true).unwrap();
        assert!(thinking.is_none());
    }

    #[test]
    fn thinking_that_fits_is_unchanged() {
        let sampling = sampling(Some("high"), None);
        let thinking = anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(8_000), false)
            .unwrap()
            .expect("thinking must be representable");
        assert!(
            thinking["budget_tokens"]
                .as_i64()
                .is_some_and(|budget| budget < 8_000),
            "{thinking}"
        );
    }

    #[test]
    fn auxiliary_thinking_subfields_survive_the_rewrite() {
        // Claude Code's live wire carries `display` next to the reasoning
        // knobs (captured 2026-11-05); it is not a reasoning parameter, so it
        // must ride along while the knobs are still rewritten.
        {
            let sampling = sampling(
                None,
                Some(serde_json::json!({
                    "type": "enabled", "budget_tokens": 31_999, "display": "omitted"
                })),
            );
            let thinking = anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(32_000), false)
                .unwrap()
                .expect("thinking must be representable");
            assert_eq!(thinking["display"], serde_json::json!("omitted"));
            assert_eq!(thinking["budget_tokens"], serde_json::json!(31_999));
        }
        // The knobs are still clamped when extras ride along.
        {
            let sampling = sampling(
                None,
                Some(serde_json::json!({
                    "type": "enabled", "budget_tokens": 50_000, "display": "omitted"
                })),
            );
            let thinking = anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(8_192), false)
                .unwrap()
                .expect("thinking must be representable");
            assert_eq!(thinking["budget_tokens"], serde_json::json!(8_191));
            assert_eq!(thinking["display"], serde_json::json!("omitted"));
        }
    }

    #[test]
    fn malformed_thinking_is_an_error_not_a_passthrough() {
        let err = intent_from_sampling(&sampling(
            None,
            Some(serde_json::json!({"type": "enabled"})),
        ))
        .unwrap_err();
        assert!(err.to_string().contains("budget_tokens"), "{err}");

        let err = intent_from_sampling(&sampling(
            None,
            Some(serde_json::json!({"type": " turbo "})),
        ))
        .unwrap_err();
        assert!(err.to_string().contains("thinking.type"), "{err}");
    }

    #[test]
    fn disabled_thinking_clears_the_intent() {
        let sampling = sampling(None, Some(serde_json::json!({"type": "disabled"})));
        assert_eq!(
            intent_from_sampling(&sampling).unwrap(),
            ReasoningIntent::default()
        );
    }

    #[test]
    fn chat_effort_derives_from_an_anthropic_budget() {
        let sampling = sampling(
            None,
            Some(serde_json::json!({"type": "enabled", "budget_tokens": 20_000})),
        );
        assert_eq!(chat_effort(&sampling).unwrap().as_deref(), Some("high"));
    }

    #[test]
    fn chat_effort_never_sees_the_thinking_object() {
        let sampling = sampling(Some("low"), None);
        assert_eq!(chat_effort(&sampling).unwrap().as_deref(), Some("low"));
    }

    #[test]
    fn gemini_budget_clamps_per_model() {
        let sampling = sampling(
            None,
            Some(serde_json::json!({"type": "enabled", "budget_tokens": 90_000})),
        );
        assert_eq!(
            gemini_budget(&sampling, "gemini-2.5-pro").unwrap(),
            Some(32_768)
        );
        assert_eq!(
            gemini_budget(&sampling, "gemini-2.5-flash").unwrap(),
            Some(24_576)
        );
    }

    #[test]
    fn no_intent_yields_no_budget() {
        let sampling = SamplingOptions::default();
        assert_eq!(gemini_budget(&sampling, "gemini-2.5-pro").unwrap(), None);
        assert!(
            anthropic_thinking(&sampling, "claude-sonnet-4-5", Some(2_048), false)
                .unwrap()
                .is_none()
        );
    }
}
