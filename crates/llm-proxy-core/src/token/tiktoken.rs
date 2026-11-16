//! Real BPE token-counting backends via `tiktoken-rs`.
//!
//! The BPE rank tables for the supported OpenAI encodings are bundled at
//! compile time, so counting requires no network access and works offline.
//! [`Counter`](super::Counter) remains the public front door: it dispatches
//! to a [`TiktokenTokenizer`] for known OpenAI model ids and falls back to a
//! [`HeuristicTokenizer`](super::tokenizer::HeuristicTokenizer)
//! (~4 chars/token) for everything else, so the
//! `/v1/messages/count_tokens` route never hard-fails on an unfamiliar model.

use std::sync::Arc;

use tiktoken_rs::CoreBPE;

use super::tokenizer::{Tokenizer, heuristic_count};

/// BPE backend using a specific tiktoken encoding.
///
/// Wrapped in [`Arc`] so cloning the tokenizer is a refcount bump.
#[derive(Clone)]
pub struct TiktokenTokenizer {
    bpe: Arc<CoreBPE>,
}

impl std::fmt::Debug for TiktokenTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiktokenTokenizer").finish_non_exhaustive()
    }
}

impl TiktokenTokenizer {
    pub(crate) fn new(bpe: Arc<CoreBPE>) -> Self {
        Self { bpe }
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        // tiktoken-rs 0.6 internally unwraps fancy-regex match results, so a
        // pathological input can panic inside the BPE encode. Catch the panic
        // and fall back to the heuristic so the route never hard-fails.
        let bpe = std::panic::AssertUnwindSafe(&self.bpe);
        match std::panic::catch_unwind(|| bpe.encode_with_special_tokens(text)) {
            Ok(encoded) => encoded.len(),
            Err(panic_payload) => {
                let msg = panic_payload
                    .downcast_ref::<&'static str>()
                    .copied()
                    .or_else(|| panic_payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("<non-string panic>");
                tracing::warn!(
                    error = %msg,
                    "tiktoken BPE encode panicked (likely catastrophic backtracking); \
                     falling back to heuristic token counting"
                );
                heuristic_count(text)
            }
        }
    }
}

/// A tiktoken BPE encoding family.
///
/// These are the four OpenAI encodings whose rank tables are bundled into the
/// binary at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// `o200k_base` -- `gpt-4o`, `o1-*`, `o3-*`.
    O200kBase,
    /// `cl100k_base` -- `gpt-4`, `gpt-3.5-turbo`, and their derivatives.
    Cl100kBase,
    /// `p50k_base` -- legacy text/code models (`text-davinci-003`, `code-*`).
    P50kBase,
    /// `r50k_base` -- base (`gpt-3`-era) models.
    R50kBase,
}

impl Encoding {
    pub(crate) fn load(self) -> Option<Arc<CoreBPE>> {
        let result = match self {
            Encoding::O200kBase => tiktoken_rs::o200k_base(),
            Encoding::Cl100kBase => tiktoken_rs::cl100k_base(),
            Encoding::P50kBase => tiktoken_rs::p50k_base(),
            Encoding::R50kBase => tiktoken_rs::r50k_base(),
        };
        match result {
            Ok(bpe) => Some(Arc::new(bpe)),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    encoding = ?self,
                    "failed to load tiktoken BPE ranks; falling back to heuristic token counting"
                );
                None
            }
        }
    }
}

/// Select the tiktoken encoding for a model id.
///
/// Mirrors OpenAI's `encoding_for_model` mapping with longest-prefix matching.
/// Prefix order matters: `gpt-4o` is checked before `gpt-4` (both start with
/// `gpt-4`) so the `o200k_base` family wins for `gpt-4o*`. Returns `None` for
/// unknown models so the caller can use the heuristic.
///
/// Non-OpenAI providers (Anthropic, Gemini, Fireworks, ...) do not publish
/// tiktoken-compatible tokenizers; they fall through to `None` here. For those,
/// provider-reported `Usage` remains the
/// authoritative count for billing; the local counter is only an estimate for
/// the `/v1/messages/count_tokens` route and pre-flight guardrails.
pub fn encoding_for_model(model: &str) -> Option<Encoding> {
    if model.starts_with("gpt-4o")
        || model.starts_with("gpt-4.1")
        || model.starts_with("gpt-4.5")
        || model.starts_with("chatgpt-4o")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
    {
        return Some(Encoding::O200kBase);
    }
    if model.starts_with("gpt-4") || model.starts_with("gpt-3.5") || model.starts_with("gpt-35") {
        return Some(Encoding::Cl100kBase);
    }
    if model.starts_with("text-davinci-003")
        || model.starts_with("text-davinci-002")
        || model.starts_with("code-davinci")
        || model.starts_with("code-cushman")
    {
        return Some(Encoding::P50kBase);
    }
    if model.starts_with("text-davinci-001")
        || model.starts_with("text-curie")
        || model.starts_with("text-babbage")
        || model.starts_with("text-ada")
        || model == "davinci"
        || model == "curie"
        || model == "babbage"
        || model == "ada"
    {
        return Some(Encoding::R50kBase);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_for_model_maps_known_families() {
        assert_eq!(encoding_for_model("gpt-4o"), Some(Encoding::O200kBase));
        assert_eq!(
            encoding_for_model("gpt-4o-2024-08-06"),
            Some(Encoding::O200kBase)
        );
        assert_eq!(encoding_for_model("o1-preview"), Some(Encoding::O200kBase));
        assert_eq!(encoding_for_model("o1-mini"), Some(Encoding::O200kBase));
        assert_eq!(encoding_for_model("o3-mini"), Some(Encoding::O200kBase));
        assert_eq!(encoding_for_model("gpt-4.1"), Some(Encoding::O200kBase));
        assert_eq!(
            encoding_for_model("gpt-4.1-mini"),
            Some(Encoding::O200kBase)
        );
        assert_eq!(encoding_for_model("gpt-4.5"), Some(Encoding::O200kBase));
        assert_eq!(
            encoding_for_model("gpt-4.5-preview"),
            Some(Encoding::O200kBase)
        );
        assert_eq!(
            encoding_for_model("chatgpt-4o-latest"),
            Some(Encoding::O200kBase)
        );
        assert_eq!(encoding_for_model("gpt-4"), Some(Encoding::Cl100kBase));
        assert_eq!(
            encoding_for_model("gpt-4-turbo"),
            Some(Encoding::Cl100kBase)
        );
        assert_eq!(
            encoding_for_model("gpt-3.5-turbo"),
            Some(Encoding::Cl100kBase)
        );
        assert_eq!(
            encoding_for_model("text-davinci-003"),
            Some(Encoding::P50kBase)
        );
        assert_eq!(
            encoding_for_model("text-davinci-002"),
            Some(Encoding::P50kBase)
        );
        assert_eq!(
            encoding_for_model("code-davinci-002"),
            Some(Encoding::P50kBase)
        );
        assert_eq!(
            encoding_for_model("code-cushman-001"),
            Some(Encoding::P50kBase)
        );
        assert_eq!(
            encoding_for_model("text-davinci-001"),
            Some(Encoding::R50kBase)
        );
        assert_eq!(
            encoding_for_model("text-curie-001"),
            Some(Encoding::R50kBase)
        );
        assert_eq!(
            encoding_for_model("text-babbage-001"),
            Some(Encoding::R50kBase)
        );
        assert_eq!(encoding_for_model("text-ada-001"), Some(Encoding::R50kBase));
        assert_eq!(encoding_for_model("davinci"), Some(Encoding::R50kBase));
        assert_eq!(encoding_for_model("curie"), Some(Encoding::R50kBase));
        assert_eq!(encoding_for_model("claude-sonnet-4-6"), None);
        assert_eq!(encoding_for_model("gemini-2.5-pro"), None);
        assert_eq!(
            encoding_for_model("accounts/fireworks/models/deepseek-v3"),
            None
        );
    }

    #[test]
    fn heuristic_count_basic() {
        assert_eq!(heuristic_count(""), 0);
        assert_eq!(heuristic_count("a"), 1);
        assert_eq!(heuristic_count("hello"), 1);
        assert_eq!(heuristic_count(&"a".repeat(20)), 5);
        assert_eq!(heuristic_count("\u{4F60}\u{597D}\u{4E16}\u{754C}"), 1);
    }

    #[test]
    fn heuristic_fallback_is_infallible() {
        assert!(heuristic_count("hello") > 0);
        assert_eq!(heuristic_count(""), 0);
        let nasty = "\u{0}".repeat(1000);
        assert!(heuristic_count(&nasty) > 0);
    }

    #[test]
    fn tiktoken_count_tokens_never_panics() {
        let bpe = Arc::new(tiktoken_rs::cl100k_base().expect("cl100k_base loads"));
        let tok = TiktokenTokenizer::new(bpe);
        let n = tok.count_tokens("The quick brown fox");
        assert!(n > 0, "real BPE count should be positive");
        assert_eq!(tok.count_tokens(""), 0);
        let weird = format!("{}{}", "\u{0}\u{1}".repeat(500), "hello world");
        let _weird_count: usize = tok.count_tokens(&weird);
    }

    #[test]
    fn catch_unwind_returns_fallback_on_panic_shape() {
        let payload = std::panic::catch_unwind(|| panic!("boom"));
        assert!(payload.is_err(), "panic should be captured");
        let err = payload.unwrap_err();
        let msg_static = err
            .downcast_ref::<&'static str>()
            .copied()
            .or_else(|| err.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic>");
        assert_eq!(msg_static, "boom");
        assert!(heuristic_count("hello") > 0);
    }
}
