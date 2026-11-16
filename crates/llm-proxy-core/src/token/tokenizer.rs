//! The token-counting backend trait and the heuristic fallback, shared by
//! both crate feature configurations.

/// Characters per token assumed by the heuristic backend.
pub(crate) const CHARS_PER_TOKEN: usize = 4;

/// Token-counting backend.
///
/// `Debug` is required so that [`super::Counter`] (which may hold
/// `Arc<TiktokenTokenizer>`) can be formatted.
pub trait Tokenizer: Send + Sync + std::fmt::Debug {
    /// Count tokens in a single text string.
    ///
    /// # Blocking
    ///
    /// This is a synchronous, CPU-bound operation: the BPE backend runs a
    /// fancy-regex pass over the input. It must NOT be called on an async
    /// runtime worker thread; callers should run it under
    /// `tokio::task::spawn_blocking`.
    fn count_tokens(&self, text: &str) -> usize;
}

/// Heuristic backend (~4 chars/token). Zero-allocation and infallible.
///
/// Used as the fallback for unknown models and whenever a BPE encoding cannot
/// be loaded. Counts Unicode code points (not bytes) for better accuracy with
/// multi-byte scripts (CJK, emoji).
#[derive(Debug, Clone, Default)]
pub struct HeuristicTokenizer;

impl Tokenizer for HeuristicTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        heuristic_count(text)
    }
}

pub(crate) fn heuristic_count(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    (text.chars().count() / CHARS_PER_TOKEN).max(1)
}
