//! Token counting utilities.
//!
//! Provides a [`Counter`] that counts tokens with a real BPE tokenizer
//! (`tiktoken`) for known OpenAI model ids and falls back to a character
//! heuristic (~4 chars/token) for everything else. BPE ranks are bundled at
//! compile time, so counting is offline-safe.

pub mod counter;
#[cfg(feature = "tiktoken")]
pub mod tiktoken;
pub mod tokenizer;

pub use counter::{Counter, MessageContent};
