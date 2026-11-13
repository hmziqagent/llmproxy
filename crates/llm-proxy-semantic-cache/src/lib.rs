//! The optional semantic response cache (plan 09 step 5, audit conflict
//! 7): near-duplicate prompts reuse stored responses, off by default.
//!
//! - **Off by default.** It is wired only when `[server.semantic_cache]`
//!   is enabled and the crate is compiled in.
//! - **Caller-scoped keys.** Entries bucket by `(caller, provider,
//!   model, shape)`; one caller's cached response is never served to
//!   another, and only requests of the same shape — route kind, tool
//!   definitions, tool choice, sampling parameters, provider hints and
//!   message roles — share a bucket. A request carrying any non-text
//!   content (an image, document, audio, a tool use or result) is never
//!   cached and never served.
//! - **Data policy.** The cache stores prompt embeddings and complete
//!   responses per caller for the configured TTL, bounded to
//!   `max_entries` per caller — see `docs/data-class-policy.md`. Nothing
//!   is stored when the feature is off.
//! - **Threshold-gated.** A stored prompt must reach the configured
//!   cosine-similarity threshold (`1.0` behaves like an exact cache).
//!   Only complete non-streaming responses are stored; a streaming lookup
//!   misses.
//!
//! Embeddings come from the [`Embedder`] trait. The default
//! [`NgramEmbedder`] is offline and deterministic — a normalized bag of
//! hashed character trigrams — so the cache works with no external
//! service; a deployment that wants true semantic matching supplies an
//! embedder backed by an embeddings model (the plan 07 endpoint families)
//! through the same trait.

#![deny(missing_docs)]

mod embedder;
mod store;

pub use embedder::{EmbedError, Embedder, Embedding, NgramEmbedder, cosine_similarity};

use std::sync::Mutex;

use async_trait::async_trait;
use llm_proxy_core::SemanticCacheConfig;
use llm_proxy_engine::{Cache, CacheEntry, CacheKey, EngineError};
use llm_proxy_protocol::core::{CoreContent, CoreRequest};
use time::OffsetDateTime;

use store::{Entry, Store};

/// The semantic cache, implementing the engine's [`Cache`] trait.
pub struct SemanticCache {
    config: SemanticCacheConfig,
    embedder: Box<dyn Embedder>,
    store: Mutex<Store>,
    /// Cache hits served since construction (observable, for tests and
    /// operators).
    hits: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for SemanticCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticCache")
            .field("threshold", &self.config.threshold)
            .field("ttl", &self.config.ttl)
            .field("max_entries", &self.config.max_entries)
            .field("embedder", &self.embedder)
            .finish_non_exhaustive()
    }
}

impl SemanticCache {
    /// A cache over `config` using `embedder`.
    #[must_use]
    pub fn new(config: SemanticCacheConfig, embedder: Box<dyn Embedder>) -> Self {
        Self {
            config,
            embedder,
            store: Mutex::new(Store::default()),
            hits: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Hits served so far.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The text a cache key embeds: every request text block, joined.
    fn prompt_text(request: &CoreRequest) -> String {
        let mut parts: Vec<String> = Vec::new();
        for content in &request.system {
            if let CoreContent::Text { text, .. } = content {
                parts.push(text.clone());
            }
        }
        for message in &request.messages {
            for content in &message.content {
                if let CoreContent::Text { text, .. } = content {
                    parts.push(text.clone());
                }
            }
        }
        parts.join("\n")
    }

    /// The request shape a bucket is keyed on: route kind, tool
    /// definitions, tool choice, sampling parameters, provider hints and
    /// message roles — everything besides the embedded text that changes
    /// what a response means. `None` when the request can never be
    /// cached: any non-text system or message content (an image, a
    /// document, audio, a tool use or result) cannot be embedded as text,
    /// and keeping the original data in memory would break the data
    /// policy. This is JSON of tool schemas and sampling numbers, not
    /// prompt text, so it lives in the bucket key only.
    fn shape(key: &CacheKey<'_>) -> Option<String> {
        let all_text = key
            .request
            .system
            .iter()
            .chain(
                key.request
                    .messages
                    .iter()
                    .flat_map(|message| message.content.iter()),
            )
            .all(|content| matches!(content, CoreContent::Text { .. }));
        if !all_text {
            return None;
        }
        let roles: Vec<String> = key
            .request
            .messages
            .iter()
            .map(|message| format!("{:?}", message.role))
            .collect();
        serde_json::to_string(&(
            key.route_kind.config_key(),
            &key.request.tools,
            &key.request.tool_choice,
            &key.request.sampling,
            &key.request.provider_hints,
            roles,
        ))
        .ok()
    }
}

#[async_trait]
impl Cache for SemanticCache {
    async fn lookup(&self, key: &CacheKey<'_>) -> Result<Option<CacheEntry>, EngineError> {
        // Only complete non-streaming responses are stored, so a streaming
        // lookup cannot be served from this cache.
        if key.streaming || !self.config.enabled {
            return Ok(None);
        }
        let Some(shape) = Self::shape(key) else {
            return Ok(None);
        };
        let prompt = Self::prompt_text(key.request);
        let embedding = self
            .embedder
            .embed(&[prompt])
            .await
            .map_err(|error| EngineError::Internal(format!("semantic cache embed: {error}")))?;
        let Some(query) = embedding.first() else {
            return Ok(None);
        };
        let now = OffsetDateTime::now_utc();
        let mut store = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        store.expire(now, self.config.ttl);
        match store.nearest(
            (key.caller, key.provider, key.model, &shape),
            query,
            self.config.threshold,
        ) {
            Some(entry) => {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::debug!(
                    caller = %key.caller,
                    model = %key.model,
                    "semantic cache hit"
                );
                Ok(Some(CacheEntry {
                    response: entry.response.clone(),
                    events: Vec::new(),
                }))
            }
            None => Ok(None),
        }
    }

    async fn store(&self, key: &CacheKey<'_>, entry: &CacheEntry) {
        if key.streaming || !self.config.enabled {
            return;
        }
        let Some(shape) = Self::shape(key) else {
            tracing::debug!("semantic cache store skipped: request has non-text content");
            return;
        };
        let prompt = Self::prompt_text(key.request);
        let embedding = match self.embedder.embed(&[prompt]).await {
            Ok(embedding) => embedding,
            Err(error) => {
                tracing::warn!(%error, "semantic cache store skipped: embedding failed");
                return;
            }
        };
        let Some(vector) = embedding.first() else {
            return;
        };
        // Only model responses are worth caching: an empty response body
        // embeds nothing and would poison the bucket.
        if vector
            .iter()
            .all(|component| component.abs() < f32::EPSILON)
        {
            return;
        }
        let mut store = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        store.insert(
            (key.caller, key.provider, key.model, &shape),
            Entry {
                embedding: vector.clone(),
                response: entry.response.clone(),
                stored_at: OffsetDateTime::now_utc(),
            },
            self.config.max_entries,
        );
    }
}
