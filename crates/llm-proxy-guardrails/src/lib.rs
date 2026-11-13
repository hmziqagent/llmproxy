//! The optional guardrail set (plan 09 step 4, audit conflict 9): guards
//! that declare when they run and how they fail, instead of one hardcoded
//! hook.
//!
//! - Every [`Guardrail`] declares a [`GuardMode`] — [`PreCall`](GuardMode)
//!   (requests only, before egress), [`PostHoc`](GuardMode) (completed
//!   responses only) or [`Buffered`](GuardMode) (both, with masking) —
//!   and a [`FailPolicy`] for its own failures: [`Open`](FailPolicy)
//!   serves the request and logs, [`Closed`](FailPolicy) refuses it.
//! - Masking replaces rule matches with placeholders in request texts and
//!   response content, through the engine's
//!   [`Guard::transform_request`](llm_proxy_engine::Guard::transform_request)
//!   and
//!   [`Guard::transform_response`](llm_proxy_engine::Guard::transform_response).
//!   A match can straddle two stream chunks, and this crate does not yet
//!   carry per-request masking state, so **a masking guard rejects
//!   streaming requests instead of masking them unsafely** — the plan 09
//!   rule. Deny-only guards let streams through. A rule that answers
//!   [`GuardVerdict::Mask`] under a guard that cannot rewrite text
//!   (deny-only) is a misconfiguration, decided by the guard's fail
//!   policy rather than a silent pass-through; a post-hoc guard applies
//!   mask verdicts to the completed response.
//!
//! The crate is opt-in: it never becomes a dependency of the engine, the
//! server or the daemon. Embedders add guards through
//! [`EngineBuilder::guard`](llm_proxy_engine::EngineBuilder::guard).

#![deny(missing_docs)]

mod pattern;
mod text;

pub use pattern::{PatternRule, RuleAction, RuleScope};
pub use text::{mask_request_texts, mask_response_texts, request_texts, response_texts};

use std::sync::Arc;

use async_trait::async_trait;
use llm_proxy_engine::{EngineError, Guard, GuardDecision};
use llm_proxy_protocol::core::{CoreRequest, CoreResponse};

/// When a guard runs (audit conflict 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuardMode {
    /// Before egress: inspects (and may reject) the request only.
    PreCall,
    /// After completion: inspects (and may reject or mask) the completed
    /// response only. Responses are finished when it sees them, so a mask
    /// verdict is applied here; requests are never inspected or rewritten.
    /// Streams skip the guard's verdict — a PostHoc rule that must see
    /// every output belongs in [`GuardMode::Buffered`].
    PostHoc,
    /// Both sides, with masking: the request is redacted before egress
    /// and the completed response before the client sees it. Streaming
    /// requests are rejected (see the module docs).
    #[default]
    Buffered,
}

/// What a guard does when its own machinery fails (a broken pattern, an
/// unreachable checker) on either pass it runs — inspection and, for
/// masking guards, the masking rewrite: serve the request anyway, or
/// refuse it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FailPolicy {
    /// Serve the request and log the guard failure: availability over
    /// safety. On a masking pass the affected text passes unmasked.
    Open,
    /// Refuse the request: safety over availability, on inspection and
    /// masking alike.
    #[default]
    Closed,
}

/// A rule failure: the guard's machinery broke (as opposed to a verdict).
#[derive(Debug, thiserror::Error)]
#[error("guard rule failed: {0}")]
pub struct RuleError(pub String);

/// What a rule decided about one piece of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardVerdict {
    /// The text may pass unchanged.
    Allow,
    /// The request or response is refused; the message reaches the client
    /// as the rejection reason.
    Reject(String),
    /// The text passes with matches replaced: the value is the redacted
    /// form of the whole input text.
    Mask(String),
}

/// The pluggable rule behind a [`Guardrail`]. Rules are pure functions of
/// text; modes, stream policies and failure handling are the wrapper's.
#[async_trait]
pub trait GuardRule: std::fmt::Debug + Send + Sync {
    /// Judge one request text (prompt, system, message contents).
    async fn check_request_text(&self, text: &str) -> Result<GuardVerdict, RuleError>;

    /// Judge one response text (content blocks).
    async fn check_response_text(&self, text: &str) -> Result<GuardVerdict, RuleError>;
}

/// One guardrail: a [`GuardRule`] plus its declared mode and fail policy,
/// implementing the engine's [`Guard`].
#[derive(Debug)]
pub struct Guardrail {
    rule: Arc<dyn GuardRule>,
    mode: GuardMode,
    fail_policy: FailPolicy,
}

impl Guardrail {
    /// A deny-only guard: it never rewrites text, so streams may pass. A
    /// rule that answers [`GuardVerdict::Mask`] is a misconfiguration
    /// here — the guard cannot redact — and the verdict is decided by the
    /// [`FailPolicy`].
    #[must_use]
    pub fn deny_only(rule: Arc<dyn GuardRule>, fail_policy: FailPolicy) -> Self {
        Self {
            rule,
            mode: GuardMode::PreCall,
            fail_policy,
        }
    }

    /// A masking guard ([`GuardMode::Buffered`]): both sides are checked
    /// and redacted, and streaming requests are rejected rather than
    /// masked unsafely across chunks.
    #[must_use]
    pub fn masking(rule: Arc<dyn GuardRule>, fail_policy: FailPolicy) -> Self {
        Self {
            rule,
            mode: GuardMode::Buffered,
            fail_policy,
        }
    }

    /// A post-hoc guard: completed responses only, which it may reject or
    /// mask; requests pass untouched.
    #[must_use]
    pub fn post_hoc(rule: Arc<dyn GuardRule>, fail_policy: FailPolicy) -> Self {
        Self {
            rule,
            mode: GuardMode::PostHoc,
            fail_policy,
        }
    }

    /// The declared mode.
    #[must_use]
    pub fn mode(&self) -> GuardMode {
        self.mode
    }

    /// The declared fail policy.
    #[must_use]
    pub fn fail_policy(&self) -> FailPolicy {
        self.fail_policy
    }

    /// The fail policy's ruling on one rule failure, shared by the
    /// inspection and masking passes so the two cannot drift: `Open`
    /// logs and lets the caller continue; `Closed` produces the
    /// guard-failed error.
    fn apply_fail_policy(&self, error: &RuleError) -> Result<(), EngineError> {
        match self.fail_policy {
            FailPolicy::Open => {
                tracing::warn!(%error, "guard rule failed; failing open");
                Ok(())
            }
            FailPolicy::Closed => Err(EngineError::Internal(format!(
                "guard failed closed: {error}"
            ))),
        }
    }

    /// Applies the fail policy to a rule failure during inspection.
    fn rule_failure(&self, error: &RuleError) -> Result<GuardDecision, EngineError> {
        self.apply_fail_policy(error).map(|()| GuardDecision::Allow)
    }

    /// A `Reject` verdict that surfaced only on the masking pass — the
    /// rule flipped since inspection. `Open` keeps the text unchanged
    /// with a warning; `Closed` refuses rather than pass text a rule
    /// would have rejected.
    fn flipped_verdict(&self, message: &str) -> Result<(), EngineError> {
        match self.fail_policy {
            FailPolicy::Open => {
                tracing::warn!(
                    reason = %message,
                    "guard verdict changed between inspection and masking; failing open"
                );
                Ok(())
            }
            FailPolicy::Closed => Err(EngineError::Internal(
                "guard verdict changed between inspection and masking".to_owned(),
            )),
        }
    }
}

#[async_trait]
impl Guard for Guardrail {
    async fn inspect_request(&self, request: &CoreRequest) -> Result<GuardDecision, EngineError> {
        if self.mode == GuardMode::PostHoc {
            return Ok(GuardDecision::Allow);
        }
        // The plan 09 rule: masking across stream chunks needs placeholder
        // handling this crate does not have, so a Buffered guard rejects
        // streaming requests instead of masking them unsafely. Deny-only
        // guards need no cross-chunk state and let streams through.
        if self.mode == GuardMode::Buffered && request.stream {
            return Ok(GuardDecision::Reject(
                "a masking guardrail is active and cannot redact streamed \
                 responses safely across chunk boundaries; retry with \
                 stream: false"
                    .to_owned(),
            ));
        }
        for text in request_texts(request) {
            let text_owned = text;
            match self.rule.check_request_text(&text_owned).await {
                Ok(GuardVerdict::Allow) => {}
                Ok(GuardVerdict::Reject(message)) => {
                    return Ok(GuardDecision::Reject(message));
                }
                // A guard that cannot rewrite text cannot honour a mask
                // verdict: leaving the text unredacted with no signal is
                // a data-leak failure mode, so the misconfiguration is
                // decided by the fail policy instead.
                Ok(GuardVerdict::Mask(_)) if self.mode != GuardMode::Buffered => {
                    return self.rule_failure(&RuleError(
                        "mask verdict from a non-masking guard".to_owned(),
                    ));
                }
                Ok(GuardVerdict::Mask(_)) => {}
                Err(error) => return self.rule_failure(&error),
            }
        }
        Ok(GuardDecision::Allow)
    }

    async fn transform_request(&self, request: &mut CoreRequest) -> Result<(), EngineError> {
        if self.mode != GuardMode::Buffered {
            return Ok(());
        }
        mask_request_texts(request, |text| async {
            let text_owned = text;
            match self.rule.check_request_text(&text_owned).await {
                Ok(GuardVerdict::Mask(redacted)) => Ok(Some(redacted)),
                Ok(GuardVerdict::Allow) => Ok(None),
                // inspect_request already vetted the request; a rule that
                // flips between the two passes is still decided by the
                // fail policy — never a quiet pass-through of unredacted
                // text.
                Ok(GuardVerdict::Reject(message)) => self.flipped_verdict(&message).map(|()| None),
                Err(error) => self.apply_fail_policy(&error).map(|()| None),
            }
        })
        .await
    }

    async fn inspect_response(
        &self,
        response: &CoreResponse,
    ) -> Result<GuardDecision, EngineError> {
        if self.mode == GuardMode::PreCall {
            return Ok(GuardDecision::Allow);
        }
        for text in response_texts(response) {
            let text_owned = text;
            match self.rule.check_response_text(&text_owned).await {
                Ok(GuardVerdict::Allow) => {}
                Ok(GuardVerdict::Reject(message)) => {
                    return Ok(GuardDecision::Reject(message));
                }
                Ok(GuardVerdict::Mask(_)) => {}
                Err(error) => return self.rule_failure(&error),
            }
        }
        Ok(GuardDecision::Allow)
    }

    async fn transform_response(&self, response: &mut CoreResponse) -> Result<(), EngineError> {
        // Responses are complete when a post-hoc guard sees them, so
        // masking is safe there: no chunk-boundary state to carry.
        if self.mode == GuardMode::PreCall {
            return Ok(());
        }
        mask_response_texts(response, |text| async {
            let text_owned = text;
            match self.rule.check_response_text(&text_owned).await {
                Ok(GuardVerdict::Mask(redacted)) => Ok(Some(redacted)),
                Ok(GuardVerdict::Allow) => Ok(None),
                Ok(GuardVerdict::Reject(message)) => self.flipped_verdict(&message).map(|()| None),
                Err(error) => self.apply_fail_policy(&error).map(|()| None),
            }
        })
        .await
    }
}
