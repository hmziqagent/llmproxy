//! The one admission step (plan 05 steps 3-7, backlog 6/18/30, conflicts 2
//! and 8): per-key allowlists, rate windows, spend budgets and concurrency
//! caps decided in one place, reserve-then-settle.
//!
//! The rules:
//!
//! - **Allowlists** (step 3). A key with `allow_models` may only run a model
//!   matching one of its globs (the requested name or the upstream name);
//!   with `allow_providers`, only a matching route. A restricted key whose
//!   request names no model fails closed. Refusals are 403. The engine asks
//!   at pipeline slot 2, through [`Admission::authorize`] — ahead of the
//!   idempotency replay, the cache and single-flight, so a key restricted
//!   at runtime is not served an answer stored for it earlier — and
//!   `reserve` checks again.
//! - **Windows** (step 5). `rpm` and `rpd` count admitted requests per
//!   calendar minute and UTC day; `tpm` counts tokens per calendar minute —
//!   the input estimate plus `max_tokens` is reserved at admission and
//!   replaced by the provider-reported total at settlement. Refusals are 429
//!   with `Retry-After` set to the window's reset.
//! - **Budgets** (step 4). A key with `usd_per_day` reserves its estimated
//!   cost (input estimate plus `max_tokens`, or the key's
//!   `assumed_max_output_tokens`, at the member's pricing) before egress.
//!   Admission requires `spent + reserved_in_flight + estimate <= budget`,
//!   so concurrent requests cannot overspend the budget by more than what
//!   is reserved. A model with no pricing cannot be estimated and goes to
//!   the key's explicit `unestimatable` policy: reject (402), or reserve a
//!   fixed amount. Refusals are 402.
//! - **What counts** (step 6). At settlement the reservation is released
//!   and the budget is charged exactly what the provider billed: the
//!   engine-computed cost of the usage the serving attempt reported —
//!   including a response the proxy's own guards then refused, which the
//!   provider billed all the same. A request that fails before any usage
//!   charges nothing (the reservation is simply released: the refund).
//!   Cache hits never reach `reserve`, so they charge nothing. Unpriced
//!   usage cannot be charged and is logged.
//! - **Concurrency** (step 7). `[provider.concurrency]` caps in-flight
//!   requests per member and per upstream model; a saturated cap answers
//!   503 with a one-second `Retry-After`, never 429, because capacity
//!   shedding misread as rate limiting arms hour-long client backoff
//!   ladders. The slot is held from admission to settlement, and moves
//!   with the ladder: at each failover the slot is re-keyed to (and the
//!   cap checked against) the member about to serve, so a capped member
//!   is capped however the request reaches it.
//!
//! Since plan 09 the counters live behind one
//! [`AdmissionBackend`](crate::admission_backend::AdmissionBackend): the
//! in-process [`MemoryAdmissionBackend`](crate::admission_backend::MemoryAdmissionBackend)
//! by default, or a shared store from the optional
//! `llm-proxy-store-redis` / `llm-proxy-store-postgres` crates, so replicas
//! enforce one budget between them. This type holds only the rules; every
//! check-and-commit stays atomic inside the backend.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llm_proxy_core::{
    AuthConfig, ClientKeySpec, KeyBudgetConfig, KeyWindowLimits, ModelPricing, ProviderRegistry,
    UnestimatableCostPolicy, glob_match,
};
use llm_proxy_protocol::core::Usage;
use rust_decimal::Decimal;
use time::OffsetDateTime;

use crate::admission_backend::{
    AdmissionBackend, MemoryAdmissionBackend, MicroUsd, ReservationDemand, Settlement, to_micro_usd,
};
use crate::error::EngineError;
use crate::traits::{Admission, AdmissionClaim, AdmissionOutcome, AdmissionTicket};

/// One key's admission policy, as configured.
#[derive(Debug, Clone, Default)]
pub struct KeyPolicy {
    /// Model globs this key may run; empty allows all.
    pub allow_models: Vec<String>,
    /// Route (provider or pool) globs this key may address; empty allows all.
    pub allow_providers: Vec<String>,
    /// Rate windows.
    pub limits: KeyWindowLimits,
    /// Spend budget.
    pub budget: Option<KeyBudgetConfig>,
}

impl KeyPolicy {
    fn restricts_anything(&self) -> bool {
        !self.allow_models.is_empty()
            || !self.allow_providers.is_empty()
            || self.limits != KeyWindowLimits::default()
            || self.budget.is_some()
    }
}

/// A member's in-flight caps.
#[derive(Debug, Clone, Default)]
struct MemberCaps {
    total: Option<u32>,
    models: HashMap<String, u32>,
}

type Clock = Box<dyn Fn() -> OffsetDateTime + Send + Sync>;

fn total_tokens(usage: &Usage) -> u64 {
    let parts = [
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens.unwrap_or(0),
        usage.cache_read_input_tokens.unwrap_or(0),
        usage.reasoning_tokens.unwrap_or(0),
    ];
    parts
        .into_iter()
        .map(|count| u64::try_from(count).unwrap_or(0))
        .sum()
}

/// Whether `value` (or `alternate`) matches any glob; an empty list allows
/// everything, and a restricted list with nothing to match fails closed.
fn allowed_by(globs: &[String], value: &str, alternate: &str) -> bool {
    if globs.is_empty() {
        return true;
    }
    [value, alternate]
        .into_iter()
        .filter(|candidate| !candidate.is_empty())
        .any(|candidate| globs.iter().any(|glob| glob_match(glob, candidate)))
}

/// The configured [`Admission`]: allowlists, windows, budgets and
/// concurrency caps from `[auth.clients]` and `[provider.concurrency]`,
/// with the counters held by an [`AdmissionBackend`].
pub struct PolicyAdmission {
    keys: HashMap<String, KeyPolicy>,
    pricing: HashMap<(String, String), ModelPricing>,
    caps: HashMap<String, MemberCaps>,
    clock: Clock,
    backend: Arc<dyn AdmissionBackend>,
}

impl std::fmt::Debug for PolicyAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyAdmission")
            .field("keys", &self.keys.keys().collect::<Vec<_>>())
            .field("priced_models", &self.pricing.len())
            .field("capped_members", &self.caps.keys().collect::<Vec<_>>())
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl PolicyAdmission {
    /// Builds the admission step from the auth section (per-key policies)
    /// and the provider registry (pricing for estimates, concurrency caps),
    /// keeping its counters in this process.
    #[must_use]
    pub fn from_config(auth: &AuthConfig, providers: &ProviderRegistry) -> Self {
        let keys = auth
            .clients
            .iter()
            .filter_map(|(id, spec)| match spec {
                ClientKeySpec::Verifier(_) => None,
                ClientKeySpec::Policy {
                    allow_models,
                    allow_providers,
                    limits,
                    budget,
                    ..
                } => Some((
                    id.clone(),
                    KeyPolicy {
                        allow_models: allow_models.clone(),
                        allow_providers: allow_providers.clone(),
                        limits: limits.clone().unwrap_or_default(),
                        budget: budget.clone(),
                    },
                )),
            })
            .collect();
        let mut pricing = HashMap::new();
        let mut caps: HashMap<String, (Option<u32>, HashMap<String, u32>)> = HashMap::new();
        for provider in providers.iter() {
            for (model, price) in &provider.pricing {
                pricing.insert(
                    (provider.name.clone(), model.as_str().to_owned()),
                    price.clone(),
                );
            }
            if let Some(concurrency) = &provider.concurrency {
                caps.insert(
                    provider.name.clone(),
                    (
                        concurrency.total,
                        concurrency
                            .models
                            .iter()
                            .map(|(model, cap)| (model.as_str().to_owned(), *cap))
                            .collect(),
                    ),
                );
            }
        }
        Self::new(keys, pricing, caps)
    }

    /// Builds from explicit parts: per-key policies, `(member, model)`
    /// pricing, and per-member `(total, per-model)` caps.
    #[must_use]
    pub fn new(
        keys: HashMap<String, KeyPolicy>,
        pricing: HashMap<(String, String), ModelPricing>,
        caps: HashMap<String, (Option<u32>, HashMap<String, u32>)>,
    ) -> Self {
        Self::over_backend(keys, pricing, caps, Arc::new(MemoryAdmissionBackend::new()))
    }

    /// Builds from the same config surfaces [`PolicyAdmission::from_config`]
    /// reads, over an [`AdmissionBackend`] — what a replica deployment
    /// (plan 09 step 1) builds: identical rules, shared counters.
    #[must_use]
    pub fn over_config_backend(
        auth: &AuthConfig,
        providers: &ProviderRegistry,
        backend: Arc<dyn AdmissionBackend>,
    ) -> Self {
        let keys = auth
            .clients
            .iter()
            .filter_map(|(id, spec)| match spec {
                ClientKeySpec::Verifier(_) => None,
                ClientKeySpec::Policy {
                    allow_models,
                    allow_providers,
                    limits,
                    budget,
                    ..
                } => Some((
                    id.clone(),
                    KeyPolicy {
                        allow_models: allow_models.clone(),
                        allow_providers: allow_providers.clone(),
                        limits: limits.clone().unwrap_or_default(),
                        budget: budget.clone(),
                    },
                )),
            })
            .collect();
        let mut pricing = HashMap::new();
        let mut caps: HashMap<String, (Option<u32>, HashMap<String, u32>)> = HashMap::new();
        for provider in providers.iter() {
            for (model, price) in &provider.pricing {
                pricing.insert(
                    (provider.name.clone(), model.as_str().to_owned()),
                    price.clone(),
                );
            }
            if let Some(concurrency) = &provider.concurrency {
                caps.insert(
                    provider.name.clone(),
                    (
                        concurrency.total,
                        concurrency
                            .models
                            .iter()
                            .map(|(model, cap)| (model.as_str().to_owned(), *cap))
                            .collect(),
                    ),
                );
            }
        }
        Self::over_backend(keys, pricing, caps, backend)
    }

    /// Builds with the counters held by an [`AdmissionBackend`] — the
    /// replica path (plan 09 step 1): a shared backend makes two instances
    /// enforce one budget between them. The rules stay here.
    #[must_use]
    pub fn over_backend(
        keys: HashMap<String, KeyPolicy>,
        pricing: HashMap<(String, String), ModelPricing>,
        caps: HashMap<String, (Option<u32>, HashMap<String, u32>)>,
        backend: Arc<dyn AdmissionBackend>,
    ) -> Self {
        Self {
            keys,
            pricing,
            caps: caps
                .into_iter()
                .map(|(member, (total, models))| (member, MemberCaps { total, models }))
                .collect(),
            clock: Box::new(OffsetDateTime::now_utc),
            backend,
        }
    }

    /// The backend holding this admission's counters (the shared store a
    /// replica deployment was built with).
    #[must_use]
    pub fn backend(&self) -> &Arc<dyn AdmissionBackend> {
        &self.backend
    }

    /// Replace the clock the windows read (tests drive it by hand).
    #[must_use]
    pub fn with_clock(
        mut self,
        clock: impl Fn() -> OffsetDateTime + Send + Sync + 'static,
    ) -> Self {
        self.clock = Box::new(clock);
        self
    }

    /// Whether this step would ever refuse anything: `false` when no key
    /// carries a policy and no provider a cap, so a builder can keep the
    /// allow-all default.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.keys
            .values()
            .all(|policy| !policy.restricts_anything())
            && self.caps.is_empty()
    }

    /// The reservation a budgeted key needs for this claim, or the refusal
    /// its unestimatable policy dictates.
    fn estimate_usd(
        &self,
        claim: &AdmissionClaim<'_>,
        budget: &KeyBudgetConfig,
    ) -> Result<MicroUsd, EngineError> {
        let Some(price) = self
            .pricing
            .get(&(claim.member.to_owned(), claim.model.to_owned()))
        else {
            return match &budget.unestimatable {
                UnestimatableCostPolicy::Reject => Err(EngineError::BudgetExhausted {
                    message: format!(
                        "cost of model \"{}\" on \"{}\" cannot be estimated (no pricing) and the key's policy rejects unestimatable requests",
                        claim.model, claim.member
                    ),
                    retry_after: None,
                }),
                UnestimatableCostPolicy::ReserveUsd { usd } => Ok(to_micro_usd(*usd)),
            };
        };
        let output = claim
            .max_output_tokens
            .unwrap_or(budget.assumed_max_output_tokens);
        let input_cost = Decimal::from(claim.estimated_input_tokens)
            .checked_mul(price.input)
            .unwrap_or(Decimal::MAX);
        let output_cost = Decimal::from(output)
            .checked_mul(price.output)
            .unwrap_or(Decimal::MAX);
        Ok(to_micro_usd(
            input_cost.checked_add(output_cost).unwrap_or(Decimal::MAX),
        ))
    }

    fn caps_for(&self, member: &str, model: &str) -> (Option<u32>, Option<u32>) {
        let caps = self.caps.get(member);
        (
            caps.and_then(|c| c.total),
            caps.and_then(|c| c.models.get(model).copied()),
        )
    }

    /// The per-key allowlist verdict for one claim: a key with a policy may
    /// only run a model and address a route its globs match. Reads config
    /// only; no counter moves.
    fn check_allowlists(&self, claim: &AdmissionClaim<'_>) -> Result<(), EngineError> {
        let Some(policy) = self.keys.get(claim.caller) else {
            return Ok(());
        };
        if !allowed_by(&policy.allow_models, claim.requested_model, claim.model) {
            return Err(EngineError::Forbidden(
                if claim.requested_model.is_empty() && claim.model.is_empty() {
                    "this key is restricted to specific models and the request names none"
                        .to_owned()
                } else {
                    format!(
                        "this key may not use model \"{}\"",
                        if claim.requested_model.is_empty() {
                            claim.model
                        } else {
                            claim.requested_model
                        }
                    )
                },
            ));
        }
        if !allowed_by(&policy.allow_providers, claim.provider, "") {
            return Err(EngineError::Forbidden(format!(
                "this key may not use provider \"{}\"",
                claim.provider
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl Admission for PolicyAdmission {
    async fn authorize(&self, claim: &AdmissionClaim<'_>) -> Result<(), EngineError> {
        self.check_allowlists(claim)
    }

    async fn reserve(&self, claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        let now = (self.clock)();
        let policy = self.keys.get(claim.caller);

        // Allowlists again, before any accounting: the engine asks at slot 2
        // (`authorize`), but callers that reserve directly — the
        // endpoint-family routes, the MCP gateway — never do.
        self.check_allowlists(claim)?;

        // Budget estimate is computed before any counter moves: it reads
        // only config.
        let (budget_usd_per_day, estimate_usd) = match policy.and_then(|p| p.budget.as_ref()) {
            Some(budget) => (Some(to_micro_usd(budget.usd_per_day)), {
                let estimate = self.estimate_usd(claim, budget)?;
                Some(estimate)
            }),
            None => (None, None),
        };
        let reserved_tokens = claim
            .estimated_input_tokens
            .saturating_add(claim.max_output_tokens.unwrap_or(0));

        let limits = policy.map_or(KeyWindowLimits::default(), |p| p.limits.clone());
        let (member_cap, model_cap) = self.caps_for(claim.member, claim.model);
        let demand = ReservationDemand {
            now,
            caller: claim.caller,
            member: claim.member,
            model: claim.model,
            rpm: limits.rpm,
            tpm: limits.tpm,
            rpd: limits.rpd,
            budget_usd_per_day,
            estimate_usd,
            reserved_tokens,
            member_cap,
            model_cap,
        };
        let lease = self.backend.admit(&demand).await?;
        Ok(AdmissionTicket {
            reserved: Some(reserved_tokens),
            lease: Some(lease),
            caller: Some(claim.caller.to_owned()),
        })
    }

    fn settle(&self, ticket: &AdmissionTicket, outcome: &AdmissionOutcome<'_>) {
        let Some(lease) = ticket.lease.as_ref() else {
            return;
        };
        let now = (self.clock)();
        let settlement = Settlement {
            now,
            succeeded: outcome.succeeded,
            usage_tokens: outcome.usage.map(total_tokens),
            charged_usd: outcome.cost.map(|cost| to_micro_usd(cost.total)),
        };
        self.backend.settle(lease, &settlement);
        if outcome.cost.is_none()
            && outcome.usage.is_some()
            && ticket.caller.as_deref().is_some_and(|caller| {
                self.keys
                    .get(caller)
                    .is_some_and(|policy| policy.budget.is_some())
            })
        {
            tracing::warn!(
                caller = ticket.caller.as_deref().unwrap_or_default(),
                lease = lease.as_str(),
                "usage settled against a budgeted key with no pricing for the model; nothing charged (unpriced)"
            );
        }
    }

    /// Step 7 under failover: the in-flight slot moves to the member the
    /// ladder is about to try, so a capped member is capped however the
    /// request reached it. The whole move — release, saturation check,
    /// acquire — happens atomically in the backend, so counts never drift.
    async fn rekey(
        &self,
        ticket: &AdmissionTicket,
        member: &str,
        model: &str,
    ) -> Result<(), EngineError> {
        let Some(lease) = ticket.lease.as_ref() else {
            return Ok(());
        };
        let (member_cap, model_cap) = self.caps_for(member, model);
        self.backend
            .rekey(lease, member, model, member_cap, model_cap)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};

    fn claim<'a>(
        caller: &'a str,
        member: &'a str,
        model: &'a str,
        input: u64,
        max_output: Option<u64>,
    ) -> AdmissionClaim<'a> {
        AdmissionClaim {
            caller,
            provider: member,
            member,
            model,
            requested_model: model,
            body_hash: "h",
            estimated_input_tokens: input,
            max_output_tokens: max_output,
        }
    }

    fn pricing(input: &str, output: &str) -> ModelPricing {
        ModelPricing {
            input: input.parse().expect("decimal"),
            output: output.parse().expect("decimal"),
            ..ModelPricing::default()
        }
    }

    fn budgeted(usd_per_day: &str) -> KeyPolicy {
        KeyPolicy {
            budget: Some(KeyBudgetConfig {
                usd_per_day: usd_per_day.parse().expect("decimal"),
                unestimatable: UnestimatableCostPolicy::Reject,
                assumed_max_output_tokens: 100,
            }),
            ..KeyPolicy::default()
        }
    }

    /// An admission with one priced model: 1 USD per 1000 input tokens and
    /// per 1000 output tokens, so a (500 in, 500 out) claim reserves 1 USD.
    fn admission(policy: KeyPolicy) -> PolicyAdmission {
        PolicyAdmission::new(
            HashMap::from([("key".to_owned(), policy)]),
            HashMap::from([(("p".to_owned(), "m".to_owned()), pricing("0.001", "0.001"))]),
            HashMap::new(),
        )
    }

    fn usage(input: i32, output: i32) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            ..Usage::default()
        }
    }

    fn cost(total: &str) -> llm_proxy_protocol::core::Cost {
        llm_proxy_protocol::core::Cost {
            input: Decimal::ZERO,
            output: Decimal::ZERO,
            cache_creation: Decimal::ZERO,
            cache_read: Decimal::ZERO,
            reasoning: Decimal::ZERO,
            total: total.parse().expect("decimal"),
        }
    }

    #[tokio::test]
    async fn concurrent_reservations_cannot_overspend_the_budget() {
        // Budget 2.5 USD; each claim reserves 1 USD; five claims race.
        let admission = Arc::new(admission(budgeted("2.5")));
        let mut handles = Vec::new();
        for _ in 0..5 {
            let admission = Arc::clone(&admission);
            handles.push(tokio::spawn(async move {
                admission
                    .reserve(&claim("key", "p", "m", 500, Some(500)))
                    .await
                    .is_ok()
            }));
        }
        let mut admitted = 0;
        for handle in handles {
            if handle.await.expect("task") {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, 2,
            "spent + in-flight reservations + this request must stay within the budget"
        );
    }

    #[tokio::test]
    async fn a_refund_restores_the_budget_when_a_request_fails_before_any_cost() {
        let admission = admission(budgeted("1"));
        let ticket = admission
            .reserve(&claim("key", "p", "m", 500, Some(500)))
            .await
            .expect("first reservation fits");
        let refused = admission
            .reserve(&claim("key", "p", "m", 500, Some(500)))
            .await
            .expect_err("the budget is fully reserved");
        assert!(matches!(refused, EngineError::BudgetExhausted { .. }));
        assert_eq!(refused.status_code(), 402);

        // Failed before any usage: refund.
        admission.settle(
            &ticket,
            &AdmissionOutcome {
                succeeded: false,
                usage: None,
                cost: None,
            },
        );
        admission
            .reserve(&claim("key", "p", "m", 500, Some(500)))
            .await
            .expect("the refund restored the full budget");
    }

    #[tokio::test]
    async fn settlement_charges_what_the_provider_billed_not_the_estimate() {
        let admission = admission(budgeted("1"));
        let ticket = admission
            .reserve(&claim("key", "p", "m", 500, Some(500)))
            .await
            .expect("reserve 1 USD");
        // The provider billed 0.1 USD (far below the 1 USD estimate).
        let billed = usage(50, 50);
        admission.settle(
            &ticket,
            &AdmissionOutcome {
                succeeded: true,
                usage: Some(&billed),
                cost: Some(&cost("0.1")),
            },
        );
        // 0.9 USD remains: a 0.9 USD claim fits, a 0.95 USD claim does not.
        admission
            .reserve(&claim("key", "p", "m", 450, Some(450)))
            .await
            .expect("0.9 fits in the 0.9 left");
        let refused = admission
            .reserve(&claim("key", "p", "m", 25, Some(25)))
            .await
            .expect_err("nothing is left once 0.9 is reserved again");
        assert!(matches!(refused, EngineError::BudgetExhausted { .. }));
    }

    #[tokio::test]
    async fn an_unpriced_model_goes_to_the_keys_explicit_policy() {
        let reject = admission(budgeted("10"));
        let refused = reject
            .reserve(&claim("key", "p", "unpriced", 10, Some(10)))
            .await
            .expect_err("reject policy fails closed");
        assert!(matches!(refused, EngineError::BudgetExhausted { .. }));

        let mut policy = budgeted("1");
        if let Some(budget) = policy.budget.as_mut() {
            budget.unestimatable = UnestimatableCostPolicy::ReserveUsd {
                usd: "0.6".parse().expect("decimal"),
            };
        }
        let reserve = admission(policy);
        reserve
            .reserve(&claim("key", "p", "unpriced", 10, Some(10)))
            .await
            .expect("the fixed reservation fits");
        let refused = reserve
            .reserve(&claim("key", "p", "unpriced", 10, Some(10)))
            .await
            .expect_err("a second fixed reservation would overspend");
        assert!(matches!(refused, EngineError::BudgetExhausted { .. }));
    }

    #[tokio::test]
    async fn allowlists_fail_closed_without_an_identifiable_model() {
        let policy = KeyPolicy {
            allow_models: vec!["claude-*".to_owned()],
            allow_providers: vec!["anthropic*".to_owned()],
            ..KeyPolicy::default()
        };
        let admission = admission(policy);
        let ok = AdmissionClaim {
            provider: "anthropic-pool",
            member: "anthropic",
            requested_model: "claude-sonnet-4-6",
            model: "claude-sonnet-4-6",
            ..claim("key", "anthropic", "claude-sonnet-4-6", 1, None)
        };
        admission
            .reserve(&ok)
            .await
            .expect("matching model and provider");

        let wrong_model = AdmissionClaim {
            requested_model: "gpt-4o",
            model: "gpt-4o",
            ..ok
        };
        let refused = admission
            .reserve(&wrong_model)
            .await
            .expect_err("model denied");
        assert!(matches!(refused, EngineError::Forbidden(_)));
        assert_eq!(refused.status_code(), 403);

        let wrong_provider = AdmissionClaim {
            provider: "openai",
            ..ok
        };
        assert!(matches!(
            admission.reserve(&wrong_provider).await,
            Err(EngineError::Forbidden(_))
        ));

        let no_model = AdmissionClaim {
            requested_model: "",
            model: "",
            ..ok
        };
        let refused = admission
            .reserve(&no_model)
            .await
            .expect_err("a restricted key with no model fails closed");
        assert!(matches!(refused, EngineError::Forbidden(_)));

        // An unrestricted key (no policy entry) runs anything.
        admission
            .reserve(&AdmissionClaim {
                caller: "other",
                ..no_model
            })
            .await
            .expect("keys without a policy are unrestricted");
    }

    #[tokio::test]
    async fn authorize_gives_the_allowlist_verdict_without_moving_a_counter() {
        // Slot 2's verdict is the allowlists `reserve` enforces, with no
        // accounting: the key's one-request minute window is still whole
        // after every authorize call below.
        let policy = KeyPolicy {
            allow_models: vec!["claude-*".to_owned()],
            allow_providers: vec!["anthropic*".to_owned()],
            limits: KeyWindowLimits {
                rpm: Some(1),
                ..KeyWindowLimits::default()
            },
            ..KeyPolicy::default()
        };
        let admission = admission(policy);
        let ok = AdmissionClaim {
            provider: "anthropic-pool",
            member: "anthropic",
            requested_model: "claude-sonnet-4-6",
            model: "claude-sonnet-4-6",
            ..claim("key", "anthropic", "claude-sonnet-4-6", 1, None)
        };
        admission
            .authorize(&ok)
            .await
            .expect("matching model and provider");

        let wrong_model = AdmissionClaim {
            requested_model: "gpt-4o",
            model: "gpt-4o",
            ..ok
        };
        let refused = admission
            .authorize(&wrong_model)
            .await
            .expect_err("model denied");
        assert!(matches!(refused, EngineError::Forbidden(_)));
        assert_eq!(refused.status_code(), 403);

        let wrong_provider = AdmissionClaim {
            provider: "openai",
            ..ok
        };
        let refused = admission
            .authorize(&wrong_provider)
            .await
            .expect_err("provider denied");
        assert!(matches!(refused, EngineError::Forbidden(_)));
        assert_eq!(refused.status_code(), 403);

        // An unrestricted key (no policy entry) is authorized for anything.
        admission
            .authorize(&AdmissionClaim {
                caller: "other",
                ..wrong_model
            })
            .await
            .expect("keys without a policy are unrestricted");

        // No counter moved: the allowed claim still fits the minute window
        // of one request, and that reservation is what spends it.
        admission
            .reserve(&ok)
            .await
            .expect("authorize spent none of the minute window");
        let refused = admission
            .reserve(&ok)
            .await
            .expect_err("the window holds exactly one request");
        assert!(
            matches!(refused, EngineError::RateLimited { .. }),
            "got {refused:?}"
        );
    }

    #[tokio::test]
    async fn rate_windows_refuse_with_the_window_reset_and_roll_over() {
        let clock = Arc::new(AtomicI64::new(1_700_000_000));
        let reader = Arc::clone(&clock);
        let policy = KeyPolicy {
            limits: KeyWindowLimits {
                rpm: Some(2),
                tpm: Some(1_000),
                rpd: Some(3),
            },
            ..KeyPolicy::default()
        };
        let admission = admission(policy).with_clock(move || {
            OffsetDateTime::from_unix_timestamp(reader.load(Ordering::SeqCst)).expect("timestamp")
        });

        let first = admission
            .reserve(&claim("key", "p", "m", 100, Some(100)))
            .await
            .expect("1/2 rpm");
        admission
            .reserve(&claim("key", "p", "m", 100, Some(100)))
            .await
            .expect("2/2 rpm");
        let refused = admission
            .reserve(&claim("key", "p", "m", 100, Some(100)))
            .await
            .expect_err("3rd in the minute");
        match refused {
            EngineError::RateLimited { retry_after, .. } => {
                let wait = retry_after.expect("window reset advised");
                assert!(wait.as_secs() >= 1 && wait.as_secs() <= 60, "got {wait:?}");
            }
            other => panic!("expected a 429, got {other:?}"),
        }

        // Next minute: rpm resets, but the day count (2) carries on.
        clock.fetch_add(60, Ordering::SeqCst);
        admission
            .reserve(&claim("key", "p", "m", 100, Some(100)))
            .await
            .expect("3/3 rpd in a fresh minute");
        let refused = admission
            .reserve(&claim("key", "p", "m", 100, Some(100)))
            .await
            .expect_err("4th in the day");
        assert!(
            matches!(&refused, EngineError::RateLimited { message, .. } if message.contains("rpd")),
            "expected the day window to refuse, got {refused:?}"
        );

        // Next day: everything resets.
        clock.fetch_add(86_400, Ordering::SeqCst);
        admission
            .reserve(&claim("key", "p", "m", 100, Some(100)))
            .await
            .expect("fresh day");

        // TPM: the fresh day's first claim reserved 200 tokens; a claim
        // estimated at 1000 would push the minute to 1200 > 1000.
        let refused = admission
            .reserve(&claim("key", "p", "m", 900, Some(100)))
            .await
            .expect_err("200 reserved + 1000 estimated exceeds tpm 1000");
        assert!(
            matches!(&refused, EngineError::RateLimited { message, .. } if message.contains("tpm")),
            "expected the token window to refuse, got {refused:?}"
        );

        // Settling the first (old-minute) lease never disturbs the current
        // minute's accounting.
        admission.settle(
            &first,
            &AdmissionOutcome {
                succeeded: true,
                usage: Some(&usage(100, 100)),
                cost: None,
            },
        );
    }

    #[tokio::test]
    async fn concurrency_caps_answer_503_with_a_short_wait_and_free_on_settle() {
        let admission = PolicyAdmission::new(
            HashMap::new(),
            HashMap::new(),
            HashMap::from([(
                "p".to_owned(),
                (Some(2), HashMap::from([("m".to_owned(), 1)])),
            )]),
        );
        let held = admission
            .reserve(&claim("anonymous", "p", "m", 1, None))
            .await
            .expect("first slot");
        let refused = admission
            .reserve(&claim("anonymous", "p", "m", 1, None))
            .await
            .expect_err("model cap of 1");
        match &refused {
            EngineError::Overloaded { retry_after, .. } => {
                assert!(
                    retry_after.is_some_and(|wait| wait <= std::time::Duration::from_secs(5)),
                    "Retry-After must be short, got {retry_after:?}"
                );
            }
            other => panic!("expected 503 Overloaded, got {other:?}"),
        }
        assert_eq!(refused.status_code(), 503);
        // Another model on the same member still fits under total=2.
        let other = admission
            .reserve(&claim("anonymous", "p", "n", 1, None))
            .await
            .expect("second member slot");
        assert!(matches!(
            admission
                .reserve(&claim("anonymous", "p", "o", 1, None))
                .await,
            Err(EngineError::Overloaded { .. })
        ));
        admission.settle(
            &held,
            &AdmissionOutcome {
                succeeded: false,
                usage: None,
                cost: None,
            },
        );
        admission.settle(
            &other,
            &AdmissionOutcome {
                succeeded: true,
                usage: None,
                cost: None,
            },
        );
        admission
            .reserve(&claim("anonymous", "p", "m", 1, None))
            .await
            .expect("slots freed at settlement");
    }

    #[tokio::test]
    async fn rekey_moves_the_in_flight_slot_to_the_member_being_tried() {
        let admission = PolicyAdmission::new(
            HashMap::new(),
            HashMap::new(),
            HashMap::from([
                ("a".to_owned(), (Some(1), HashMap::new())),
                ("b".to_owned(), (Some(1), HashMap::new())),
            ]),
        );
        let ticket = admission
            .reserve(&claim("anonymous", "a", "m", 1, None))
            .await
            .expect("slot on a");

        // Re-keying to the member already held is a no-op, not a second
        // slot: the cap of 1 is still saturated.
        admission
            .rekey(&ticket, "a", "m")
            .await
            .expect("same member: nothing to move");
        assert!(matches!(
            admission
                .reserve(&claim("anonymous", "a", "m", 1, None))
                .await,
            Err(EngineError::Overloaded { .. })
        ));

        // The ladder fails over to "b": the slot moves.
        admission
            .rekey(&ticket, "b", "m")
            .await
            .expect("the slot moves to b");
        // "a" was released by the move, so a fresh reservation fits there —
        // the failed-over request no longer squats on its first member.
        let second = admission
            .reserve(&claim("anonymous", "a", "m", 1, None))
            .await
            .expect("a's slot was released by the move");
        // "b" is held by the moved lease, so moving the second lease there
        // is refused.
        assert!(matches!(
            admission.rekey(&second, "b", "m").await,
            Err(EngineError::Overloaded { .. })
        ));

        // Settling the moved lease frees "b", the member it ended on.
        admission.settle(
            &ticket,
            &AdmissionOutcome {
                succeeded: false,
                usage: None,
                cost: None,
            },
        );
        admission
            .rekey(&second, "b", "m")
            .await
            .expect("settlement freed b, the member the lease ended on");
        admission.settle(
            &second,
            &AdmissionOutcome {
                succeeded: false,
                usage: None,
                cost: None,
            },
        );
    }

    #[tokio::test]
    async fn two_admissions_over_one_backend_share_one_budget() {
        // The plan 09 property, at the seam: two PolicyAdmission instances
        // (two replicas) built over one shared backend enforce one budget
        // between them, without either seeing the other's process.
        let backend: Arc<dyn AdmissionBackend> = Arc::new(MemoryAdmissionBackend::new());
        let keys = HashMap::from([("key".to_owned(), budgeted("1"))]);
        let pricing =
            HashMap::from([(("p".to_owned(), "m".to_owned()), pricing("0.001", "0.001"))]);
        let replica_a = Arc::new(PolicyAdmission::over_backend(
            keys.clone(),
            pricing.clone(),
            HashMap::new(),
            Arc::clone(&backend),
        ));
        let replica_b = Arc::new(PolicyAdmission::over_backend(
            keys,
            pricing,
            HashMap::new(),
            Arc::clone(&backend),
        ));

        let ticket = replica_a
            .reserve(&claim("key", "p", "m", 500, Some(500)))
            .await
            .expect("replica a reserves the whole budget");
        let refused = replica_b
            .reserve(&claim("key", "p", "m", 1, Some(1)))
            .await
            .expect_err("replica b must see replica a's reservation");
        assert!(matches!(refused, EngineError::BudgetExhausted { .. }));

        // Refunding on one replica releases it for the other.
        replica_a.settle(
            &ticket,
            &AdmissionOutcome {
                succeeded: false,
                usage: None,
                cost: None,
            },
        );
        replica_b
            .reserve(&claim("key", "p", "m", 500, Some(500)))
            .await
            .expect("the refund is visible to replica b");
    }

    #[test]
    fn noop_detection_ignores_policies_that_restrict_nothing() {
        let empty = PolicyAdmission::new(
            HashMap::from([("key".to_owned(), KeyPolicy::default())]),
            HashMap::new(),
            HashMap::new(),
        );
        assert!(empty.is_noop());
        assert!(!admission(budgeted("1")).is_noop());
    }

    #[tokio::test]
    async fn a_seeded_day_budget_refuses_when_spent() {
        // Fixed epoch for both the seed and the windows, so the day
        // boundary cannot move under the test (midnight UTC or otherwise).
        const NOW: i64 = 1_700_000_000;
        let today = OffsetDateTime::from_unix_timestamp(NOW).expect("timestamp");
        let backend = Arc::new(MemoryAdmissionBackend::new());
        // Startup seeding: today's ledger already shows the whole 1 USD
        // budget spent (1_000_000 micro-USD) across 4 requests.
        backend.seed_day(today, "key", 4, 1_000_000);
        let admission = PolicyAdmission::over_backend(
            HashMap::from([("key".to_owned(), budgeted("1"))]),
            HashMap::from([(("p".to_owned(), "m".to_owned()), pricing("0.001", "0.001"))]),
            HashMap::new(),
            Arc::clone(&backend) as Arc<dyn AdmissionBackend>,
        )
        .with_clock(move || OffsetDateTime::from_unix_timestamp(NOW).expect("timestamp"));

        let refused = admission
            .reserve(&claim("key", "p", "m", 500, Some(500)))
            .await
            .expect_err("the seeded spend consumed the whole day budget");
        assert!(matches!(refused, EngineError::BudgetExhausted { .. }));
        assert_eq!(refused.status_code(), 402);

        // The same seed dated yesterday rolls over: today starts empty.
        backend.seed_day(
            today - std::time::Duration::from_secs(86_400),
            "key",
            4,
            1_000_000,
        );
        admission
            .reserve(&claim("key", "p", "m", 500, Some(500)))
            .await
            .expect("yesterday's seeded spend must not carry into today");

        // The seeded request count feeds the rpd window: at its cap, refuse.
        let counted = Arc::new(MemoryAdmissionBackend::new());
        counted.seed_day(today, "key", 3, 0);
        let limited = PolicyAdmission::over_backend(
            HashMap::from([(
                "key".to_owned(),
                KeyPolicy {
                    limits: KeyWindowLimits {
                        rpd: Some(3),
                        ..KeyWindowLimits::default()
                    },
                    ..KeyPolicy::default()
                },
            )]),
            HashMap::new(),
            HashMap::new(),
            Arc::clone(&counted) as Arc<dyn AdmissionBackend>,
        )
        .with_clock(move || OffsetDateTime::from_unix_timestamp(NOW).expect("timestamp"));
        let refused = limited
            .reserve(&claim("key", "p", "m", 10, Some(10)))
            .await
            .expect_err("the seeded day already used every rpd slot");
        assert!(
            matches!(&refused, EngineError::RateLimited { message, .. } if message.contains("rpd")),
            "expected the seeded rpd counter to refuse, got {refused:?}"
        );
    }
}
