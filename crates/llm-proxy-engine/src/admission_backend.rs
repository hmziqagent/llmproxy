//! The shared-state seam behind [`Admission`](crate::Admission) (plan 09
//! step 1, audit conflict 13): one trait, [`AdmissionBackend`], owns every
//! limit, window, budget and concurrency counter the one admission step
//! enforces. The default backend is the in-process
//! [`MemoryAdmissionBackend`] (one `Mutex`, exactly the pre-plan-09
//! behaviour); the optional `llm-proxy-store-redis` and
//! `llm-proxy-store-postgres` crates implement the same trait against a
//! shared store so replicas enforce one budget between them. The rules —
//! allowlists, estimates, refusals, statuses — stay in
//! [`PolicyAdmission`](crate::PolicyAdmission), so no backend can drift
//! into its own admission policy.
//!
//! Money crosses this seam as [`MicroUsd`] (scaled fixed-point, 1e-6 USD),
//! never a float, matching the workspace's money rule.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use time::{Date, OffsetDateTime};

use crate::error::EngineError;

/// Scaled money unit used by admission backends: micro-USD (`1e-6` USD).
pub type MicroUsd = i64;

/// Converts a fixed-point USD amount to micro-USD, saturating at the
/// `i64` bounds rather than overflowing.
#[must_use]
pub fn to_micro_usd(amount: Decimal) -> MicroUsd {
    const SCALE: u32 = 6;
    let scaled = amount.saturating_mul(Decimal::from(10_u64.pow(SCALE)));
    scaled
        .trunc()
        .to_i64()
        .unwrap_or(if scaled.is_sign_negative() {
            MicroUsd::MIN
        } else {
            MicroUsd::MAX
        })
}

/// Converts micro-USD back to a fixed-point USD amount.
#[must_use]
pub fn from_micro_usd(amount: MicroUsd) -> Decimal {
    Decimal::from(amount) / Decimal::from(1_000_000)
}

/// An opaque handle to one backend reservation. Backends choose the id
/// shape (an in-process counter, a UUID, a store key); it is only handed
/// back to the backend that issued it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LeaseId(Arc<str>);

impl LeaseId {
    /// Wraps a backend-chosen id string.
    #[must_use]
    pub fn new(id: impl Into<Arc<str>>) -> Self {
        Self(id.into())
    }

    /// The backend-chosen id string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// What one reservation asks a backend to enforce and record, atomically:
/// every check and every counter mutation in [`AdmissionBackend::admit`]
/// happens as one unit, so concurrent reservations cannot overspend a
/// budget by more than what is reserved.
#[derive(Debug, Clone)]
pub struct ReservationDemand<'a> {
    /// The instant the reservation is made at (the wrapper's clock).
    pub now: OffsetDateTime,
    /// The caller identity (key id or `"anonymous"`).
    pub caller: &'a str,
    /// The member provider the reservation starts on.
    pub member: &'a str,
    /// The upstream model the attempt bills as.
    pub model: &'a str,
    /// Requests-per-minute cap, when the key carries one.
    pub rpm: Option<u32>,
    /// Tokens-per-minute cap, when the key carries one.
    pub tpm: Option<u64>,
    /// Requests-per-day cap, when the key carries one.
    pub rpd: Option<u64>,
    /// The key's daily spend budget, in micro-USD, when it carries one.
    pub budget_usd_per_day: Option<MicroUsd>,
    /// The reservation's USD estimate in micro-USD. `Some` only when the
    /// key carries a budget (that is what makes it demand-able); the
    /// backend must refuse when `spent + reserved + estimate` would exceed
    /// the budget.
    pub estimate_usd: Option<MicroUsd>,
    /// Tokens to reserve against the tpm window (input estimate plus
    /// `max_tokens`).
    pub reserved_tokens: u64,
    /// The member's total in-flight cap, when configured.
    pub member_cap: Option<u32>,
    /// The member's per-model in-flight cap, when configured.
    pub model_cap: Option<u32>,
}

impl ReservationDemand<'_> {
    /// Whether any per-caller window or budget applies: only then does the
    /// backend track this caller's counters at all (keys without a policy
    /// are unrestricted, exactly as before plan 09).
    #[must_use]
    pub fn tracks_caller(&self) -> bool {
        self.rpm.is_some()
            || self.tpm.is_some()
            || self.rpd.is_some()
            || self.budget_usd_per_day.is_some()
    }
}

/// The outcome one reservation settled with.
#[derive(Debug, Clone, Copy)]
pub struct Settlement {
    /// The instant settlement happens at (the wrapper's clock).
    pub now: OffsetDateTime,
    /// Whether the request completed.
    pub succeeded: bool,
    /// The provider-reported total token count, when the attempt produced
    /// usage; charged into the minute window the reservation was made in.
    pub usage_tokens: Option<u64>,
    /// What the provider billed, in micro-USD, when a cost was computable;
    /// charged to the day's spend. `None` means unpriced or failed-before-
    /// cost: the reservation release alone is the refund.
    pub charged_usd: Option<MicroUsd>,
}

/// Holds limits, windows, budgets and in-flight counters for the one
/// admission step. Implementations must make each method atomic: a
/// concurrent [`AdmissionBackend::admit`] never observes a half-applied
/// reservation, and a refusal must leave no counter behind.
///
/// [`AdmissionBackend::settle`] is called fire-and-forget from sync code;
/// network-backed implementations may complete asynchronously (for example
/// by spawning the write) but must keep reservations bounded — a lease
/// that is never settled (a crashed replica) must expire on its own so
/// the budget it held is released.
#[async_trait]
pub trait AdmissionBackend: std::fmt::Debug + Send + Sync {
    /// Atomically check every limit in `demand` and commit the
    /// reservation, returning its lease id.
    ///
    /// # Errors
    ///
    /// [`EngineError::RateLimited`] when an rpm/tpm/rpd window is full,
    /// [`EngineError::BudgetExhausted`] when the daily budget would be
    /// overspent, [`EngineError::Overloaded`] when an in-flight cap is
    /// saturated — the same refusals the in-process backend answers with.
    async fn admit(&self, demand: &ReservationDemand<'_>) -> Result<LeaseId, EngineError>;

    /// Release the reservation and charge the outcome. Called exactly once
    /// per successful `admit`; an unknown lease is logged and ignored.
    fn settle(&self, lease: &LeaseId, settlement: &Settlement);

    /// Move the reservation's in-flight slot to `member`/`model` (failover):
    /// release the old slot, check and acquire the new one, atomically.
    ///
    /// # Errors
    ///
    /// [`EngineError::Overloaded`] when the new member or model is at its
    /// cap, in which case the ladder skips it and the old slot stands.
    async fn rekey(
        &self,
        lease: &LeaseId,
        member: &str,
        model: &str,
        member_cap: Option<u32>,
        model_cap: Option<u32>,
    ) -> Result<(), EngineError>;
}

/// Per-minute counters for one caller.
#[derive(Debug, Default)]
struct MinuteWindow {
    minute: i64,
    requests: u64,
    tokens: u64,
    reserved_tokens: u64,
}

/// Per-day counters for one caller.
#[derive(Debug)]
struct DayWindow {
    day: Date,
    requests: u64,
    spent: MicroUsd,
    reserved: MicroUsd,
}

#[derive(Debug)]
struct KeyState {
    minute: MinuteWindow,
    day: DayWindow,
}

impl KeyState {
    fn fresh(now: OffsetDateTime) -> Self {
        Self {
            minute: MinuteWindow {
                minute: unix_minute(now),
                ..MinuteWindow::default()
            },
            day: DayWindow {
                day: now.date(),
                requests: 0,
                spent: 0,
                reserved: 0,
            },
        }
    }

    /// Rolls the windows forward when the clock crossed a boundary.
    fn roll(&mut self, now: OffsetDateTime) {
        let minute = unix_minute(now);
        if self.minute.minute != minute {
            self.minute = MinuteWindow {
                minute,
                ..MinuteWindow::default()
            };
        }
        if self.day.day != now.date() {
            self.day = DayWindow {
                day: now.date(),
                requests: 0,
                spent: 0,
                reserved: 0,
            };
        }
    }
}

/// What one in-process reservation holds, so settlement can release
/// exactly it.
#[derive(Debug)]
struct Lease {
    caller: String,
    member: String,
    model: String,
    minute: i64,
    day: Date,
    reserved_tokens: u64,
    reserved_usd: MicroUsd,
    holds_member_slot: bool,
    holds_model_slot: bool,
}

#[derive(Debug, Default)]
struct State {
    keys: HashMap<String, KeyState>,
    leases: HashMap<String, Lease>,
    inflight_members: HashMap<String, u32>,
    inflight_models: HashMap<(String, String), u32>,
}

/// The default [`AdmissionBackend`]: every counter in one process, behind
/// one `Mutex`, so a reservation is atomic across all four rules. This is
/// exactly the admission state the proxy kept before the replica stores
/// existed; a single node runs on it with no external services.
#[derive(Debug, Default)]
pub struct MemoryAdmissionBackend {
    state: std::sync::Mutex<State>,
    next_lease: std::sync::atomic::AtomicU64,
}

impl MemoryAdmissionBackend {
    /// An empty backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("admission backend state poisoned; recovering");
            poisoned.into_inner()
        })
    }

    /// Seeds one caller's day window with counts read back from durable
    /// state at startup (the usage ledger): today's paid-request count and
    /// spend, so a restart cannot reset a `usd_per_day` budget or an rpd
    /// counter. Only the day window is seeded — minute windows cover 60
    /// seconds, shorter than any restart gap, so rpm/tpm restart empty.
    /// Seeding runs before the server accepts traffic, so the values replace
    /// whatever the fresh window holds.
    pub fn seed_day(&self, now: OffsetDateTime, caller: &str, requests: u64, spent: MicroUsd) {
        let mut state = self.lock();
        let key = state
            .keys
            .entry(caller.to_owned())
            .or_insert_with(|| KeyState::fresh(now));
        key.roll(now);
        key.day.requests = requests;
        key.day.spent = spent;
    }
}

/// Seconds until the next calendar minute starts.
pub(crate) fn until_next_minute(now: OffsetDateTime) -> Duration {
    let elapsed = now.unix_timestamp().rem_euclid(60);
    Duration::from_secs((60 - elapsed).clamp(1, 60).unsigned_abs())
}

/// Seconds until the next UTC day starts.
pub(crate) fn until_next_day(now: OffsetDateTime) -> Duration {
    let (hours, minutes, seconds) = now.time().as_hms();
    let elapsed = i64::from(hours) * 3600 + i64::from(minutes) * 60 + i64::from(seconds);
    Duration::from_secs((86_400 - elapsed).clamp(1, 86_400).unsigned_abs())
}

pub(crate) fn unix_minute(now: OffsetDateTime) -> i64 {
    now.unix_timestamp().div_euclid(60)
}

fn member_saturated(member: &str, cap: u32) -> EngineError {
    EngineError::Overloaded {
        message: format!("provider \"{member}\" is at its in-flight cap of {cap}"),
        retry_after: Some(Duration::from_secs(1)),
    }
}

fn model_saturated(model: &str, member: &str, cap: u32) -> EngineError {
    EngineError::Overloaded {
        message: format!("model \"{model}\" on \"{member}\" is at its in-flight cap of {cap}"),
        retry_after: Some(Duration::from_secs(1)),
    }
}

#[async_trait]
impl AdmissionBackend for MemoryAdmissionBackend {
    #[expect(
        clippy::too_many_lines,
        reason = "the four rules are checked and committed under one lock; splitting them would scatter the atomic reservation"
    )]
    async fn admit(&self, demand: &ReservationDemand<'_>) -> Result<LeaseId, EngineError> {
        let now = demand.now;
        let mut state = self.lock();
        let State {
            keys,
            leases,
            inflight_members,
            inflight_models,
        } = &mut *state;

        if demand.tracks_caller() {
            let key = keys
                .entry(demand.caller.to_owned())
                .or_insert_with(|| KeyState::fresh(now));
            key.roll(now);
            if let Some(rpm) = demand.rpm
                && key.minute.requests >= u64::from(rpm)
            {
                return Err(EngineError::RateLimited {
                    message: format!("rpm limit {rpm} reached for this key"),
                    retry_after: Some(until_next_minute(now)),
                });
            }
            if let Some(rpd) = demand.rpd
                && key.day.requests >= rpd
            {
                return Err(EngineError::RateLimited {
                    message: format!("rpd limit {rpd} reached for this key"),
                    retry_after: Some(until_next_day(now)),
                });
            }
            if let Some(tpm) = demand.tpm {
                let projected = key
                    .minute
                    .tokens
                    .saturating_add(key.minute.reserved_tokens)
                    .saturating_add(demand.reserved_tokens);
                if projected > tpm {
                    return Err(EngineError::RateLimited {
                        message: format!("tpm limit {tpm} would be exceeded for this key"),
                        retry_after: Some(until_next_minute(now)),
                    });
                }
            }
            if let (Some(budget), Some(estimate)) = (demand.budget_usd_per_day, demand.estimate_usd)
            {
                let committed = key.day.spent.saturating_add(key.day.reserved);
                let projected = committed.saturating_add(estimate);
                if projected > budget {
                    return Err(EngineError::BudgetExhausted {
                        message: format!(
                            "daily budget of {} USD would be exceeded for this key (spent {}, \
                             reserved in flight {}, this request {})",
                            from_micro_usd(budget).normalize(),
                            from_micro_usd(key.day.spent).normalize(),
                            from_micro_usd(key.day.reserved).normalize(),
                            from_micro_usd(estimate).normalize(),
                        ),
                        retry_after: Some(until_next_day(now)),
                    });
                }
            }
        }

        // Concurrency caps on the member and its model.
        let model_key = (demand.member.to_owned(), demand.model.to_owned());
        if let Some(cap) = demand.member_cap {
            let inflight = inflight_members.get(demand.member).copied().unwrap_or(0);
            if inflight >= cap {
                return Err(member_saturated(demand.member, cap));
            }
        }
        if let Some(cap) = demand.model_cap {
            let inflight = inflight_models.get(&model_key).copied().unwrap_or(0);
            if inflight >= cap {
                return Err(model_saturated(demand.model, demand.member, cap));
            }
        }

        // Every check passed: commit the whole reservation at once.
        let mut lease = Lease {
            caller: demand.caller.to_owned(),
            member: demand.member.to_owned(),
            model: demand.model.to_owned(),
            minute: unix_minute(now),
            day: now.date(),
            reserved_tokens: 0,
            reserved_usd: 0,
            holds_member_slot: demand.member_cap.is_some(),
            holds_model_slot: demand.model_cap.is_some(),
        };
        if demand.tracks_caller()
            && let Some(key) = keys.get_mut(demand.caller)
        {
            key.minute.requests += 1;
            key.day.requests += 1;
            key.minute.reserved_tokens = key
                .minute
                .reserved_tokens
                .saturating_add(demand.reserved_tokens);
            lease.reserved_tokens = demand.reserved_tokens;
            if let Some(estimate) = demand.estimate_usd {
                key.day.reserved = key.day.reserved.saturating_add(estimate);
                lease.reserved_usd = estimate;
            }
        }
        if lease.holds_member_slot {
            *inflight_members
                .entry(demand.member.to_owned())
                .or_insert(0) += 1;
        }
        if lease.holds_model_slot {
            *inflight_models.entry(model_key).or_insert(0) += 1;
        }
        let id = self
            .next_lease
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let lease_id = LeaseId::new(id.to_string());
        leases.insert(lease_id.as_str().to_owned(), lease);
        Ok(lease_id)
    }

    fn settle(&self, lease: &LeaseId, settlement: &Settlement) {
        let now = settlement.now;
        let mut state = self.lock();
        let Some(lease_record) = state.leases.remove(lease.as_str()) else {
            tracing::warn!(
                lease = lease.as_str(),
                "admission settle for an unknown lease; ignoring"
            );
            return;
        };
        if lease_record.holds_member_slot
            && let Some(count) = state.inflight_members.get_mut(&lease_record.member)
        {
            *count = count.saturating_sub(1);
        }
        if lease_record.holds_model_slot
            && let Some(count) = state
                .inflight_models
                .get_mut(&(lease_record.member.clone(), lease_record.model.clone()))
        {
            *count = count.saturating_sub(1);
        }
        let Some(key) = state.keys.get_mut(&lease_record.caller) else {
            return;
        };
        key.roll(now);
        // Tokens: release the estimate and charge the provider-reported
        // total, but only into the minute the request was admitted in — a
        // window that already rolled over has no limit left to protect.
        if key.minute.minute == lease_record.minute {
            key.minute.reserved_tokens = key
                .minute
                .reserved_tokens
                .saturating_sub(lease_record.reserved_tokens);
            if let Some(tokens) = settlement.usage_tokens {
                key.minute.tokens = key.minute.tokens.saturating_add(tokens);
            }
        }
        // Money: release the reservation from the day it was made (if that
        // day is still current) and charge what the provider billed to
        // today — spend is never dropped at a midnight boundary.
        if key.day.day == lease_record.day {
            key.day.reserved = key
                .day
                .reserved
                .saturating_sub(lease_record.reserved_usd)
                .max(0);
        }
        if let Some(charged) = settlement.charged_usd {
            key.day.spent = key.day.spent.saturating_add(charged);
        }
    }

    async fn rekey(
        &self,
        lease: &LeaseId,
        member: &str,
        model: &str,
        member_cap: Option<u32>,
        model_cap: Option<u32>,
    ) -> Result<(), EngineError> {
        let mut state = self.lock();
        let (old_member, old_model, held_member_slot, held_model_slot) = {
            let Some(lease_record) = state.leases.get(lease.as_str()) else {
                tracing::warn!(
                    lease = lease.as_str(),
                    "admission rekey for an unknown lease; ignoring"
                );
                return Ok(());
            };
            if lease_record.member == member && lease_record.model == model {
                return Ok(());
            }
            (
                lease_record.member.clone(),
                lease_record.model.clone(),
                lease_record.holds_member_slot,
                lease_record.holds_model_slot,
            )
        };
        if let Some(cap) = member_cap {
            let inflight = state.inflight_members.get(member).copied().unwrap_or(0);
            if inflight >= cap {
                return Err(member_saturated(member, cap));
            }
        }
        if let Some(cap) = model_cap {
            let key = (member.to_owned(), model.to_owned());
            let inflight = state.inflight_models.get(&key).copied().unwrap_or(0);
            if inflight >= cap {
                return Err(model_saturated(model, member, cap));
            }
        }
        if held_member_slot && let Some(count) = state.inflight_members.get_mut(&old_member) {
            *count = count.saturating_sub(1);
        }
        if held_model_slot
            && let Some(count) = state.inflight_models.get_mut(&(old_member, old_model))
        {
            *count = count.saturating_sub(1);
        }
        if let Some(lease_record) = state.leases.get_mut(lease.as_str()) {
            lease_record.member = member.to_owned();
            lease_record.model = model.to_owned();
            lease_record.holds_member_slot = member_cap.is_some();
            lease_record.holds_model_slot = model_cap.is_some();
        }
        if member_cap.is_some() {
            *state.inflight_members.entry(member.to_owned()).or_insert(0) += 1;
        }
        if model_cap.is_some() {
            *state
                .inflight_models
                .entry((member.to_owned(), model.to_owned()))
                .or_insert(0) += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn micro_usd_conversion_round_trips_within_scale() {
        let amount = Decimal::new(1_234_567, 6); // 1.234567 USD
        let micros = to_micro_usd(amount);
        assert_eq!(micros, 1_234_567);
        assert_eq!(from_micro_usd(micros).normalize(), amount.normalize());
    }

    #[test]
    fn micro_usd_conversion_saturates_instead_of_overflowing() {
        let huge = Decimal::from(i64::MAX) * Decimal::from(10);
        assert_eq!(to_micro_usd(huge), MicroUsd::MAX);
        let tiny = Decimal::from(i64::MIN) * Decimal::from(10);
        assert_eq!(to_micro_usd(tiny), MicroUsd::MIN);
    }

    #[test]
    fn micro_usd_truncates_sub_micro_amounts_downward() {
        let amount = Decimal::new(15_005, 5); // 0.15005 USD
        assert_eq!(to_micro_usd(amount), 150_050);
        // 0.1500005 USD truncates to 0.150000.
        let sub = Decimal::new(1_500_005, 7);
        assert_eq!(to_micro_usd(sub), 150_000);
    }
}
