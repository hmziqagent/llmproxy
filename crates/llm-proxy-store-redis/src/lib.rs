//! Replica-shared proxy state in Redis (plan 09 step 1, audit conflict
//! 13): the admission counters, upstream cooldowns and usage events of
//! every instance that points at the same Redis, so two replicas enforce
//! one budget between them.
//!
//! Everything here implements an engine trait —
//! [`AdmissionBackend`](llm_proxy_engine::AdmissionBackend) through
//! [`RedisAdmissionBackend`], [`Cooldowns`](llm_proxy_engine::Cooldowns)
//! through [`RedisCooldowns`], and
//! [`EventBus`](llm_proxy_storage::EventBus) through [`RedisEventBus`] —
//! so the rules (allowlists, estimates, statuses) stay in the engine and
//! no second admission policy exists. Use [`replica_admission`] to build
//! the engine-side [`PolicyAdmission`](llm_proxy_engine::PolicyAdmission)
//! over this store.
//!
//! # Atomicity model
//!
//! Redis executes one command atomically, and every counter move here is
//! a single `INCR`/`INCRBY`. A reservation is *reserve, check, roll back
//! on refusal*: between the increment and the check another replica can
//! briefly observe a counter that is about to be rolled back. The error
//! direction is fail-closed — a concurrent admission may see a fuller
//! window than it truly has and refuse — and no two admitted requests can
//! both fit a budget that only fits one, because the check runs after the
//! increments of both are visible. Leases that are never settled (a
//! crashed replica) expire on a TTL, so the budget they held is released
//! without operator action.
//!
//! Settlement is ordered, not transactional. An admission reads the
//! reserved counter (the `INCR` reply) strictly before it reads `spent`,
//! so a settle charges spend first and releases the reservation second:
//! no interleaving of a settle and an admission can observe the
//! reservation released while the charge is still missing — the one
//! state that would let an admission under-count a request that is
//! about to be billed. The other direction, a brief double count of
//! `spent + reserved`, only makes the budget briefly stricter for the
//! settle window, which is fail-closed. The token window makes its
//! release and charge in one net `INCRBY`, so no dipped value exists at
//! all. Settle-vs-admit is covered by this ordering, not by a
//! transaction. A settle claims its lease by the `DEL` reply count, so
//! two concurrent settles of one lease release its counters exactly
//! once.
//!
//! Money moves as integer micro-USD (`INCRBY`), never a float, matching
//! the engine's money rule.

#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use llm_proxy_engine::{
    AdmissionBackend, EngineError, KeyPolicy, LeaseId, MicroUsd, PolicyAdmission,
    ReservationDemand, Settlement,
};
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

mod cooldowns;
mod event_bus;

/// An in-process Redis server speaking real RESP2 over TCP (the
/// `test-server` feature): for this crate's tests and downstream
/// integration tests in environments with no Redis binary. Never enabled
/// in production builds.
#[cfg(feature = "test-server")]
pub mod test_server;

pub use cooldowns::RedisCooldowns;
pub use event_bus::RedisEventBus;

/// Errors from the Redis replica store.
#[derive(Debug, thiserror::Error)]
pub enum RedisStoreError {
    /// The configured URL is not a valid Redis connection info.
    #[error("invalid Redis URL: {0}")]
    InvalidUrl(String),
    /// Connecting to Redis failed.
    #[error("connecting to Redis: {0}")]
    Connect(redis::RedisError),
    /// A command against Redis failed.
    #[error("Redis command failed: {0}")]
    Command(#[from] redis::RedisError),
    /// A stored lease record could not be decoded.
    #[error("stored admission lease is corrupt: {0}")]
    CorruptLease(#[from] serde_json::Error),
}

/// A shared connection to the replica Redis plus the key prefix every
/// instance must agree on. Clone-cheap; the underlying connection manager
/// reconnects on failure.
#[derive(Debug, Clone)]
pub struct RedisStore {
    conn: ConnectionManager,
    prefix: Arc<str>,
}

impl RedisStore {
    /// Connect to `url` (for example `redis://127.0.0.1:6379`). Every
    /// replica must use the same `prefix` to share state; different
    /// prefixes are independent deployments.
    ///
    /// # Errors
    ///
    /// [`RedisStoreError`] when the URL is invalid or the connection
    /// cannot be established.
    pub async fn connect(url: &str, prefix: &str) -> Result<Self, RedisStoreError> {
        let client = redis::Client::open(url)
            .map_err(|error| RedisStoreError::InvalidUrl(format!("{url}: {error}")))?;
        let conn = client
            .get_connection_manager()
            .await
            .map_err(RedisStoreError::Connect)?;
        Ok(Self {
            conn,
            prefix: if prefix.is_empty() {
                "llm-proxy".to_owned()
            } else {
                prefix.to_owned()
            }
            .into(),
        })
    }
}

/// The persisted form of one reservation: everything settlement needs to
/// release exactly what was reserved, stored under the lease id with a
/// TTL so a crashed replica's reservation cannot hold a budget forever.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeaseRecord {
    caller: String,
    member: String,
    model: String,
    minute: i64,
    day: String,
    reserved_tokens: u64,
    reserved_usd: MicroUsd,
    holds_member_slot: bool,
    holds_model_slot: bool,
}

/// How long an unsettled lease holds its reservation before expiring.
const LEASE_TTL: Duration = Duration::from_secs(3600);
/// Counters for a calendar minute outlive the minute itself by this much.
const MINUTE_TTL: Duration = Duration::from_secs(120);
/// Day counters outlive the day so late settlements still land.
const DAY_TTL: Duration = Duration::from_secs(48 * 3600);
/// In-flight slots expire only as a crash leak bound.
const INFLIGHT_TTL: Duration = Duration::from_secs(3600);

fn day_stamp(now: OffsetDateTime) -> String {
    format!(
        "{:04}{:02}{:02}",
        now.year(),
        u8::from(now.month()),
        now.day()
    )
}

fn unix_minute(now: OffsetDateTime) -> i64 {
    now.unix_timestamp().div_euclid(60)
}

fn until_next_minute(now: OffsetDateTime) -> Duration {
    let elapsed = now.unix_timestamp().rem_euclid(60);
    Duration::from_secs((60 - elapsed).clamp(1, 60).unsigned_abs())
}

fn until_next_day(now: OffsetDateTime) -> Duration {
    let (hours, minutes, seconds) = now.time().as_hms();
    let elapsed = i64::from(hours) * 3600 + i64::from(minutes) * 60 + i64::from(seconds);
    Duration::from_secs((86_400 - elapsed).clamp(1, 86_400).unsigned_abs())
}

fn rate_limited(message: String, retry_after: Duration) -> EngineError {
    EngineError::RateLimited {
        message,
        retry_after: Some(retry_after),
    }
}

fn store_error(error: &redis::RedisError) -> EngineError {
    EngineError::Internal(format!("admission store: {error}"))
}

/// Which committed increments a refused admission must still roll back.
/// Every refusal first undoes its own check's increment inline; this
/// releases the counters the checks *before* it committed, so a refusal
/// leaves no counter behind.
#[derive(Clone, Copy, Default)]
struct CommittedWindows {
    rpm: bool,
    rpd: bool,
    tpm: bool,
    reserved_usd: bool,
}

impl CommittedWindows {
    /// Everything `admit_windows` commits on success — what a concurrency
    /// refusal after the window checks must roll back.
    const ALL: Self = Self {
        rpm: true,
        rpd: true,
        tpm: true,
        reserved_usd: true,
    };
}

/// The Redis-backed [`AdmissionBackend`]: windows, budgets, concurrency
/// caps and leases held in the shared store.
#[derive(Debug, Clone)]
pub struct RedisAdmissionBackend {
    store: RedisStore,
}

impl RedisAdmissionBackend {
    /// Over a connected store.
    #[must_use]
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    /// The settlement write, as an async operation: the sync
    /// [`AdmissionBackend::settle`] spawns it, and callers that need the
    /// write to have happened await this directly.
    ///
    /// # Errors
    ///
    /// [`RedisStoreError`] when Redis fails; the lease is deleted before
    /// any counter moves, so a failure here leaks nothing but may leave
    /// the reservation in flight until its TTL.
    pub async fn settle_async(
        &self,
        lease: &LeaseId,
        settlement: &Settlement,
    ) -> Result<(), RedisStoreError> {
        let record_key = self.lease_key(lease);
        let mut conn = self.store.conn.clone();
        let raw: Option<String> = conn.get(&record_key).await?;
        let Some(raw) = raw else {
            tracing::warn!(
                lease = lease.as_str(),
                "admission settle for an unknown or expired lease; ignoring"
            );
            return Ok(());
        };
        let record: LeaseRecord = serde_json::from_str(&raw)?;
        // Claim the lease by the delete count: exactly one concurrent
        // settle of this lease gets past this line, so its counters are
        // released once, never twice.
        let removed: i64 = conn.del(&record_key).await?;
        if removed == 0 {
            return Ok(());
        }

        // Money: charge to today first, release from the day it was
        // reserved on second (spend is never dropped at a midnight
        // boundary). Spent is charged before the reservation is released
        // so a concurrent admission — which reads `reserved` strictly
        // before `spent` — can only over-count the exposure, never see
        // the reservation gone while the charge is missing (see the
        // atomicity model above).
        if let Some(charged) = settlement.charged_usd {
            let today = self.spent_key(&record.caller, &day_stamp(settlement.now));
            let _: i64 = conn.incr(&today, charged).await?;
            let _: bool = conn.expire(&today, DAY_TTL.as_secs() as i64).await?;
        }
        if record.reserved_usd != 0 {
            let key = self.reserved_key(&record.caller, &record.day);
            if conn.exists(&key).await? {
                let _: i64 = conn.decr(&key, record.reserved_usd).await?;
            }
        }

        // Token window: release the estimate and charge the reported
        // total as one net `INCRBY` into the minute the reservation was
        // made in, so a concurrent admission never observes a dipped
        // window between the release and the charge. The key may have
        // expired with the minute; a nonzero net move recreates it, and
        // charging into a gone window is how the usage lands.
        let token_key = self.token_key(&record.caller, record.minute);
        let key_existed = record.reserved_tokens > 0 && conn.exists(&token_key).await?;
        let delta = settlement.usage_tokens.map_or(0, |tokens| tokens as i64)
            - if key_existed {
                record.reserved_tokens as i64
            } else {
                0
            };
        if delta != 0 || key_existed {
            let _: i64 = conn.incr(&token_key, delta).await?;
        }
        if settlement.usage_tokens.is_some() {
            let _: bool = conn.expire(&token_key, MINUTE_TTL.as_secs() as i64).await?;
        }

        self.release_slots(&mut conn, &record).await?;
        Ok(())
    }

    fn lease_key(&self, lease: &LeaseId) -> String {
        format!("{}:lease:{}", self.store.prefix, lease.as_str())
    }

    fn token_key(&self, caller: &str, minute: i64) -> String {
        format!("{}:ad:tok:{caller}:{minute}", self.store.prefix)
    }

    fn reserved_key(&self, caller: &str, day: &str) -> String {
        format!("{}:ad:resv:{caller}:{day}", self.store.prefix)
    }

    fn spent_key(&self, caller: &str, day: &str) -> String {
        format!("{}:ad:spent:{caller}:{day}", self.store.prefix)
    }

    fn member_slot_key(&self, member: &str) -> String {
        format!("{}:ad:infl:m:{member}", self.store.prefix)
    }

    fn model_slot_key(&self, member: &str, model: &str) -> String {
        format!("{}:ad:infl:mm:{member}:{model}", self.store.prefix)
    }

    async fn release_slots(
        &self,
        conn: &mut ConnectionManager,
        record: &LeaseRecord,
    ) -> Result<(), RedisStoreError> {
        if record.holds_member_slot {
            let key = self.member_slot_key(&record.member);
            if conn.exists(&key).await? {
                let _: i64 = conn.decr(&key, 1).await?;
            }
        }
        if record.holds_model_slot {
            let key = self.model_slot_key(&record.member, &record.model);
            if conn.exists(&key).await? {
                let _: i64 = conn.decr(&key, 1).await?;
            }
        }
        Ok(())
    }

    /// The window and budget checks of one admission (rpm, rpd, tpm, the
    /// daily budget), each a reserve-check-roll-back sequence on the same
    /// connection; a refusal leaves no counter behind.
    async fn admit_windows(
        &self,
        conn: &mut ConnectionManager,
        demand: &ReservationDemand<'_>,
        minute: i64,
        day: &str,
    ) -> Result<(), EngineError> {
        if demand.tracks_caller() {
            // rpm: one request count per calendar minute.
            if let Some(rpm) = demand.rpm {
                let key = format!("{}:ad:rq:m:{}:{}", self.store.prefix, demand.caller, minute);
                let count: i64 = conn
                    .incr(&key, 1)
                    .await
                    .map_err(|error| store_error(&error))?;
                let _: bool = conn
                    .expire(&key, MINUTE_TTL.as_secs() as i64)
                    .await
                    .map_err(|error| store_error(&error))?;
                if count > i64::from(rpm) {
                    let _: i64 = conn
                        .decr(&key, 1)
                        .await
                        .map_err(|error| store_error(&error))?;
                    return Err(rate_limited(
                        format!("rpm limit {rpm} reached for this key"),
                        until_next_minute(demand.now),
                    ));
                }
            }
            // rpd: one request count per UTC day.
            if let Some(rpd) = demand.rpd {
                let key = format!("{}:ad:rq:d:{}:{}", self.store.prefix, demand.caller, day);
                let count: i64 = conn
                    .incr(&key, 1)
                    .await
                    .map_err(|error| store_error(&error))?;
                let _: bool = conn
                    .expire(&key, DAY_TTL.as_secs() as i64)
                    .await
                    .map_err(|error| store_error(&error))?;
                if count > i64::try_from(rpd).unwrap_or(i64::MAX) {
                    let _: i64 = conn
                        .decr(&key, 1)
                        .await
                        .map_err(|error| store_error(&error))?;
                    // The rpm increment above was already committed.
                    self.rollback_windows(
                        conn,
                        demand,
                        CommittedWindows {
                            rpm: true,
                            ..CommittedWindows::default()
                        },
                    )
                    .await;
                    return Err(rate_limited(
                        format!("rpd limit {rpd} reached for this key"),
                        until_next_day(demand.now),
                    ));
                }
            }
            // tpm: token demand (reserved now, replaced by usage at settle).
            if let Some(tpm) = demand.tpm {
                let key = self.token_key(demand.caller, minute);
                let tokens: i64 = conn
                    .incr(&key, demand.reserved_tokens as i64)
                    .await
                    .map_err(|error| store_error(&error))?;
                let _: bool = conn
                    .expire(&key, MINUTE_TTL.as_secs() as i64)
                    .await
                    .map_err(|error| store_error(&error))?;
                if tokens > 0 && u64::try_from(tokens).unwrap_or(0) > tpm {
                    let _: i64 = conn
                        .decr(&key, demand.reserved_tokens as i64)
                        .await
                        .map_err(|error| store_error(&error))?;
                    // The rpm and rpd increments above were already committed.
                    self.rollback_windows(
                        conn,
                        demand,
                        CommittedWindows {
                            rpm: true,
                            rpd: true,
                            ..CommittedWindows::default()
                        },
                    )
                    .await;
                    return Err(rate_limited(
                        format!("tpm limit {tpm} would be exceeded for this key"),
                        until_next_minute(demand.now),
                    ));
                }
            }
            self.admit_budget(conn, demand, day).await?;
        }
        Ok(())
    }

    /// The daily-budget check: reserve the estimate, then refuse (and
    /// roll the reservation back) when `spent + reserved + estimate`
    /// would exceed the day's budget.
    async fn admit_budget(
        &self,
        conn: &mut ConnectionManager,
        demand: &ReservationDemand<'_>,
        day: &str,
    ) -> Result<(), EngineError> {
        // Budget: spent + reserved + estimate must fit the day.
        if let (Some(budget), Some(estimate)) = (demand.budget_usd_per_day, demand.estimate_usd) {
            let resv = self.reserved_key(demand.caller, day);
            let reserved: i64 = conn
                .incr(&resv, estimate)
                .await
                .map_err(|error| store_error(&error))?;
            let _: bool = conn
                .expire(&resv, DAY_TTL.as_secs() as i64)
                .await
                .map_err(|error| store_error(&error))?;
            let spent: Option<i64> = conn
                .get(self.spent_key(demand.caller, day))
                .await
                .map_err(|error| store_error(&error))?;
            let spent = spent.unwrap_or(0);
            if spent.saturating_add(reserved) > budget {
                let _: i64 = conn
                    .decr(&resv, estimate)
                    .await
                    .map_err(|error| store_error(&error))?;
                // The rpm, rpd and tpm increments from admit_windows were
                // already committed on this connection.
                self.rollback_windows(
                    conn,
                    demand,
                    CommittedWindows {
                        rpm: true,
                        rpd: true,
                        tpm: true,
                        reserved_usd: false,
                    },
                )
                .await;
                return Err(EngineError::BudgetExhausted {
                    message: format!(
                        "daily budget of {} USD would be exceeded for this key (spent {}, \
                 reserved in flight {}, this request {})",
                        llm_proxy_engine::from_micro_usd(budget).normalize(),
                        llm_proxy_engine::from_micro_usd(spent).normalize(),
                        llm_proxy_engine::from_micro_usd((reserved - estimate).max(0)).normalize(),
                        llm_proxy_engine::from_micro_usd(estimate).normalize(),
                    ),
                    retry_after: Some(until_next_day(demand.now)),
                });
            }
        }
        Ok(())
    }

    /// The concurrency-cap checks of one admission; a refusal rolls the
    /// window increments back — the request never happened.
    async fn admit_caps(
        &self,
        conn: &mut ConnectionManager,
        demand: &ReservationDemand<'_>,
    ) -> Result<(), EngineError> {
        // Concurrency caps on the member and its model. A refusal rolls
        // the window increments back — the request never happened.
        if demand.member_cap.is_some() {
            let key = self.member_slot_key(demand.member);
            let count: i64 = conn
                .incr(&key, 1)
                .await
                .map_err(|error| store_error(&error))?;
            let _: bool = conn
                .expire(&key, INFLIGHT_TTL.as_secs() as i64)
                .await
                .map_err(|error| store_error(&error))?;
            if demand.member_cap.is_some_and(|cap| count > i64::from(cap)) {
                let _: i64 = conn
                    .decr(&key, 1)
                    .await
                    .map_err(|error| store_error(&error))?;
                self.rollback_windows(conn, demand, CommittedWindows::ALL)
                    .await;
                let cap = demand.member_cap.unwrap_or(0);
                return Err(EngineError::Overloaded {
                    message: format!(
                        "provider \"{}\" is at its in-flight cap of {cap}",
                        demand.member
                    ),
                    retry_after: Some(Duration::from_secs(1)),
                });
            }
        }
        if let Some(cap) = demand.model_cap {
            let key = self.model_slot_key(demand.member, demand.model);
            let count: i64 = conn
                .incr(&key, 1)
                .await
                .map_err(|error| store_error(&error))?;
            let _: bool = conn
                .expire(&key, INFLIGHT_TTL.as_secs() as i64)
                .await
                .map_err(|error| store_error(&error))?;
            if count > i64::from(cap) {
                let _: i64 = conn
                    .decr(&key, 1)
                    .await
                    .map_err(|error| store_error(&error))?;
                if demand.member_cap.is_some() {
                    let member_key = self.member_slot_key(demand.member);
                    let _: i64 = conn
                        .decr(&member_key, 1)
                        .await
                        .map_err(|error| store_error(&error))?;
                }
                self.rollback_windows(conn, demand, CommittedWindows::ALL)
                    .await;
                return Err(EngineError::Overloaded {
                    message: format!(
                        "model \"{}\" on \"{}\" is at its in-flight cap of {cap}",
                        demand.model, demand.member
                    ),
                    retry_after: Some(Duration::from_secs(1)),
                });
            }
        }
        Ok(())
    }

    /// Undoes the committed increments of a refused reservation: the
    /// checks before the refusing one (a concurrency slot, or a later
    /// window/budget check) have already moved counters on this connection.
    async fn rollback_windows(
        &self,
        conn: &mut ConnectionManager,
        demand: &ReservationDemand<'_>,
        parts: CommittedWindows,
    ) {
        let caller = demand.caller;
        fn undo(result: Result<i64, redis::RedisError>) {
            if let Err(error) = result {
                tracing::warn!(%error, "admission rollback failed");
            }
        }
        if parts.rpm && demand.rpm.is_some() {
            let key = format!(
                "{}:ad:rq:m:{caller}:{}",
                self.store.prefix,
                unix_minute(demand.now)
            );
            undo(conn.decr(&key, 1).await);
        }
        if parts.rpd && demand.rpd.is_some() {
            let key = format!(
                "{}:ad:rq:d:{caller}:{}",
                self.store.prefix,
                day_stamp(demand.now)
            );
            undo(conn.decr(&key, 1).await);
        }
        if parts.tpm && demand.tpm.is_some() && demand.reserved_tokens > 0 {
            let key = self.token_key(caller, unix_minute(demand.now));
            undo(conn.decr(&key, demand.reserved_tokens as i64).await);
        }
        if parts.reserved_usd
            && let Some(estimate) = demand.estimate_usd
        {
            let key = self.reserved_key(caller, &day_stamp(demand.now));
            undo(conn.decr(&key, estimate).await);
        }
    }
}

#[async_trait]
impl AdmissionBackend for RedisAdmissionBackend {
    async fn admit(&self, demand: &ReservationDemand<'_>) -> Result<LeaseId, EngineError> {
        let now = demand.now;
        let mut conn = self.store.conn.clone();
        let minute = unix_minute(now);
        let day = day_stamp(now);

        self.admit_windows(&mut conn, demand, minute, &day).await?;

        self.admit_caps(&mut conn, demand).await?;

        // Every check passed: persist the lease. The TTL is the crash
        // bound — a replica that dies mid-request releases its budget
        // within the hour without operator action.
        let id = Uuid::new_v4();
        let record = LeaseRecord {
            caller: demand.caller.to_owned(),
            member: demand.member.to_owned(),
            model: demand.model.to_owned(),
            minute,
            day,
            reserved_tokens: if demand.tracks_caller() {
                demand.reserved_tokens
            } else {
                0
            },
            reserved_usd: match (demand.tracks_caller(), demand.estimate_usd) {
                (true, estimate) => estimate.unwrap_or(0),
                (false, _) => 0,
            },
            holds_member_slot: demand.member_cap.is_some(),
            holds_model_slot: demand.model_cap.is_some(),
        };
        let record_key = self.lease_key(&LeaseId::new(id.to_string()));
        let payload = serde_json::to_string(&record).map_err(|error| {
            EngineError::Internal(format!("serializing admission lease: {error}"))
        })?;
        let _: () = conn
            .set_ex(record_key, payload, LEASE_TTL.as_secs().max(1))
            .await
            .map_err(|error| store_error(&error))?;
        Ok(LeaseId::new(id.to_string()))
    }

    fn settle(&self, lease: &LeaseId, settlement: &Settlement) {
        // Fire-and-forget, mirroring the trait contract: never block the
        // calling task. Without a runtime (settlement outside async
        // context) the write is dropped; the lease TTL bounds the leak.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let backend = self.clone();
                let lease = lease.clone();
                let settlement = *settlement;
                handle.spawn(async move {
                    if let Err(error) = backend.settle_async(&lease, &settlement).await {
                        tracing::warn!(
                            %error,
                            lease = lease.as_str(),
                            "admission settlement write failed; reservation expires with its TTL"
                        );
                    }
                });
            }
            Err(_) => {
                tracing::warn!(
                    lease = lease.as_str(),
                    "admission settlement outside a runtime was dropped; reservation expires with its TTL"
                );
            }
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
        let mut conn = self.store.conn.clone();
        let record_key = self.lease_key(lease);
        let raw: Option<String> = conn
            .get(&record_key)
            .await
            .map_err(|error| store_error(&error))?;
        let Some(raw) = raw else {
            tracing::warn!(
                lease = lease.as_str(),
                "admission rekey for an unknown or expired lease; ignoring"
            );
            return Ok(());
        };
        let mut record: LeaseRecord = serde_json::from_str(&raw).map_err(|error| {
            EngineError::Internal(format!("deserializing admission lease: {error}"))
        })?;
        if record.member == member && record.model == model {
            return Ok(());
        }

        // Acquire the new slots first; on saturation nothing has moved.
        if let Some(cap) = member_cap {
            let key = self.member_slot_key(member);
            let count: i64 = conn
                .incr(&key, 1)
                .await
                .map_err(|error| store_error(&error))?;
            if count > i64::from(cap) {
                let _: i64 = conn
                    .decr(&key, 1)
                    .await
                    .map_err(|error| store_error(&error))?;
                return Err(EngineError::Overloaded {
                    message: format!("provider \"{member}\" is at its in-flight cap of {cap}"),
                    retry_after: Some(Duration::from_secs(1)),
                });
            }
        }
        if let Some(cap) = model_cap {
            let key = self.model_slot_key(member, model);
            let count: i64 = conn
                .incr(&key, 1)
                .await
                .map_err(|error| store_error(&error))?;
            if count > i64::from(cap) {
                let _: i64 = conn
                    .decr(&key, 1)
                    .await
                    .map_err(|error| store_error(&error))?;
                if member_cap.is_some() {
                    let member_key = self.member_slot_key(member);
                    let _: i64 = conn
                        .decr(&member_key, 1)
                        .await
                        .map_err(|error| store_error(&error))?;
                }
                return Err(EngineError::Overloaded {
                    message: format!(
                        "model \"{model}\" on \"{member}\" is at its in-flight cap of {cap}"
                    ),
                    retry_after: Some(Duration::from_secs(1)),
                });
            }
        }

        // Release the old slots and record the move, keeping the lease's
        // TTL (a moving lease must not outlive its crash bound).
        self.release_slots(&mut conn, &record)
            .await
            .map_err(|error: RedisStoreError| EngineError::Internal(error.to_string()))?;
        record.member = member.to_owned();
        record.model = model.to_owned();
        record.holds_member_slot = member_cap.is_some();
        record.holds_model_slot = model_cap.is_some();
        let payload = serde_json::to_string(&record).map_err(|error| {
            EngineError::Internal(format!("serializing admission lease: {error}"))
        })?;
        let options = redis::SetOptions::default().with_expiration(redis::SetExpiry::KEEPTTL);
        let _: Option<String> = conn
            .set_options(record_key, payload, options)
            .await
            .map_err(|error| store_error(&error))?;
        Ok(())
    }
}

/// Builds the engine's [`PolicyAdmission`] over the Redis backend: the
/// same rules the single-node proxy runs, with the counters shared by
/// every replica that connects to the same store.
pub fn replica_admission(
    keys: HashMap<String, KeyPolicy>,
    pricing: HashMap<(String, String), llm_proxy_core::ModelPricing>,
    caps: HashMap<String, (Option<u32>, HashMap<String, u32>)>,
    store: RedisStore,
) -> PolicyAdmission {
    PolicyAdmission::over_backend(
        keys,
        pricing,
        caps,
        Arc::new(RedisAdmissionBackend::new(store)),
    )
}

/// The daemon's replica path: the same config surfaces
/// [`PolicyAdmission::from_config`](llm_proxy_engine::PolicyAdmission::from_config)
/// reads — per-key policies from `[auth.clients]`, pricing and concurrency
/// caps from the provider registry — over the shared store.
#[must_use]
pub fn replica_admission_over_config(
    auth: &llm_proxy_core::AuthConfig,
    providers: &llm_proxy_core::ProviderRegistry,
    store: RedisStore,
) -> PolicyAdmission {
    PolicyAdmission::over_config_backend(
        auth,
        providers,
        Arc::new(RedisAdmissionBackend::new(store)),
    )
}
