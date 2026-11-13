//! Replica-shared proxy state in Postgres (plan 09 step 1, audit conflict
//! 13): the admission counters, upstream cooldowns and usage events of
//! every instance that points at the same database, so two replicas
//! enforce one budget between them.
//!
//! Everything here implements an engine trait —
//! [`AdmissionBackend`](llm_proxy_engine::AdmissionBackend) through
//! [`PostgresAdmissionBackend`], [`Cooldowns`](llm_proxy_engine::Cooldowns)
//! through [`PostgresCooldowns`], and
//! [`EventBus`](llm_proxy_storage::EventBus) through
//! [`PostgresEventBus`] — so the rules (allowlists, estimates, statuses)
//! stay in the engine and no second admission policy exists. Use
//! [`replica_admission`] to build the engine-side
//! [`PolicyAdmission`](llm_proxy_engine::PolicyAdmission) over this store.
//!
//! # Atomicity model
//!
//! One admission is one transaction: the window counters, the in-flight
//! slots and the lease row are committed together or not at all, so a
//! refusal leaves no counter behind and two replicas can never both admit
//! against capacity that fits only one. Money moves as integer micro-USD
//! columns, never a float, matching the engine's money rule.
//!
//! # Verification status
//!
//! The integration tests in `tests/` need a live Postgres and are
//! `#[ignore]`d: set `LLM_PROXY_TEST_POSTGRES` to a connection string
//! (for example `postgres://localhost/llm_proxy_test`) to run them. This
//! environment has no Postgres server, so they were compiled but not
//! executed here.

#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use llm_proxy_engine::{
    AdmissionBackend, EngineError, KeyPolicy, LeaseId, MicroUsd, PolicyAdmission,
    ReservationDemand, Settlement,
};
use time::OffsetDateTime;
use uuid::Uuid;

mod cooldowns;
mod event_bus;
mod schema;

pub use cooldowns::PostgresCooldowns;
pub use event_bus::PostgresEventBus;
pub use schema::SCHEMA_SQL;

/// Errors from the Postgres replica store.
#[derive(Debug, thiserror::Error)]
pub enum PostgresStoreError {
    /// The configured connection string is not valid.
    #[error("invalid Postgres config: {0}")]
    InvalidConfig(String),
    /// Connecting failed.
    #[error("connecting to Postgres: {0}")]
    Connect(#[source] tokio_postgres::Error),
    /// Creating the schema failed.
    #[error("creating the replica schema: {0}")]
    Schema(#[source] tokio_postgres::Error),
    /// A query failed.
    #[error("Postgres query failed: {0}")]
    Query(#[from] tokio_postgres::Error),
    /// The connection pool could not provide a connection.
    #[error("Postgres connection pool: {0}")]
    Pool(String),
}

/// A connection pool to the replica Postgres (deadpool-managed: guards
/// are checked out per operation and recycled on drop, so a transaction
/// always owns its connection exclusively).
#[derive(Clone)]
pub struct PostgresStore {
    pool: deadpool_postgres::Pool,
}

/// The default number of connections each process keeps to the replica.
const DEFAULT_POOL_SIZE: usize = 8;

impl std::fmt::Debug for PostgresStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresStore").finish_non_exhaustive()
    }
}

impl PostgresStore {
    /// Connect using a Postgres connection string (for example
    /// `postgres://user:pass@host:5432/db`) and create the schema if it
    /// does not exist. Call once at startup.
    ///
    /// # Errors
    ///
    /// [`PostgresStoreError`] on an invalid config, a failed connection,
    /// or a failed schema creation.
    pub async fn connect(config: &str) -> Result<Self, PostgresStoreError> {
        Self::connect_with_pool_size(config, DEFAULT_POOL_SIZE).await
    }

    /// Same, with an explicit pool size (tests use one).
    ///
    /// # Errors
    ///
    /// [`PostgresStoreError`] on a failed connection or schema creation.
    pub async fn connect_with_pool_size(
        config: &str,
        pool_size: usize,
    ) -> Result<Self, PostgresStoreError> {
        let pg_config: tokio_postgres::Config = config
            .parse()
            .map_err(|error| PostgresStoreError::InvalidConfig(format!("{config}: {error}")))?;
        let manager = deadpool_postgres::Manager::new(pg_config, tokio_postgres::NoTls);
        let pool = deadpool_postgres::Pool::builder(manager)
            .max_size(pool_size.max(1))
            .build()
            .map_err(|error| PostgresStoreError::InvalidConfig(error.to_string()))?;
        let store = Self { pool };
        store.ensure_schema().await?;
        Ok(store)
    }

    /// Checks one connection out of the pool, waiting while every
    /// connection is in use.
    ///
    /// # Errors
    ///
    /// [`PostgresStoreError::Pool`] when the pool cannot provide a
    /// connection.
    pub(crate) async fn connection(&self) -> Result<deadpool_postgres::Client, PostgresStoreError> {
        self.pool
            .get()
            .await
            .map_err(|error| PostgresStoreError::Pool(error.to_string()))
    }

    /// Creates the replica tables and indexes if they do not exist.
    ///
    /// Schema creation is serialized across every process sharing the
    /// database with a transaction-scoped advisory lock: Postgres's
    /// `CREATE TABLE IF NOT EXISTS` checks the catalog and inserts
    /// non-atomically across transactions, so two replicas creating the
    /// schema at the same time collide in `pg_type` (SQLSTATE 23505)
    /// even though each statement alone is idempotent. The lock is
    /// transaction-scoped, so it releases at commit even if a creator
    /// dies mid-DDL. Found by running the store against a real Postgres;
    /// `tests/replica_state.rs` reproduces it by connecting several
    /// stores concurrently.
    ///
    /// # Errors
    ///
    /// [`PostgresStoreError::Schema`] when the DDL fails.
    pub async fn ensure_schema(&self) -> Result<(), PostgresStoreError> {
        let mut client = self.connection().await?;
        let tx = client
            .transaction()
            .await
            .map_err(PostgresStoreError::Schema)?;
        tx.execute("SELECT pg_advisory_xact_lock($1)", &[&SCHEMA_LOCK_KEY])
            .await
            .map_err(PostgresStoreError::Schema)?;
        tx.batch_execute(SCHEMA_SQL)
            .await
            .map_err(PostgresStoreError::Schema)?;
        tx.commit().await.map_err(PostgresStoreError::Schema)?;
        Ok(())
    }

    /// Number of usage events whose JSON payload matches
    /// `payload->>'field' = value` — the feed check for drain jobs and
    /// tests.
    ///
    /// # Errors
    ///
    /// [`PostgresStoreError::Query`] when the count fails.
    pub async fn count_usage_events_where(
        &self,
        field: &str,
        value: &str,
    ) -> Result<u64, PostgresStoreError> {
        let client = self.connection().await?;
        let row = client
            .query_one(
                "SELECT COUNT(*) FROM usage_events WHERE payload->>$1 = $2",
                &[&field, &value],
            )
            .await?;
        let count: i64 = row.get(0);
        Ok(u64::try_from(count.max(0)).unwrap_or(0))
    }

    /// Deletes window rows that can no longer be queried: minute rows
    /// older than two minutes, day rows older than one day, and leases
    /// older than the lease TTL. An expired lease held in-flight slots,
    /// reserved tokens and reserved budget until its owner died; deleting
    /// it without releasing those would leak them permanently (after `cap`
    /// crashed leases a member sits at its in-flight cap forever), so the
    /// purge releases exactly what [`PostgresAdmissionBackend::settle`]
    /// would have — in one transaction, locking the lease rows so a
    /// concurrent settle of the same lease cannot double-release. Call on
    /// a timer.
    ///
    /// # Errors
    ///
    /// [`PostgresStoreError::Query`] when the release or deletes fail.
    pub async fn purge_expired(&self) -> Result<u64, PostgresStoreError> {
        let minute_cut = unix_minute(OffsetDateTime::now_utc()) - 2;
        let today = OffsetDateTime::now_utc().date();
        let day_cut = pg_date(today.previous_day().unwrap_or(today));
        let lease_cut = OffsetDateTime::now_utc() - LEASE_TTL;
        let mut client = self.connection().await?;
        let tx = client.transaction().await?;
        let expired = tx
            .query(
                "SELECT id FROM admission_leases WHERE created_at < $1 FOR UPDATE",
                &[&lease_cut],
            )
            .await?;
        let ids: Vec<uuid::Uuid> = expired.iter().map(|row| row.get(0)).collect();
        if !ids.is_empty() {
            tx.execute(
                "UPDATE admission_minutes w \
                 SET tokens = GREATEST(w.tokens - l.reserved_tokens, 0) \
                 FROM admission_leases l \
                 WHERE l.id = ANY($1) AND l.reserved_tokens > 0 \
                 AND w.caller = l.caller AND w.minute = l.minute",
                &[&ids],
            )
            .await?;
            tx.execute(
                "UPDATE admission_days d \
                 SET reserved_usd = GREATEST(d.reserved_usd - l.reserved_usd, 0) \
                 FROM admission_leases l \
                 WHERE l.id = ANY($1) AND l.reserved_usd <> 0 \
                 AND d.caller = l.caller AND d.day = l.day",
                &[&ids],
            )
            .await?;
            tx.execute(
                "UPDATE admission_inflight_members m \
                 SET count = GREATEST(m.count - 1, 0) \
                 FROM admission_leases l \
                 WHERE l.id = ANY($1) AND l.holds_member_slot AND m.member = l.member",
                &[&ids],
            )
            .await?;
            tx.execute(
                "UPDATE admission_inflight_models g \
                 SET count = GREATEST(g.count - 1, 0) \
                 FROM admission_leases l \
                 WHERE l.id = ANY($1) AND l.holds_model_slot \
                 AND g.member = l.member AND g.model = l.model",
                &[&ids],
            )
            .await?;
        }
        let rows = tx
            .execute("DELETE FROM admission_leases WHERE id = ANY($1)", &[&ids])
            .await?;
        tx.commit().await?;
        let total = rows;
        let rows = client
            .execute(
                "DELETE FROM admission_minutes WHERE minute < $1",
                &[&minute_cut],
            )
            .await?;
        let total = total.saturating_add(rows);
        let rows = client
            .execute("DELETE FROM admission_days WHERE day < $1", &[&day_cut])
            .await?;
        Ok(total.saturating_add(rows))
    }
}

type PgDate = tokio_postgres::types::Date<time::Date>;

fn pg_date(day: time::Date) -> PgDate {
    tokio_postgres::types::Date::Value(day)
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

/// How long an unsettled lease holds its reservation before the purge
/// drops it (a crashed replica's crash bound).
const LEASE_TTL: Duration = Duration::from_secs(3600);

/// The advisory-lock key serializing schema creation across every
/// process sharing the database (see [`PostgresStore::ensure_schema`]).
/// Arbitrary but fixed; collisions with unrelated applications are
/// improbable and only cause brief waiting.
const SCHEMA_LOCK_KEY: i64 = 910_109;

fn store_error(error: &tokio_postgres::Error) -> EngineError {
    EngineError::Internal(format!("admission store: {error}"))
}

fn postgres_store_error(error: &PostgresStoreError) -> EngineError {
    EngineError::Internal(format!("admission store: {error}"))
}

/// The Postgres-backed [`AdmissionBackend`]: windows, budgets,
/// concurrency caps and leases held in the shared database, one
/// transaction per admission.
#[derive(Debug, Clone)]
pub struct PostgresAdmissionBackend {
    store: PostgresStore,
}

impl PostgresAdmissionBackend {
    /// Over a connected store.
    #[must_use]
    pub fn new(store: PostgresStore) -> Self {
        Self { store }
    }

    /// The settlement write, as an async operation: the sync
    /// [`AdmissionBackend::settle`] spawns it; callers that need the write
    /// to have happened await this directly.
    ///
    /// # Errors
    ///
    /// [`PostgresStoreError`] when the transaction fails; the lease row is
    /// deleted before any counter moves, so a failure here leaks nothing
    /// but may leave the reservation in flight until the purge drops it.
    pub async fn settle_async(
        &self,
        lease: &LeaseId,
        settlement: &Settlement,
    ) -> Result<(), PostgresStoreError> {
        let id = parse_lease_id(lease)?;
        let mut conn = self.store.connection().await?;
        let tx = conn.transaction().await?;
        let row = tx
            .query_opt(
                "SELECT caller, member, model, minute, day, reserved_tokens, reserved_usd, \
                 holds_member_slot, holds_model_slot \
                 FROM admission_leases WHERE id = $1 FOR UPDATE",
                &[&id],
            )
            .await?;
        let Some(row) = row else {
            tracing::warn!(
                lease = lease.as_str(),
                "admission settle for an unknown or purged lease; ignoring"
            );
            tx.rollback().await?;
            return Ok(());
        };
        let caller: String = row.get(0);
        let member: String = row.get(1);
        let model: String = row.get(2);
        let minute: i64 = row.get(3);
        let lease_day: PgDate = row.get(4);
        let reserved_tokens: i64 = row.get(5);
        let reserved_usd: i64 = row.get(6);
        let holds_member_slot: bool = row.get(7);
        let holds_model_slot: bool = row.get(8);
        tx.execute("DELETE FROM admission_leases WHERE id = $1", &[&id])
            .await?;

        // Token window: release the estimate, charge the reported total,
        // into the minute the reservation was made in.
        if reserved_tokens > 0 || settlement.usage_tokens.is_some() {
            tx.execute(
                "INSERT INTO admission_minutes (caller, minute, requests, tokens) \
                 VALUES ($1, $2, 0, $3) \
                 ON CONFLICT (caller, minute) DO UPDATE \
                 SET tokens = GREATEST(admission_minutes.tokens - $4, 0) + $3",
                &[
                    &caller,
                    &minute,
                    &(i64::try_from(settlement.usage_tokens.unwrap_or(0)).unwrap_or(0)),
                    &reserved_tokens,
                ],
            )
            .await?;
        }

        // Money: release from the day it was reserved on; charge what the
        // provider billed to today.
        if reserved_usd != 0 {
            tx.execute(
                "UPDATE admission_days SET reserved_usd = GREATEST(reserved_usd - $1, 0) \
                 WHERE caller = $2 AND day = $3",
                &[&reserved_usd, &caller, &lease_day],
            )
            .await?;
        }
        if let Some(charged) = settlement.charged_usd {
            let today = pg_date(settlement.now.date());
            tx.execute(
                "INSERT INTO admission_days (caller, day, requests, spent_usd, reserved_usd) \
                 VALUES ($1, $2, 0, $3, 0) \
                 ON CONFLICT (caller, day) DO UPDATE \
                 SET spent_usd = admission_days.spent_usd + $3",
                &[&caller, &today, &charged],
            )
            .await?;
        }

        if holds_member_slot {
            tx.execute(
                "UPDATE admission_inflight_members \
                 SET count = GREATEST(count - 1, 0) WHERE member = $1",
                &[&member],
            )
            .await?;
        }
        if holds_model_slot {
            tx.execute(
                "UPDATE admission_inflight_models \
                 SET count = GREATEST(count - 1, 0) WHERE member = $1 AND model = $2",
                &[&member, &model],
            )
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }
    /// The rpm/rpd/tpm window checks inside one admission's transaction.
    async fn check_windows(
        &self,
        tx: &tokio_postgres::Transaction<'_>,
        demand: &ReservationDemand<'_>,
        minute: i64,
        day: &PgDate,
    ) -> Result<(), EngineError> {
        if let Some(rpm) = demand.rpm {
            let row = tx
                .query_one(
                    "INSERT INTO admission_minutes (caller, minute, requests, tokens)                      VALUES ($1, $2, 1, 0)                      ON CONFLICT (caller, minute) DO UPDATE                      SET requests = admission_minutes.requests + 1                      RETURNING requests",
                    &[&demand.caller, &minute],
                )
                .await
                .map_err(|error| store_error(&error))?;
            let requests: i64 = row.get(0);
            if requests > i64::from(rpm) {
                return Err(EngineError::RateLimited {
                    message: format!("rpm limit {rpm} reached for this key"),
                    retry_after: Some(until_next_minute(demand.now)),
                });
            }
        }
        if let Some(rpd) = demand.rpd {
            let row = tx
                .query_one(
                    "INSERT INTO admission_days                      (caller, day, requests, spent_usd, reserved_usd)                      VALUES ($1, $2, 1, 0, 0)                      ON CONFLICT (caller, day) DO UPDATE                      SET requests = admission_days.requests + 1                      RETURNING requests",
                    &[&demand.caller, day],
                )
                .await
                .map_err(|error| store_error(&error))?;
            let requests: i64 = row.get(0);
            if requests > i64::try_from(rpd).unwrap_or(i64::MAX) {
                return Err(EngineError::RateLimited {
                    message: format!("rpd limit {rpd} reached for this key"),
                    retry_after: Some(until_next_day(demand.now)),
                });
            }
        }
        if let Some(tpm) = demand.tpm {
            let reserved_tokens = i64::try_from(demand.reserved_tokens).unwrap_or(i64::MAX);
            let row = tx
                .query_one(
                    "INSERT INTO admission_minutes (caller, minute, requests, tokens)                      VALUES ($1, $2, 0, $3)                      ON CONFLICT (caller, minute) DO UPDATE                      SET tokens = admission_minutes.tokens + $3                      RETURNING tokens",
                    &[&demand.caller, &minute, &reserved_tokens],
                )
                .await
                .map_err(|error| store_error(&error))?;
            let tokens: i64 = row.get(0);
            if tokens > 0 && u64::try_from(tokens).unwrap_or(0) > tpm {
                return Err(EngineError::RateLimited {
                    message: format!("tpm limit {tpm} would be exceeded for this key"),
                    retry_after: Some(until_next_minute(demand.now)),
                });
            }
        }
        Ok(())
    }

    /// The daily-budget check inside one admission's transaction.
    async fn check_budget(
        &self,
        tx: &tokio_postgres::Transaction<'_>,
        demand: &ReservationDemand<'_>,
        day: &PgDate,
    ) -> Result<(), EngineError> {
        if let (Some(budget), Some(estimate)) = (demand.budget_usd_per_day, demand.estimate_usd) {
            let row = tx
                .query_one(
                    "INSERT INTO admission_days                      (caller, day, requests, spent_usd, reserved_usd)                      VALUES ($1, $2, 0, 0, $3)                      ON CONFLICT (caller, day) DO UPDATE                      SET reserved_usd = admission_days.reserved_usd + $3                      RETURNING spent_usd, reserved_usd",
                    &[&demand.caller, day, &estimate],
                )
                .await
                .map_err(|error| store_error(&error))?;
            let spent: i64 = row.get(0);
            let reserved: i64 = row.get(1);
            if spent.saturating_add(reserved) > budget {
                return Err(EngineError::BudgetExhausted {
                    message: format!(
                        "daily budget of {} USD would be exceeded for this key (spent {},                          reserved in flight {}, this request {})",
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

    /// The concurrency-cap checks inside one admission's transaction.
    async fn check_caps(
        &self,
        tx: &tokio_postgres::Transaction<'_>,
        demand: &ReservationDemand<'_>,
    ) -> Result<(), EngineError> {
        if let Some(cap) = demand.member_cap {
            let row = tx
                .query_one(
                    "INSERT INTO admission_inflight_members (member, count) VALUES ($1, 1)                      ON CONFLICT (member) DO UPDATE                      SET count = admission_inflight_members.count + 1                      RETURNING count",
                    &[&demand.member],
                )
                .await
                .map_err(|error| store_error(&error))?;
            let count: i64 = row.get(0);
            if count > i64::from(cap) {
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
            let row = tx
                .query_one(
                    "INSERT INTO admission_inflight_models (member, model, count)                      VALUES ($1, $2, 1)                      ON CONFLICT (member, model) DO UPDATE                      SET count = admission_inflight_models.count + 1                      RETURNING count",
                    &[&demand.member, &demand.model],
                )
                .await
                .map_err(|error| store_error(&error))?;
            let count: i64 = row.get(0);
            if count > i64::from(cap) {
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

    /// Acquires (and checks) the member's in-flight slot inside `tx`.
    async fn acquire_member_slot(
        &self,
        tx: &tokio_postgres::Transaction<'_>,
        member: &str,
        cap: Option<u32>,
    ) -> Result<(), EngineError> {
        if let Some(cap) = cap {
            let row = tx
                .query_one(
                    "INSERT INTO admission_inflight_members (member, count) VALUES ($1, 1) \
                     ON CONFLICT (member) DO UPDATE \
                     SET count = admission_inflight_members.count + 1 \
                     RETURNING count",
                    &[&member],
                )
                .await
                .map_err(|error| store_error(&error))?;
            let count: i64 = row.get(0);
            if count > i64::from(cap) {
                return Err(EngineError::Overloaded {
                    message: format!("provider \"{member}\" is at its in-flight cap of {cap}"),
                    retry_after: Some(Duration::from_secs(1)),
                });
            }
        }
        Ok(())
    }

    /// Acquires (and checks) the member-model's in-flight slot inside `tx`.
    async fn acquire_model_slot(
        &self,
        tx: &tokio_postgres::Transaction<'_>,
        member: &str,
        model: &str,
        cap: Option<u32>,
    ) -> Result<(), EngineError> {
        if let Some(cap) = cap {
            let row = tx
                .query_one(
                    "INSERT INTO admission_inflight_models (member, model, count) \
                     VALUES ($1, $2, 1) \
                     ON CONFLICT (member, model) DO UPDATE \
                     SET count = admission_inflight_models.count + 1 \
                     RETURNING count",
                    &[&member, &model],
                )
                .await
                .map_err(|error| store_error(&error))?;
            let count: i64 = row.get(0);
            if count > i64::from(cap) {
                return Err(EngineError::Overloaded {
                    message: format!(
                        "model \"{model}\" on \"{member}\" is at its in-flight cap of {cap}"
                    ),
                    retry_after: Some(Duration::from_secs(1)),
                });
            }
        }
        Ok(())
    }

    /// The rekey's re-pricing half (fix 05-7) inside `tx`: refuse with
    /// `BudgetExhausted` when the grown reservation alone would exceed the
    /// caller's day budget, otherwise move the reservation on the lease's
    /// own day row — the day settlement releases it from. A refusal leaves
    /// the whole move to the caller's transaction rollback.
    async fn check_rekey_budget(
        &self,
        tx: &tokio_postgres::Transaction<'_>,
        // The lease as locked: caller, its reservation day and the USD it
        // still holds there.
        lease: (&str, &PgDate, i64),
        new: MicroUsd,
        budget_usd_per_day: Option<MicroUsd>,
    ) -> Result<(), EngineError> {
        let (caller, lease_day, old_reserved) = lease;
        let now = OffsetDateTime::now_utc();
        let today = pg_date(now.date());
        let held = if lease_day == &today { old_reserved } else { 0 };
        let day_row = tx
            .query_opt(
                "SELECT spent_usd, reserved_usd FROM admission_days \
                 WHERE caller = $1 AND day = $2",
                &[&caller, &today],
            )
            .await
            .map_err(|error| store_error(&error))?;
        let (spent, reserved_today) = day_row.map_or((0, 0), |day_row| {
            (day_row.get::<_, i64>(0), day_row.get::<_, i64>(1))
        });
        let without = reserved_today.saturating_sub(held);
        let projected = without.saturating_add(new);
        if let Some(budget) = budget_usd_per_day
            && new > held
            && spent.saturating_add(projected) > budget
        {
            return Err(EngineError::BudgetExhausted {
                message: format!(
                    "daily budget of {} USD would be exceeded for this key (spent {}, \
                     reserved in flight {}, this request {})",
                    llm_proxy_engine::from_micro_usd(budget).normalize(),
                    llm_proxy_engine::from_micro_usd(spent).normalize(),
                    llm_proxy_engine::from_micro_usd(without).normalize(),
                    llm_proxy_engine::from_micro_usd(new).normalize(),
                ),
                retry_after: Some(until_next_day(now)),
            });
        }
        let delta = new - old_reserved;
        if delta != 0 {
            tx.execute(
                "UPDATE admission_days \
                 SET reserved_usd = GREATEST(reserved_usd + $1, 0) \
                 WHERE caller = $2 AND day = $3",
                &[&delta, &caller, lease_day],
            )
            .await
            .map_err(|error| store_error(&error))?;
        }
        Ok(())
    }

    /// Inserts the lease row for an admission that passed every check;
    /// the purge drops it after the TTL even if the owning replica
    /// crashes mid-request.
    async fn commit_lease(
        &self,
        tx: &tokio_postgres::Transaction<'_>,
        demand: &ReservationDemand<'_>,
        minute: i64,
        day: &PgDate,
    ) -> Result<LeaseId, EngineError> {
        let id = Uuid::new_v4();
        tx.execute(
            "INSERT INTO admission_leases \
             (id, caller, member, model, minute, day, reserved_tokens, reserved_usd, \
              holds_member_slot, holds_model_slot) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            &[
                &id,
                &demand.caller,
                &demand.member,
                &demand.model,
                &minute,
                day,
                &(i64::try_from(demand.reserved_tokens).unwrap_or(0)),
                &(demand.estimate_usd.unwrap_or(0)),
                &demand.member_cap.is_some(),
                &demand.model_cap.is_some(),
            ],
        )
        .await
        .map_err(|error| store_error(&error))?;
        Ok(LeaseId::new(id.to_string()))
    }
}

fn parse_lease_id(lease: &LeaseId) -> Result<Uuid, PostgresStoreError> {
    lease.as_str().parse().map_err(|error| {
        PostgresStoreError::InvalidConfig(format!(
            "lease id {} was not issued by this backend: {}",
            lease.as_str(),
            error
        ))
    })
}

#[async_trait]
impl AdmissionBackend for PostgresAdmissionBackend {
    async fn admit(&self, demand: &ReservationDemand<'_>) -> Result<LeaseId, EngineError> {
        let now = demand.now;
        let minute = unix_minute(now);
        let day = pg_date(now.date());
        let mut conn = self
            .store
            .connection()
            .await
            .map_err(|error| postgres_store_error(&error))?;
        let tx = conn
            .transaction()
            .await
            .map_err(|error| store_error(&error))?;

        if demand.tracks_caller() {
            self.check_windows(&tx, demand, minute, &day).await?;
            self.check_budget(&tx, demand, &day).await?;
        }
        self.check_caps(&tx, demand).await?;

        // Every check passed: record the lease and commit. An uncommitted
        // transaction rolls back on drop, so a refusal leaves nothing.
        let id = self.commit_lease(&tx, demand, minute, &day).await?;
        tx.commit().await.map_err(|error| store_error(&error))?;
        Ok(id)
    }

    fn settle(&self, lease: &LeaseId, settlement: &Settlement) {
        // Fire-and-forget per the trait contract: never block the calling
        // task. Without a runtime the write is dropped; the lease TTL plus
        // the purge bound the leak.
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
        new_estimate_usd: Option<MicroUsd>,
        budget_usd_per_day: Option<MicroUsd>,
    ) -> Result<(), EngineError> {
        let id = parse_lease_id(lease).map_err(|error| EngineError::Internal(error.to_string()))?;
        let mut conn = self
            .store
            .connection()
            .await
            .map_err(|error| postgres_store_error(&error))?;
        let tx = conn
            .transaction()
            .await
            .map_err(|error| store_error(&error))?;
        let row = tx
            .query_opt(
                "SELECT member, model, holds_member_slot, holds_model_slot, \
                 caller, reserved_usd, day \
                 FROM admission_leases WHERE id = $1 FOR UPDATE",
                &[&id],
            )
            .await
            .map_err(|error| store_error(&error))?;
        let Some(row) = row else {
            tracing::warn!(
                lease = lease.as_str(),
                "admission rekey for an unknown or purged lease; ignoring"
            );
            tx.rollback().await.map_err(|error| store_error(&error))?;
            return Ok(());
        };
        let old_member: String = row.get(0);
        let old_model: String = row.get(1);
        let held_member_slot: bool = row.get(2);
        let held_model_slot: bool = row.get(3);
        let caller: String = row.get(4);
        let old_reserved: i64 = row.get(5);
        let lease_day: PgDate = row.get(6);
        if old_member == member && old_model == model {
            tx.rollback().await.map_err(|error| store_error(&error))?;
            return Ok(());
        }

        // Re-price the day reservation inside this transaction (fix 05-7):
        // the whole move rolls back when the grown reservation would exceed
        // the budget. The reservation counts against the day it was made
        // on — the day settlement releases it from.
        if let Some(new) = new_estimate_usd {
            self.check_rekey_budget(
                &tx,
                (&caller, &lease_day, old_reserved),
                new,
                budget_usd_per_day,
            )
            .await?;
        }

        self.acquire_member_slot(&tx, member, member_cap).await?;
        self.acquire_model_slot(&tx, member, model, model_cap)
            .await?;
        if held_member_slot {
            tx.execute(
                "UPDATE admission_inflight_members \
                 SET count = GREATEST(count - 1, 0) WHERE member = $1",
                &[&old_member],
            )
            .await
            .map_err(|error| store_error(&error))?;
        }
        if held_model_slot {
            tx.execute(
                "UPDATE admission_inflight_models \
                 SET count = GREATEST(count - 1, 0) WHERE member = $1 AND model = $2",
                &[&old_member, &old_model],
            )
            .await
            .map_err(|error| store_error(&error))?;
        }
        tx.execute(
            "UPDATE admission_leases SET member = $1, model = $2, \
             holds_member_slot = $3, holds_model_slot = $4, reserved_usd = $5 WHERE id = $6",
            &[
                &member,
                &model,
                &member_cap.is_some(),
                &model_cap.is_some(),
                &new_estimate_usd.unwrap_or(old_reserved),
                &id,
            ],
        )
        .await
        .map_err(|error| store_error(&error))?;
        tx.commit().await.map_err(|error| store_error(&error))?;
        Ok(())
    }
}

/// Builds the engine's [`PolicyAdmission`] over the Postgres backend: the
/// same rules the single-node proxy runs, with the counters shared by
/// every replica that connects to the same database.
pub fn replica_admission(
    keys: HashMap<String, KeyPolicy>,
    pricing: HashMap<(String, String), llm_proxy_core::ModelPricing>,
    caps: HashMap<String, (Option<u32>, HashMap<String, u32>)>,
    store: PostgresStore,
) -> PolicyAdmission {
    PolicyAdmission::over_backend(
        keys,
        pricing,
        caps,
        Arc::new(PostgresAdmissionBackend::new(store)),
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
    store: PostgresStore,
) -> PolicyAdmission {
    PolicyAdmission::over_config_backend(
        auth,
        providers,
        Arc::new(PostgresAdmissionBackend::new(store)),
    )
}
