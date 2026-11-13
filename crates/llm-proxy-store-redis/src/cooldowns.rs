//! Shared upstream cooldowns in Redis (plan 09 step 1): one replica's
//! cooldown is every replica's cooldown, and extending one never shortens
//! it — writes are one server-side script, the Redis counterpart of the
//! Postgres store's `GREATEST` upserts.
//!
//! Cooldown records are plain string keys (value: unix-ms reset). Reads
//! delete expired records lazily, so storage stays bounded.

use std::time::Duration;

use async_trait::async_trait;
use llm_proxy_engine::Cooldowns;
use time::OffsetDateTime;

use crate::RedisStore;

/// How long a cooldown record outlives its own reset; the record is
/// deleted lazily once read, and this bounds storage for members whose
/// cooldown is never read again.
const COOLDOWN_GRACE_MS: u64 = 3_600_000;

/// The cooldown write as one atomic server-side script: store `ARGV[1]`
/// unless an equal or later reset is already stored, with `ARGV[2]`
/// seconds of TTL. Redis has no atomic set-if-greater command. The same
/// script also reads legacy values left by older builds (plain integer
/// ms) without change.
pub(crate) const SET_IF_GREATER: &str = r#"
local cur = redis.call('GET', KEYS[1])
if cur and tonumber(cur) >= tonumber(ARGV[1]) then return 0 end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
return 1
"#;

fn unix_ms(at: OffsetDateTime) -> i64 {
    at.unix_timestamp_nanos().div_euclid(1_000_000) as i64
}

fn provider_key(prefix: &str, provider: &str) -> String {
    format!("{prefix}:cd:p:{provider}")
}

fn credential_key(prefix: &str, provider: &str, credential_id: &str) -> String {
    format!("{prefix}:cd:c:{provider}:{credential_id}")
}

/// The ids that ever cooled for a provider (SADD is idempotent); bounded
/// by the member's credential pool size.
fn credential_ids_key(prefix: &str, provider: &str) -> String {
    format!("{prefix}:cd:ids:{provider}")
}

/// The Redis-backed [`Cooldowns`]: a provider or credential cooling on one
/// replica is skipped by every replica.
#[derive(Debug, Clone)]
pub struct RedisCooldowns {
    store: RedisStore,
}

impl RedisCooldowns {
    /// Over a connected store.
    #[must_use]
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    /// Reads one cooldown record, deleting it lazily once expired.
    async fn read_until(&self, key: &str, now: OffsetDateTime) -> Option<i64> {
        use redis::AsyncCommands;
        let mut conn = self.store.conn.clone();
        let until_ms: Option<i64> = conn.get(key).await.ok()?;
        let until_ms = until_ms?;
        if until_ms <= unix_ms(now) {
            if let Err(error) = conn.del::<&str, i64>(key).await {
                tracing::warn!(%error, "deleting an expired cooldown record failed");
            }
            None
        } else {
            Some(until_ms)
        }
    }

    /// Writes `until` unless a later reset is already stored (extend,
    /// never shorten): one atomic server-side script, so two replicas
    /// writing concurrently can never leave the shorter reset in place.
    async fn write_until(&self, key: &str, until: OffsetDateTime) {
        let now = OffsetDateTime::now_utc();
        let mut conn = self.store.conn.clone();
        let ttl = ttl_until(until, now);
        let result = redis::cmd("EVAL")
            .arg(SET_IF_GREATER)
            .arg(1)
            .arg(key)
            .arg(unix_ms(until))
            .arg(ttl.as_secs().max(1))
            .query_async::<i64>(&mut conn)
            .await;
        if let Err(error) = result {
            tracing::warn!(%error, "writing cooldown to Redis failed");
        }
    }
}

fn ttl_until(until: OffsetDateTime, now: OffsetDateTime) -> Duration {
    let remaining = (until - now).whole_milliseconds().max(0) + i128::from(COOLDOWN_GRACE_MS);
    Duration::from_millis(u64::try_from(remaining).unwrap_or(COOLDOWN_GRACE_MS))
}

#[async_trait]
impl Cooldowns for RedisCooldowns {
    async fn cooldown_provider(&self, provider: &str, until: OffsetDateTime) {
        self.write_until(&provider_key(&self.store.prefix, provider), until)
            .await;
    }

    async fn cooldown_credential(
        &self,
        provider: &str,
        credential_id: &str,
        until: OffsetDateTime,
    ) {
        use redis::AsyncCommands;
        let mut conn = self.store.conn.clone();
        if let Err(error) = conn
            .sadd::<_, _, usize>(
                credential_ids_key(&self.store.prefix, provider),
                credential_id,
            )
            .await
        {
            tracing::warn!(%error, "recording a credential cooldown id failed");
        }
        self.write_until(
            &credential_key(&self.store.prefix, provider, credential_id),
            until,
        )
        .await;
    }

    async fn clear_provider(&self, provider: &str) {
        use redis::AsyncCommands;
        let key = provider_key(&self.store.prefix, provider);
        let mut conn = self.store.conn.clone();
        let result: std::result::Result<i64, redis::RedisError> = conn.del(key).await;
        if let Err(error) = result {
            tracing::warn!(%error, "clearing provider cooldown in Redis failed");
        }
    }

    async fn provider_available(&self, provider: &str, now: OffsetDateTime) -> bool {
        let key = provider_key(&self.store.prefix, provider);
        self.read_until(&key, now).await.is_none()
    }

    async fn cooling_credentials(&self, provider: &str, now: OffsetDateTime) -> Vec<String> {
        use redis::AsyncCommands;
        let mut conn = self.store.conn.clone();
        let ids: Vec<String> = conn
            .smembers(credential_ids_key(&self.store.prefix, provider))
            .await
            .unwrap_or_default();
        let now_ms = unix_ms(now);
        let mut cooling = Vec::new();
        for id in ids {
            let key = credential_key(&self.store.prefix, provider, &id);
            let until: Option<i64> = conn.get(key).await.ok().flatten();
            if until.is_some_and(|until_ms| until_ms > now_ms) {
                cooling.push(id);
            }
        }
        cooling.sort();
        cooling
    }

    async fn provider_remaining(&self, provider: &str, now: OffsetDateTime) -> Option<Duration> {
        let key = provider_key(&self.store.prefix, provider);
        let until_ms = self.read_until(&key, now).await?;
        let remaining = until_ms - unix_ms(now);
        (remaining > 0).then(|| Duration::from_millis(u64::try_from(remaining).unwrap_or(0)))
    }

    async fn soonest_reset(&self, providers: &[&str], now: OffsetDateTime) -> Option<Duration> {
        let now_ms = unix_ms(now);
        let mut soonest: Option<i64> = None;
        for provider in providers {
            let key = provider_key(&self.store.prefix, provider);
            if let Some(until_ms) = self.read_until(&key, now).await
                && soonest.is_none_or(|best| until_ms < best)
            {
                soonest = Some(until_ms);
            }
        }
        soonest.map(|until_ms| {
            Duration::from_millis(u64::try_from((until_ms - now_ms).max(0)).unwrap_or(0))
        })
    }
}
