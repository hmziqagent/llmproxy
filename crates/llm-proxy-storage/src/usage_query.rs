//! Read-only queries over the persistent usage ledger (the `requests`
//! table written by [`crate::SqliteEventBus`]). The admin API crate surfaces
//! these over HTTP; nothing here writes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::Connection;
use rusqlite::OpenFlags;
use rusqlite::types::Value as SqlValue;
use serde::Serialize;
use time::OffsetDateTime;

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Ceiling on the number of rows one `/usage/requests` page may return.
pub const MAX_PAGE_LIMIT: i64 = 1000;

/// Failures from querying the usage ledger.
#[derive(Debug, thiserror::Error)]
pub enum UsageQueryError {
    /// The SQLite query failed.
    #[error("usage ledger query failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A stored cost value is not a valid decimal.
    #[error("usage ledger cost value is not a valid decimal: {0}")]
    Cost(String),
    /// The blocking query task failed.
    #[error("usage ledger query task failed: {0}")]
    Task(String),
}

/// The grouping key for usage summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupBy {
    /// Group by provider name.
    Provider,
    /// Group by requested model.
    Model,
    /// Group by caller identity.
    Caller,
}

impl GroupBy {
    const fn column(self) -> &'static str {
        match self {
            Self::Provider => "provider",
            Self::Model => "model_requested",
            Self::Caller => "caller",
        }
    }

    /// Parse the query-parameter spelling (`provider`, `model`, `caller`).
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "provider" => Some(Self::Provider),
            "model" => Some(Self::Model),
            "caller" => Some(Self::Caller),
            _ => None,
        }
    }

    /// The query-parameter spelling of this grouping.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Provider => "provider",
            Self::Model => "model",
            Self::Caller => "caller",
        }
    }
}

/// Token totals over a summary window.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TokenCounts {
    /// Input tokens.
    pub input: i64,
    /// Output tokens.
    pub output: i64,
    /// Reasoning tokens.
    pub reasoning: i64,
    /// Cache-read tokens.
    pub cache_read: i64,
    /// Cache-creation tokens.
    pub cache_creation: i64,
}

/// Cost totals over a summary window, as decimal strings.
#[derive(Debug, Clone, Serialize)]
pub struct CostTotals {
    /// Input cost.
    pub input: String,
    /// Output cost.
    pub output: String,
    /// Cache-read cost.
    pub cache_read: String,
    /// Cache-creation cost.
    pub cache_creation: String,
    /// Reasoning cost.
    pub reasoning: String,
    /// Sum of all cost components.
    pub total: String,
}

/// One summary group: the grouped key plus its totals.
#[derive(Debug, Serialize)]
pub struct SummaryGroup {
    /// The grouped key (`None` groups rows with a NULL key).
    pub key: Option<String>,
    /// Completed requests.
    /// Completed requests.
    pub requests: i64,
    /// Failed requests.
    /// Failed requests.
    pub errors: i64,
    /// Token totals, when usage was reported. Served copies (cache hits,
    /// shared calls, replays) are excluded: their one upstream call's tokens
    /// are counted on the row that made the call.
    pub tokens: TokenCounts,
    /// Cost totals over the priced rows. `None` when no row in the group
    /// carries a computed cost: unpriced is a state, never a silent $0
    /// (BUG-24, plan 05 step 9).
    pub cost: Option<CostTotals>,
    /// Completed rows with reported usage but no computed cost: usage that
    /// happened on an unpriced model. Served copies never count: they
    /// charge nothing by design, so they are not unpriced either.
    pub unpriced: i64,
    /// `cost` covers only priced rows when true: the group also carries
    /// unpriced completions, so the total is partial, not the full cost.
    pub cost_partial: bool,
    /// Requests answered from the response cache (plan 05 step 6): they
    /// cost nothing by design.
    pub cache_hits: i64,
}

/// One request's ledger row (identifiers and totals only; no bodies).
#[derive(Debug, Serialize)]
pub struct RequestRow {
    /// Proxy-generated request identifier.
    pub request_id: String,
    /// When the request was received, UNIX-epoch milliseconds.
    pub received_at_ms: Option<i64>,
    /// When the response completed or failed, if it did.
    pub completed_at_ms: Option<i64>,
    /// Authenticated caller identity, when known.
    /// The caller identity (`None` groups rows with no caller).
    pub caller: Option<String>,
    /// Provider name from the URL.
    pub provider: Option<String>,
    /// Endpoint family (`chat_completions` or `messages`).
    pub route_kind: Option<String>,
    /// Client dialect (`openai_chat` or `anthropic`).
    pub client_protocol: Option<String>,
    /// Whether the request asked for a stream.
    pub streaming: Option<bool>,
    /// The model the client asked for.
    pub model_requested: Option<String>,
    /// The model that served the request, when aliased.
    pub model_upstream: Option<String>,
    /// Token totals, when usage was reported.
    pub tokens: Option<TokenCounts>,
    /// How the usage numbers were obtained.
    pub usage_provenance: Option<String>,
    /// Computed cost, when pricing was configured.
    pub cost: Option<CostTotals>,
    /// Why the model stopped generating.
    pub stop_reason: Option<String>,
    /// Error kind for failed rows.
    pub error_kind: Option<String>,
    /// HTTP status returned to the client.
    pub http_status: Option<i64>,
    /// Request latency in milliseconds.
    pub latency_ms: Option<i64>,
    /// SHA-256 of the raw client body.
    pub body_hash: Option<String>,
    /// Whether the response came from the response cache.
    pub cache_hit: Option<bool>,
    /// Lifecycle state derived from the row (plan 07 step 8, backlog 27):
    /// `in_flight` while no completion is recorded (a row the ledger wrote
    /// at `RequestReceived` whose request is still running), `failed` when an
    /// error_kind is set, `completed` otherwise.
    pub state: String,
    /// For `in_flight` rows: milliseconds since the request was received,
    /// measured at query time. `None` for settled rows (`latency_ms` is the
    /// settled fact there).
    pub elapsed_ms: Option<i64>,
}

/// One page of request rows plus the cursor for the next page.
#[derive(Debug, Serialize)]
pub struct RequestsPage {
    /// The rows in this page.
    pub rows: Vec<RequestRow>,
    /// Cursor for the next page; `None` on the last page.
    pub next_cursor: Option<String>,
}

/// Per-caller request and error counts with last-seen time.
#[derive(Debug, Serialize)]
pub struct CallerSummary {
    /// Authenticated caller identity, when known.
    /// The caller identity (`None` groups rows with no caller).
    pub caller: Option<String>,
    /// Completed requests.
    pub requests: i64,
    /// Failed requests.
    pub errors: i64,
    /// When this caller was last seen, UNIX-epoch milliseconds.
    pub last_seen_ms: Option<i64>,
}

/// Filters for the request-row query.
#[derive(Debug, Default)]
pub struct RequestFilter {
    /// Only rows at or after this UNIX-epoch millisecond timestamp.
    pub since_ms: Option<i64>,
    /// Only rows before this UNIX-epoch millisecond timestamp.
    pub until_ms: Option<i64>,
    /// Only rows for this provider.
    /// Provider name from the URL.
    pub provider: Option<String>,
    /// Page size (bounded by [`MAX_PAGE_LIMIT`]).
    pub limit: i64,
    /// Cursor: `(received_at_ms, request_id)` of the last row of the
    /// previous page.
    pub cursor: Option<(i64, String)>,
}

/// One caller's day-window totals read back from the `requests` table: the
/// count of rows the caller paid for (served copies excluded) and the exact
/// decimal sum of their `cost_total`. What startup seeding
/// (`MemoryAdmissionBackend::seed_day`) feeds so a restart cannot reset a
/// day budget or an rpd counter.
#[derive(Debug, Serialize)]
pub struct CallerDaySpend {
    /// Authenticated caller identity; `None` for rows written without one.
    pub caller: Option<String>,
    /// Request rows in the window with `cache_hit = 0`; rows with a NULL
    /// cost count here too.
    pub requests: i64,
    /// Exact sum of `cost_total` over the window's non-cache-hit rows.
    pub spend: rust_decimal::Decimal,
}

/// Per-caller day-window spend over `requests` rows from `since_unix_ms`
/// (00:00 UTC of the day being seeded). Costs are summed in Rust as exact
/// decimals, never a SQL float; a cache hit is neither a paid request nor
/// spend. Rows without a caller group under `None`.
///
/// # Errors
///
/// [`UsageQueryError`] when the ledger cannot be read or a stored cost is
/// not a valid decimal.
pub fn spend_since(
    path: &Path,
    since_unix_ms: i64,
) -> Result<Vec<CallerDaySpend>, UsageQueryError> {
    let conn = UsageReader::open(path)?;
    let mut stmt = conn.prepare(
        "SELECT caller, \
         COALESCE(SUM(CASE WHEN COALESCE(cache_hit, 0) = 0 THEN 1 ELSE 0 END), 0) \
         FROM requests \
         WHERE COALESCE(received_at, completed_at) >= ?1 \
         GROUP BY caller",
    )?;
    let mut rows: Vec<CallerDaySpend> = Vec::new();
    let mapped = stmt.query_map([since_unix_ms], |row| {
        Ok(CallerDaySpend {
            caller: row.get(0)?,
            requests: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
            spend: rust_decimal::Decimal::ZERO,
        })
    })?;
    for row in mapped {
        rows.push(row?);
    }
    // Exact decimal sums happen in Rust over the stored strings (the same
    // discipline as `sum_costs_by_group`).
    let mut stmt = conn.prepare(
        "SELECT caller, cost_total FROM requests \
         WHERE COALESCE(received_at, completed_at) >= ?1 \
           AND COALESCE(cache_hit, 0) = 0 AND cost_total IS NOT NULL",
    )?;
    let mut spends: HashMap<Option<String>, rust_decimal::Decimal> = HashMap::new();
    let mapped = stmt.query_map([since_unix_ms], |row| {
        Ok((
            row.get::<_, Option<String>>(0)?,
            row.get::<_, Option<String>>(1)?,
        ))
    })?;
    for row in mapped {
        let (caller, cost_total) = row?;
        let entry = spends.entry(caller).or_insert(rust_decimal::Decimal::ZERO);
        *entry += parse_cost(cost_total)?;
    }
    for row in &mut rows {
        if let Some(total) = spends.get(&row.caller) {
            row.spend = *total;
        }
    }
    Ok(rows)
}

/// Read-only query interface over the SQLite usage ledger.
#[derive(Debug, Clone)]
pub struct UsageReader {
    path: PathBuf,
}

fn offset_to_unix_millis(ts: OffsetDateTime) -> i64 {
    ts.unix_timestamp()
        .saturating_mul(1000)
        .saturating_add(i64::from(ts.millisecond()))
}

/// The current UNIX-epoch time in milliseconds.
pub fn now_unix_millis() -> i64 {
    offset_to_unix_millis(OffsetDateTime::now_utc())
}

impl UsageReader {
    /// Open the reader over the ledger at `path` (read-only).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Aggregated usage grouped by `group` since `since_ms`.
    ///
    /// # Errors
    ///
    /// [`UsageQueryError`] when the ledger cannot be read.
    pub async fn summary(
        &self,
        since_ms: i64,
        group: GroupBy,
    ) -> Result<Vec<SummaryGroup>, UsageQueryError> {
        let path = self.path.clone();
        run_blocking(move || summary_sync(&path, since_ms, group)).await
    }

    /// One page of request rows matching `filter`.
    ///
    /// # Errors
    ///
    /// [`UsageQueryError`] when the ledger cannot be read.
    pub async fn requests(&self, filter: RequestFilter) -> Result<RequestsPage, UsageQueryError> {
        let path = self.path.clone();
        run_blocking(move || requests_sync(&path, &filter)).await
    }

    /// Per-caller totals.
    ///
    /// # Errors
    ///
    /// [`UsageQueryError`] when the ledger cannot be read.
    pub async fn callers(&self) -> Result<Vec<CallerSummary>, UsageQueryError> {
        let path = self.path.clone();
        run_blocking(move || callers_sync(&path)).await
    }

    fn open(path: &Path) -> Result<Connection, UsageQueryError> {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        conn.busy_timeout(SQLITE_BUSY_TIMEOUT)?;
        Ok(conn)
    }
}

async fn run_blocking<T, F>(f: F) -> Result<T, UsageQueryError>
where
    F: FnOnce() -> Result<T, UsageQueryError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|error| UsageQueryError::Task(error.to_string()))?
}

fn parse_cost(value: Option<String>) -> Result<rust_decimal::Decimal, UsageQueryError> {
    match value {
        None => Ok(rust_decimal::Decimal::ZERO),
        Some(text) => rust_decimal::Decimal::from_str_exact(&text)
            .map_err(|_parse_error| UsageQueryError::Cost(text)),
    }
}

fn summary_sync(
    path: &Path,
    since_ms: i64,
    group: GroupBy,
) -> Result<Vec<SummaryGroup>, UsageQueryError> {
    let conn = UsageReader::open(path)?;
    let key = group.column();
    // Served copies (`cache_hit = 1`) count as requests and hits but never
    // as consumed tokens, cost or unpriced completions: their one upstream
    // call's usage and cost live on the row of the request that made it
    // (plan 05 step 6 — cache hits charge nothing).
    let sql = format!(
        "SELECT {key} AS k, COUNT(*), \
         SUM(CASE WHEN error_kind IS NOT NULL THEN 1 ELSE 0 END), \
         COALESCE(SUM(CASE WHEN COALESCE(cache_hit, 0) = 0 \
                THEN COALESCE(input_tokens, 0) ELSE 0 END), 0), \
         COALESCE(SUM(CASE WHEN COALESCE(cache_hit, 0) = 0 \
                THEN COALESCE(output_tokens, 0) ELSE 0 END), 0), \
         COALESCE(SUM(CASE WHEN COALESCE(cache_hit, 0) = 0 \
                THEN COALESCE(reasoning_tokens, 0) ELSE 0 END), 0), \
         COALESCE(SUM(CASE WHEN COALESCE(cache_hit, 0) = 0 \
                THEN COALESCE(cache_read_tokens, 0) ELSE 0 END), 0), \
         COALESCE(SUM(CASE WHEN COALESCE(cache_hit, 0) = 0 \
                THEN COALESCE(cache_creation_tokens, 0) ELSE 0 END), 0), \
         COALESCE(SUM(CASE WHEN error_kind IS NULL AND input_tokens IS NOT NULL \
                AND cost_total IS NULL AND COALESCE(cache_hit, 0) = 0 THEN 1 ELSE 0 END), 0), \
         COALESCE(SUM(CASE WHEN COALESCE(cache_hit, 0) = 1 THEN 1 ELSE 0 END), 0) \
         FROM requests \
         WHERE COALESCE(received_at, completed_at) >= ?1 \
         GROUP BY {key} ORDER BY 2 DESC"
    );
    let mut groups: Vec<SummaryGroup> = Vec::new();
    {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([since_ms], |row| {
            Ok(SummaryGroup {
                key: row.get(0)?,
                requests: row.get(1)?,
                errors: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
                tokens: TokenCounts {
                    input: row.get(3)?,
                    output: row.get(4)?,
                    reasoning: row.get(5)?,
                    cache_read: row.get(6)?,
                    cache_creation: row.get(7)?,
                },
                cost: None,
                unpriced: row.get(8)?,
                cost_partial: false,
                cache_hits: row.get(9)?,
            })
        })?;
        for row in rows {
            groups.push(row?);
        }
    }
    // Only groups with at least one priced row get cost totals; a group of
    // unpriced rows keeps `cost: None` instead of a fabricated $0.
    let costs = sum_costs_by_group(&conn, key, since_ms)?;
    for (group_key, total) in costs {
        if let Some(group_row) = groups.iter_mut().find(|g| g.key == group_key) {
            group_row.cost = Some(cost_totals_from(total));
        }
    }
    // A priced sum sitting next to unpriced completions is incomplete: flag
    // it so a partial total is never presented as the full cost.
    for group in &mut groups {
        group.cost_partial = group.cost.is_some() && group.unpriced > 0;
    }
    Ok(groups)
}

fn sum_costs_by_group(
    conn: &Connection,
    key: &str,
    since_ms: i64,
) -> Result<HashMap<Option<String>, [rust_decimal::Decimal; 6]>, UsageQueryError> {
    let sql = format!(
        "SELECT {key} AS k, cost_input, cost_output, cost_cache_read, \
         cost_cache_creation, cost_reasoning, cost_total \
         FROM requests \
         WHERE COALESCE(received_at, completed_at) >= ?1 AND cost_total IS NOT NULL"
    );
    let mut sums: HashMap<Option<String>, [rust_decimal::Decimal; 6]> = HashMap::new();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([since_ms], |row| {
        Ok((
            row.get::<_, Option<String>>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, Option<String>>(6)?,
        ))
    })?;
    for row in rows {
        let (key, input, output, cache_read, cache_creation, reasoning, total) = row?;
        let entry = sums.entry(key).or_insert([rust_decimal::Decimal::ZERO; 6]);
        let parts = [
            parse_cost(input)?,
            parse_cost(output)?,
            parse_cost(cache_read)?,
            parse_cost(cache_creation)?,
            parse_cost(reasoning)?,
            parse_cost(total)?,
        ];
        for (slot, part) in entry.iter_mut().zip(parts) {
            *slot += part;
        }
    }
    Ok(sums)
}

fn cost_totals_from(sums: [rust_decimal::Decimal; 6]) -> CostTotals {
    let show = |d: rust_decimal::Decimal| d.normalize().to_string();
    CostTotals {
        input: show(sums[0]),
        output: show(sums[1]),
        cache_read: show(sums[2]),
        cache_creation: show(sums[3]),
        reasoning: show(sums[4]),
        total: show(sums[5]),
    }
}

fn map_request_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(RequestRow, i64)> {
    let input: Option<i64> = row.get(10)?;
    let output: Option<i64> = row.get(11)?;
    let reasoning: Option<i64> = row.get(12)?;
    let cache_read: Option<i64> = row.get(13)?;
    let cache_creation: Option<i64> = row.get(14)?;
    let cost_input: Option<String> = row.get(16)?;
    let cost_output: Option<String> = row.get(17)?;
    let cost_cache_creation: Option<String> = row.get(18)?;
    let cost_cache_read: Option<String> = row.get(19)?;
    let cost_reasoning: Option<String> = row.get(20)?;
    let cost_total: Option<String> = row.get(21)?;
    let streaming: Option<i64> = row.get(7)?;
    let tokens = input.map(|input| TokenCounts {
        input,
        output: output.unwrap_or(0),
        reasoning: reasoning.unwrap_or(0),
        cache_read: cache_read.unwrap_or(0),
        cache_creation: cache_creation.unwrap_or(0),
    });
    let cost = cost_total.map(|total| CostTotals {
        input: cost_input.unwrap_or_else(|| "0".to_owned()),
        output: cost_output.unwrap_or_else(|| "0".to_owned()),
        cache_read: cost_cache_read.unwrap_or_else(|| "0".to_owned()),
        cache_creation: cost_cache_creation.unwrap_or_else(|| "0".to_owned()),
        reasoning: cost_reasoning.unwrap_or_else(|| "0".to_owned()),
        total,
    });
    // Lifecycle state (plan 07 step 8): computed in SQL from the same row so
    // the state and the timestamps it rests on cannot drift apart. Elapsed
    // time for in-flight rows is measured at query time (?1).
    let state: String = row.get(29)?;
    let row_out = RequestRow {
        request_id: row.get(0)?,
        received_at_ms: row.get(1)?,
        completed_at_ms: row.get(2)?,
        caller: row.get(3)?,
        provider: row.get(4)?,
        route_kind: row.get(5)?,
        client_protocol: row.get(6)?,
        streaming: streaming.map(|flag| flag != 0),
        model_requested: row.get(8)?,
        model_upstream: row.get(9)?,
        tokens,
        usage_provenance: row.get(15)?,
        cost,
        stop_reason: row.get(22)?,
        error_kind: row.get(23)?,
        http_status: row.get(24)?,
        latency_ms: row.get(25)?,
        body_hash: row.get(26)?,
        cache_hit: row.get::<_, Option<i64>>(27)?.map(|flag| flag != 0),
        state,
        elapsed_ms: row.get(30)?,
    };
    let ts: Option<i64> = row.get(28)?;
    Ok((row_out, ts.unwrap_or(i64::MIN)))
}

fn requests_sync(path: &Path, filter: &RequestFilter) -> Result<RequestsPage, UsageQueryError> {
    let conn = UsageReader::open(path)?;
    // In-flight elapsed time is measured once per query, at query time (an
    // integer literal, so it cannot perturb the bound-parameter order).
    let now_ms = now_unix_millis();
    let mut clauses: Vec<&'static str> = Vec::new();
    let mut params: Vec<SqlValue> = Vec::new();
    if let Some(since) = filter.since_ms {
        clauses.push("COALESCE(received_at, completed_at) >= ?");
        params.push(SqlValue::Integer(since));
    }
    if let Some(until) = filter.until_ms {
        clauses.push("COALESCE(received_at, completed_at) < ?");
        params.push(SqlValue::Integer(until));
    }
    if let Some(provider) = &filter.provider {
        clauses.push("provider = ?");
        params.push(SqlValue::Text(provider.clone()));
    }
    if let Some((cursor_ts, cursor_id)) = &filter.cursor {
        clauses.push(
            "(COALESCE(received_at, completed_at) < ? \
              OR (COALESCE(received_at, completed_at) = ? AND request_id < ?))",
        );
        params.push(SqlValue::Integer(*cursor_ts));
        params.push(SqlValue::Integer(*cursor_ts));
        params.push(SqlValue::Text(cursor_id.clone()));
    }
    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };
    let sql = format!(
        "SELECT request_id, received_at, completed_at, caller, provider, route_kind, \
         client_protocol, streaming, model_requested, model_upstream, input_tokens, \
         output_tokens, reasoning_tokens, cache_read_tokens, cache_creation_tokens, \
         usage_provenance, cost_input, cost_output, cost_cache_creation, cost_cache_read, \
         cost_reasoning, cost_total, stop_reason, error_kind, http_status, latency_ms, \
         body_hash, cache_hit, COALESCE(received_at, completed_at) AS ts, \
         CASE WHEN completed_at IS NULL THEN 'in_flight' \
              WHEN error_kind IS NOT NULL THEN 'failed' \
              ELSE 'completed' END AS state, \
         CASE WHEN completed_at IS NULL THEN {now_ms} - received_at ELSE NULL END AS elapsed_ms \
         FROM requests {where_sql} \
         ORDER BY ts DESC, request_id DESC \
         LIMIT ?"
    );
    params.push(SqlValue::Integer(filter.limit + 1));
    let mut fetched: Vec<(RequestRow, i64)> = Vec::new();
    {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(params), map_request_row)?;
        for row in rows {
            fetched.push(row?);
        }
    }
    let has_more = fetched.len() > usize::try_from(filter.limit).unwrap_or(0);
    if has_more {
        fetched.truncate(usize::try_from(filter.limit).unwrap_or(0));
    }
    let next_cursor = if has_more {
        fetched
            .last()
            .map(|(row, ts)| format!("{ts}:{}", row.request_id))
    } else {
        None
    };
    Ok(RequestsPage {
        rows: fetched.into_iter().map(|(row, _ts)| row).collect(),
        next_cursor,
    })
}

fn callers_sync(path: &Path) -> Result<Vec<CallerSummary>, UsageQueryError> {
    let conn = UsageReader::open(path)?;
    let mut stmt = conn.prepare(
        "SELECT caller, COUNT(*), \
         SUM(CASE WHEN error_kind IS NOT NULL THEN 1 ELSE 0 END), \
         MAX(COALESCE(received_at, completed_at)) \
         FROM requests GROUP BY caller ORDER BY 4 DESC NULLS LAST",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(CallerSummary {
            caller: row.get(0)?,
            requests: row.get(1)?,
            errors: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
            last_seen_ms: row.get(3)?,
        })
    })?;
    let mut callers = Vec::new();
    for row in rows {
        callers.push(row?);
    }
    Ok(callers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EventBus, ProxyEvent, RequestReceived, ResponseCompleted, RetentionPolicy, SqliteEventBus,
    };
    use llm_proxy_protocol::core::{Cost, ModelRef, StopReason, Usage, UsageProvenance};
    use rust_decimal::Decimal;

    fn seed_day_row(
        bus: &crate::SqliteEventBus,
        id: &str,
        caller: &str,
        cost_total: Option<Decimal>,
        cache_hit: bool,
    ) {
        let now = OffsetDateTime::now_utc();
        bus.emit(&ProxyEvent::RequestReceived(RequestReceived {
            request_id: id.to_owned(),
            timestamp: now,
            caller: caller.to_owned(),
            provider: "prov-a".to_owned(),
            route_kind: "messages".to_owned(),
            client_protocol: "anthropic".to_owned(),
            model: ModelRef {
                requested: "claude-3".to_owned(),
                upstream: None,
            },
            streaming: false,
            body_hash: "digest".to_owned(),
            session_id: None,
        }));
        bus.emit(&ProxyEvent::ResponseCompleted(ResponseCompleted {
            request_id: id.to_owned(),
            timestamp: now,
            provider: "prov-a".to_owned(),
            upstream_message_id: None,
            model: ModelRef {
                requested: "claude-3".to_owned(),
                upstream: None,
            },
            usage: Usage {
                input_tokens: 100,
                output_tokens: 10,
                provenance: UsageProvenance::ProviderReported,
                ..Usage::default()
            },
            cost: cost_total.map(|total| Cost {
                input: Decimal::ZERO,
                output: Decimal::ZERO,
                cache_creation: Decimal::ZERO,
                cache_read: Decimal::ZERO,
                reasoning: Decimal::ZERO,
                total,
            }),
            stop_reason: StopReason::EndTurn,
            latency_ms: 12,
            credential_id: None,
            cache_hit,
        }));
    }

    #[tokio::test]
    async fn spend_since_sums_exact_decimals_and_skips_cache_hits() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("usage.db");
        let bus = SqliteEventBus::open(
            &db,
            RetentionPolicy {
                max_age: None,
                max_rows: None,
                prune_every: std::time::Duration::from_secs(3600),
            },
        )
        .expect("open ledger");
        let priced = Decimal::from_str_exact("0.000123456789012345678901234").expect("decimal");
        seed_day_row(&bus, "r1", "key-a", Some(priced), false);
        // A NULL cost contributes zero spend but still counts as a request.
        seed_day_row(&bus, "r2", "key-a", None, false);
        // A served copy charges nothing and is not a request the key paid for.
        seed_day_row(
            &bus,
            "r3",
            "key-a",
            Some(Decimal::from_str_exact("5").expect("decimal")),
            true,
        );
        tokio::task::spawn_blocking(move || EventBus::close(&bus))
            .await
            .expect("flush the ledger");

        let since = OffsetDateTime::now_utc()
            .replace_time(time::Time::MIDNIGHT)
            .unix_timestamp()
            .saturating_mul(1000);
        let rows = spend_since(&db, since).expect("day-window spend query");

        assert_eq!(rows.len(), 1, "one caller group");
        assert_eq!(rows[0].caller.as_deref(), Some("key-a"));
        assert_eq!(
            rows[0].requests, 2,
            "the cache hit is not a request the key paid for; the NULL cost still counts"
        );
        assert_eq!(
            rows[0].spend, priced,
            "the cost sum must keep full decimal precision and exclude the cache hit"
        );
    }

    #[test]
    fn read_only_connections_refuse_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("usage.db");
        {
            let writer = Connection::open(&db).expect("create db");
            writer
                .execute_batch("CREATE TABLE t (a INTEGER);")
                .expect("seed table");
        }
        let reader = UsageReader::open(&db).expect("open read-only");
        let attempt = reader.execute("INSERT INTO t (a) VALUES (1)", []);
        assert!(
            attempt.is_err(),
            "SQLITE_OPEN_READ_ONLY must refuse writes at the connection level"
        );
        let count: i64 = reader
            .query_row("SELECT COUNT(*) FROM t", [], |row| row.get(0))
            .expect("read on the read-only connection");
        assert_eq!(count, 0);
    }

    #[test]
    fn group_by_parses_known_values_and_rejects_unknown() {
        assert_eq!(GroupBy::parse("provider"), Some(GroupBy::Provider));
        assert_eq!(GroupBy::parse("model"), Some(GroupBy::Model));
        assert_eq!(GroupBy::parse("caller"), Some(GroupBy::Caller));
        assert_eq!(GroupBy::parse("team"), None);
        assert_eq!(GroupBy::Provider.as_str(), "provider");
        assert_eq!(GroupBy::Model.as_str(), "model");
        assert_eq!(GroupBy::Caller.as_str(), "caller");
    }
}
