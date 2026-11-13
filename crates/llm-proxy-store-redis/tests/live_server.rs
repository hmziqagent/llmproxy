//! Plan 09 step 1's done-when against a **real** Redis server: two
//! instances sharing the store enforce one budget between them, windows
//! and cooldowns are shared, and usage events land in the stream.
//!
//! These tests need a live Redis and are `#[ignore]`d. Run them with:
//!
//! ```text
//! LLM_PROXY_TEST_REDIS=redis://localhost:6399 \
//!     cargo test -p llm-proxy-store-redis --test live_server -- --ignored
//! ```
//!
//! The default test run uses the in-repo minimal RESP server
//! (`tests/support/`); these live tests are the same properties checked
//! against the real thing, so wire-level divergences surface here.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use llm_proxy_core::{KeyBudgetConfig, KeyWindowLimits, ModelPricing, UnestimatableCostPolicy};
use llm_proxy_engine::{
    Admission, AdmissionClaim, AdmissionOutcome, Cooldowns, EngineError, KeyPolicy,
};
use llm_proxy_protocol::core::ModelRef;
use llm_proxy_storage::{EventBus, ProxyEvent, RequestReceived};
use llm_proxy_store_redis::{RedisCooldowns, RedisEventBus, RedisStore, replica_admission};
use time::OffsetDateTime;

fn live_url() -> Option<String> {
    std::env::var("LLM_PROXY_TEST_REDIS").ok()
}

/// A unique caller per run so repeated runs never inherit counters.
fn fresh_caller(prefix: &str) -> String {
    format!(
        "{prefix}-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos() % 1_000_000_000_000
    )
}

fn claim<'a>(caller: &'a str) -> AdmissionClaim<'a> {
    AdmissionClaim {
        caller,
        provider: "p",
        member: "p",
        model: "m",
        requested_model: "m",
        body_hash: "h",
        estimated_input_tokens: 500,
        max_output_tokens: Some(500),
    }
}

fn pricing() -> HashMap<(String, String), ModelPricing> {
    let price = ModelPricing {
        input: "0.001".parse().expect("decimal"),
        output: "0.001".parse().expect("decimal"),
        ..ModelPricing::default()
    };
    HashMap::from([(("p".to_owned(), "m".to_owned()), price)])
}

fn budgeted_keys(caller: &str, usd: &str) -> HashMap<String, KeyPolicy> {
    HashMap::from([(
        caller.to_owned(),
        KeyPolicy {
            budget: Some(KeyBudgetConfig {
                usd_per_day: usd.parse().expect("decimal"),
                unestimatable: UnestimatableCostPolicy::Reject,
                assumed_max_output_tokens: 100,
            }),
            ..KeyPolicy::default()
        },
    )])
}

async fn store() -> RedisStore {
    let url = live_url().expect("LLM_PROXY_TEST_REDIS must be set");
    RedisStore::connect(&url, "livesrv").await.expect("connect")
}

/// The done-when of plan 09 step 1, against the real server: two
/// replica admissions over one store enforce one budget between them —
/// concurrently — and a refund on one replica releases budget for the
/// other once the fire-and-forget settlement lands.
#[tokio::test]
#[ignore = "needs a live Redis; set LLM_PROXY_TEST_REDIS"]
async fn two_instances_sharing_a_real_redis_enforce_one_budget_between_them() {
    let store = store().await;
    let caller = fresh_caller("budget");
    let keys = budgeted_keys(&caller, "2.5");
    let replica_a = Arc::new(replica_admission(
        keys.clone(),
        pricing(),
        HashMap::new(),
        store.clone(),
    ));
    let replica_b = Arc::new(replica_admission(keys, pricing(), HashMap::new(), store));

    let mut handles = Vec::new();
    for index in 0..5 {
        let admission = if index % 2 == 0 {
            Arc::clone(&replica_a)
        } else {
            Arc::clone(&replica_b)
        };
        let caller = caller.clone();
        handles.push(tokio::spawn(async move {
            admission.reserve(&claim(&caller)).await.ok()
        }));
    }
    let mut tickets = Vec::new();
    for handle in handles {
        if let Some(ticket) = handle.await.expect("task") {
            tickets.push(ticket);
        }
    }
    assert_eq!(
        tickets.len(),
        2,
        "the real server bounds both replicas to the one budget"
    );

    replica_a.settle(
        &tickets.remove(0),
        &AdmissionOutcome {
            succeeded: false,
            usage: None,
            cost: None,
        },
    );
    let released = async {
        loop {
            if replica_b.reserve(&claim(&caller)).await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), released)
        .await
        .expect("a refund on one replica releases budget on the other (real server)");
}

#[tokio::test]
#[ignore = "needs a live Redis; set LLM_PROXY_TEST_REDIS"]
async fn rate_windows_and_cooldowns_are_shared_through_the_real_server() {
    let store = store().await;
    let caller = fresh_caller("win");
    let keys = HashMap::from([(
        caller.clone(),
        KeyPolicy {
            limits: KeyWindowLimits {
                rpm: Some(2),
                tpm: None,
                rpd: None,
            },
            ..KeyPolicy::default()
        },
    )]);
    let replica_a = replica_admission(keys.clone(), pricing(), HashMap::new(), store.clone());
    let replica_b = replica_admission(keys, pricing(), HashMap::new(), store.clone());

    replica_a
        .reserve(&claim(&caller))
        .await
        .expect("1/2 rpm on replica a");
    replica_b
        .reserve(&claim(&caller))
        .await
        .expect("2/2 rpm across both");
    match replica_a.reserve(&claim(&caller)).await {
        Err(EngineError::RateLimited {
            message,
            retry_after,
        }) => {
            assert!(message.contains("rpm"), "got {message}");
            let wait = retry_after.expect("window reset advised");
            assert!(wait.as_secs() >= 1 && wait.as_secs() <= 60, "got {wait:?}");
        }
        other => panic!("the shared window must refuse the third, got {other:?}"),
    }

    // Cooldowns cross replicas through the real server.
    let a = RedisCooldowns::new(store.clone());
    let b = RedisCooldowns::new(store);
    let now = OffsetDateTime::now_utc();
    let member = fresh_caller("member");
    a.cooldown_provider(&member, now + Duration::from_secs(30))
        .await;
    assert!(
        !b.provider_available(&member, now).await,
        "the other replica sees the cooldown"
    );
    a.cooldown_credential(&member, "key-0", now + Duration::from_secs(10))
        .await;
    assert_eq!(
        b.cooling_credentials(&member, now).await,
        vec!["key-0".to_owned()],
        "credential cooldowns cross replicas too"
    );
}

#[tokio::test]
#[ignore = "needs a live Redis; set LLM_PROXY_TEST_REDIS"]
async fn usage_events_reach_the_real_stream() {
    let store = store().await;
    let bus = RedisEventBus::new(store, "livesrv:usage:check");
    let request_id = fresh_caller("req");
    bus.emit(&ProxyEvent::RequestReceived(RequestReceived {
        request_id: request_id.clone(),
        timestamp: OffsetDateTime::now_utc(),
        caller: "key".to_owned(),
        provider: "p".to_owned(),
        route_kind: "messages".to_owned(),
        client_protocol: "anthropic".to_owned(),
        model: ModelRef {
            requested: "m".to_owned(),
            upstream: None,
        },
        streaming: false,
        body_hash: "abc".to_owned(),
        session_id: None,
    }));
    bus.close();
    // The XADD either landed or the writer logged a drop; assert it
    // landed by reading the stream length back through a fresh store.
    let landed = async {
        loop {
            let mut conn = conn_of().await;
            let len: i64 = {
                use redis::AsyncCommands;
                conn.xlen("livesrv:usage:check").await.unwrap_or(0)
            };
            if len >= 1 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), landed)
        .await
        .expect("the event reaches the real stream");
}

async fn conn_of() -> redis::aio::ConnectionManager {
    let url = live_url().expect("LLM_PROXY_TEST_REDIS must be set");
    let client = redis::Client::open(url).expect("client");
    client.get_connection_manager().await.expect("conn")
}

/// Fix 09-3 against the real server: the Lua set-if-greater script keeps
/// concurrent cooldown writes monotonic — sixteen writers over two
/// replicas in shuffled order leave the longest reset stored, and the
/// real server's own script engine (not the in-repo stand-in) is what
/// made it so.
#[tokio::test]
#[ignore = "needs a live Redis; set LLM_PROXY_TEST_REDIS"]
async fn live_cooldown_script_is_monotonic() {
    let store_a = store().await;
    let store_b = store().await;
    let a = Arc::new(RedisCooldowns::new(store_a));
    let b = Arc::new(RedisCooldowns::new(store_b));

    let member = fresh_caller("cdmono");
    let now = OffsetDateTime::now_utc();
    let untils: Vec<OffsetDateTime> = (1..=16_u64)
        .map(|offset| now + Duration::from_secs(offset))
        .collect();
    let longest = *untils.iter().max().expect("sixteen resets");

    let order = [3, 11, 7, 0, 14, 5, 9, 2, 15, 6, 12, 1, 8, 13, 4, 10];
    let mut handles = Vec::new();
    for (slot, index) in order.iter().enumerate() {
        let cooldowns = if slot % 2 == 0 {
            Arc::clone(&a)
        } else {
            Arc::clone(&b)
        };
        let member = member.clone();
        let until = untils[*index];
        handles.push(tokio::spawn(async move {
            cooldowns.cooldown_provider(&member, until).await;
        }));
    }
    for handle in handles {
        handle.await.expect("writer task");
    }

    let key = format!("livesrv:cd:p:{member}");
    let stored: Option<String> = {
        use redis::AsyncCommands;
        let mut conn = conn_of().await;
        conn.get(&key).await.expect("read the cooldown key")
    };
    let stored_ms: i64 = stored
        .expect("the longest cooldown is stored")
        .parse()
        .expect("the stored reset is unix ms");
    assert_eq!(
        Some(stored_ms),
        i64::try_from(longest.unix_timestamp_nanos().div_euclid(1_000_000)).ok(),
        "the real script stored the longest of the sixteen resets"
    );
    assert!(
        !b.provider_available(&member, now + Duration::from_secs(8))
            .await,
        "the other replica still sees the provider cooling"
    );
}
