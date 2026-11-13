//! Plan 09 step 1's done-when against a live Postgres: two instances
//! sharing the replica stores enforce one budget between them, share rate
//! windows and cooldowns, and append to one usage table.
//!
//! These tests need a Postgres server and are `#[ignore]`d. Run them with:
//!
//! ```text
//! LLM_PROXY_TEST_POSTGRES=postgres://localhost/llm_proxy_test \
//!     cargo test -p llm-proxy-store-postgres --test replica_state -- --ignored
//! ```

#![cfg(test)]
use std::collections::HashMap;
use std::time::Duration;

use llm_proxy_core::{KeyBudgetConfig, ModelPricing, UnestimatableCostPolicy};
use llm_proxy_engine::{Admission, AdmissionClaim, AdmissionOutcome, Cooldowns, EngineError};
use llm_proxy_protocol::core::ModelRef;
use llm_proxy_storage::{EventBus, ProxyEvent, RequestReceived};
use llm_proxy_store_postgres::{
    PostgresCooldowns, PostgresEventBus, PostgresStore, replica_admission,
};
use time::OffsetDateTime;

fn test_url() -> Option<String> {
    std::env::var("LLM_PROXY_TEST_POSTGRES").ok()
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

async fn store() -> PostgresStore {
    let url = test_url().expect("LLM_PROXY_TEST_POSTGRES must be set");
    PostgresStore::connect(&url)
        .await
        .expect("connect to test Postgres")
}

/// The done-when of plan 09 step 1: two replica admissions over one
/// shared database enforce one budget between them.
#[tokio::test]
#[ignore = "needs a live Postgres; set LLM_PROXY_TEST_POSTGRES"]
async fn two_instances_sharing_the_store_enforce_one_budget_between_them() {
    let store = store().await;
    let caller = format!(
        "budget-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let keys = HashMap::from([(
        caller.clone(),
        llm_proxy_engine::KeyPolicy {
            budget: Some(KeyBudgetConfig {
                usd_per_day: "2.5".parse().expect("decimal"),
                unestimatable: UnestimatableCostPolicy::Reject,
                assumed_max_output_tokens: 100,
            }),
            ..llm_proxy_engine::KeyPolicy::default()
        },
    )]);
    let replica_a = std::sync::Arc::new(replica_admission(
        keys.clone(),
        pricing(),
        HashMap::new(),
        store.clone(),
    ));
    let replica_b = std::sync::Arc::new(replica_admission(keys, pricing(), HashMap::new(), store));

    let mut handles = Vec::new();
    for index in 0..5 {
        let admission = if index % 2 == 0 {
            std::sync::Arc::clone(&replica_a)
        } else {
            std::sync::Arc::clone(&replica_b)
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
    assert_eq!(tickets.len(), 2, "the shared budget bounds both replicas");

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
        .expect("a refund on one replica releases budget on the other");
}

#[tokio::test]
#[ignore = "needs a live Postgres; set LLM_PROXY_TEST_POSTGRES"]
async fn rate_windows_are_shared_across_instances() {
    let store = store().await;
    let caller = format!("win-{}", OffsetDateTime::now_utc().unix_timestamp_nanos());
    let keys = HashMap::from([(
        caller.clone(),
        llm_proxy_engine::KeyPolicy {
            limits: llm_proxy_core::KeyWindowLimits {
                rpm: Some(2),
                tpm: None,
                rpd: None,
            },
            ..llm_proxy_engine::KeyPolicy::default()
        },
    )]);
    let replica_a = replica_admission(keys.clone(), pricing(), HashMap::new(), store.clone());
    let replica_b = replica_admission(keys, pricing(), HashMap::new(), store);

    replica_a
        .reserve(&claim(&caller))
        .await
        .expect("1/2 rpm on replica a");
    replica_b
        .reserve(&claim(&caller))
        .await
        .expect("2/2 rpm across both");
    assert!(
        matches!(
            replica_a.reserve(&claim(&caller)).await,
            Err(EngineError::RateLimited { .. })
        ),
        "the shared window refuses the third request"
    );
}

#[tokio::test]
#[ignore = "needs a live Postgres; set LLM_PROXY_TEST_POSTGRES"]
async fn purging_expired_leases_releases_their_inflight_slots_and_reserved_budget() {
    // A crashed replica leaves leases that are never settled. The TTL purge
    // must release what those leases held — the in-flight concurrency slot
    // and the day's reserved budget — not merely delete the rows, or `cap`
    // crashed leases park the member at its cap forever and the budget
    // stays reserved until the day row itself is purged.
    let store = store().await;
    // A fresh caller AND member per run: the in-flight counters live in the
    // shared database, so reusing a member would inherit earlier runs'
    // slots.
    let nonce = OffsetDateTime::now_utc().unix_timestamp_nanos();
    let caller = format!("purge-{nonce}");
    let member = format!("purge-member-{nonce}");
    let keys = HashMap::from([(
        caller.clone(),
        llm_proxy_engine::KeyPolicy {
            budget: Some(KeyBudgetConfig {
                usd_per_day: "1.0".parse().expect("decimal"),
                unestimatable: UnestimatableCostPolicy::Reject,
                assumed_max_output_tokens: 100,
            }),
            ..llm_proxy_engine::KeyPolicy::default()
        },
    )]);
    let price = ModelPricing {
        input: "0.001".parse().expect("decimal"),
        output: "0.001".parse().expect("decimal"),
        ..ModelPricing::default()
    };
    let pricing = HashMap::from([((member.clone(), "m".to_owned()), price)]);
    let caps = HashMap::from([(member.clone(), (Some(1_u32), HashMap::new()))]);
    let admission = replica_admission(keys, pricing, caps, store.clone());
    let claim = AdmissionClaim {
        caller: &caller,
        provider: &member,
        member: &member,
        model: "m",
        requested_model: "m",
        body_hash: "h",
        estimated_input_tokens: 500,
        max_output_tokens: Some(500),
    };

    // One admitted, never-settled lease: the whole budget reserved, the
    // only member slot held.
    admission
        .reserve(&claim)
        .await
        .expect("the first reservation takes the slot and the budget");

    // Age that lease past the TTL, exactly as a crashed replica's lease
    // looks when the purge runs (scoped to this caller so parallel tests'
    // live leases are untouched).
    let url = test_url().expect("LLM_PROXY_TEST_POSTGRES must be set");
    let (client, connection) = tokio_postgres::connect(&url, tokio_postgres::NoTls)
        .await
        .expect("raw connect");
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            tracing::warn!(%error, "raw test connection ended");
        }
    });
    let rows = client
        .execute(
            "UPDATE admission_leases SET created_at = now() - interval '2 hours' \
             WHERE caller = $1",
            &[&caller],
        )
        .await
        .expect("age the lease");
    assert_eq!(rows, 1, "exactly this caller's lease was aged");
    drop(client);

    store.purge_expired().await.expect("the purge runs");

    admission
        .reserve(&claim)
        .await
        .expect("the purged lease must have released its slot and budget");
}

#[tokio::test]
#[ignore = "needs a live Postgres; set LLM_PROXY_TEST_POSTGRES"]
async fn cooldowns_written_by_one_replica_are_seen_by_the_other() {
    let store = store().await;
    let a = PostgresCooldowns::new(store.clone());
    let b = PostgresCooldowns::new(store);
    let now = OffsetDateTime::now_utc();
    let name = format!("member-{}", now.unix_timestamp_nanos());
    a.cooldown_provider(&name, now + Duration::from_secs(30))
        .await;
    assert!(!b.provider_available(&name, now).await);
    assert!(
        b.provider_available(&name, now + Duration::from_secs(31))
            .await
    );
}

#[tokio::test]
#[ignore = "needs a live Postgres; set LLM_PROXY_TEST_POSTGRES"]
async fn usage_events_append_to_the_shared_table() {
    let store = store().await;
    let bus = PostgresEventBus::new(store.clone());
    let request_id = format!("req-{}", OffsetDateTime::now_utc().unix_timestamp_nanos());
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
    let lands = async {
        loop {
            if let Ok(1) = store
                .count_usage_events_where("request_id", &request_id)
                .await
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), lands)
        .await
        .expect("the event reaches the shared table");
}

/// Per-member output prices in USD per token, for the rekey-repricing
/// mirror of the in-process backend (fix 05-7).
fn reprice_pricing(a_output: &str, b_output: &str) -> HashMap<(String, String), ModelPricing> {
    HashMap::from([
        (
            ("a".to_owned(), "m".to_owned()),
            ModelPricing {
                input: "0".parse().expect("decimal"),
                output: a_output.parse().expect("decimal"),
                ..ModelPricing::default()
            },
        ),
        (
            ("b".to_owned(), "m".to_owned()),
            ModelPricing {
                input: "0".parse().expect("decimal"),
                output: b_output.parse().expect("decimal"),
                ..ModelPricing::default()
            },
        ),
    ])
}

/// The shared database mirrors the in-process reference: a failover to a
/// pricier member grows the day reservation to the new member's estimate,
/// and a reservation that alone would exceed the budget refuses, leaving
/// the lease on its original estimate.
#[tokio::test]
#[ignore = "needs a live Postgres; set LLM_PROXY_TEST_POSTGRES"]
async fn a_rekey_to_a_pricier_member_moves_the_shared_day_reservation() {
    let store = store().await;
    let caller = format!("rekey-{}", OffsetDateTime::now_utc().unix_timestamp_nanos());
    let keys = HashMap::from([(
        caller.clone(),
        llm_proxy_engine::KeyPolicy {
            budget: Some(KeyBudgetConfig {
                usd_per_day: "0.011".parse().expect("decimal"),
                unestimatable: UnestimatableCostPolicy::Reject,
                assumed_max_output_tokens: 100,
            }),
            ..llm_proxy_engine::KeyPolicy::default()
        },
    )]);
    let admission = std::sync::Arc::new(replica_admission(
        keys,
        reprice_pricing("0.000002", "0.00002"),
        HashMap::new(),
        store,
    ));

    // a's estimate (0.001) fits twice; the lease then fails over to b,
    // whose estimate (0.01) fills the budget exactly.
    let first = admission
        .reserve(&claim(&caller))
        .await
        .expect("a's estimate fits");
    admission
        .reserve(&claim(&caller))
        .await
        .expect("a's estimate fits twice");

    admission
        .rekey(&first, "b", "m")
        .await
        .expect("b's pricier estimate still fits once");

    let refused = admission
        .reserve(&claim(&caller))
        .await
        .expect_err("a's estimate no longer fits beside b's");
    assert_eq!(refused.status_code(), 402);

    // Settlement releases what the lease now holds: b's estimate. The
    // settlement write is fire-and-forget, so poll until the refund lands.
    admission.settle(
        &first,
        &AdmissionOutcome {
            succeeded: false,
            usage: None,
            cost: None,
        },
    );
    let released = async {
        loop {
            if admission.reserve(&claim(&caller)).await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), released)
        .await
        .expect("the refund released b's re-priced estimate");
}

#[tokio::test]
#[ignore = "needs a live Postgres; set LLM_PROXY_TEST_POSTGRES"]
async fn a_rekey_that_would_exceed_the_shared_budget_is_refused() {
    let store = store().await;
    let caller = format!(
        "rekey-refused-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let keys = HashMap::from([(
        caller.clone(),
        llm_proxy_engine::KeyPolicy {
            budget: Some(KeyBudgetConfig {
                usd_per_day: "0.01".parse().expect("decimal"),
                unestimatable: UnestimatableCostPolicy::Reject,
                assumed_max_output_tokens: 100,
            }),
            ..llm_proxy_engine::KeyPolicy::default()
        },
    )]);
    let admission = std::sync::Arc::new(replica_admission(
        keys,
        reprice_pricing("0.000002", "0.00004"),
        HashMap::new(),
        store,
    ));

    let ticket = admission
        .reserve(&claim(&caller))
        .await
        .expect("a's estimate fits");
    let refused = admission
        .rekey(&ticket, "b", "m")
        .await
        .expect_err("b's estimate alone would exceed the budget");
    assert!(matches!(refused, EngineError::BudgetExhausted { .. }));

    // Nothing moved: a's estimate still fits beside the untouched
    // reservation.
    admission
        .reserve(&claim(&caller))
        .await
        .expect("a's estimate still fits beside the untouched reservation");

    // Settlement releases the original estimate, not b's; poll until the
    // fire-and-forget write lands.
    admission.settle(
        &ticket,
        &AdmissionOutcome {
            succeeded: false,
            usage: None,
            cost: None,
        },
    );
    let released = async {
        loop {
            if admission.reserve(&claim(&caller)).await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), released)
        .await
        .expect("the refund released the original estimate");
}
