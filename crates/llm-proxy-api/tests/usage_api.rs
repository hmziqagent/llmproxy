//! Integration tests for the `/usage` query surface over a real SQLite
//! ledger seeded through `SqliteEventBus`.

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::Request;
    use axum::http::StatusCode;
    use llm_proxy_protocol::core::{Cost, ModelRef, StopReason, Usage, UsageProvenance};
    use llm_proxy_storage::{
        EventBus, ProxyEvent, RequestReceived, ResponseCompleted, ResponseFailed, SqliteEventBus,
    };
    use rust_decimal::Decimal;
    use serde_json::Value;
    use time::OffsetDateTime;
    use tower::ServiceExt;

    pub(super) const ADMIN_TOKEN: &str = "test-admin-token";

    pub(super) struct Ledger {
        _dir: tempfile::TempDir,
        pub(super) path: std::path::PathBuf,
        t0: OffsetDateTime,
    }

    pub(super) fn received_event(
        id: &str,
        caller: &str,
        provider: &str,
        model: &str,
        ts: OffsetDateTime,
    ) -> ProxyEvent {
        ProxyEvent::RequestReceived(RequestReceived {
            request_id: id.to_owned(),
            timestamp: ts,
            caller: caller.to_owned(),
            provider: provider.to_owned(),
            route_kind: "messages".to_owned(),
            client_protocol: "anthropic".to_owned(),
            model: ModelRef {
                requested: model.to_owned(),
                upstream: None,
            },
            streaming: true,
            body_hash: format!("sha256-digest-{id}"),
            session_id: None,
        })
    }

    pub(super) fn completed_event(
        id: &str,
        provider: &str,
        model: &str,
        ts: OffsetDateTime,
        total: &str,
    ) -> ProxyEvent {
        ProxyEvent::ResponseCompleted(ResponseCompleted {
            request_id: id.to_owned(),
            timestamp: ts,
            provider: provider.to_owned(),
            upstream_message_id: None,
            model: ModelRef {
                requested: model.to_owned(),
                upstream: None,
            },
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
                reasoning_tokens: Some(7),
                cache_creation_input_tokens: Some(30),
                cache_read_input_tokens: Some(200),
                provenance: UsageProvenance::ProviderReported,
            },
            cost: Some(Cost {
                input: Decimal::from_str_exact(total).expect("decimal fixture"),
                output: Decimal::ZERO,
                cache_creation: Decimal::ZERO,
                cache_read: Decimal::ZERO,
                reasoning: Decimal::ZERO,
                total: Decimal::from_str_exact(total).expect("decimal fixture"),
            }),
            stop_reason: StopReason::EndTurn,
            latency_ms: 4321,
            credential_id: None,
            cache_hit: false,
        })
    }

    fn completed_event_without_cost(
        id: &str,
        provider: &str,
        model: &str,
        ts: OffsetDateTime,
    ) -> ProxyEvent {
        let mut event = completed_event(id, provider, model, ts, "0");
        if let ProxyEvent::ResponseCompleted(done) = &mut event {
            done.cost = None;
        }
        event
    }

    pub(super) fn failed_event(id: &str, provider: Option<&str>, ts: OffsetDateTime) -> ProxyEvent {
        ProxyEvent::ResponseFailed(ResponseFailed {
            request_id: id.to_owned(),
            timestamp: ts,
            provider: provider.map(str::to_owned),
            model: None,
            error_kind: "UpstreamError".to_owned(),
            message: "sanitized failure text".to_owned(),
            http_status: 502,
            latency_ms: 99,
        })
    }

    pub(super) async fn seed() -> Ledger {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("usage.db");
        let bus = SqliteEventBus::open(
            &path,
            llm_proxy_storage::RetentionPolicy {
                max_age: None,
                max_rows: None,
                prune_every: std::time::Duration::from_secs(3600),
            },
        )
        .expect("open ledger");

        let t0 = OffsetDateTime::now_utc().saturating_sub(time::Duration::seconds(60));
        let second = std::time::Duration::from_secs(1);
        let events = [
            received_event("req-1", "laptop", "prov-a", "claude-3", t0),
            completed_event(
                "req-1",
                "prov-a",
                "claude-3",
                t0 + second * 4,
                "0.1000000000000000000000001",
            ),
            received_event("req-2", "laptop", "prov-a", "gpt-4", t0 + second),
            completed_event(
                "req-2",
                "prov-a",
                "gpt-4",
                t0 + second * 5,
                "0.2000000000000000000000002",
            ),
            received_event("req-3", "phone", "prov-b", "claude-3", t0 + second * 2),
            failed_event("req-3", Some("prov-b"), t0 + second * 6),
            // prov-free serves usage with no pricing entry: unpriced.
            received_event(
                "req-4",
                "tablet",
                "prov-free",
                "free-model",
                t0 + second / 2,
            ),
            completed_event_without_cost("req-4", "prov-free", "free-model", t0 + second / 2),
            failed_event("orphan-1", None, t0 + second * 3),
            received_event(
                "old-1",
                "laptop",
                "prov-old",
                "claude-3",
                t0.saturating_sub(time::Duration::seconds(7200)),
            ),
            completed_event(
                "old-1",
                "prov-old",
                "claude-3",
                t0.saturating_sub(time::Duration::seconds(7190)),
                "0.5",
            ),
        ];
        for event in &events {
            bus.emit(event);
        }
        let closer = std::sync::Arc::new(bus);
        tokio::task::spawn_blocking(move || closer.close())
            .await
            .expect("close joined");
        Ledger {
            _dir: dir,
            path,
            t0,
        }
    }

    pub(super) fn app_for(path: &std::path::Path) -> axum::Router {
        llm_proxy_api::UsageApi::new(path.to_owned(), ADMIN_TOKEN).router()
    }

    /// Seeds a ledger with one priced completion and two served copies of it
    /// (cache hits: usage kept for the record, no cost), for the summary's
    /// served-copy accounting.
    async fn seed_with_served_copies() -> Ledger {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("usage.db");
        let bus = SqliteEventBus::open(
            &path,
            llm_proxy_storage::RetentionPolicy {
                max_age: None,
                max_rows: None,
                prune_every: std::time::Duration::from_secs(3600),
            },
        )
        .expect("open ledger");
        let t0 = OffsetDateTime::now_utc();
        let second = std::time::Duration::from_secs(1);
        let mut events = vec![
            received_event("orig", "laptop", "prov-a", "claude-3", t0),
            completed_event(
                "orig",
                "prov-a",
                "claude-3",
                t0 + second,
                "0.1000000000000000000000001",
            ),
        ];
        for (id, offset) in [("copy-1", 2), ("copy-2", 3)] {
            events.push(received_event(
                id,
                "laptop",
                "prov-a",
                "claude-3",
                t0 + second * offset,
            ));
            let mut served = completed_event(id, "prov-a", "claude-3", t0 + second * offset, "0");
            if let ProxyEvent::ResponseCompleted(done) = &mut served {
                done.cost = None;
                done.cache_hit = true;
            }
            events.push(served);
        }
        for event in &events {
            bus.emit(event);
        }
        let closer = std::sync::Arc::new(bus);
        tokio::task::spawn_blocking(move || closer.close())
            .await
            .expect("close joined");
        Ledger {
            _dir: dir,
            path,
            t0,
        }
    }

    #[tokio::test]
    async fn served_copies_never_inflate_tokens_cost_or_unpriced_counts() {
        // Plan 05 step 6: cache hits charge nothing and are recorded as
        // hits. One upstream call served to three requests must show one
        // call's tokens and cost, two hits, and no unpriced completions.
        let ledger = seed_with_served_copies().await;
        let (status, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?group_by=provider",
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let group = &body["groups"][0];
        assert_eq!(group["requests"], 3, "every served request is a request");
        assert_eq!(group["cache_hits"], 2);
        assert_eq!(
            group["tokens"]["input"], 100,
            "one upstream call's tokens, not three"
        );
        assert_eq!(group["tokens"]["output"], 50);
        assert_eq!(
            group["cost"]["total"], "0.1000000000000000000000001",
            "one upstream call's cost, not three"
        );
        assert_eq!(
            group["unpriced"], 0,
            "a served copy has no cost by design; it is not an unpriced completion"
        );
    }

    /// Seeds one ledger with three provider groups: all priced (prov-a),
    /// all unpriced (prov-free), and prov-mixed — one priced completion and
    /// one unpriced completion — for the summary's `cost_partial` flag.
    async fn seed_with_mixed_pricing_group() -> Ledger {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("usage.db");
        let bus = SqliteEventBus::open(
            &path,
            llm_proxy_storage::RetentionPolicy {
                max_age: None,
                max_rows: None,
                prune_every: std::time::Duration::from_secs(3600),
            },
        )
        .expect("open ledger");
        let t0 = OffsetDateTime::now_utc();
        let second = std::time::Duration::from_secs(1);
        let events = [
            received_event("mix-a", "laptop", "prov-a", "claude-3", t0),
            completed_event(
                "mix-a",
                "prov-a",
                "claude-3",
                t0 + second,
                "0.1000000000000000000000001",
            ),
            received_event(
                "mix-free",
                "tablet",
                "prov-free",
                "free-model",
                t0 + second * 2,
            ),
            completed_event_without_cost("mix-free", "prov-free", "free-model", t0 + second * 2),
            received_event(
                "mix-priced",
                "phone",
                "prov-mixed",
                "claude-3",
                t0 + second * 3,
            ),
            completed_event(
                "mix-priced",
                "prov-mixed",
                "claude-3",
                t0 + second * 4,
                "0.0400000000000000000000004",
            ),
            received_event(
                "mix-unpriced",
                "phone",
                "prov-mixed",
                "free-model",
                t0 + second * 5,
            ),
            completed_event_without_cost(
                "mix-unpriced",
                "prov-mixed",
                "free-model",
                t0 + second * 6,
            ),
        ];
        for event in &events {
            bus.emit(event);
        }
        let closer = std::sync::Arc::new(bus);
        tokio::task::spawn_blocking(move || closer.close())
            .await
            .expect("close joined");
        Ledger {
            _dir: dir,
            path,
            t0,
        }
    }

    /// Plan 05 step 9: a group mixing priced and unpriced completions must
    /// not present its priced-only sum as the full cost.
    #[tokio::test]
    async fn a_group_mixing_priced_and_unpriced_rows_flags_its_cost_partial() {
        let ledger = seed_with_mixed_pricing_group().await;
        let (status, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?group_by=provider",
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let groups = body["groups"].as_array().expect("groups array");
        let find = |key: &str| {
            groups
                .iter()
                .find(|g| g["key"] == key)
                .unwrap_or_else(|| panic!("group {key} missing: {groups:?}"))
        };

        let mixed = find("prov-mixed");
        assert_eq!(mixed["requests"], 2);
        assert_eq!(
            mixed["cost"]["total"], "0.0400000000000000000000004",
            "cost covers only the priced row's exact decimal"
        );
        assert_eq!(mixed["unpriced"], 1);
        assert_eq!(
            mixed["cost_partial"], true,
            "a priced-only sum next to an unpriced completion is partial"
        );

        let prov_a = find("prov-a");
        assert_eq!(prov_a["unpriced"], 0);
        assert_eq!(
            prov_a["cost_partial"], false,
            "a fully priced group's sum is complete"
        );

        let prov_free = find("prov-free");
        assert!(
            prov_free["cost"].is_null(),
            "an all-unpriced group keeps null cost, got {}",
            prov_free["cost"]
        );
        assert_eq!(prov_free["unpriced"], 1);
        assert_eq!(
            prov_free["cost_partial"], false,
            "null cost has nothing to be partial about"
        );
    }

    async fn get_json(app: axum::Router, uri: &str, token: Option<&str>) -> (StatusCode, Value) {
        let mut builder = Request::builder().uri(uri);
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let response = app
            .oneshot(builder.body(Body::empty()).expect("request builds"))
            .await
            .expect("oneshot");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let json = serde_json::from_slice(&bytes).expect("body is JSON");
        (status, json)
    }

    #[tokio::test]
    async fn usage_routes_refuse_missing_and_wrong_tokens_without_echoing_them() {
        let ledger = seed().await;
        for path in ["/usage/summary", "/usage/requests", "/usage/keys"] {
            let (status, body) = get_json(app_for(&ledger.path), path, None).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{path} without token");
            assert_eq!(
                body["error"]["message"], "missing or invalid admin token",
                "uniform 401 body for {path}"
            );

            let (status, body_text) = {
                let (status, body) =
                    get_json(app_for(&ledger.path), path, Some("wrong-token")).await;
                (status, body.to_string())
            };
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{path} with wrong token");
            assert!(
                !body_text.contains("wrong-token") && !body_text.contains(ADMIN_TOKEN),
                "the 401 body must not echo any presented or configured token: {body_text}"
            );

            let (status, _) = get_json(app_for(&ledger.path), path, Some(ADMIN_TOKEN)).await;
            assert_eq!(status, StatusCode::OK, "{path} with the admin token");
        }
    }

    /// Backlog 37: every admin API response — successes and the 401 refusal
    /// alike — must forbid intermediary caching. A CDN that caches an
    /// authenticated response turns the token check into a one-time check.
    #[tokio::test]
    async fn every_admin_response_forbids_caching() {
        let ledger = seed().await;
        for path in ["/usage/summary", "/usage/requests", "/usage/keys"] {
            for token in [Some(ADMIN_TOKEN), Some("wrong-token"), None] {
                let app = app_for(&ledger.path);
                let builder = Request::builder().uri(path);
                let builder = match token {
                    Some(token) => builder.header("authorization", format!("Bearer {token}")),
                    None => builder,
                };
                let response = app
                    .oneshot(builder.body(Body::empty()).expect("request builds"))
                    .await
                    .expect("oneshot");
                let status = response.status();
                let cache_control = response
                    .headers()
                    .get("cache-control")
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or_default();
                let pragma = response
                    .headers()
                    .get("pragma")
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or_default();
                assert!(
                    status.is_success() || status == StatusCode::UNAUTHORIZED,
                    "{path}: unexpected status {status}"
                );
                assert_eq!(
                    cache_control, "no-store",
                    "{path} ({status}) must carry Cache-Control: no-store, got {cache_control:?}"
                );
                assert_eq!(
                    pragma, "no-cache",
                    "{path} ({status}) must carry Pragma: no-cache, got {pragma:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn summary_groups_by_provider_with_exact_decimal_costs() {
        let ledger = seed().await;
        let (status, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?group_by=provider",
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let groups = body["groups"].as_array().expect("groups array");
        let find = |key: &str| {
            groups
                .iter()
                .find(|g| g["key"] == key)
                .unwrap_or_else(|| panic!("group {key} missing: {groups:?}"))
        };

        let prov_a = find("prov-a");
        assert_eq!(prov_a["requests"], 2);
        assert_eq!(prov_a["errors"], 0, "both prov-a requests completed");
        assert_eq!(
            prov_a["tokens"]["input"], 200,
            "100 input tokens per request"
        );
        assert_eq!(prov_a["tokens"]["output"], 100);
        assert_eq!(prov_a["tokens"]["reasoning"], 14);
        assert_eq!(prov_a["tokens"]["cache_read"], 400);
        assert_eq!(prov_a["tokens"]["cache_creation"], 60);
        assert_eq!(
            prov_a["cost"]["total"], "0.3000000000000000000000003",
            "cost sums must be exact decimals — a float sum of these two values \
             yields 0.30000000000000004"
        );
        assert_eq!(
            prov_a["cost"]["input"], "0.3000000000000000000000003",
            "input-cost column sums exactly too"
        );

        let prov_b = find("prov-b");
        assert_eq!(prov_b["requests"], 1);
        assert_eq!(prov_b["errors"], 1);

        // BUG-24: usage on an unpriced model is an explicit unpriced state,
        // never a silent $0.
        let prov_free = find("prov-free");
        assert_eq!(prov_free["requests"], 1);
        assert!(
            prov_free["cost"].is_null(),
            "a group of unpriced rows must report null cost, got {}",
            prov_free["cost"]
        );
        assert_eq!(prov_free["unpriced"], 1);
        assert_eq!(prov_free["cache_hits"], 0);
        // Priced groups stay priced, with the same exact decimals as before.
        assert_eq!(prov_a["unpriced"], 0);

        let null_group = groups
            .iter()
            .find(|g| g["key"].is_null())
            .expect("failure-only rows group under a null provider");
        assert_eq!(null_group["requests"], 1);
        assert_eq!(null_group["errors"], 1);

        assert_eq!(body["group_by"], "provider");
    }

    #[tokio::test]
    async fn summary_groups_by_caller_and_model() {
        let ledger = seed().await;
        let (_, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?group_by=caller",
            Some(ADMIN_TOKEN),
        )
        .await;
        let groups = &body["groups"];
        let laptop = groups
            .as_array()
            .expect("groups")
            .iter()
            .find(|g| g["key"] == "laptop")
            .expect("laptop group");
        assert_eq!(
            laptop["requests"], 3,
            "req-1, req-2 and the two-hour-old row"
        );
        assert_eq!(laptop["errors"], 0);

        let (_, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?group_by=model",
            Some(ADMIN_TOKEN),
        )
        .await;
        let model_groups = body["groups"].as_array().expect("groups");
        let claude = model_groups
            .iter()
            .find(|g| g["key"] == "claude-3")
            .expect("claude-3 group");
        assert_eq!(claude["requests"], 3, "req-1, req-3 and old-1");
        let gpt = model_groups
            .iter()
            .find(|g| g["key"] == "gpt-4")
            .expect("gpt-4 group");
        assert_eq!(gpt["requests"], 1);
    }

    #[tokio::test]
    async fn summary_window_excludes_rows_older_than_it() {
        let ledger = seed().await;
        let (_, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?window=1h",
            Some(ADMIN_TOKEN),
        )
        .await;
        let groups = body["groups"].as_array().expect("groups");
        assert!(
            groups.iter().all(|g| g["key"] != "prov-old"),
            "the two-hour-old row must fall outside a 1h window: {groups:?}"
        );
        let (_, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?window=4h",
            Some(ADMIN_TOKEN),
        )
        .await;
        let groups = body["groups"].as_array().expect("groups");
        assert!(
            groups.iter().any(|g| g["key"] == "prov-old"),
            "a 4h window covers the two-hour-old row"
        );

        let (status, _) = get_json(
            app_for(&ledger.path),
            "/usage/summary?window=7d",
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "\"7d\" must be an accepted window");
    }

    #[tokio::test]
    async fn summary_rejects_bad_window_and_group_by() {
        let ledger = seed().await;
        let (status, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?window=abc",
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].to_string().contains("window"));

        let (status, _) = get_json(
            app_for(&ledger.path),
            "/usage/summary?window=0s",
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "zero window is not a window"
        );

        let (status, body) = get_json(
            app_for(&ledger.path),
            "/usage/summary?group_by=team",
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].to_string().contains("group_by"));
    }

    #[tokio::test]
    async fn requests_pages_newest_first_with_cursor_and_provider_filter() {
        let ledger = seed().await;
        let (status, page1) = get_json(
            app_for(&ledger.path),
            "/usage/requests?limit=2",
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let rows1 = page1["rows"].as_array().expect("rows");
        assert_eq!(rows1.len(), 2, "limit is honored");
        let first_id = rows1[0]["request_id"].as_str().expect("id");
        assert_eq!(
            first_id, "orphan-1",
            "newest first: orphan-1 completed latest among the seeded rows"
        );
        let cursor = page1["next_cursor"].as_str().expect("cursor present");
        assert_eq!(rows1[1]["request_id"], "req-3");

        let (status, page2) = get_json(
            app_for(&ledger.path),
            &format!("/usage/requests?limit=10&cursor={cursor}"),
            Some(ADMIN_TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let rows2 = page2["rows"].as_array().expect("rows");
        assert!(
            rows2
                .iter()
                .all(|r| r["request_id"] != "orphan-1" && r["request_id"] != "req-3"),
            "cursor paging must not repeat rows: {rows2:?}"
        );

        let (_, filtered) = get_json(
            app_for(&ledger.path),
            "/usage/requests?provider=prov-a",
            Some(ADMIN_TOKEN),
        )
        .await;
        let filtered_rows = filtered["rows"].as_array().expect("rows");
        assert!(
            !filtered_rows.is_empty() && filtered_rows.iter().all(|r| r["provider"] == "prov-a"),
            "provider filter: {filtered_rows:?}"
        );
    }

    #[tokio::test]
    async fn requests_honors_since_and_until_bounds() {
        let ledger = seed().await;
        let until = (ledger.t0 + std::time::Duration::from_secs(3))
            .format(&time::format_description::well_known::Rfc3339)
            .expect("rfc3339");
        let (_, body) = get_json(
            app_for(&ledger.path),
            &format!("/usage/requests?until={until}"),
            Some(ADMIN_TOKEN),
        )
        .await;
        let ids: Vec<&str> = body["rows"]
            .as_array()
            .expect("rows")
            .iter()
            .filter_map(|r| r["request_id"].as_str())
            .collect();
        assert_eq!(
            ids,
            vec!["req-3", "req-2", "req-4", "req-1", "old-1"],
            "until is exclusive on the ledger time axis (orphan-1 at t0+3s is out)"
        );

        let since = (ledger.t0 + std::time::Duration::from_secs(1))
            .format(&time::format_description::well_known::Rfc3339)
            .expect("rfc3339");
        let (_, body) = get_json(
            app_for(&ledger.path),
            &format!("/usage/requests?since={since}"),
            Some(ADMIN_TOKEN),
        )
        .await;
        let ids: Vec<&str> = body["rows"]
            .as_array()
            .expect("rows")
            .iter()
            .filter_map(|r| r["request_id"].as_str())
            .collect();
        assert_eq!(
            ids,
            vec!["orphan-1", "req-3", "req-2"],
            "since is inclusive"
        );
    }

    #[tokio::test]
    async fn requests_rejects_invalid_parameters() {
        let ledger = seed().await;
        let cases = [
            "/usage/requests?limit=0",
            "/usage/requests?limit=1001",
            "/usage/requests?limit=-5",
            "/usage/requests?since=not-a-time",
            "/usage/requests?until=not-a-time",
            "/usage/requests?cursor=garbage",
            "/usage/requests?cursor=123:",
            "/usage/requests?since=2026-01-02T03:04:05Z&until=2026-01-01T00:00:00Z",
        ];
        for uri in cases {
            let (status, body) = get_json(app_for(&ledger.path), uri, Some(ADMIN_TOKEN)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {body}");
        }
    }

    #[tokio::test]
    async fn requests_rows_carry_digests_but_never_bodies() {
        let ledger = seed().await;
        let (_, body) = get_json(app_for(&ledger.path), "/usage/requests", Some(ADMIN_TOKEN)).await;
        let text = body.to_string();
        assert!(
            text.contains("sha256-digest-"),
            "rows expose the body digest for correlation"
        );
        for row in body["rows"].as_array().expect("rows") {
            let keys: Vec<&str> = row
                .as_object()
                .expect("row object")
                .keys()
                .map(String::as_str)
                .collect();
            assert!(
                !keys
                    .iter()
                    .any(|k| *k == "body" || *k == "request_body" || *k == "message"),
                "raw rows must never carry request bodies or sanitized messages: {keys:?}"
            );
        }
    }

    #[tokio::test]
    async fn keys_lists_callers_with_counts_and_last_seen() {
        let ledger = seed().await;
        let (status, body) =
            get_json(app_for(&ledger.path), "/usage/keys", Some(ADMIN_TOKEN)).await;
        assert_eq!(status, StatusCode::OK);
        let callers = body["callers"].as_array().expect("callers");
        let find = |name: &str| {
            callers
                .iter()
                .find(|c| c["caller"] == name)
                .unwrap_or_else(|| panic!("caller {name} missing: {callers:?}"))
        };
        let laptop = find("laptop");
        assert_eq!(laptop["requests"], 3, "req-1, req-2, old-1");
        assert_eq!(laptop["errors"], 0);
        let phone = find("phone");
        assert_eq!(phone["requests"], 1);
        assert_eq!(phone["errors"], 1);
        assert!(laptop["last_seen_ms"].is_u64(), "last_seen_ms is populated");

        let dated: Vec<i64> = callers
            .iter()
            .filter_map(|c| c["last_seen_ms"].as_i64())
            .collect();
        assert!(
            !dated.is_empty(),
            "callers with ledger activity must carry last_seen_ms: {callers:?}"
        );
        assert_eq!(
            dated,
            {
                let mut sorted = dated.clone();
                sorted.sort_by(|a, b| b.cmp(a));
                sorted
            },
            "callers appear most-recently-seen first"
        );
        let order: Vec<&str> = callers
            .iter()
            .map(|c| c["caller"].as_str().unwrap_or("<null>"))
            .collect();
        assert_eq!(
            order,
            vec!["<null>", "phone", "laptop", "tablet"],
            "caller ordering follows last-seen timestamps, null caller included"
        );
    }

    #[tokio::test]
    async fn summary_serves_while_the_ledger_writer_is_active() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("usage.db");
        let bus = std::sync::Arc::new(
            SqliteEventBus::open(
                &path,
                llm_proxy_storage::RetentionPolicy {
                    max_age: None,
                    max_rows: None,
                    prune_every: std::time::Duration::from_secs(3600),
                },
            )
            .expect("open ledger"),
        );
        bus.emit(&received_event(
            "live-1",
            "laptop",
            "prov-live",
            "claude-3",
            OffsetDateTime::now_utc(),
        ));

        let mut seen = false;
        for _ in 0..200 {
            let (status, body) = get_json(
                app_for(&path),
                "/usage/summary?group_by=provider",
                Some(ADMIN_TOKEN),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "query against live writer");
            let found = body["groups"]
                .as_array()
                .is_some_and(|g| g.iter().any(|x| x["key"] == "prov-live"));
            if found {
                seen = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            seen,
            "the row must become visible while the writer is active"
        );
        let closer = std::sync::Arc::clone(&bus);
        tokio::task::spawn_blocking(move || closer.close())
            .await
            .expect("close joined");
    }
}

#[cfg(test)]
mod metrics_route {
    use super::tests::ADMIN_TOKEN;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use llm_proxy_core::Metrics;
    use std::sync::Arc;
    use std::time::Duration;
    use tower::ServiceExt;

    /// Backlog 21: the admin surface serves the runtime counters as
    /// Prometheus text, behind the admin token like every other route.
    #[tokio::test]
    async fn metrics_serves_prometheus_text_behind_the_admin_token() {
        let ledger = super::tests::seed().await;
        let metrics = Arc::new(Metrics::new());
        metrics.record_request(false);
        metrics.record_failure();
        metrics.record_success("prov-a", "gpt-4o", Duration::from_millis(5));
        let app = llm_proxy_api::UsageApi::new(ledger.path.clone(), ADMIN_TOKEN)
            .with_metrics(Arc::clone(&metrics))
            .router();

        // No or wrong token: the same 401 as every other admin route.
        for token in [None, Some("wrong-token")] {
            let mut builder = Request::builder().uri("/metrics");
            if let Some(token) = token {
                builder = builder.header("authorization", format!("Bearer {token}"));
            }
            let response = app
                .clone()
                .oneshot(builder.body(Body::empty()).expect("request builds"))
                .await
                .expect("oneshot");
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                    .body(Body::empty())
                    .expect("request builds"),
            )
            .await
            .expect("oneshot");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/plain; version=0.0.4; charset=utf-8"
        );
        // no-store applies here like everywhere on the admin surface.
        assert_eq!(response.headers().get("cache-control").unwrap(), "no-store");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&bytes).into_owned();
        assert!(
            text.contains("llm_proxy_requests_total 1"),
            "the scraped exposition carries the live counter: {text}"
        );
        assert!(
            text.contains("llm_proxy_requests_failed_total 1"),
            "the scraped exposition carries the live counter: {text}"
        );
        assert!(
            text.contains(r#"llm_proxy_model_requests_total{provider="prov-a",model="gpt-4o"} 1"#),
            "the scraped exposition carries per-model counts: {text}"
        );
    }

    /// Without metrics attached there is no /metrics route at all: the
    /// surface keeps its exact shape unless the operator's server wired one.
    #[tokio::test]
    async fn without_metrics_attached_the_route_does_not_exist() {
        let ledger = super::tests::seed().await;
        let app = llm_proxy_api::UsageApi::new(ledger.path, ADMIN_TOKEN).router();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                    .body(Body::empty())
                    .expect("request builds"),
            )
            .await
            .expect("oneshot");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}

#[cfg(test)]
mod in_flight_visibility {
    use super::tests::{ADMIN_TOKEN, app_for, completed_event, failed_event, received_event};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use llm_proxy_storage::{EventBus, RetentionPolicy, SqliteEventBus};
    use serde_json::Value;
    use std::time::Duration;
    use time::OffsetDateTime;
    use tower::ServiceExt;

    /// Backlog 27: /usage/requests already returns rows written at
    /// RequestReceived; now each row says which state it is in, and
    /// in-flight rows carry elapsed time measured at query time.
    #[tokio::test]
    async fn request_rows_report_state_and_in_flight_elapsed_time() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("usage.db");
        let bus = SqliteEventBus::open(
            &path,
            RetentionPolicy {
                max_age: None,
                max_rows: None,
                prune_every: Duration::from_secs(3600),
            },
        )
        .expect("open ledger");
        let t0 = OffsetDateTime::now_utc();
        let second = Duration::from_secs(1);

        // A request whose received row exists and never completed: in flight.
        bus.emit(&received_event(
            "live-1", "laptop", "prov-a", "claude-3", t0,
        ));
        // One that completed.
        bus.emit(&received_event(
            "done-1",
            "laptop",
            "prov-a",
            "claude-3",
            t0 + second,
        ));
        bus.emit(&completed_event(
            "done-1",
            "prov-a",
            "claude-3",
            t0 + 2 * second,
            "0.5",
        ));
        // One that failed.
        bus.emit(&received_event(
            "sad-1",
            "laptop",
            "prov-a",
            "claude-3",
            t0 + 3 * second,
        ));
        bus.emit(&failed_event("sad-1", Some("prov-a"), t0 + 4 * second));
        let closer = std::sync::Arc::new(bus);
        tokio::task::spawn_blocking(move || closer.close())
            .await
            .expect("close joined");

        let app = app_for(&path);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/usage/requests")
                    .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                    .body(Body::empty())
                    .expect("request builds"),
            )
            .await
            .expect("oneshot");
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body: Value = serde_json::from_slice(&bytes).expect("body is JSON");

        let find = |id: &str| {
            body["rows"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["request_id"] == id)
                .unwrap_or_else(|| panic!("{id} missing from rows: {body}"))
                .clone()
        };
        let live = find("live-1");
        assert_eq!(live["state"], "in_flight", "row: {live}");
        let elapsed = live["elapsed_ms"].as_i64().expect("elapsed is present");
        assert!(
            elapsed >= 0,
            "elapsed time is measured at query time: {live}"
        );
        let done = find("done-1");
        assert_eq!(done["state"], "completed", "row: {done}");
        assert!(
            done["elapsed_ms"].is_null(),
            "settled rows carry latency_ms, not elapsed: {done}"
        );
        let sad = find("sad-1");
        assert_eq!(sad["state"], "failed", "row: {sad}");
        assert!(sad["elapsed_ms"].is_null(), "row: {sad}");
    }
}
