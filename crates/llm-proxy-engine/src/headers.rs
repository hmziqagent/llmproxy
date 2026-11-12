//! Inbound header handling at the engine boundary: which client headers the
//! per-provider allowlist lets through to the upstream hop.

use std::collections::HashMap;

use crate::traits::InboundHeaders;

/// Applies the per-provider allowlist to the inbound headers: every
/// allowlisted name that is present (and not on the never-forwarded list)
/// is selected verbatim for the upstream request.
pub fn select_forwarded_headers(
    headers: &InboundHeaders,
    allowlist: &[String],
) -> HashMap<String, String> {
    let mut selected = HashMap::new();
    for name in allowlist {
        let lower = name.to_ascii_lowercase();
        if llm_proxy_core::provider_config::is_never_forwarded_header(&lower) {
            if headers.contains_key(&lower) {
                tracing::warn!(
                    header = %lower,
                    "inbound header is on the never-forwarded list; dropping"
                );
            }
            continue;
        }
        let Some(value) = headers.get(&lower) else {
            continue;
        };
        if value.contains('\r') || value.contains('\n') {
            tracing::warn!(header = %lower, "inbound header value contains CR/LF; dropping");
            continue;
        }
        selected.insert(lower, value.clone());
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> InboundHeaders {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn allowlisted_headers_are_selected_verbatim() {
        let inbound = headers(&[
            ("anthropic-beta", "claude-3-7-20250219"),
            ("x-app", "code-helper"),
            ("authorization", "Bearer client-secret"),
        ]);
        let allowlist = vec!["anthropic-beta".to_owned(), "x-session-id".to_owned()];
        let selected = select_forwarded_headers(&inbound, &allowlist);
        assert_eq!(
            selected.get("anthropic-beta").map(String::as_str),
            Some("claude-3-7-20250219")
        );
        assert!(
            !selected.contains_key("x-session-id"),
            "absent header is skipped"
        );
    }

    #[test]
    fn credentials_and_browser_headers_never_forward_even_if_allowlisted() {
        let inbound = headers(&[
            ("authorization", "Bearer x"),
            ("x-api-key", "x"),
            ("origin", "https://app.example"),
            ("sec-fetch-mode", "cors"),
            ("anthropic-beta", "b1"),
        ]);
        let allowlist: Vec<String> = [
            "authorization",
            "x-api-key",
            "origin",
            "sec-fetch-mode",
            "anthropic-beta",
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        let selected = select_forwarded_headers(&inbound, &allowlist);
        assert_eq!(
            selected.len(),
            1,
            "only the safe header survives: {selected:?}"
        );
        assert!(selected.contains_key("anthropic-beta"));
    }

    #[test]
    fn header_values_with_control_characters_are_dropped() {
        let inbound = headers(&[("x-session-id", "abc\r\nx-injected: 1")]);
        let allowlist = vec!["x-session-id".to_owned()];
        assert!(select_forwarded_headers(&inbound, &allowlist).is_empty());
    }

    #[test]
    fn select_forwarded_headers_enforces_the_shared_never_forwarded_set() {
        let mut names: Vec<&str> =
            llm_proxy_core::provider_config::NEVER_FORWARDED_HEADERS.to_vec();
        names.extend([
            "sec-fetch-mode",
            "access-control-allow-origin",
            "browser-use",
        ]);
        for name in names {
            let inbound = headers(&[(name, "value")]);
            let allowlist = vec![name.to_owned()];
            let selected = select_forwarded_headers(&inbound, &allowlist);
            assert!(
                selected.is_empty(),
                "never-forwarded header {name} must be dropped, got {selected:?}"
            );
        }
    }

    #[test]
    fn unlisted_inbound_headers_are_not_forwarded() {
        let inbound = headers(&[("x-stainless-lang", "js"), ("anthropic-beta", "b")]);
        let allowlist = vec!["anthropic-beta".to_owned()];
        let selected = select_forwarded_headers(&inbound, &allowlist);
        assert!(!selected.contains_key("x-stainless-lang"));
    }
}
