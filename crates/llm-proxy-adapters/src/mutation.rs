//! The proxy's request mutations run in a fixed order (audit conflict 5,
//! plan 06 step 6). Every adapter applies the same phases to a same-format
//! (passthrough) request, in this order:
//!
//! 1. **Route edits** — the model alias. Top-level routing fields only.
//! 2. **Dialect edits** — reasoning-parameter translation, stripping
//!    foreign fields the target wire format does not define, and (for
//!    cross-format requests) encoding the normalized core request under
//!    the field policy. On a same-format route these never alter prompt
//!    content: `tools`, `system` and `messages` survive byte-for-byte.
//! 3. **Operator hygiene** (opt-in, `[provider.prompt_cache]`, plan 06
//!    step 7) — volatile-byte removal and cache-marker injection at the
//!    provider edge, applied last, immediately before serialization.
//!
//! The invariant the order enforces: nothing changes the bytes of the
//! prompt prefix before the last cache breakpoint unless the operator
//! opted in. Phases 1-2 touch routing and dialect fields; phase 3 is the
//! only phase that may touch content, and only per its explicit list. A
//! client that sets its own cache markers keeps them — hygiene never moves
//! or duplicates a marker the client placed.

use llm_proxy_core::PromptCacheConfig;
use serde_json::Value;

/// Anthropic's prompt-cache breakpoint budget: at most four `cache_control`
/// markers per request.
const ANTHROPIC_MAX_BREAKPOINTS: usize = 4;

/// Phase 3, at the provider edge (plan 06 step 7): strip the configured
/// volatile pointers, then inject cache markers when the operator opted in.
/// Marker injection is meaningful only where the wire format has cache
/// markers (Anthropic Messages); other protocols strip only.
pub(crate) fn apply_prompt_hygiene(
    request: &mut Value,
    config: &PromptCacheConfig,
    inject_markers: bool,
) {
    for pointer in &config.strip_volatile {
        strip_pointer(request, pointer);
    }
    if inject_markers {
        inject_anthropic_markers(request);
    }
}

/// Removes the value at `pointer` (RFC 6901) when it resolves; an
/// unresolvable pointer is a no-op, so an operator may list fields some
/// requests do not carry.
pub(crate) fn strip_pointer(request: &mut Value, pointer: &str) {
    let Some((parent_pointer, leaf)) = split_pointer(pointer) else {
        return;
    };
    let Some(parent) = request.pointer_mut(&parent_pointer) else {
        return;
    };
    match parent {
        Value::Object(map) => {
            map.remove(&leaf);
        }
        Value::Array(items) => {
            if let Ok(index) = leaf.parse::<usize>()
                && index < items.len()
            {
                items.remove(index);
            }
        }
        _ => {}
    }
}

/// Splits `/a/b/3` into the parent pointer `/a/b` and the unescaped leaf
/// token `3`; `/user` splits into the root pointer and `user`. A pointer
/// without a leading slash, with an empty leaf, or with an invalid `~`
/// escape (RFC 6901: `~` must be followed by `0` or `1`) in the leaf or in
/// any parent segment is unusable — such a pointer never names a document
/// location, so stripping it must not fall back to a literal-key match.
fn split_pointer(pointer: &str) -> Option<(String, String)> {
    let trimmed = pointer.strip_prefix('/')?;
    let (parent, leaf) = match trimmed.rsplit_once('/') {
        Some((parent, leaf)) => (format!("/{parent}"), leaf),
        None => (String::new(), trimmed),
    };
    if leaf.is_empty() || has_invalid_escape(leaf) || has_invalid_escape(&parent) {
        return None;
    }
    // `~1` is `/` and `~0` is `~` in an RFC 6901 reference token.
    Some((parent, leaf.replace("~1", "/").replace("~0", "~")))
}

/// RFC 6901: within a reference token, `~` must be followed by `0` (a
/// literal `~`) or `1` (a literal `/`); any other follower, or a dangling
/// `~` at the end of a token, is an invalid escape.
fn has_invalid_escape(token: &str) -> bool {
    let mut chars = token.chars().peekable();
    while let Some(current) = chars.next() {
        if current == '~' && !matches!(chars.peek(), Some(&'0') | Some(&'1')) {
            return true;
        }
    }
    false
}

/// Phase 3, markers (Anthropic Messages): place `cache_control` breakpoints
/// on the stable prefix within the four-marker budget — the last tool, the
/// last system block, then the last content block of earlier messages,
/// walking backwards; the newest message is never marked (it changes every
/// turn). Markers the client already set count against the budget and are
/// kept where they are.
pub(crate) fn inject_anthropic_markers(request: &mut Value) {
    // A client may carry more markers than the provider's budget allows:
    // its own markers all stay (the budget for new ones is simply spent).
    let mut budget = ANTHROPIC_MAX_BREAKPOINTS.saturating_sub(existing_markers(request));

    if budget > 0
        && let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut)
        && let Some(last_tool) = tools.last_mut()
    {
        budget -= mark_block(last_tool, budget);
    }
    if budget > 0
        && let Some(system) = request.get_mut("system").and_then(Value::as_array_mut)
        && let Some(last_block) = system.last_mut()
    {
        budget -= mark_block(last_block, budget);
    }
    if budget > 0
        && let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut)
    {
        // Newest message excluded: mark the last content block of each
        // earlier message, backwards, until the budget is spent.
        for message in messages.iter_mut().rev().skip(1) {
            if budget == 0 {
                break;
            }
            let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for block in blocks.iter_mut().rev() {
                if budget == 0 {
                    break;
                }
                budget -= mark_block(block, budget);
            }
        }
    }
}

/// Counts the `cache_control` markers the request already carries: client
/// markers count against the provider's breakpoint budget.
fn existing_markers(request: &Value) -> usize {
    let mut count = 0;
    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        count += tools.iter().filter(|tool| has_marker(tool)).count();
    }
    if let Some(system) = request.get("system").and_then(Value::as_array) {
        count += system.iter().filter(|block| has_marker(block)).count();
    }
    if let Some(messages) = request.get("messages").and_then(Value::as_array) {
        for message in messages {
            if let Some(blocks) = message.get("content").and_then(Value::as_array) {
                count += blocks.iter().filter(|block| has_marker(block)).count();
            }
        }
    }
    count
}

fn has_marker(block: &Value) -> bool {
    block.get("cache_control").is_some_and(|v| !v.is_null())
}

/// Marks one block when it does not already carry a marker. Returns 1 when
/// the budget was spent here, 0 otherwise.
fn mark_block(block: &mut Value, budget: usize) -> usize {
    if budget == 0 || has_marker(block) {
        return 0;
    }
    if let Some(map) = block.as_object_mut() {
        map.insert(
            "cache_control".to_owned(),
            serde_json::json!({"type": "ephemeral"}),
        );
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_removes_resolvable_pointers_and_ignores_the_rest() {
        let mut request = json!({
            "model": "claude-sonnet-4-6",
            "metadata": {"user_id": "turn-42"},
            "user": "correlation-token",
            "messages": [{"role": "user", "content": "hello"}]
        });
        strip_pointer(&mut request, "/metadata/user_id");
        strip_pointer(&mut request, "/user");
        strip_pointer(&mut request, "/not/there");
        strip_pointer(&mut request, "no-leading-slash");
        assert_eq!(
            request,
            json!({
                "model": "claude-sonnet-4-6",
                "metadata": {},
                "messages": [{"role": "user", "content": "hello"}]
            })
        );
    }

    #[test]
    fn strip_ignores_a_pointer_with_a_bad_escape() {
        // RFC 6901: `~` must be followed by `0` or `1`, so `/a~2` never
        // names a key — it must not strip a key that is literally `a~2`.
        let mut request = json!({"a~2": 1, "b": 2});
        strip_pointer(&mut request, "/a~2");
        assert_eq!(
            request,
            json!({"a~2": 1, "b": 2}),
            "a bad escape never matches a literal key"
        );
    }

    #[test]
    fn markers_land_on_the_stable_prefix_within_the_budget() {
        let mut request = json!({
            "system": [{"type": "text", "text": "you are a proxy"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "turn 1"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "reply 1"}]},
                {"role": "user", "content": [{"type": "text", "text": "turn 2 (newest)"}]}
            ]
        });
        inject_anthropic_markers(&mut request);
        // System block marked; walking back from the newest: the assistant
        // message's block, then turn 1 — the newest message unmarked.
        assert!(request["system"][0]["cache_control"].is_object());
        assert!(request["messages"][1]["content"][0]["cache_control"].is_object());
        assert!(request["messages"][0]["content"][0]["cache_control"].is_object());
        assert!(
            request["messages"][2]["content"][0]["cache_control"].is_null(),
            "the newest message is never marked"
        );
    }

    #[test]
    fn client_markers_count_against_the_budget_and_are_kept() {
        let mut request = json!({
            "system": [
                {"type": "text", "text": "you are a proxy", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"type": "text", "text": "second block"}
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "turn 1"}]},
                {"role": "user", "content": [{"type": "text", "text": "turn 2"}]},
                {"role": "user", "content": [{"type": "text", "text": "turn 3"}]},
                {"role": "user", "content": [{"type": "text", "text": "turn 4 (newest)"}]}
            ]
        });
        inject_anthropic_markers(&mut request);
        // One client marker already spent; three left: last system block,
        // then the two messages before the newest.
        assert_eq!(
            request["system"][0]["cache_control"]["ttl"],
            json!("1h"),
            "the client's marker is kept verbatim"
        );
        assert!(request["system"][1]["cache_control"].is_object());
        assert!(request["messages"][2]["content"][0]["cache_control"].is_object());
        assert!(request["messages"][1]["content"][0]["cache_control"].is_object());
        assert!(request["messages"][0]["content"][0]["cache_control"].is_null());
        assert!(request["messages"][3]["content"][0]["cache_control"].is_null());
    }

    #[test]
    fn more_client_markers_than_the_budget_add_none_and_do_not_panic() {
        // Five client markers against Anthropic's four-marker budget: the
        // client's markers are all kept and none are added — the subtraction
        // on the budget must saturate, not underflow.
        let mut request = json!({
            "system": [
                {"type": "text", "text": "one", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "two", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "m1", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "m2", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "m3", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "unmarked"}
                ]}
            ]
        });
        inject_anthropic_markers(&mut request);
        assert!(
            request["system"][0]["cache_control"].is_object()
                && request["system"][1]["cache_control"].is_object(),
            "every client marker survives verbatim"
        );
        for (message, block) in [(0usize, 0usize), (0, 1), (1, 0)] {
            assert!(
                request["messages"][message]["content"][block]["cache_control"].is_object(),
                "every client marker survives verbatim"
            );
        }
        assert!(
            request["messages"][1]["content"][1]["cache_control"].is_null(),
            "no marker is injected once the client spent the budget and more"
        );
        assert_eq!(existing_markers(&request), 5);
    }

    #[test]
    fn tools_take_the_first_breakpoint() {
        let mut request = json!({
            "tools": [
                {"name": "a", "description": "first"},
                {"name": "b", "description": "last"}
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "only turn"}]}
            ]
        });
        inject_anthropic_markers(&mut request);
        assert!(
            request["tools"][1]["cache_control"].is_object(),
            "the last tool is marked"
        );
        // No system; the only message is the newest, so nothing else is
        // marked.
        assert!(request["messages"][0]["content"][0]["cache_control"].is_null());
    }
}
