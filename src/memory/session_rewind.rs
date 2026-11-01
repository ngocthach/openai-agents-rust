//! Rolling back items a retried model request already wrote to a session. Port of
//! `rewind_session_items`, `wait_for_session_cleanup` and their private helpers in
//! `agents.run_internal.session_persistence`.

use std::collections::BTreeSet;
use std::time::Duration;

use super::{Session, SessionRunContext};
use crate::items::InputItem;
use crate::run_internal::oai_conversation::{ServerConversationTracker, fingerprint};

/// The content fingerprint rewinds match items by, or `None` for an item that cannot be
/// serialized. `ignore_ids` drops the item id first, for stores that reassign ids on write.
/// Port of `fingerprint_input_item`.
pub fn fingerprint_input_item(item: &InputItem, ignore_ids: bool) -> Option<String> {
    fingerprint(&item.0, ignore_ids)
}

async fn get_items(
    session: &dyn Session,
    ctx: Option<SessionRunContext<'_>>,
    limit: Option<usize>,
) -> crate::error::Result<Vec<InputItem>> {
    match ctx {
        Some(ctx) => session.get_items_with_context(ctx, limit).await,
        None => session.get_items(limit).await,
    }
}

async fn pop_item(
    session: &dyn Session,
    ctx: Option<SessionRunContext<'_>>,
) -> crate::error::Result<Option<InputItem>> {
    match ctx {
        Some(ctx) => session.pop_item_with_context(ctx).await,
        None => session.pop_item().await,
    }
}

async fn add_items(
    session: &dyn Session,
    ctx: Option<SessionRunContext<'_>>,
    items: Vec<InputItem>,
) -> crate::error::Result<()> {
    match ctx {
        Some(ctx) => session.add_items_with_context(ctx, items).await,
        None => session.add_items(items).await,
    }
}

/// Best-effort restoration for items popped during a failed rewind attempt.
async fn restore_popped(
    session: &dyn Session,
    ctx: Option<SessionRunContext<'_>>,
    popped: Vec<InputItem>,
) {
    if popped.is_empty() {
        return;
    }
    let restored: Vec<InputItem> = popped.into_iter().rev().collect();
    if let Err(error) = add_items(session, ctx, restored).await {
        crate::logger::log_model_and_tool_action_warning(
            "Failed to restore session items after a rewind mismatch",
            &error,
        );
    }
}

/// Remove an exact serialized suffix from the session tail, aborting when the tail diverges.
async fn rewind_tail_suffix(
    session: &dyn Session,
    ctx: Option<SessionRunContext<'_>>,
    expected: &[String],
    ignore_ids: bool,
    mismatch_warning: &str,
    pop_failure_warning: &str,
) -> bool {
    if expected.is_empty() {
        return true;
    }
    let tail = match get_items(session, ctx, Some(expected.len())).await {
        Ok(tail) => tail,
        Err(error) => {
            crate::logger::log_model_and_tool_action_warning(pop_failure_warning, &error);
            return false;
        }
    };
    if tail.len() != expected.len() {
        log::warn!("{mismatch_warning}");
        return false;
    }
    let tail_fingerprints: Vec<String> = tail
        .iter()
        .filter_map(|item| fingerprint(&item.0, ignore_ids))
        .collect();
    if tail_fingerprints != expected {
        log::warn!("{mismatch_warning}");
        return false;
    }
    let mut popped: Vec<InputItem> = Vec::new();
    for want in expected.iter().rev() {
        let item = match pop_item(session, ctx).await {
            Ok(item) => item,
            Err(error) => {
                restore_popped(session, ctx, popped).await;
                crate::logger::log_model_and_tool_action_warning(pop_failure_warning, &error);
                return false;
            }
        };
        let Some(item) = item else {
            restore_popped(session, ctx, popped).await;
            log::warn!("{mismatch_warning}");
            return false;
        };
        let matches = fingerprint(&item.0, ignore_ids).as_deref() == Some(want.as_str());
        popped.push(item);
        if !matches {
            restore_popped(session, ctx, popped).await;
            log::warn!("{mismatch_warning}");
            return false;
        }
    }
    true
}

/// The contiguous retry-owned tail suffix that can be safely stripped.
fn collect_retry_owned_tail(
    session_items: &[InputItem],
    tracker: &ServerConversationTracker,
    ignore_ids: bool,
) -> Vec<String> {
    let state = tracker.state();
    let mut stray: Vec<String> = Vec::new();
    for item in session_items.iter().rev() {
        let id = item.0.get("id").and_then(serde_json::Value::as_str);
        if id.is_some_and(|id| state.server_item_ids.contains(id)) {
            stray.reverse();
            return stray;
        }
        match fingerprint(&item.0, ignore_ids) {
            Some(fp) if state.sent_item_fingerprints.contains(&fp) => stray.push(fp),
            _ => {
                log::warn!(
                    "Skipping stray session cleanup because the current tail contains items \
                     unrelated to this retry"
                );
                return Vec::new();
            }
        }
    }
    if !stray.is_empty() {
        log::warn!(
            "Skipping stray session cleanup because no known server item was found before the \
             session boundary"
        );
    }
    Vec::new()
}

/// Roll back items recently persisted to `session` when a conversation retry is needed, so the
/// retry does not accumulate duplicate inputs on lock errors. Best effort: a tail that does not
/// end with exactly `items` is left alone.
///
/// With a `tracker`, items the retry itself wrote after the last known server item are
/// stripped as well. `ctx` scopes every read, pop and restore the way the runner's own session
/// calls are scoped. Port of `rewind_session_items`.
pub async fn rewind_session_items(
    session: &dyn Session,
    items: &[InputItem],
    tracker: Option<&ServerConversationTracker>,
    ctx: Option<SessionRunContext<'_>>,
) {
    if items.is_empty() {
        return;
    }
    let ignore_ids = session.ignores_item_ids_for_matching();
    let targets: Vec<String> = items
        .iter()
        .filter_map(|item| fingerprint(&item.0, ignore_ids))
        .collect();
    if targets.is_empty() {
        return;
    }
    log::debug!(
        "Rewinding session items due to conversation retry (targets={})",
        targets.len()
    );
    if !(crate::debug::dont_log_model_data() || crate::debug::dont_log_tool_data()) {
        for (i, target) in targets.iter().enumerate() {
            let head: String = target.chars().take(300).collect();
            log::debug!("Rewind target {i} (first 300 chars): {head}");
        }
    }
    let rewound = rewind_tail_suffix(
        session,
        ctx,
        &targets,
        ignore_ids,
        "Skipping session rewind because the current tail does not match the retry-owned suffix",
        "Failed to rewind session item",
    )
    .await;
    if !rewound {
        return;
    }
    wait_for_session_cleanup(session, &targets, 5, ignore_ids, ctx).await;
    let Some(tracker) = tracker else {
        return;
    };
    let latest = match get_items(session, ctx, Some(1)).await {
        Ok(latest) => latest,
        Err(error) => {
            crate::logger::log_model_and_tool_action_debug(
                "Failed to peek session items while rewinding",
                &error,
            );
            return;
        }
    };
    let Some(latest) = latest.first() else {
        return;
    };
    let known = tracker.state().server_item_ids;
    if latest
        .0
        .get("id")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|id| known.contains(id))
    {
        return;
    }
    let session_items = match get_items(session, ctx, None).await {
        Ok(all) => all,
        Err(error) => {
            crate::logger::log_model_and_tool_action_debug(
                "Failed to inspect session tail while stripping stray items",
                &error,
            );
            return;
        }
    };
    let stray = collect_retry_owned_tail(&session_items, tracker, ignore_ids);
    if stray.is_empty() {
        return;
    }
    log::debug!(
        "Stripping {} retry-owned conversation items until the session tail reaches a known \
         server item",
        stray.len()
    );
    rewind_tail_suffix(
        session,
        ctx,
        &stray,
        ignore_ids,
        "Skipping stray session cleanup because the current tail no longer matches retry-owned \
         conversation items",
        "Failed to strip stray session item",
    )
    .await;
}

/// Confirm that rewound items are no longer present in the session tail, so the store stays
/// consistent before the next retry attempt begins. `serialized_targets` are item fingerprints.
/// Sleeps `0.1s * (attempt + 1)` between checks. Port of `wait_for_session_cleanup`.
pub async fn wait_for_session_cleanup(
    session: &dyn Session,
    serialized_targets: &[String],
    max_attempts: u32,
    ignore_ids: bool,
    ctx: Option<SessionRunContext<'_>>,
) {
    if serialized_targets.is_empty() {
        return;
    }
    let window = serialized_targets.len() + 2;
    for attempt in 0..max_attempts {
        let delay = Duration::from_millis(100 * u64::from(attempt + 1));
        let tail = match get_items(session, ctx, Some(window)).await {
            Ok(tail) => tail,
            Err(error) => {
                crate::logger::log_model_and_tool_action_debug(
                    &format!("Failed to verify session cleanup (attempt {})", attempt + 1),
                    &error,
                );
                tokio::time::sleep(delay).await;
                continue;
            }
        };
        let present: BTreeSet<String> = tail
            .iter()
            .filter_map(|item| fingerprint(&item.0, ignore_ids))
            .collect();
        if !serialized_targets.iter().any(|t| present.contains(t)) {
            return;
        }
        tokio::time::sleep(delay).await;
    }
    log::debug!(
        "Session cleanup verification exhausted attempts; targets may still linger temporarily"
    );
}
