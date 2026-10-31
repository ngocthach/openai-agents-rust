//! Recoverable Session appends. Port of the ordinary (non-held) pending write of
//! `save_result_to_session` and of `resume_pending_session_write`.
//!
//! A resumed turn's batch is recorded on the checkpoint before the Session is offered it, with
//! digests of the history tail read first. A failed append leaves the record behind, and the
//! next run settles it before any other work: an unchanged tail means the append never landed
//! and is repeated, an exact `tail + batch` means only the acknowledgement was lost, and
//! anything else is ambiguous and refused rather than guessed at.

use std::sync::Arc;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{RunLoop, strip_item_id};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;
use crate::memory::{Session, compaction_item_digest};

/// Whether a pending write record is the withheld batch of an interruption.
pub(super) fn is_held(record: &Value) -> bool {
    record.get("held").and_then(Value::as_bool).unwrap_or(false)
}

fn digests(session: &Arc<dyn Session>, items: &[InputItem]) -> Vec<String> {
    let ignore_ids = session.ignores_item_ids_for_matching();
    items
        .iter()
        .map(|item| {
            let item = if ignore_ids {
                strip_item_id(item.clone())
            } else {
                item.clone()
            };
            let digest = Sha256::digest(compaction_item_digest(&item).as_bytes());
            digest.iter().map(|b| format!("{b:02x}")).collect()
        })
        .collect()
}

fn unreconcilable() -> AgentsError {
    AgentsError::user(
        "Cannot reconcile the pending Session write: history changed or is ambiguous. Repair \
         the original Session before resuming; do not rerun the completed tool.",
    )
}

impl<C: Send + Sync + 'static> RunLoop<C> {
    /// Whether the checkpoint carries an ordinary pending write awaiting settlement.
    pub(super) fn has_recoverable_write(&self) -> bool {
        self.pending_write.as_ref().is_some_and(|p| !is_held(p))
    }

    /// Record `items` as the pending write, then settle it. A failed append keeps the record.
    pub(super) async fn append_recoverably(
        &mut self,
        session: &Arc<dyn Session>,
        items: Vec<InputItem>,
        saved_through: usize,
    ) -> Result<()> {
        // Whether the batch carries local continuation outputs: the session defers compaction
        // for those until the next response settles.
        let has_local_tool_outputs = items.iter().any(|i| {
            i.item_type()
                .is_some_and(|t| t.ends_with("_output") || t == "mcp_approval_response")
        });
        self.pending_write = Some(json!({
            "session_id": session.session_id(),
            "items": items.into_iter().map(|i| i.0).collect::<Vec<_>>(),
            "before": Value::Null,
            "persisted_count": saved_through,
            "response_id": self.model_responses.last().and_then(|r| r.response_id.clone()),
            "store": self.last_store,
            "has_local_tool_outputs": has_local_tool_outputs,
        }));
        let digests = if session.wants_model_exchange() {
            self.exchange_digests(session)
        } else {
            Vec::new()
        };
        if !digests.is_empty()
            && let Some(record) = self.pending_write.as_mut()
        {
            record["compaction_model_exchange"] = json!({
                "item_digests": digests,
                "reasoning_item_id_policy": (self.reasoning_item_id_policy
                    == crate::run_config::ReasoningItemIdPolicy::Omit)
                    .then_some("omit"),
            });
        }
        self.settle_recoverable_write(false).await
    }

    /// Settle the ordinary pending write against the attached Session. Port of
    /// `resume_pending_session_write`.
    pub(super) async fn settle_recoverable_write(&mut self, on_resume: bool) -> Result<()> {
        let Some(mut pending) = self.pending_write.clone().filter(|p| !is_held(p)) else {
            return Ok(());
        };
        let expected_id = pending
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let session = match &self.session {
            Some(session) if session.session_id() == expected_id => session.clone(),
            _ => {
                return Err(AgentsError::user(
                    "Resume the pending Session write with the original Session and session ID.",
                ));
            }
        };
        let items: Vec<InputItem> = pending
            .get("items")
            .and_then(Value::as_array)
            .map(|a| a.iter().cloned().map(InputItem).collect())
            .unwrap_or_default();
        let acknowledged = pending
            .get("append_acknowledged")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let before: Option<Vec<String>> =
            pending.get("before").and_then(Value::as_array).map(|a| {
                a.iter()
                    .filter_map(|d| d.as_str().map(str::to_string))
                    .collect()
            });
        let append;
        match &before {
            None if !acknowledged => {
                // No append has started. The record is already retained if this read fails.
                let tail = session.get_items(Some(items.len() + 1)).await?;
                pending["before"] = json!(digests(&session, &tail));
                self.pending_write = Some(pending.clone());
                append = true;
            }
            _ => {
                let before = before.unwrap_or_default();
                let batch = digests(&session, &items);
                let expected: Vec<String> = before.iter().chain(batch.iter()).cloned().collect();
                let tail = session.get_items(Some(expected.len())).await?;
                let observed = digests(&session, &tail);
                let committed = observed == expected;
                let mut unchanged = if before.is_empty() {
                    observed.is_empty()
                } else {
                    observed.len() >= before.len()
                        && observed[observed.len() - before.len()..] == before[..]
                };
                // Only multi-request Conversations writes admit partially committed chunks.
                // Other backends and single-request writes keep their atomic recovery contract.
                let mut partial = false;
                let max_chunk = crate::memory::openai_conversations_chunk_size();
                if session.manages_history_remotely() && !acknowledged && batch.len() > max_chunk {
                    let before_count = before.len();
                    let complete_before = before_count < batch.len() + 1;
                    // A short snapshot captured all prior history: the count must agree with
                    // zero progress as well as with each candidate chunk prefix. A full-size
                    // snapshot may omit older items, so repeated content stays ambiguous.
                    if complete_before {
                        unchanged = unchanged && observed.len() == before_count;
                    }
                    partial = (max_chunk..batch.len()).step_by(max_chunk).any(|count| {
                        let width = before_count + count;
                        (!complete_before || observed.len() == width)
                            && observed.len() >= width
                            && observed[observed.len() - width..]
                                .iter()
                                .eq(before.iter().chain(batch[..count].iter()))
                    });
                }
                if (acknowledged && !committed)
                    || (!acknowledged && (partial || committed == unchanged))
                {
                    return Err(unreconcilable());
                }
                append = !acknowledged && unchanged;
            }
        }
        if append {
            let batch = items.clone();
            let target = session.clone();
            crate::memory::await_session_mutation(async move { target.add_items(batch).await })
                .await?;
        }
        if let Some(exchange) = pending.get("compaction_model_exchange") {
            let digests: Vec<String> = exchange
                .get("item_digests")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let omit = exchange
                .get("reasoning_item_id_policy")
                .and_then(Value::as_str)
                == Some("omit");
            self.restored_model_exchange = Some((digests, omit));
        }
        let saved_through = pending
            .get("persisted_count")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(usize::MAX);
        let response_id = pending
            .get("response_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let store = pending.get("store").and_then(Value::as_bool);
        self.input_saved_to_session = true;
        self.session_saved_items = self
            .session_saved_items
            .max(saved_through.min(self.generated_items.len()));
        // The session hears about the response its batch belongs to before the record is
        // dropped: the record stays, acknowledged, until that succeeds, so a failed compaction
        // (or one the checkpoint must redo after a rollback) is retried alone.
        if response_id.is_some() && (on_resume || self.inline_notice) {
            pending["append_acknowledged"] = json!(true);
            let local = pending
                .get("has_local_tool_outputs")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            self.pending_write = Some(pending);
            self.settle_notified = response_id.clone();
            let held = on_resume.then_some((response_id, store));
            self.notice_has_local = Some(local);
            if let Err(error) = self.notify_session(held).await {
                self.settle_notified = None;
                return Err(error);
            }
        }
        self.pending_write = None;
        Ok(())
    }
}
