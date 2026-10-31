//! Ported from the Python SDK's `tests/test_run_impl_resume_paths.py`: a resumed approval whose
//! Session append fails (atomically or after committing) must be recoverable before the next
//! model call, from the live state or its JSON form, streamed or not.
//!
//! Python retries the very `RunState` object it passed to the failed run. A Rust run consumes
//! its state, so the failed run is given a `RunCheckpoint` and the retry starts from that slot.

mod py_core6_resume_support;

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::run::RunCheckpoint;
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunOptions, RunState, Runner};
use py_core6_resume_support::*;
use serde_json::{Value, json};

macro_rules! ck {
    ($label:expr, $cond:expr) => {
        if !$cond {
            return Err(format!("{}: {}", $label, stringify!($cond)));
        }
    };
}

async fn recovered_case(
    failing_streamed: bool,
    retry_streamed: bool,
    json: bool,
    failure: &'static str,
) -> std::result::Result<(), String> {
    let label = format!("fs={failing_streamed} rs={retry_streamed} json={json} {failure}");
    let p = approved_session_state(failing_streamed).await;
    p.session.fail_next(failure);
    let slot = RunCheckpoint::new();
    let err = drive(
        &p.agent,
        p.state,
        Some(p.session.clone()),
        failing_streamed,
        &slot,
    )
    .await
    .err()
    .ok_or_else(|| format!("{label}: the append must fail"))?;
    ck!(label, is_session_failure(&err));
    ck!(label, *p.effects.lock().unwrap() == vec![7]);
    ck!(label, p.model.calls().len() == 1);
    let mut state = slot.take().ok_or("no checkpoint")?;
    if json {
        state = round_trip(&state);
    }
    let result = drive(
        &p.agent,
        state,
        Some(p.session.clone()),
        retry_streamed,
        &RunCheckpoint::new(),
    )
    .await
    .map_err(|e| format!("{label}: retry failed: {e}"))?;
    ck!(label, result.final_output == Some(json!("done")));
    ck!(label, *p.effects.lock().unwrap() == vec![7]);
    ck!(label, charge_pair(&p.session.items()) == pair());
    ck!(label, charge_pair(&result.input_list) == pair());
    drive(
        &p.agent,
        "What was the receipt?",
        Some(p.session.clone()),
        retry_streamed,
        &RunCheckpoint::new(),
    )
    .await
    .map_err(|e| format!("{label}: follow-up failed: {e}"))?;
    let last = p.model.calls().pop().unwrap();
    ck!(label, charge_pair(&last.request.input) == pair());
    let payload = result.state.unwrap().to_json().unwrap();
    ck!(label, payload.get("pending_session_write").is_none());
    Ok(())
}

#[tokio::test]
async fn resumed_session_append_is_recovered_before_next_model() {
    let mut failures = Vec::new();
    for (fs, rs) in [(false, false), (false, true), (true, false), (true, true)] {
        for json in [false, true] {
            for failure in ["before", "after"] {
                if let Err(e) = recovered_case(fs, rs, json, failure).await {
                    failures.push(e);
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[tokio::test]
async fn resumed_session_append_rejects_ambiguous_recovery() {
    for retry_streamed in [false, true] {
        for mismatch in ["missing", "different-id", "changed-tail"] {
            let p = approved_session_state(false).await;
            let state = failed_resume(&p, p.state.clone(), false, "before").await;
            let state = round_trip(&state);
            let supplied: Option<Arc<FailingSession>> = match mismatch {
                "missing" => None,
                "different-id" => {
                    let other = FailingSession::new("other");
                    for item in p.session.items() {
                        other.push(item.0);
                    }
                    Some(other)
                }
                _ => {
                    p.session
                        .push(json!({"role": "user", "content": "another writer"}));
                    Some(p.session.clone())
                }
            };
            let before = p.session.items();
            let err = drive(
                &p.agent,
                state,
                supplied,
                retry_streamed,
                &RunCheckpoint::new(),
            )
            .await
            .err()
            .unwrap_or_else(|| panic!("{mismatch}: must refuse"));
            assert_user_error(err, "pending Session write");
            assert_eq!(p.model.calls().len(), 1, "{mismatch}");
            assert_eq!(*p.effects.lock().unwrap(), vec![7], "{mismatch}");
            assert_eq!(p.session.items(), before, "{mismatch}");
        }
    }
}

#[tokio::test]
async fn resumed_session_append_survives_repeated_failure_and_late_input() {
    let p = approved_session_state(false).await;
    let mut state = p.state.clone();
    for _ in 0..2 {
        state = round_trip(&failed_resume(&p, state, false, "before").await);
        assert_eq!(p.model.calls().len(), 1);
        assert_eq!(*p.effects.lock().unwrap(), vec![7]);
    }
    state.add_input("What was the receipt?").unwrap();
    let result = drive(
        &p.agent,
        state,
        Some(p.session.clone()),
        true,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output, Some(json!("done")));
    let stored = p.session.items();
    let output_index = stored
        .iter()
        .position(|i| i.0.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .unwrap();
    let late_index = stored
        .iter()
        .position(|i| i.0.get("content") == Some(&json!("What was the receipt?")))
        .unwrap();
    assert!(output_index < late_index);
    assert_eq!(*p.effects.lock().unwrap(), vec![7]);
}

/// Python's `_partially_approved_session_state`: two gated calls in one response, only the
/// first approved.
async fn partially_approved_session_state(streamed: bool) -> Paused {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let effects = Arc::new(Mutex::new(Vec::new()));
    let notify = FunctionTool::new_no_args("notify", "", |_c| async move {
        panic!("the unresolved approval must not execute");
        #[allow(unreachable_code)]
        Ok("never")
    })
    .with_needs_approval(true);
    let model = scripted(vec![
        vec![
            call("charge", json!({"amount": 7}), "charge-1"),
            call("notify", json!({}), "notify-1"),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("payment")
        .with_model_instance(model.clone())
        .with_tool(charge_tool(&effects))
        .with_tool(notify);
    let session = FailingSession::new("resume");
    let paused = drive(
        &agent,
        "charge 7 and notify",
        Some(session.clone()),
        streamed,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    let mut state = round_trip(paused.state.as_ref().unwrap());
    let charge = state
        .get_interruptions()
        .into_iter()
        .find(|i| i.call_id() == "charge-1")
        .unwrap();
    state.approve(&charge, false);
    Paused {
        agent,
        model,
        session,
        state,
        effects,
    }
}

#[tokio::test]
async fn renewed_interruption_recovers_failed_resumed_session_append() {
    for streamed in [false, true] {
        for json_trip in [false, true] {
            let label = format!("streamed={streamed} json={json_trip}");
            let p = partially_approved_session_state(streamed).await;
            let mut state = failed_resume(&p, p.state.clone(), streamed, "before").await;
            assert_eq!(*p.effects.lock().unwrap(), vec![7], "{label}");
            assert_eq!(
                charge_pair(&p.session.items()),
                vec!["function_call"],
                "{label}"
            );
            if json_trip {
                state = round_trip(&state);
            }
            let pending = drive(
                &p.agent,
                state,
                Some(p.session.clone()),
                streamed,
                &RunCheckpoint::new(),
            )
            .await
            .unwrap();
            let mut pending_state = pending.state.unwrap();
            let remaining = pending_state.get_interruptions();
            assert_eq!(
                remaining
                    .iter()
                    .map(|i| i.call_id().to_string())
                    .collect::<Vec<_>>(),
                vec!["notify-1"],
                "{label}"
            );
            assert_eq!(p.model.calls().len(), 1, "{label}");
            pending_state.reject(&remaining[0], false, Some("declined".into()));
            let result = drive(
                &p.agent,
                pending_state,
                Some(p.session.clone()),
                streamed,
                &RunCheckpoint::new(),
            )
            .await
            .unwrap();
            assert_eq!(result.final_output, Some(json!("done")), "{label}");
            assert_eq!(*p.effects.lock().unwrap(), vec![7], "{label}");
            assert_eq!(charge_pair(&p.session.items()), pair(), "{label}");
            assert_eq!(charge_pair(&result.input_list), pair(), "{label}");
        }
    }
}

#[tokio::test]
async fn failed_streamed_result_checkpoint_retains_detached_pending_write() {
    let p = approved_session_state(true).await;
    let state = p.state.clone();
    p.session.fail_next("before");
    let mut streamed = Runner::run_streamed_with_options(
        &p.agent,
        state,
        RunOptions::new(()).session(p.session.clone()),
    );
    let mut failed = false;
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let Err(e) = event {
                assert!(is_session_failure(&e), "{e}");
                failed = true;
            }
        }
    }
    assert!(failed);
    let snapshot = streamed
        .to_state()
        .expect("a failed streamed run keeps its state");
    let mut payload = snapshot.to_json().unwrap();
    payload["pending_session_write"]["items"][0]["output"] = json!("changed snapshot");
    assert_eq!(
        snapshot.to_json().unwrap()["pending_session_write"]["items"][0]["output"],
        json!("receipt-7")
    );
    drive(
        &p.agent,
        snapshot,
        Some(p.session.clone()),
        false,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    assert_eq!(*p.effects.lock().unwrap(), vec![7]);
    assert_eq!(p.model.calls().len(), 2);
    assert_eq!(charge_pair(&p.session.items()), pair());
}

/// Python's `_relabel_as_older_schema`: downgrade only what the destination label cannot
/// represent, then relabel.
fn relabel_as_older_schema(payload: &mut Value, version: &str) {
    let (major, minor) = version.split_once('.').unwrap();
    let minor: u32 = minor.parse().unwrap();
    assert_eq!(major, "1");
    if minor < 18 {
        if let Some(entries) = payload["context"]
            .as_object_mut()
            .and_then(|c| c.remove("function_tool_approvals"))
            .and_then(|e| e.as_array().cloned())
        {
            for entry in entries {
                payload["context"]["approvals"][entry["tool_key"].as_str().unwrap()] =
                    entry["decision"].clone();
            }
        }
        if let Some(response) = payload
            .get_mut("last_processed_response")
            .and_then(Value::as_object_mut)
        {
            response.remove("mcp_tool_bindings");
        }
    }
    payload["$schemaVersion"] = json!(version);
}

#[tokio::test]
async fn pending_session_write_rejects_invalid_serialized_checkpoint() {
    let cases = [
        "old-schema",
        "batch-shape",
        "compaction-under-1.17",
        "held-shape",
        "held-with-before",
        "held-under-1-17",
        "held-under-1-18",
        "held-keys-without-held",
        "policy-shape",
        "response-boundary-shape",
        "response-boundary-missing",
        "response-boundary-outside-batch",
        "acknowledgement-type",
        "acknowledgement-without-before",
        "compaction-exchange-type",
        "compaction-exchange-policy",
        "acknowledgement-under-1.17",
    ];
    let mut failures = Vec::new();
    for invalid in cases {
        let p = approved_session_state(false).await;
        let state = failed_resume(&p, p.state.clone(), false, "before").await;
        let mut payload = state.to_json().unwrap();
        let current_turn = payload["current_turn"].clone();
        let items_len = payload["pending_session_write"]["items"]
            .as_array()
            .unwrap()
            .len();
        let pending = |payload: &mut Value| payload["pending_session_write"].take();
        let _ = pending;
        match invalid {
            "old-schema" => relabel_as_older_schema(&mut payload, "1.16"),
            "compaction-under-1.17" => relabel_as_older_schema(&mut payload, "1.17"),
            "acknowledgement-type" => {
                payload["pending_session_write"]["append_acknowledged"] = json!("true")
            }
            "acknowledgement-without-before" => {
                payload["pending_session_write"]["append_acknowledged"] = json!(true);
                payload["pending_session_write"]["before"] = Value::Null;
            }
            "compaction-exchange-type" => {
                payload["pending_session_write"]["compaction_model_exchange"] = json!({
                    "item_digests": "not a digest list",
                    "reasoning_item_id_policy": null,
                })
            }
            "compaction-exchange-policy" => {
                payload["pending_session_write"]["compaction_model_exchange"] = json!({
                    "item_digests": [],
                    "reasoning_item_id_policy": "unknown",
                })
            }
            "acknowledgement-under-1.17" => {
                relabel_as_older_schema(&mut payload, "1.17");
                for key in ["response_id", "store", "has_local_tool_outputs"] {
                    payload["pending_session_write"]
                        .as_object_mut()
                        .unwrap()
                        .remove(key);
                }
                payload["pending_session_write"]["append_acknowledged"] = json!(true);
            }
            "batch-shape" => payload["pending_session_write"]["items"] = json!("not an item batch"),
            "held-shape" => payload["pending_session_write"]["held"] = json!("yes"),
            "held-keys-without-held" => {
                payload["pending_session_write"]["reasoning_item_id_policy"] = json!("omit")
            }
            "policy-shape" => {
                payload["pending_session_write"]["held"] = json!(true);
                payload["pending_session_write"]["before"] = Value::Null;
                payload["pending_session_write"]["reasoning_item_id_policy"] = json!("banana");
            }
            "response-boundary-shape"
            | "response-boundary-missing"
            | "response-boundary-outside-batch" => {
                payload["pending_session_write"]["held"] = json!(true);
                payload["pending_session_write"]["before"] = Value::Null;
                let boundary = if invalid == "response-boundary-shape" {
                    json!({"turn": -1, "start": 0})
                } else {
                    json!({"turn": current_turn, "start": items_len + 1})
                };
                if invalid != "response-boundary-missing" {
                    payload["pending_session_write"]["current_response"] = boundary;
                }
            }
            "held-under-1-17" => {
                relabel_as_older_schema(&mut payload, "1.17");
                payload["pending_session_write"]["held"] = json!(true);
                payload["pending_session_write"]["before"] = Value::Null;
            }
            "held-under-1-18" => {
                relabel_as_older_schema(&mut payload, "1.18");
                payload["pending_session_write"]["held"] = json!(true);
                payload["pending_session_write"]["before"] = Value::Null;
                payload["pending_session_write"]["current_response"] =
                    json!({"turn": current_turn, "start": 0});
            }
            _ => payload["pending_session_write"]["held"] = json!(true),
        }
        match RunState::from_json(payload) {
            Err(e) if e.to_string().contains("pending Session write is invalid") => {}
            other => failures.push(format!("{invalid}: {:?}", other.map(|_| ()))),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[tokio::test]
async fn pending_session_write_without_the_held_key_keeps_its_meaning() {
    let p = approved_session_state(false).await;
    let state = failed_resume(&p, p.state.clone(), false, "before").await;
    let mut payload = state.to_json().unwrap();
    assert!(payload["pending_session_write"].get("held").is_none());
    relabel_as_older_schema(&mut payload, "1.17");
    for key in ["response_id", "store", "has_local_tool_outputs"] {
        payload["pending_session_write"]
            .as_object_mut()
            .unwrap()
            .remove(key);
    }
    let restored = RunState::from_json(payload).unwrap();
    let result = drive(
        &p.agent,
        restored,
        Some(p.session.clone()),
        false,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output, Some(json!("done")));
    assert_eq!(*p.effects.lock().unwrap(), vec![7]);
    assert_eq!(charge_pair(&p.session.items()), pair());
}

#[tokio::test]
async fn compaction_metadata_stays_readable_under_1_18() {
    for acknowledged in [false, true] {
        let p = approved_session_state(false).await;
        let state = failed_resume(&p, p.state.clone(), false, "before").await;
        let mut payload = state.to_json().unwrap();
        let pending = payload["pending_session_write"]
            .as_object()
            .unwrap()
            .clone();
        assert!(!pending.contains_key("held"));
        for key in ["response_id", "store", "has_local_tool_outputs"] {
            assert!(pending.contains_key(key), "{key}");
        }
        relabel_as_older_schema(&mut payload, "1.18");
        if acknowledged {
            payload["pending_session_write"]["append_acknowledged"] = json!(true);
            payload["pending_session_write"]["compaction_model_exchange"] = json!({
                "item_digests": pending["before"].clone(),
                "reasoning_item_id_policy": null,
            });
        }
        let restored = RunState::from_json(payload).unwrap();
        let record = restored
            .pending_session_write
            .clone()
            .expect("pending write restored");
        assert_eq!(
            record
                .get("append_acknowledged")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            acknowledged
        );
        if acknowledged {
            continue;
        }
        let result = drive(
            &p.agent,
            restored,
            Some(p.session.clone()),
            false,
            &RunCheckpoint::new(),
        )
        .await
        .unwrap();
        assert_eq!(result.final_output, Some(json!("done")));
        assert_eq!(*p.effects.lock().unwrap(), vec![7]);
        assert_eq!(charge_pair(&p.session.items()), pair());
    }
}

#[tokio::test]
async fn resumed_session_append_partial_commit_fails_closed() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let effects = Arc::new(Mutex::new(Vec::new()));
    let model = scripted(vec![
        vec![
            call("charge", json!({"amount": 7}), "charge-1"),
            call("charge", json!({"amount": 7}), "charge-2"),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::<()>::new("payment")
        .with_model_instance(model.clone())
        .with_tool(charge_tool(&effects));
    let session = FailingSession::new("resume");
    let paused = drive(
        &agent,
        "charge twice",
        Some(session.clone()),
        false,
        &RunCheckpoint::new(),
    )
    .await
    .unwrap();
    let mut state = round_trip(paused.state.as_ref().unwrap());
    for interruption in state.get_interruptions() {
        state.approve(&interruption, false);
    }
    session.fail_next("partial");
    let slot = RunCheckpoint::new();
    let err = drive(&agent, state, Some(session.clone()), false, &slot)
        .await
        .err()
        .unwrap();
    assert!(is_session_failure(&err), "{err}");
    let before = session.items();
    let restored = round_trip(&slot.take().unwrap());
    let err = drive(
        &agent,
        restored,
        Some(session.clone()),
        true,
        &RunCheckpoint::new(),
    )
    .await
    .err()
    .expect("a partially committed batch fails closed");
    assert_user_error(err, "history changed or is ambiguous");
    assert_eq!(*effects.lock().unwrap(), vec![7, 7]);
    assert_eq!(model.calls().len(), 1);
    assert_eq!(session.items(), before);
}
