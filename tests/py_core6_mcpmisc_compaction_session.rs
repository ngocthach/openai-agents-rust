//! Parity ports from `tests/memory/test_openai_responses_compaction_session.py`: the replay of
//! server-stamped `created_by` metadata, rollback budgets on the automatic path, invalidation of
//! the response chain by `pop_item` / `clear_session`, mutations waiting for an in-flight
//! compaction, orphaned assistant ids in compacted output, and a failed compaction keeping its
//! "must force" marker.
#![allow(
    clippy::field_reassign_with_default,
    clippy::await_holding_lock,
    clippy::type_complexity,
    clippy::cloned_ref_to_slice_refs
)]
#![cfg(feature = "sqlite")]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents_sdk::memory::{
    CompactionMode, OpenAIResponsesCompactionArgs, OpenAIResponsesCompactionSession, Session,
    SessionRunCompletion, SqliteSession,
};
use openai_agents_sdk::models::openai_responses::OpenAIResponsesModel;
use openai_agents_sdk::testing::assistant_message;
use openai_agents_sdk::{Agent, InputItem, OpenAIClient, RunConfig, RunOptions, Runner};
use serde_json::{Value, json};

mod py_core6_mcpmisc_support;
mod py_mcpmem_support;
use py_core6_mcpmisc_support::*;
use py_mcpmem_support::{ListSession, Reply, Request, serve};

fn force() -> OpenAIResponsesCompactionArgs {
    OpenAIResponsesCompactionArgs {
        force: true,
        ..Default::default()
    }
}

fn response_args(id: &str, force: bool) -> OpenAIResponsesCompactionArgs {
    OpenAIResponsesCompactionArgs {
        response_id: Some(id.into()),
        force,
        ..Default::default()
    }
}

// test_compaction_replay_strips_created_by (automatic, manual, stored)
async fn created_by_case(history_source: &str) {
    let compacted = json!({
        "id": "cmp_synthetic", "type": "compaction",
        "encrypted_content": "synthetic-encrypted-content", "created_by": "server",
    });
    let inputs = Arc::new(Mutex::new(Vec::<Value>::new()));
    let log = inputs.clone();
    let reply = compacted.clone();
    let base = serve(Arc::new(move |req: &Request| {
        log.lock().unwrap().push(req.body["input"].clone());
        if req.path.ends_with("/compact") {
            return Reply::json(json!({
                "id": "cmp_response", "object": "response.compaction", "created_at": 1,
                "output": [reply.clone()],
            }));
        }
        Reply::json(json!({
            "id": "resp_1", "object": "response", "created_at": 1, "status": "completed",
            "model": "gpt-5", "parallel_tool_calls": true, "tool_choice": "auto", "tools": [],
            "output": [{
                "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "42", "annotations": []}],
            }],
            "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        }))
    }))
    .await;
    let client = OpenAIClient::new("synthetic-key").with_base_url(format!("{base}/v1"));
    let underlying = Arc::new(SqliteSession::in_memory("created-by-replay").unwrap());
    let session = Arc::new(
        OpenAIResponsesCompactionSession::new("created-by-replay", underlying.clone())
            .unwrap()
            .with_client(client.clone())
            .with_model("gpt-5")
            .unwrap()
            .with_compaction_mode(CompactionMode::Input)
            .with_should_trigger_compaction(|ctx| !ctx.compaction_candidate_items.is_empty()),
    );
    let dyn_session: Arc<dyn Session> = session.clone();
    let agent = Agent::<()>::new("test")
        .with_model_instance(Arc::new(OpenAIResponsesModel::new("gpt-5", client)));
    let config = RunConfig::<()>::default();
    let options = || {
        RunOptions::new(())
            .session(dyn_session.clone())
            .run_config(config.clone())
    };
    match history_source {
        "stored" => underlying
            .add_items(vec![InputItem::from_value(compacted.clone())])
            .await
            .unwrap(),
        "manual" => {
            session.add_items(vec![user("How many?")]).await.unwrap();
            session.run_compaction(force()).await.unwrap();
        }
        _ => {
            Runner::run_with_options(&agent, "How many?", options())
                .await
                .unwrap();
        }
    }
    let stored = values(&underlying.get_items(None).await.unwrap());
    assert_eq!(stored, vec![compacted.clone()], "{history_source}");
    if history_source == "manual" {
        // Repeated manual compaction reuses cached output without a history reload.
        session.add_items(vec![user("More context")]).await.unwrap();
        session.run_compaction(force()).await.unwrap();
    }
    let result = Runner::run_with_options(&agent, "And urgent?", options())
        .await
        .unwrap();
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("42"),
        "{history_source}"
    );

    // Inspect encoded inputs to both Responses and the next compact request.
    let replayed: Vec<Value> = inputs
        .lock()
        .unwrap()
        .iter()
        .flat_map(|input| input.as_array().cloned().unwrap_or_default())
        .filter(|item| item["type"] == "compaction")
        .collect();
    let expected = json!({
        "id": "cmp_synthetic", "type": "compaction",
        "encrypted_content": "synthetic-encrypted-content",
    });
    let count = if history_source == "manual" { 3 } else { 2 };
    assert_eq!(replayed, vec![expected; count], "{history_source}");
    assert_eq!(
        values(&underlying.get_items(None).await.unwrap())[0],
        compacted
    );
}

#[tokio::test]
async fn compaction_replay_strips_created_by() {
    openai_agents_sdk::set_tracing_disabled(true);
    for source in ["automatic", "manual", "stored"] {
        created_by_case(source).await;
    }
}

// test_automatic_compaction_rollback_budget (native sqlite / list)
#[tokio::test]
async fn automatic_compaction_rollback_budget() {
    openai_agents_sdk::set_tracing_disabled(true);
    for native in [false, true] {
        let endpoint = Compaction::replying(vec![]).await;
        let backend: Arc<dyn Session> = if native {
            Arc::new(SqliteSession::in_memory("test").unwrap())
        } else {
            Arc::new(ListSession::new("test"))
        };
        let session: Arc<dyn Session> = Arc::new(
            compaction_session("test", backend.clone(), &endpoint, CompactionMode::Auto)
                .with_max_rollback_items(1)
                .unwrap()
                .with_should_trigger_compaction(|_| true),
        );
        let model = model_with(texts(&["reply"]));
        let agent = Agent::<()>::new("test").with_model_instance(model);
        let outcome = Runner::run_with_options(
            &agent,
            "hello",
            RunOptions::new(()).session(session.clone()),
        )
        .await;
        if native {
            outcome.unwrap();
            assert_eq!(endpoint.requests().len(), 1);
            assert!(backend.get_items(None).await.unwrap().is_empty());
        } else {
            let error = outcome.unwrap_err().to_string();
            assert!(error.contains("exceeds max_rollback_items"), "{error}");
            assert!(endpoint.requests().is_empty());
            assert_eq!(backend.get_items(None).await.unwrap().len(), 2);
        }
    }
}

// pop / clear invalidate the response chain
fn chain_session(
    endpoint: &Compaction,
    history: Vec<InputItem>,
    deferring: Arc<AtomicBool>,
) -> (Arc<OpenAIResponsesCompactionSession>, Arc<ListSession>) {
    let underlying = Arc::new(ListSession::with_history("test", history));
    let session = Arc::new(
        compaction_session(
            "test",
            underlying.clone(),
            endpoint,
            CompactionMode::PreviousResponseId,
        )
        .with_should_trigger_compaction(move |_| deferring.load(Ordering::SeqCst)),
    );
    (session, underlying)
}

async fn seed_deferred_response(
    session: &OpenAIResponsesCompactionSession,
    deferring: &AtomicBool,
) {
    // An unstored response id, then deferred work through the runner's completion hook.
    session
        .run_compaction(OpenAIResponsesCompactionArgs {
            response_id: Some("resp-old".into()),
            store: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();
    deferring.store(true, Ordering::SeqCst);
    session
        .on_run_completed(&SessionRunCompletion {
            response_id: Some("resp-old".into()),
            has_local_tool_outputs: true,
            ..Default::default()
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn pop_item_invalidates_response_chain() {
    let endpoint = Compaction::replying(vec![]).await;
    let item = InputItem::from_value(json!({"role": "assistant", "content": "remove me"}));
    let flag = Arc::new(AtomicBool::new(false));
    let (session, _) = chain_session(&endpoint, vec![item.clone()], flag.clone());
    seed_deferred_response(&session, &flag).await;

    assert_eq!(session.pop_item().await.unwrap(), Some(item));
    assert!(session.get_items(None).await.unwrap().is_empty());
    let error = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires a response_id"), "{error}");
    assert!(endpoint.requests().is_empty());

    session
        .run_compaction(response_args("resp-new", true))
        .await
        .unwrap();
    assert_eq!(endpoint.requests().len(), 1);
    assert_eq!(endpoint.requests()[0]["previous_response_id"], "resp-new");
}

#[tokio::test]
async fn empty_pop_preserves_response_chain() {
    let endpoint = Compaction::replying(vec![]).await;
    let flag = Arc::new(AtomicBool::new(false));
    let (session, _) = chain_session(&endpoint, Vec::new(), flag.clone());
    seed_deferred_response(&session, &flag).await;

    assert_eq!(session.pop_item().await.unwrap(), None);
    session.run_compaction(force()).await.unwrap();
    assert_eq!(endpoint.requests().len(), 1);
    assert_eq!(endpoint.requests()[0]["previous_response_id"], "resp-old");
}

#[tokio::test]
async fn clear_session_invalidates_response_chain() {
    let endpoint = Compaction::replying(vec![]).await;
    let item = InputItem::from_value(json!({"role": "assistant", "content": "remove me"}));
    let flag = Arc::new(AtomicBool::new(false));
    let (session, _) = chain_session(&endpoint, vec![item], flag.clone());
    seed_deferred_response(&session, &flag).await;

    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
    let error = session
        .run_compaction(force())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires a response_id"), "{error}");
    assert!(endpoint.requests().is_empty());

    session
        .run_compaction(response_args("resp-new", true))
        .await
        .unwrap();
    assert_eq!(endpoint.requests()[0]["previous_response_id"], "resp-new");
}

#[tokio::test]
async fn run_compaction_auto_uses_input_after_clear() {
    let endpoint = Compaction::replying(vec![]).await;
    let item = InputItem::from_value(json!({"role": "assistant", "content": "old turn"}));
    let underlying = Arc::new(ListSession::with_history("test", vec![item]));
    let session = compaction_session("test", underlying, &endpoint, CompactionMode::Auto);
    // Default auto mode follows the stored response chain first.
    session
        .run_compaction(response_args("resp-old", true))
        .await
        .unwrap();
    let first = &endpoint.requests()[0];
    assert_eq!(first["previous_response_id"], "resp-old");
    assert!(first.get("input").is_none());

    session.clear_session().await.unwrap();
    let new_items = vec![
        InputItem::from_value(json!({"role": "user", "content": "fresh start"})),
        InputItem::from_value(json!({"role": "assistant", "content": "fresh reply"})),
    ];
    session.add_items(new_items.clone()).await.unwrap();
    // After clearing the stored response, auto mode compacts from the current local input.
    session.run_compaction(force()).await.unwrap();
    assert_eq!(endpoint.requests().len(), 2);
    let second = &endpoint.requests()[1];
    assert!(second.get("previous_response_id").is_none());
    assert_eq!(second["input"], json!(values(&new_items)));
}

// test_add_waits_for_in_flight_compaction_and_survives / test_clear_waits_...
async fn mutation_waits_case(clear: bool) {
    let old =
        InputItem::from_value(json!({"type": "message", "role": "assistant", "content": "old"}));
    let concurrent =
        InputItem::from_value(json!({"type": "message", "role": "user", "content": "concurrent"}));
    let compacted = json!({"type": "compaction", "summary": "compacted"});
    let underlying = Arc::new(ListSession::with_history("serialized", vec![old]));
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(AtomicBool::new(false));
    let (signal, gate) = (entered.clone(), release.clone());
    let reply = compacted.clone();
    let endpoint = Compaction::with(move |_| {
        signal.notify_one();
        while !gate.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        json!({"id": "cmp", "output": [reply.clone()], "usage": usage()})
    })
    .await;
    let session = Arc::new(compaction_session(
        "serialized",
        underlying.clone(),
        &endpoint,
        CompactionMode::Input,
    ));
    let compacting = {
        let session = session.clone();
        tokio::spawn(async move { session.run_compaction(force()).await })
    };
    entered.notified().await;
    let mutating = {
        let session = session.clone();
        let concurrent = concurrent.clone();
        tokio::spawn(async move {
            if clear {
                session.clear_session().await
            } else {
                session.add_items(vec![concurrent]).await
            }
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(
        !mutating.is_finished(),
        "the mutation must wait for the compaction (clear={clear})"
    );
    release.store(true, Ordering::SeqCst);
    compacting.await.unwrap().unwrap();
    mutating.await.unwrap().unwrap();
    let stored = values(&underlying.get_items(None).await.unwrap());
    if clear {
        assert!(stored.is_empty());
    } else {
        assert_eq!(stored, vec![compacted, concurrent.as_value().clone()]);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn add_and_clear_wait_for_an_in_flight_compaction() {
    openai_agents_sdk::set_tracing_disabled(true);
    mutation_waits_case(false).await;
    mutation_waits_case(true).await;
}

// TestCompactionStripsOrphanedIds
#[tokio::test]
async fn run_compaction_strips_orphaned_assistant_ids() {
    for with_reasoning in [false, true] {
        let output = if with_reasoning {
            vec![
                json!({"type": "reasoning", "id": "rs_111", "content": "thinking..."}),
                json!({"type": "message", "role": "assistant", "id": "msg_aaa", "content": "answer"}),
            ]
        } else {
            (1..=3)
                .map(|i| json!({"type": "message", "role": "assistant", "id": format!("msg_{i}"), "content": format!("summary {i}")}))
                .collect()
        };
        let endpoint = Compaction::replying(output).await;
        let history: Vec<InputItem> = (0..10)
            .map(|i| {
                InputItem::from_value(
                    json!({"type": "message", "role": "assistant", "content": format!("m{i}")}),
                )
            })
            .collect();
        let underlying = Arc::new(ListSession::with_history("test", history));
        let session =
            compaction_session("test", underlying.clone(), &endpoint, CompactionMode::Auto);
        session
            .run_compaction(response_args("resp-123", false))
            .await
            .unwrap();
        let stored = values(&underlying.get_items(None).await.unwrap());
        if with_reasoning {
            let assistant: Vec<&Value> =
                stored.iter().filter(|i| i["role"] == "assistant").collect();
            assert_eq!(assistant[0]["id"], "msg_aaa");
        } else {
            assert!(!stored.is_empty());
            assert!(stored.iter().all(|i| i.get("id").is_none()), "{stored:?}");
        }
    }
}

// test_run_compaction_retains_deferred_marker_when_api_call_fails
#[tokio::test]
async fn failed_compaction_keeps_its_forced_marker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let base = serve(Arc::new(move |_req: &Request| {
        if counter.fetch_add(1, Ordering::SeqCst) == 0 {
            return Reply::status(
                500,
                json!({"error": {"message": "compact API blew up"}}).to_string(),
            );
        }
        Reply::json(json!({"id": "cmp", "output": [], "usage": usage()}))
    }))
    .await;
    let hook = Arc::new(AtomicBool::new(true));
    let flag = hook.clone();
    let client = OpenAIClient::new("sk-test").with_base_url(format!("{base}/v1"));
    let session = OpenAIResponsesCompactionSession::new("test", Arc::new(ListSession::new("test")))
        .unwrap()
        .with_client(client)
        .with_should_trigger_compaction(move |_| flag.load(Ordering::SeqCst));
    // A turn with local tool outputs defers compaction for this response.
    session
        .on_run_completed(&SessionRunCompletion {
            response_id: Some("resp-handoff".into()),
            has_local_tool_outputs: true,
            ..Default::default()
        })
        .await
        .unwrap();
    hook.store(false, Ordering::SeqCst);

    let completion = SessionRunCompletion {
        response_id: Some("resp-delegate".into()),
        ..Default::default()
    };
    assert!(session.on_run_completed(&completion).await.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // The hook still declines, so only the retained marker can force the retry.
    session.on_run_completed(&completion).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    // The marker is cleared once the forced compaction succeeded.
    session.on_run_completed(&completion).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let _ = assistant_message("unused");
}
