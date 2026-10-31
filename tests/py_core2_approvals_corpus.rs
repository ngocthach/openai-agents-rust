//! Ports of the fixture-driven half of `tests/test_run_state_compatibility_corpus.py`.
//!
//! The Python corpus under `tests/fixtures/run_state/` is copied verbatim into
//! `tests/fixtures/py_run_state_corpus/`. Every historical snapshot must be readable, rewrite to
//! the current schema, and rewrite idempotently; the pending approval recorded by the v0.19.4
//! writer must resume through `Runner` under an approve and a reject decision.

#![allow(clippy::field_reassign_with_default)]

use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunState, Runner};
use serde_json::{Value, json};

fn corpus(path: &str) -> Value {
    let full = format!(
        "{}/tests/fixtures/py_run_state_corpus/{path}",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(full).unwrap()).unwrap()
}

/// Read, rewrite, read the rewrite, and rewrite again: the second rewrite must equal the first.
fn assert_rewrites_idempotently(path: &str) -> Value {
    let payload = corpus(path);
    let first = RunState::from_json(payload.clone())
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_json()
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(
        first["$schemaVersion"],
        json!(CURRENT_SCHEMA_VERSION),
        "{path}"
    );
    let second = RunState::from_json(first.clone())
        .unwrap_or_else(|e| panic!("{path} (rewrite): {e}"))
        .to_json()
        .unwrap();
    assert_eq!(first, second, "{path}: the rewrite is not idempotent");
    // Durable topology survives the upgrade.
    assert_eq!(
        first["current_agent"]["name"], payload["current_agent"]["name"],
        "{path}"
    );
    assert_eq!(
        first["generated_items"].as_array().map(Vec::len),
        payload["generated_items"].as_array().map(Vec::len),
        "{path}"
    );
    assert_eq!(first["current_turn"], payload["current_turn"], "{path}");
    first
}

#[test]
fn historical_fixture_corpus_matches_supported_schema_versions() {
    let sources = corpus("sources.json");
    assert_eq!(sources["baseline"], "v0.19.4");
    let mut versions: Vec<String> = sources["versions"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    versions.push(CURRENT_SCHEMA_VERSION.to_string());
    for version in &versions {
        let mut minimal = corpus("minimal/v1_20.json");
        minimal["$schemaVersion"] = json!(version);
        if version == "1.0" || version == "1.1" {
            continue; // read below from their own fixtures
        }
        assert!(RunState::from_json(minimal).is_ok(), "{version}");
    }
    let features: Vec<&str> = sources["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["version"].as_str().unwrap())
        .collect();
    for version in sources["versions"].as_object().unwrap().keys() {
        if version != "1.0" && version != "1.1" {
            assert!(
                features.contains(&version.as_str()),
                "{version} has no feature fixture"
            );
        }
    }
}

#[test]
fn historical_minimal_run_state_rewrites_idempotently() {
    let sources = corpus("sources.json");
    for (version, entry) in sources["versions"].as_object().unwrap() {
        let fixture = entry["fixture"].as_str().unwrap();
        assert_eq!(corpus(fixture)["$schemaVersion"], json!(version));
        assert_rewrites_idempotently(fixture);
    }
}

#[test]
fn historical_feature_run_state_rewrites_semantically() {
    let sources = corpus("sources.json");
    for entry in sources["features"].as_array().unwrap() {
        let fixture = entry["fixture"].as_str().unwrap();
        assert_eq!(
            corpus(fixture)["$schemaVersion"],
            entry["version"],
            "{fixture}"
        );
        assert_rewrites_idempotently(fixture);
    }
}

#[test]
fn invalid_run_state_fixtures_fail_without_disclosing_values() {
    let sentinel = "RUNSTATE_SECRET_SENTINEL_42";
    for (fixture, needle) in [
        ("negative/missing_version.json", "schema version"),
        ("negative/future_version.json", "schema version"),
    ] {
        let payload = corpus(fixture);
        assert!(payload.to_string().contains(sentinel));
        let err = RunState::from_json(payload).unwrap_err().to_string();
        assert!(err.to_lowercase().contains(needle), "{fixture}: {err}");
        assert!(!err.contains(sentinel), "{fixture}: {err}");
    }
    // The current agent is resolved against the agent graph only when a run resumes.
    let state = RunState::from_json(corpus("negative/malformed_current_agent.json"));
    if let Err(err) = &state {
        assert!(!err.to_string().contains(sentinel));
    }
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("unused")]);
    let agent = Agent::<()>::new("compat-agent").with_model_instance(model);
    if let Ok(state) = state {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt
            .block_on(Runner::run(&agent, state))
            .unwrap_err()
            .to_string();
        assert!(err.contains("not found"), "{err}");
        assert!(!err.contains(sentinel), "{err}");
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct AccountArgs {
    account_id: String,
}

#[tokio::test]
async fn historical_approval_decisions_control_resumed_runs() {
    for approve in [true, false] {
        let executed = Arc::new(Mutex::new(Vec::<String>::new()));
        let e = executed.clone();
        let tool = FunctionTool::new(
            "historical_approval",
            "historical",
            move |_c, a: AccountArgs| {
                let e = e.clone();
                async move {
                    e.lock().unwrap().push(a.account_id.clone());
                    Ok(format!("approved:{}", a.account_id))
                }
            },
        )
        .with_needs_approval(true);
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("done")]);
        let agent = Agent::<()>::new("compat-agent")
            .with_tool(tool)
            .with_model_instance(model.clone());
        let mut state =
            RunState::from_json(corpus("resume/v1_13_pending_tool_approval.json")).unwrap();
        let pending = state.get_interruptions();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id(), "historical-approval-1");
        if approve {
            state.approve(&pending[0], false);
        } else {
            state.reject(&pending[0], false, Some("historical denial".into()));
        }
        let result = Runner::run(&agent, state).await.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        let outputs: Vec<Value> = result
            .new_items
            .iter()
            .filter_map(|i| match &i.kind {
                RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item["output"].clone()),
                _ => None,
            })
            .collect();
        if approve {
            assert_eq!(*executed.lock().unwrap(), vec!["account-1"]);
            assert_eq!(outputs, vec![json!("approved:account-1")]);
        } else {
            assert!(executed.lock().unwrap().is_empty());
            assert_eq!(outputs, vec![json!("historical denial")]);
        }
    }
}

#[tokio::test]
async fn canonical_invocation_identity_decision_controls_the_resumed_run() {
    // The `canonical_invocation_identity` case: the 1.15 fixture binds an approval of
    // `function-request-1` to its exact invocation, so the run that makes that call executes it
    // without asking again.
    let count = Arc::new(Mutex::new(0usize));
    let c = count.clone();
    let tool = FunctionTool::new("lookup_account", "lookup", move |_c, a: AccountArgs| {
        let c = c.clone();
        async move {
            *c.lock().unwrap() += 1;
            Ok(format!("approved:{}", a.account_id))
        }
    })
    .with_needs_approval(true);
    let model = Arc::new(FakeModel::new());
    let mut call = openai_agents_sdk::items::FunctionCall::new(
        "lookup_account",
        r#"{"account_id":"account-1"}"#,
    );
    call.call_id = "function-request-1".into();
    call.status = Some("completed".into());
    model.add_turn(vec![openai_agents_sdk::items::OutputItem::FunctionCall(
        call,
    )]);
    model.add_turn(vec![assistant_message("resume complete")]);
    let agent = Agent::<()>::new("compat-agent")
        .with_tool(tool)
        .with_model_instance(model);
    let state =
        RunState::from_json(corpus("features/v1_15_canonical_invocation_identity.json")).unwrap();
    let result = Runner::run(&agent, state).await.unwrap();
    assert!(result.interruptions.is_empty());
    assert_eq!(*count.lock().unwrap(), 1);
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("resume complete")
    );
    assert!(result.new_items.iter().any(|i| matches!(&i.kind,
        RunItemKind::ToolCallOutput { raw_item, .. } if raw_item["output"] == "approved:account-1")));
}
