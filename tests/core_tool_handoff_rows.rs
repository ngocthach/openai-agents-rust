//! The caller-side halves of the MCP, Codex and session-write rows: the tool metadata the
//! agent now carries, the duplicate-Codex-tool fail-fast, the reserved MCP tool names, and the
//! client helpers the voice and websocket paths route through.

use std::sync::Arc;

#[cfg(feature = "mcp")]
use openai_agents_sdk::handoff;
use openai_agents_sdk::models::OpenAIClient;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, RunContext};
use serde_json::json;

fn tool(name: &'static str) -> FunctionTool<()> {
    FunctionTool::<()>::new_no_args(name, name, |_c| async { Ok("ok") })
}

#[test]
fn a_function_tool_carries_an_mcp_title() {
    let plain = tool("search");
    assert_eq!(plain.mcp_title, None);
    let titled = tool("search").with_mcp_title(Some("Full-text search".into()));
    assert_eq!(titled.mcp_title.as_deref(), Some("Full-text search"));
    // The title survives a clone, like every other tool field.
    assert_eq!(
        titled.clone().mcp_title.as_deref(),
        Some("Full-text search")
    );
}

#[tokio::test]
async fn an_ordinary_duplicate_tool_name_is_not_a_codex_collision() {
    // Two same-named non-Codex tools are the caller's business; only a Codex tool collision
    // is fatal, because a Codex tool owns per-instance state.
    let agent = Arc::new(
        Agent::<()>::new("A")
            .with_tool(tool("run"))
            .with_tool(tool("run")),
    );
    let tools = agent.get_all_tools(&RunContext::new(())).await.unwrap();
    assert_eq!(tools.len(), 2);
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn the_agents_own_tool_and_handoff_names_are_reserved_for_mcp() {
    let target = Agent::<()>::new("Billing Agent");
    let agent = Agent::<()>::new("A")
        .with_tool(tool("search"))
        .with_tool(tool("fetch"))
        .with_handoff(handoff(target));
    let reserved = agent.mcp_reserved_tool_names(&RunContext::new(())).await;
    assert!(reserved.contains("search"));
    assert!(reserved.contains("fetch"));
    // The handoff's tool name is reserved too, so a derived MCP tool cannot shadow it.
    assert!(
        reserved.contains("transfer_to_billing_agent"),
        "{reserved:?}"
    );
    assert_eq!(reserved.len(), 3);
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn a_disabled_handoff_does_not_reserve_its_tool_name() {
    let target = Agent::<()>::new("Billing Agent");
    let agent = Agent::<()>::new("A").with_handoff(handoff(target).with_is_enabled(false));
    let reserved = agent.mcp_reserved_tool_names(&RunContext::new(())).await;
    assert!(reserved.is_empty(), "{reserved:?}");
}

#[test]
fn the_client_derives_a_websocket_base_url_from_its_http_one() {
    let client = OpenAIClient::new("sk-test");
    assert_eq!(client.websocket_base_url(), "wss://api.openai.com/v1");

    let local = OpenAIClient::new("sk-test").with_base_url("http://127.0.0.1:8080/v1/");
    assert_eq!(local.websocket_base_url(), "ws://127.0.0.1:8080/v1");
    assert_eq!(
        local.websocket_url("realtime"),
        "ws://127.0.0.1:8080/v1/realtime"
    );

    // An explicit websocket base URL wins over the derived one.
    let explicit = OpenAIClient::new("sk-test")
        .with_base_url("https://proxy.example/v1")
        .with_websocket_base_url("wss://ws.example/v2");
    assert_eq!(explicit.websocket_base_url(), "wss://ws.example/v2");
    assert_eq!(
        explicit.websocket_url("/realtime"),
        "wss://ws.example/v2/realtime"
    );

    // Default query parameters reach the handshake URL, as they do every HTTP request.
    let versioned = OpenAIClient::new("sk-test")
        .with_base_url("https://proxy.example/v1")
        .with_default_query("api-version", "2024-10-01");
    assert_eq!(
        versioned.websocket_url("realtime"),
        "wss://proxy.example/v1/realtime?api-version=2024-10-01"
    );
}

#[tokio::test]
async fn the_websocket_handshake_headers_carry_the_clients_auth() {
    let client = OpenAIClient::new("sk-test")
        .with_organization("org_1")
        .with_header("X-Trace", "abc");
    let headers = client
        .websocket_headers("wss://api.openai.com/v1/realtime")
        .await
        .unwrap();
    assert_eq!(headers["authorization"], "Bearer sk-test");
    assert_eq!(headers["openai-organization"], "org_1");
    assert_eq!(headers["x-trace"], "abc");
}

#[tokio::test]
async fn a_session_write_survives_a_dropped_caller() {
    use openai_agents_sdk::memory::await_session_mutation;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let done = Arc::new(AtomicUsize::new(0));
    let d = done.clone();
    let mutation = async move {
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        d.fetch_add(1, Ordering::SeqCst);
        Ok(json!("written"))
    };
    // The caller gives up after 5ms; the mutation still runs to completion.
    let fut = await_session_mutation(mutation);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(5), fut)
            .await
            .is_err()
    );
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    assert_eq!(done.load(Ordering::SeqCst), 1);
}

/// Output guardrails plus a server-managed conversation: rejected only when the model cannot
/// warn and ignore. Port of `validate_output_guardrails_with_server_managed_conversation`.
mod output_guardrails_with_server_conversation {
    use super::*;
    use openai_agents_sdk::models::interface::Model;
    use openai_agents_sdk::models::openai_chatcompletions::OpenAIChatCompletionsModel;
    use openai_agents_sdk::testing::{FakeModel, assistant_message};
    use openai_agents_sdk::{
        AgentsError, GuardrailFunctionOutput, OutputGuardrail, RunOptions, Runner,
    };

    fn guardrail() -> OutputGuardrail<()> {
        OutputGuardrail::new("never_trips", |_c, _a, _out: serde_json::Value| async {
            Ok(GuardrailFunctionOutput::new(json!({}), false))
        })
    }

    #[test]
    fn only_the_chat_completions_model_reports_itself_as_one() {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let chat = OpenAIChatCompletionsModel::new("gpt-4o", OpenAIClient::new("sk-test"));
        assert!(chat.is_chat_completions());
        assert!(!FakeModel::new().is_chat_completions());
    }

    #[tokio::test]
    async fn a_responses_model_still_rejects_the_combination() {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![assistant_message("hi")]);
        let agent = Agent::<()>::new("A")
            .with_output_guardrail(guardrail())
            .with_model_instance(model.clone());
        let err =
            Runner::run_with_options(&agent, "hi", RunOptions::new(()).conversation_id("conv_1"))
                .await
                .unwrap_err();
        assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
        assert!(
            err.to_string()
                .contains("Output guardrails cannot be combined")
        );
    }

    #[tokio::test]
    async fn a_chat_completions_model_is_exempt() {
        openai_agents_sdk::tracing::set_tracing_disabled(true);
        // The run is rejected before any model call, so an unreachable endpoint is enough to
        // show the guard no longer fires: the error must not be the guardrail `UserError`.
        let model = Arc::new(OpenAIChatCompletionsModel::new(
            "gpt-4o",
            OpenAIClient::new("sk-test").with_base_url("http://127.0.0.1:1/v1"),
        ));
        let agent = Agent::<()>::new("A")
            .with_output_guardrail(guardrail())
            .with_model_instance(model.clone());
        let err =
            Runner::run_with_options(&agent, "hi", RunOptions::new(()).conversation_id("conv_1"))
                .await
                .unwrap_err();
        assert!(
            !err.to_string()
                .contains("Output guardrails cannot be combined"),
            "the Chat Completions exemption did not apply: {err}"
        );
    }
}
