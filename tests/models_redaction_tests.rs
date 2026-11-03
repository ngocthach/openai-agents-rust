//! Model span errors and SDK logs honor the sensitive-data settings.
//!
//! `set_trace_processors` and the `debug` flags are process-global, so these tests live in their
//! own integration-test binary.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::models::interface::{Model, ModelRequest, ModelTracing};
use openai_agents_sdk::tracing::{
    InMemoryTraceProcessor, REDACTED_TRACE_ERROR_MESSAGE, TracingProcessor, set_trace_processors,
    trace,
};
use openai_agents_sdk::{OpenAIChatCompletionsModel, OpenAIClient, OpenAIResponsesModel};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const SECRET: &str = "sk-leaked-transcript-fragment";

/// A server that answers every request with a 400 whose body carries `SECRET`.
async fn failing_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let mut tmp = [0u8; 8192];
            let _ = sock.read(&mut tmp).await;
            let body = format!("{{\"error\":{{\"message\":\"{SECRET}\"}}}}");
            let resp = format!(
                "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
    });
    format!("http://{addr}/v1")
}

fn span_errors(items: &[Value]) -> Vec<Value> {
    items
        .iter()
        .filter(|i| i["object"] == "trace.span")
        .filter_map(|i| i.get("error").filter(|e| !e.is_null()).cloned())
        .collect()
}

async fn collect_span_errors(
    model: Arc<dyn Model>,
    tracing: ModelTracing,
    processor: &Arc<InMemoryTraceProcessor>,
) -> Vec<Value> {
    processor.clear();
    trace("redaction")
        .run(async {
            let request = ModelRequest {
                tracing,
                ..Default::default()
            };
            assert!(model.get_response(request).await.is_err());
        })
        .await;
    span_errors(&processor.items())
}

// One test: trace processors are global, so these cannot run in parallel.
#[tokio::test]
async fn model_spans_follow_trace_include_sensitive_data() {
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    let base_url = failing_server().await;
    let client = OpenAIClient::new("k").with_base_url(&base_url);

    for model in [
        Arc::new(OpenAIResponsesModel::new("gpt-4o", client.clone())) as Arc<dyn Model>,
        Arc::new(OpenAIChatCompletionsModel::new("gpt-4o", client.clone())) as Arc<dyn Model>,
    ] {
        let errors =
            collect_span_errors(model.clone(), ModelTracing::EnabledWithoutData, &processor).await;
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(errors[0]["message"], "Error getting response");
        assert_eq!(errors[0]["data"]["error"], REDACTED_TRACE_ERROR_MESSAGE);

        let errors = collect_span_errors(model, ModelTracing::Enabled, &processor).await;
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0]["data"]["error"]
                .as_str()
                .unwrap()
                .contains(SECRET),
            "{errors:?}"
        );
    }

    // The generation span drops `metadata` unless the run includes data.
    let model =
        OpenAIChatCompletionsModel::new("gpt-4o", OpenAIClient::new("k").with_base_url(&base_url));
    let settings = openai_agents_sdk::ModelSettings {
        metadata: Some(
            [("user".to_string(), SECRET.to_string())]
                .into_iter()
                .collect(),
        ),
        ..Default::default()
    };

    let config_for = |tracing| {
        let processor = processor.clone();
        let settings = settings.clone();
        let model = model.clone();
        async move {
            processor.clear();
            trace("metadata")
                .run(async {
                    let request = ModelRequest {
                        model_settings: settings,
                        tracing,
                        ..Default::default()
                    };
                    let _ = model.get_response(request).await;
                })
                .await;
            processor
                .items()
                .into_iter()
                .find(|i| i["span_data"]["type"] == "generation")
                .expect("generation span")["span_data"]["model_config"]
                .clone()
        }
    };

    let without = config_for(ModelTracing::EnabledWithoutData).await;
    assert!(without.get("metadata").is_none(), "{without}");
    assert_eq!(without["base_url"], format!("{base_url}"));

    let with = config_for(ModelTracing::Enabled).await;
    assert_eq!(with["metadata"]["user"], SECRET);
}

#[test]
fn redacting_log_helpers_follow_the_debug_flags() {
    // A unique probe message: `log` is process-global, and the tracing test in this binary
    // logs through the same target in parallel.
    const PROBE: &str = "redaction probe";

    #[derive(Default)]
    struct Collect(Mutex<Vec<String>>);
    impl log::Log for Collect {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            let line = record.args().to_string();
            if record.target() == openai_agents_sdk::logger::LOG_TARGET && line.starts_with(PROBE) {
                self.0.lock().unwrap().push(line);
            }
        }
        fn flush(&self) {}
    }
    static SINK: std::sync::OnceLock<Collect> = std::sync::OnceLock::new();
    let sink: &'static Collect = SINK.get_or_init(Collect::default);
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);

    openai_agents_sdk::debug::set_dont_log_model_data(true);
    openai_agents_sdk::logger::log_model_action_error(PROBE, &SECRET);
    openai_agents_sdk::debug::set_dont_log_model_data(false);
    openai_agents_sdk::logger::log_model_action_error(PROBE, &SECRET);
    openai_agents_sdk::debug::reset_dont_log_model_data();

    openai_agents_sdk::debug::set_dont_log_tool_data(true);
    // A mixed model/tool failure is redacted when either policy is on.
    openai_agents_sdk::logger::log_model_and_tool_action_warning(PROBE, &SECRET);
    openai_agents_sdk::debug::reset_dont_log_tool_data();

    let lines = sink.0.lock().unwrap().clone();
    assert_eq!(lines, [PROBE, &format!("{PROBE}: {SECRET}"), PROBE]);
}
