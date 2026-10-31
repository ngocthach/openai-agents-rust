//! Port of `tests/test_tracing.py`: span and trace lifecycle, nesting, disabled propagation,
//! metadata and the trace-level API key.

#[path = "py_core3_tracing_support/mod.rs"]
mod support;

use std::sync::Arc;

use openai_agents_sdk::tracing::{
    Span, SpanData, Trace, TracingProcessor, agent_span, function_span, generation_span,
    handoff_span, set_trace_processors, span, trace,
};
use serde_json::{Value, json};
use support::{NoApiKey, Pipeline, SpanRecorder, lock_async};

const TRACE_ID: &str = "trace_0123456789abcdef0123456789abcdef";

fn with_recorder() -> (Arc<SpanRecorder>, Pipeline) {
    let recorder = SpanRecorder::new();
    let pipeline = Pipeline::install(vec![recorder.clone() as Arc<dyn TracingProcessor>]);
    (recorder, pipeline)
}

fn custom_data(name: &str) -> SpanData {
    SpanData::Custom {
        name: name.to_string(),
        data: json!({}),
    }
}

/// A generation span with a fixed id and no data, as the async tests use.
fn generation_with_id(id: &str) -> Span {
    span(SpanData::Generation {
        input: None,
        output: None,
        model: None,
        model_config: None,
        usage: None,
    })
    .span_id(id)
    .build()
}

// ---------------------------------------------------------------------------------------------
// tree shape
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn test_simple_tracing() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    let x = trace("test").trace_id(TRACE_ID).build();
    x.start();
    let span_1 = span(SpanData::Agent {
        name: "agent_1".into(),
        handoffs: None,
        tools: None,
        output_type: None,
    })
    .span_id("span_1")
    .parent_trace(x.clone())
    .build();
    span_1.start();
    span_1.finish();

    let span_2 = span(custom_data("custom_1"))
        .span_id("span_2")
        .parent_trace(x.clone())
        .build();
    span_2.start();
    let span_3 = span(custom_data("custom_2"))
        .span_id("span_3")
        .parent_span(span_2.clone())
        .build();
    span_3.start();
    span_3.finish();
    span_2.finish();
    x.finish();

    assert_eq!(
        recorder.normalized(true, false),
        vec![json!({
            "workflow_name": "test",
            "children": [
                {"type": "agent", "id": "span_1", "data": {"name": "agent_1"}},
                {
                    "type": "custom",
                    "id": "span_2",
                    "data": {"name": "custom_1", "data": {}},
                    "children": [
                        {"type": "custom", "id": "span_3", "data": {"name": "custom_2", "data": {}}}
                    ],
                },
            ],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_ctxmanager_spans() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    trace("test")
        .trace_id(TRACE_ID)
        .group_id("456")
        .run(async {
            span(custom_data("custom_1"))
                .span_id("span_1")
                .build()
                .run(async {
                    span(custom_data("custom_2"))
                        .span_id("span_1_inner")
                        .build()
                        .run(async {})
                        .await;
                })
                .await;
            span(custom_data("custom_2"))
                .span_id("span_2")
                .build()
                .run(async {})
                .await;
        })
        .await;

    assert_eq!(
        recorder.normalized(true, false),
        vec![json!({
            "workflow_name": "test",
            "group_id": "456",
            "children": [
                {
                    "type": "custom",
                    "id": "span_1",
                    "data": {"name": "custom_1", "data": {}},
                    "children": [
                        {"type": "custom", "id": "span_1_inner", "data": {"name": "custom_2", "data": {}}}
                    ],
                },
                {"type": "custom", "id": "span_2", "data": {"name": "custom_2", "data": {}}},
            ],
        })]
    );
}

async fn run_subtask(span_id: &str) {
    generation_with_id(span_id)
        .run(async {
            tokio::time::sleep(std::time::Duration::from_micros(100)).await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_async_tracing() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    trace("test")
        .trace_id(TRACE_ID)
        .group_id("group_456")
        .run(async {
            run_subtask("span_1").await;
            run_subtask("span_2").await;
        })
        .await;

    assert_eq!(
        recorder.normalized(true, false),
        vec![json!({
            "workflow_name": "test",
            "group_id": "group_456",
            "children": [
                {"type": "generation", "id": "span_1"},
                {"type": "generation", "id": "span_2"},
            ],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_complex_async_tracing() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    // Python repeats this 300 times to flush out ordering races between concurrent spans.
    for _ in 0..300 {
        recorder.clear();
        trace("test")
            .trace_id(TRACE_ID)
            .group_id("456")
            .run(async {
                let parallel_a = async {
                    tokio::join!(run_subtask("span_1"), run_subtask("span_2"));
                };
                let parallel_b = async {
                    tokio::join!(run_subtask("span_3"), run_subtask("span_4"));
                };
                tokio::join!(parallel_a, parallel_b);

                let child_a = async {
                    generation_with_id("span_5")
                        .run(async { run_subtask("span_6").await })
                        .await;
                };
                let child_b = async {
                    generation_with_id("span_7")
                        .run(async { run_subtask("span_8").await })
                        .await;
                };
                tokio::join!(child_a, child_b);
            })
            .await;

        assert_eq!(
            recorder.normalized(true, false),
            vec![json!({
                "workflow_name": "test",
                "group_id": "456",
                "children": [
                    {"type": "generation", "id": "span_1"},
                    {"type": "generation", "id": "span_2"},
                    {"type": "generation", "id": "span_3"},
                    {"type": "generation", "id": "span_4"},
                    {
                        "type": "generation",
                        "id": "span_5",
                        "children": [{"type": "generation", "id": "span_6"}],
                    },
                    {
                        "type": "generation",
                        "id": "span_7",
                        "children": [{"type": "generation", "id": "span_8"}],
                    },
                ],
            })]
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn test_spans_with_setters() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    trace("test")
        .trace_id(TRACE_ID)
        .group_id("456")
        .run(async {
            let span_a = span(SpanData::Agent {
                name: "agent_1".into(),
                handoffs: None,
                tools: None,
                output_type: None,
            })
            .build();
            span_a
                .clone()
                .run(async {
                    span_a.update_data(|d| {
                        if let SpanData::Agent { name, .. } = d {
                            *name = "agent_2".into();
                        }
                    });
                    let span_b = function_span("function_1", None);
                    span_b
                        .clone()
                        .run(async {
                            span_b.update_data(|d| {
                                if let SpanData::Function { input, output, .. } = d {
                                    *input = Some("i".into());
                                    *output = Some("o".into());
                                }
                            });
                        })
                        .await;
                    let span_c = generation_span(None, None, None, false);
                    span_c
                        .clone()
                        .run(async {
                            span_c.update_data(|d| {
                                if let SpanData::Generation { input, .. } = d {
                                    *input = Some(json!([{"foo": "bar"}]));
                                }
                            });
                        })
                        .await;
                    handoff_span("agent_1", Some("agent_2")).run(async {}).await;
                })
                .await;
        })
        .await;

    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "test",
            "group_id": "456",
            "children": [{
                "type": "agent",
                "data": {"name": "agent_2"},
                "children": [
                    {"type": "function", "data": {"name": "function_1", "input": "i", "output": "o"}},
                    {"type": "generation", "data": {"input": [{"foo": "bar"}]}},
                    {"type": "handoff", "data": {"from_agent": "agent_1", "to_agent": "agent_2"}},
                ],
            }],
        })]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_custom_span_records_output_in_data() {
    let _serial = lock_async().await;
    let (_recorder, _pipeline) = with_recorder();

    let cell: std::sync::Mutex<Option<Span>> = std::sync::Mutex::new(None);
    trace("test")
        .run(async {
            let s = span(SpanData::Custom {
                name: "database_query".into(),
                data: json!({"operation": "SELECT", "table": "users"}),
            })
            .build();
            *cell.lock().unwrap() = Some(s.clone());
            s.clone()
                .run(async {
                    s.update_data(|d| {
                        if let SpanData::Custom { data, .. } = d {
                            data["output"] = json!({"count": 2});
                        }
                    });
                })
                .await;
        })
        .await;

    let s = cell.lock().unwrap().clone().expect("span");
    let exported = s.export().expect("exported span");
    assert_eq!(
        exported["span_data"],
        json!({
            "type": "custom",
            "name": "database_query",
            "data": {"operation": "SELECT", "table": "users", "output": {"count": 2}},
        })
    );
}

// ---------------------------------------------------------------------------------------------
// disabled propagation
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn test_disabled_tracing() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    trace("test")
        .trace_id(TRACE_ID)
        .group_id("456")
        .disabled(true)
        .run(async {
            agent_span("agent_1", None, None, None)
                .run(async {
                    function_span("function_1", None).run(async {}).await;
                })
                .await;
        })
        .await;

    assert!(recorder.traces().is_empty(), "no traces");
    assert!(recorder.ordered_spans().is_empty(), "no spans");
}

#[tokio::test(flavor = "current_thread")]
async fn test_enabled_trace_disabled_span() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    trace("test")
        .trace_id(TRACE_ID)
        .run(async {
            agent_span("agent_1", None, None, None)
                .run(async {
                    span(SpanData::Function {
                        name: "function_1".into(),
                        input: None,
                        output: None,
                        mcp_data: None,
                    })
                    .disabled(true)
                    .build()
                    .run(async {
                        generation_span(None, None, None, false).run(async {}).await;
                    })
                    .await;
                })
                .await;
        })
        .await;

    assert_eq!(
        recorder.normalized(false, false),
        vec![json!({
            "workflow_name": "test",
            "children": [{"type": "agent", "data": {"name": "agent_1"}}],
        })]
    );
}

// ---------------------------------------------------------------------------------------------
// lifecycle events
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn test_start_and_end_called_manual() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    let x = trace("test").trace_id(TRACE_ID).build();
    x.start();
    let span_1 = span(custom_data("agent_1"))
        .span_id("span_1")
        .parent_trace(x.clone())
        .build();
    span_1.start();
    span_1.finish();
    let span_2 = span(custom_data("custom_1"))
        .span_id("span_2")
        .parent_trace(x.clone())
        .build();
    span_2.start();
    let span_3 = span(custom_data("custom_2"))
        .span_id("span_3")
        .parent_span(span_2.clone())
        .build();
    span_3.start();
    span_3.finish();
    span_2.finish();
    x.finish();

    assert_eq!(
        recorder.events(),
        [
            "trace_start",
            "span_start", // span_1
            "span_end",   // span_1
            "span_start", // span_2
            "span_start", // span_3
            "span_end",   // span_3
            "span_end",   // span_2
            "trace_end",
        ]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_start_and_end_called_ctxmanager() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    trace("test")
        .trace_id(TRACE_ID)
        .group_id("456")
        .run(async {
            span(custom_data("custom_1"))
                .span_id("span_1")
                .build()
                .run(async {
                    span(custom_data("custom_2"))
                        .span_id("span_1_inner")
                        .build()
                        .run(async {})
                        .await;
                })
                .await;
            span(custom_data("custom_2"))
                .span_id("span_2")
                .build()
                .run(async {})
                .await;
        })
        .await;

    assert_eq!(
        recorder.events(),
        [
            "trace_start",
            "span_start", // span_1
            "span_start", // span_1_inner
            "span_end",   // span_1_inner
            "span_end",   // span_1
            "span_start", // span_2
            "span_end",   // span_2
            "trace_end",
        ]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_start_and_end_called_async_ctxmanager() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    trace("test")
        .trace_id(TRACE_ID)
        .group_id("group_456")
        .run(async {
            run_subtask("span_1").await;
            run_subtask("span_2").await;
        })
        .await;

    assert_eq!(
        recorder.events(),
        [
            "trace_start",
            "span_start", // span_1
            "span_end",   // span_1
            "span_start", // span_2
            "span_end",   // span_2
            "trace_end",
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// no-op spans
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn test_noop_span_doesnt_record() {
    let _serial = lock_async().await;
    let (recorder, _pipeline) = with_recorder();

    let t = trace("test").disabled(true).build();
    let cell: std::sync::Mutex<Option<Span>> = std::sync::Mutex::new(None);
    t.clone()
        .run(async {
            let s = custom_data_span("span_1");
            *cell.lock().unwrap() = Some(s.clone());
            s.clone()
                .run(async {
                    s.set_error("test", None);
                })
                .await;
        })
        .await;
    let s = cell.lock().unwrap().clone().expect("span");

    assert!(recorder.traces().is_empty());
    assert!(recorder.ordered_spans().is_empty());
    assert!(t.export().is_none());
    assert!(s.export().is_none());
    assert!(s.started_at().is_none());
    assert!(s.ended_at().is_none());
    assert!(s.error().is_none());
}

fn custom_data_span(name: &str) -> Span {
    span(custom_data(name)).build()
}

#[tokio::test(flavor = "current_thread")]
async fn test_multiple_span_start_finish_doesnt_crash() {
    let _serial = lock_async().await;
    let (_recorder, _pipeline) = with_recorder();

    trace("test")
        .trace_id(TRACE_ID)
        .group_id("456")
        .run(async {
            let s = custom_data_span("span_1");
            s.clone()
                .run(async {
                    s.start();
                })
                .await;
            s.finish();
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_noop_parent_is_noop_child() {
    let _serial = lock_async().await;
    let (_recorder, _pipeline) = with_recorder();

    let tr = trace("test").disabled(true).build();
    let s = span(custom_data("span_1")).parent_trace(tr.clone()).build();
    s.start();
    s.finish();
    assert!(s.export().is_none());

    let s2 = span(custom_data("span_2")).parent_span(s.clone()).build();
    s2.start();
    s2.finish();
    assert!(s2.export().is_none());
}

// ---------------------------------------------------------------------------------------------
// keys and metadata
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn test_trace_and_spans_use_tracing_config_key() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (_recorder, _pipeline) = with_recorder();

    let tr = trace("test").tracing_api_key("tracing-key").build();
    assert_eq!(tr.tracing_api_key(), Some("tracing-key"));
    tr.clone()
        .run(async {
            let s = custom_data_span("span_with_key");
            assert_eq!(s.tracing_api_key(), Some("tracing-key"));
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_trace_metadata_propagates_to_spans() {
    let _serial = lock_async().await;
    let (_recorder, _pipeline) = with_recorder();

    let metadata: serde_json::Map<String, Value> =
        json!({"source": "run"}).as_object().cloned().unwrap();
    let current = trace("test").metadata(metadata.clone()).build();
    current
        .clone()
        .run(async {
            let direct_child = span(custom_data("direct_child"))
                .parent_trace(current.clone())
                .build();
            assert_eq!(direct_child.trace_metadata(), Some(&metadata));
            let export = direct_child.export().expect("exported");
            assert!(export.get("metadata").is_none());

            let parent = custom_data_span("parent");
            parent
                .clone()
                .run(async {
                    assert_eq!(parent.trace_metadata(), Some(&metadata));
                    let export = parent.export().expect("exported");
                    assert!(export.get("metadata").is_none());
                    let child = span(custom_data("child"))
                        .parent_span(parent.clone())
                        .build();
                    assert_eq!(child.trace_metadata(), Some(&metadata));
                    assert!(child.export().expect("exported").get("metadata").is_none());
                })
                .await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_agent_span_metadata_exports_with_routing_metadata() {
    let _serial = lock_async().await;
    let (_recorder, _pipeline) = with_recorder();

    let metadata = json!({
        "agent_harness_id": "harness_123",
        "agent_id": "agent_123",
        "agent_task_id": "task_123",
        "tenant_id": "tenant_123",
        "user_id": "user_123",
    })
    .as_object()
    .cloned()
    .unwrap();
    let agent_cell: std::sync::Mutex<Option<Span>> = std::sync::Mutex::new(None);
    trace("test")
        .metadata(metadata)
        .run(async {
            let agent = agent_span("agent", None, None, None);
            *agent_cell.lock().unwrap() = Some(agent.clone());
            agent
                .clone()
                .run(async {
                    agent.set_data_metadata(Some(
                        json!({"usage": {
                            "requests": 1,
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                            "cached_input_tokens": 3,
                        }})
                        .as_object()
                        .cloned()
                        .unwrap(),
                    ));
                })
                .await;
        })
        .await;

    let agent = agent_cell.lock().unwrap().clone().expect("agent span");
    let export = agent.export().expect("exported");
    assert_eq!(
        export["metadata"],
        json!({
            "agent_harness_id": "harness_123",
            "usage": {
                "requests": 1,
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "cached_input_tokens": 3,
            },
        })
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_processor_can_lookup_trace_metadata_by_span_trace_id() {
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct MetadataPropagatingProcessor {
        trace_metadata_by_id: Mutex<HashMap<String, serde_json::Map<String, Value>>>,
        looked_up: Mutex<Option<serde_json::Map<String, Value>>>,
        span_trace_metadata: Mutex<Option<serde_json::Map<String, Value>>>,
    }

    impl TracingProcessor for MetadataPropagatingProcessor {
        fn on_trace_start(&self, trace: &Trace) {
            if let Some(metadata) = trace.metadata() {
                self.trace_metadata_by_id
                    .lock()
                    .unwrap()
                    .insert(trace.trace_id().to_string(), metadata.clone());
            }
        }
        fn on_trace_end(&self, _trace: &Trace) {}
        fn on_span_start(&self, _span: &Span) {}
        fn on_span_end(&self, span: &Span) {
            if span.data().span_type() != "agent" {
                return;
            }
            *self.looked_up.lock().unwrap() = self
                .trace_metadata_by_id
                .lock()
                .unwrap()
                .get(span.trace_id())
                .cloned();
            *self.span_trace_metadata.lock().unwrap() = span.trace_metadata().cloned();
        }
    }

    let _serial = lock_async().await;
    let processor = Arc::new(MetadataPropagatingProcessor::default());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    let metadata = json!({"user_id": "u_123", "chat_type": "support"})
        .as_object()
        .cloned()
        .unwrap();
    trace("workflow")
        .metadata(metadata.clone())
        .run(async {
            agent_span("agent", None, None, None).run(async {}).await;
        })
        .await;
    set_trace_processors(Vec::new());

    assert_eq!(*processor.looked_up.lock().unwrap(), Some(metadata.clone()));
    assert_eq!(
        *processor.span_trace_metadata.lock().unwrap(),
        Some(metadata)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_trace_to_json_only_includes_tracing_api_key_when_requested() {
    let _serial = lock_async().await;
    let _key = NoApiKey::new();
    let (_recorder, _pipeline) = with_recorder();

    let tr = trace("test").tracing_api_key("secret-key").build();
    let default_json = tr.to_json(false).expect("trace json");
    assert!(default_json.get("tracing_api_key").is_none());
    let with_key = tr.to_json(true).expect("trace json");
    assert_eq!(with_key["tracing_api_key"], "secret-key");
}
