//! Ported from the Python SDK's `tests/test_pretty_print.py`.
//!
//! Rust has no runtime class name for a deserialized structured output, so where Python renders
//! `Final output (Foo):` the Rust port renders `Final output (dict):` and pretty-prints the JSON.
//! Everything else — layout, counts, indentation and terminal-control escaping — matches.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::{Agent, RunOptions, Runner};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Foo {
    bar: String,
}

fn model_with(output: Vec<openai_agents_sdk::items::OutputItem>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model.enqueue(ScriptedStep::output(output)).unwrap();
    model
}

#[tokio::test]
async fn pretty_result() {
    disable_tracing();
    let model = model_with(vec![assistant_message("Hi there")]);
    let agent = Agent::<()>::new("test_agent").with_model_instance(model);
    let result = Runner::run(&agent, "Hello").await.unwrap();

    assert_eq!(
        result.pretty_print(),
        "RunResult:\n\
         - Last agent: Agent(name=\"test_agent\", ...)\n\
         - Final output (str):\n\
         \x20   Hi there\n\
         - 1 new item(s)\n\
         - 1 raw response(s)\n\
         - 0 input guardrail result(s)\n\
         - 0 output guardrail result(s)\n\
         (See `RunResult` for more details)"
    );
}

#[tokio::test]
async fn pretty_run_result_streaming() {
    disable_tracing();
    let model = model_with(vec![assistant_message("Hi there")]);
    let agent = Agent::<()>::new("test_agent").with_model_instance(model);
    let mut streamed = Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }

    assert_eq!(
        streamed.pretty_print(),
        "RunResultStreaming:\n\
         - Current agent: Agent(name=\"test_agent\", ...)\n\
         - Current turn: 1\n\
         - Max turns: 10\n\
         - Is complete: True\n\
         - Final output (str):\n\
         \x20   Hi there\n\
         - 1 new item(s)\n\
         - 1 raw response(s)\n\
         - 0 input guardrail result(s)\n\
         - 0 output guardrail result(s)\n\
         (See `RunResultStreaming` for more details)"
    );
}

#[tokio::test]
async fn pretty_run_result_structured_output() {
    disable_tracing();
    let model = model_with(vec![
        assistant_message("Test"),
        assistant_message(json!({"bar": "Hi there"}).to_string()),
    ]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model)
        .with_output_type::<Foo>();
    let result = Runner::run(&agent, "Hello").await.unwrap();

    let rendered = result.pretty_print();
    assert!(
        rendered.contains("- Final output (dict):\n    {\n      \"bar\": \"Hi there\"\n    }"),
        "{rendered}"
    );
    assert!(rendered.contains("- 2 new item(s)"), "{rendered}");
    assert!(rendered.contains("- 1 raw response(s)"), "{rendered}");
}

#[tokio::test]
async fn pretty_run_result_streaming_structured_output() {
    disable_tracing();
    let model = model_with(vec![
        assistant_message("Test"),
        assistant_message(json!({"bar": "Hi there"}).to_string()),
    ]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model)
        .with_output_type::<Foo>();
    let mut streamed = Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            event.unwrap();
        }
    }

    let rendered = streamed.pretty_print();
    assert!(rendered.starts_with("RunResultStreaming:"), "{rendered}");
    assert!(
        rendered.contains("- Final output (dict):\n    {\n      \"bar\": \"Hi there\"\n    }"),
        "{rendered}"
    );
    assert!(rendered.contains("- 2 new item(s)"), "{rendered}");
}

#[tokio::test]
async fn pretty_run_result_list_structured_output() {
    disable_tracing();
    let payload = json!({"response": [{"bar": "Hi there"}, {"bar": "Hi there 2"}]}).to_string();
    let model = model_with(vec![assistant_message("Test"), assistant_message(&payload)]);
    let agent = Agent::<()>::new("test_agent")
        .with_model_instance(model)
        .with_output_type::<Vec<Foo>>();
    let result = Runner::run(&agent, "Hello").await.unwrap();

    let rendered = result.pretty_print();
    assert!(rendered.contains("- Final output (list):"), "{rendered}");
    assert!(rendered.contains("\"bar\": \"Hi there 2\""), "{rendered}");
    assert!(rendered.contains("- 2 new item(s)"), "{rendered}");
}

/// Python's `test_result_str_escapes_terminal_controls`, non-streamed and streamed.
#[tokio::test]
async fn result_str_escapes_terminal_controls() {
    disable_tracing();
    let payload = "Hello\u{1b}[2J\u{1b}]0;title\u{7}\u{8}\u{0}\u{7f}\u{9b}31m\u{9d}0;title\u{9c}\u{b}\u{c}\u{1c}\u{1d}\u{1e}\u{85}";
    let escaped =
        r"Hello\x1b[2J\x1b]0;title\x07\x08\x00\x7f\x9b31m\x9d0;title\x9c\x0b\x0c\x1c\x1d\x1e\x85";
    let agent_name = "agent\u{1b}[2J\u{9b}31m";

    for streaming in [false, true] {
        let model = model_with(vec![assistant_message(payload)]);
        let agent = Agent::<()>::new(agent_name).with_model_instance(model);
        let rendered = if streaming {
            let mut streamed =
                Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
            {
                let mut events = std::pin::pin!(streamed.stream_events());
                while let Some(event) = events.next().await {
                    event.unwrap();
                }
            }
            assert_eq!(streamed.final_output_text().as_deref(), Some(payload));
            streamed.pretty_print()
        } else {
            let result = Runner::run(&agent, "Hello").await.unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some(payload));
            result.pretty_print()
        };

        assert!(
            rendered.contains(r#"Agent(name="agent\x1b[2J\x9b31m", ...)"#),
            "{rendered}"
        );
        assert!(rendered.contains(escaped), "{rendered}");
        // The agent's own name is left intact.
        assert_eq!(agent.name, agent_name);
        // Nothing below space survives except newline and tab, and no C1 controls remain.
        assert!(
            rendered.chars().all(|c| c >= ' ' || c == '\n' || c == '\t'),
            "{rendered}"
        );
        assert!(
            !rendered.chars().any(|c| ('\u{7f}'..='\u{9f}').contains(&c)),
            "{rendered}"
        );
    }
}

#[tokio::test]
async fn result_str_preserves_readable_multiline_output() {
    disable_tracing();
    let payload = "Hello, 世界 👋\r\n\tCafé\n\nlast line";
    for streaming in [false, true] {
        let model = model_with(vec![assistant_message(payload)]);
        let agent = Agent::<()>::new("test_agent").with_model_instance(model);
        let rendered = if streaming {
            let mut streamed =
                Runner::run_streamed_with_options(&agent, "Hello", RunOptions::new(()));
            {
                let mut events = std::pin::pin!(streamed.stream_events());
                while let Some(event) = events.next().await {
                    event.unwrap();
                }
            }
            assert_eq!(streamed.final_output_text().as_deref(), Some(payload));
            streamed.pretty_print()
        } else {
            let result = Runner::run(&agent, "Hello").await.unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some(payload));
            result.pretty_print()
        };
        assert!(
            rendered
                .contains("    Hello, 世界 👋\n    \tCafé\n    \n    last line\n- 1 new item(s)"),
            "{rendered:?}"
        );
    }
}
