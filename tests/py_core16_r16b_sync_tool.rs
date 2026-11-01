//! Ports of `tests/test_function_tool.py::test_sync_function_runs_via_to_thread` and
//! `test_sync_function_does_not_block_event_loop`: a synchronous `#[function_tool]` body runs off
//! the async worker, so a blocking body does not stall other tasks. The tests run on a
//! single-threaded runtime, where a body run inline would freeze every other task.

#![cfg(feature = "macros")]

use std::time::{Duration, Instant};

use openai_agents_sdk::function_tool;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::ToolOutput;
use openai_agents_sdk::tool_context::ToolContext;

/// Blocks its thread for 200 ms.
#[function_tool]
fn slow_sync() -> String {
    std::thread::sleep(Duration::from_millis(200));
    "ok".to_string()
}

/// Reports the name of the thread it runs on.
#[function_tool]
fn thread_name() -> String {
    std::thread::current().name().unwrap_or("").to_string()
}

async fn invoke(tool: &openai_agents_sdk::tool::FunctionTool<()>) -> String {
    let ctx = ToolContext::direct(RunContext::new(()), "agent", &tool.name, "{}");
    match tool.invoke(ctx, "{}".to_string()).await.unwrap() {
        ToolOutput::Text(text) => text,
        other => panic!("text output expected, got {other:?}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn a_blocking_sync_tool_does_not_stall_other_tasks() {
    let tool = slow_sync_tool::<()>();
    let started = Instant::now();
    let background = tokio::spawn(async {
        tokio::time::sleep(Duration::from_millis(10)).await;
        Instant::now()
    });
    let (result, finished_background) = tokio::join!(invoke(&tool), background);
    assert_eq!(result, "ok");
    let background_done = finished_background.unwrap().duration_since(started);
    assert!(
        background_done < Duration::from_millis(150),
        "the other task finished after {background_done:?}, so the sync body blocked the runtime"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_sync_tool_body_runs_on_the_blocking_pool() {
    let tool = thread_name_tool::<()>();
    let name = invoke(&tool).await;
    assert_ne!(name, "", "the body ran on a pool thread");
    assert!(
        !name.starts_with("py_core16"),
        "the body ran on the test's own thread: {name}"
    );
}
