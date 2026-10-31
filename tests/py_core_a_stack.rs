//! Stack use of whole runs. A runner future is polled on whichever thread drives it; tokio worker
//! threads and test threads default to 2 MiB, so a run has to fit comfortably below that.
//!
//! Each scenario runs on a thread with a fixed small stack (1 MiB). A stack overflow aborts the
//! process, so the scenarios cannot be probed in-process: `measure_minimum_stack` (ignored by
//! default, run it with `-- --ignored --nocapture`) re-executes this binary at decreasing stack
//! sizes to report how little each scenario needs.

use std::sync::Arc;

use futures::StreamExt;
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn agents(model: &Arc<FakeModel>) -> Agent<()> {
    let tool = FunctionTool::new("lookup", "lookup", |_c, _a: NoArgs| async move {
        Ok("found".to_string())
    });
    let target = Arc::new(Agent::<()>::new("target").with_model_instance(model.clone()));
    Agent::<()>::new("triage")
        .with_tool(tool)
        .with_handoff(handoff(target))
        .with_model_instance(model.clone())
}

fn script(model: &Arc<FakeModel>) {
    model.add_turn(vec![function_call("lookup", "{}")]);
    model.add_turn(vec![function_call("transfer_to_target", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
}

#[cfg(feature = "sandbox")]
mod sandbox_scenario {
    use std::sync::Arc;

    use async_trait::async_trait;
    use openai_agents_sdk::sandbox::manifest::Manifest;
    use openai_agents_sdk::sandbox::session::{ExecRequest, SandboxClient, SandboxSession};
    use openai_agents_sdk::sandbox::state::SandboxSessionState;
    use openai_agents_sdk::sandbox::types::ExecResult;
    use openai_agents_sdk::sandbox::{ErrorCode, SandboxError};

    #[derive(Debug)]
    struct IdleSession {
        manifest: Manifest,
    }

    #[async_trait]
    impl SandboxSession for IdleSession {
        fn id(&self) -> &str {
            "idle"
        }
        fn manifest(&self) -> &Manifest {
            &self.manifest
        }
        fn backend_id(&self) -> &str {
            "fake"
        }
        async fn start(&self) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn stop(&self) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn shutdown(&self) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn running(&self) -> Result<bool, SandboxError> {
            Ok(false)
        }
        async fn exec(&self, _request: ExecRequest) -> Result<ExecResult, SandboxError> {
            Ok(ExecResult {
                exit_code: 0,
                stdout: Vec::new(),
                stderr: Vec::new(),
            })
        }
        async fn read_file(&self, path: &str) -> Result<Vec<u8>, SandboxError> {
            Err(SandboxError::new(
                ErrorCode::WorkspaceReadNotFound,
                format!("no file at {path}"),
            ))
        }
        async fn write_file(&self, _path: &str, _contents: &[u8]) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn hydrate_workspace(&self, _archive: &[u8]) -> Result<(), SandboxError> {
            Ok(())
        }
    }

    #[derive(Debug)]
    pub struct IdleClient;

    #[async_trait]
    impl SandboxClient for IdleClient {
        fn backend_id(&self) -> &str {
            "fake"
        }
        async fn create(
            &self,
            manifest: Option<Manifest>,
        ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
            Ok(Arc::new(IdleSession {
                manifest: manifest.unwrap_or_default(),
            }))
        }
        async fn delete(&self, _session: Arc<dyn SandboxSession>) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn resume(
            &self,
            _state: &SandboxSessionState,
            manifest: Manifest,
        ) -> Result<Arc<dyn SandboxSession>, SandboxError> {
            Ok(Arc::new(IdleSession { manifest }))
        }
    }
}

#[cfg(feature = "sandbox")]
async fn sandbox_scenario(streamed: bool) {
    use openai_agents_sdk::sandbox::{SandboxAgent, SandboxRunConfig, SandboxRunner};

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call("lookup", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    let tool = FunctionTool::new("lookup", "lookup", |_c, _a: NoArgs| async move {
        Ok("found".to_string())
    });
    let agent = SandboxAgent::<()>::new("sandbox")
        .with_model_instance(model)
        .with_instructions("Base instructions.")
        .with_capabilities(Vec::new())
        .with_tool(tool);
    let config = SandboxRunConfig::with_client(Arc::new(sandbox_scenario::IdleClient));
    let options = openai_agents_sdk::RunOptions::new(());
    if streamed {
        let mut run = SandboxRunner::run_streamed_with_options(&agent, "go", &config, options)
            .await
            .unwrap();
        {
            let mut events = std::pin::pin!(run.streaming.stream_events());
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        run.finish().await.unwrap();
    } else {
        let result = SandboxRunner::run_with_options(&agent, "go", &config, options)
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
    }
}

fn scenario(name: &str) {
    // The tracing-enabled scenarios exercise every span the run opens.
    openai_agents_sdk::tracing::set_tracing_disabled(!name.ends_with("_traced"));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let model = Arc::new(FakeModel::new());
        script(&model);
        let agent = agents(&model);
        match name {
            #[cfg(feature = "sandbox")]
            "sandbox" | "sandbox_traced" => sandbox_scenario(false).await,
            #[cfg(feature = "sandbox")]
            "sandbox_streamed" => sandbox_scenario(true).await,
            "run" | "run_traced" => {
                let result = Runner::run(&agent, "go").await.unwrap();
                assert_eq!(result.final_output_text().as_deref(), Some("done"));
            }
            "streamed" => {
                let mut stream = Runner::run_streamed(&agent, "go");
                {
                    let mut events = std::pin::pin!(stream.stream_events());
                    while let Some(event) = events.next().await {
                        event.unwrap();
                    }
                }
                assert_eq!(stream.final_output_text().as_deref(), Some("done"));
            }
            other => panic!("unknown scenario {other}"),
        }
    });
}

fn on_stack(kib: usize, name: &'static str) {
    std::thread::Builder::new()
        .stack_size(kib * 1024)
        .spawn(move || scenario(name))
        .unwrap()
        .join()
        .unwrap();
}

const BUDGET_KIB: usize = 1024;

#[test]
fn a_run_with_a_tool_call_and_a_handoff_fits_in_one_mebibyte() {
    on_stack(BUDGET_KIB, "run");
}

#[test]
fn a_traced_run_fits_in_one_mebibyte() {
    on_stack(BUDGET_KIB, "run_traced");
}

#[cfg(feature = "sandbox")]
#[test]
fn a_sandbox_run_fits_in_one_mebibyte() {
    on_stack(BUDGET_KIB, "sandbox");
}

#[cfg(feature = "sandbox")]
#[test]
fn a_traced_sandbox_run_fits_in_one_mebibyte() {
    on_stack(BUDGET_KIB, "sandbox_traced");
}

#[cfg(feature = "sandbox")]
#[test]
fn a_streamed_sandbox_run_fits_in_one_mebibyte() {
    on_stack(BUDGET_KIB, "sandbox_streamed");
}

#[test]
fn a_streamed_run_with_a_tool_call_and_a_handoff_fits_in_one_mebibyte() {
    on_stack(BUDGET_KIB, "streamed");
}

/// Child entry point of `measure_minimum_stack`.
#[test]
#[ignore = "re-executed by measure_minimum_stack"]
fn stack_probe_child() {
    let (Ok(kib), Ok(name)) = (
        std::env::var("STACK_PROBE_KIB"),
        std::env::var("STACK_PROBE_SCENARIO"),
    ) else {
        return;
    };
    let name: &'static str = Box::leak(name.into_boxed_str());
    on_stack(kib.parse().unwrap(), name);
}

#[test]
#[ignore = "diagnostic: prints the smallest stack each scenario runs on"]
fn measure_minimum_stack() {
    let exe = std::env::current_exe().unwrap();
    let mut scenarios = vec!["run", "run_traced", "streamed"];
    if cfg!(feature = "sandbox") {
        scenarios.extend(["sandbox", "sandbox_traced", "sandbox_streamed"]);
    }
    for name in scenarios {
        let mut minimum = None;
        for kib in [
            4096usize, 2048, 1536, 1280, 1024, 896, 768, 640, 512, 448, 384, 320, 256, 192, 128,
        ] {
            let ok = std::process::Command::new(&exe)
                .args([
                    "--ignored",
                    "--exact",
                    "stack_probe_child",
                    "--test-threads=1",
                ])
                .env("STACK_PROBE_KIB", kib.to_string())
                .env("STACK_PROBE_SCENARIO", name)
                .output()
                .unwrap()
                .status
                .success();
            if ok {
                minimum = Some(kib);
            } else {
                break;
            }
        }
        println!("{name}: runs on a {minimum:?} KiB stack");
    }
}
