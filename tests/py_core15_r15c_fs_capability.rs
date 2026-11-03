//! Port of `test_configure_tools_can_customize_approvals_after_clone`
//! (`tests/sandbox/capabilities/test_filesystem_capability.py`): a configurator wrapped around the
//! filesystem capability can give `view_image` a per-call approval rule while the other tool and
//! the rest of the capability stay as they were.
#![cfg(feature = "sandbox")]

use std::sync::Arc;

use openai_agents_sdk::RunContext;
use openai_agents_sdk::sandbox::SandboxClient;
use openai_agents_sdk::sandbox::capabilities::{
    Capability, CapabilityBinding, ConfiguredCapability, Filesystem,
};
use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::tool::Tool;

#[tokio::test]
async fn a_configurator_can_give_view_image_a_path_based_approval_rule() {
    let client = UnixLocalSandboxClient::new();
    let session = client.create(None).await.unwrap();
    session.start().await.unwrap();
    let binding = CapabilityBinding::new(session.clone());

    let configured = ConfiguredCapability::<()>::new(Arc::new(Filesystem::new()), |tools| {
        tools
            .into_iter()
            .map(|tool| match tool {
                Tool::Function(function) if function.name == "view_image" => {
                    Tool::Function(Arc::new((*function).clone().with_needs_approval_fn(
                        |_ctx, args, _call_id| async move {
                            args["path"]
                                .as_str()
                                .is_some_and(|path| path.starts_with("sensitive/"))
                        },
                    )))
                }
                other => other,
            })
            .collect()
    });
    let capability = configured.clone();

    let tools = Capability::<()>::tools(&capability, &binding).unwrap();
    assert_eq!(
        tools.iter().map(Tool::name).collect::<Vec<_>>(),
        ["view_image", "apply_patch"]
    );
    let Tool::Function(view_image) = &tools[0] else {
        panic!("view_image is a function tool");
    };
    let ctx = RunContext::new(());
    assert!(
        view_image
            .evaluate_needs_approval(&ctx, r#"{"path": "sensitive/plan.png"}"#, "c1")
            .await
    );
    assert!(
        !view_image
            .evaluate_needs_approval(&ctx, r#"{"path": "public/plan.png"}"#, "c2")
            .await
    );

    client.delete(session).await.unwrap();
}
