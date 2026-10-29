//! `#[function_tool]` attribute macro.
#![cfg(feature = "macros")]

use std::sync::Arc;

use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, RunOptions, Runner, function_tool};

/// Get the weather for a city.
///
/// # Arguments
/// * `city` - The city to look up.
/// * `days` - How many days to forecast.
#[function_tool]
async fn get_weather(city: String, days: Option<u32>) -> Result<String, AgentsError> {
    Ok(format!("{city}: sunny for {} days", days.unwrap_or(1)))
}

struct User {
    name: String,
}

/// Greet the current user.
#[function_tool(name = "greet")]
fn greet_user(ctx: ToolContext<User>, #[arg(desc = "Greeting word")] word: String) -> String {
    format!("{word}, {}!", ctx.context().name)
}

#[tokio::test]
async fn generates_tools_from_functions() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let tool = get_weather_tool::<()>();
    assert_eq!(tool.name, "get_weather");
    assert_eq!(tool.description, "Get the weather for a city.");
    let props = &tool.params_json_schema["properties"];
    assert_eq!(props["city"]["description"], "The city to look up.");
    assert_eq!(props["days"]["description"], "How many days to forecast.");
    assert_eq!(
        tool.params_json_schema["required"],
        serde_json::json!(["city", "days"])
    );
    // The function itself still works.
    assert_eq!(
        get_weather("Oslo".into(), None).await.unwrap(),
        "Oslo: sunny for 1 days"
    );

    let greet = greet_user_tool();
    assert_eq!(greet.name, "greet");
    assert_eq!(
        greet.params_json_schema["properties"]["word"]["description"],
        "Greeting word"
    );

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![
        function_call("get_weather", r#"{"city":"Hanoi","days":3}"#),
        function_call("greet", r#"{"word":"Hello"}"#),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<User>::new("A")
        .with_tool(get_weather_tool::<User>())
        .with_tool(greet)
        .with_model_instance(model);
    let r = Runner::run_with_options(
        &agent,
        "hi",
        RunOptions::new(User {
            name: "Thach".into(),
        }),
    )
    .await
    .unwrap();
    let outputs: Vec<_> = r
        .new_items
        .iter()
        .filter_map(|i| i.tool_output().cloned())
        .collect();
    assert_eq!(outputs[0], "Hanoi: sunny for 3 days");
    assert_eq!(outputs[1], "Hello, Thach!");
}
