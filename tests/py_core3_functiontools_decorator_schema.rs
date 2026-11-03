//! Ports of the open rows of `tests/test_function_schema.py` and
//! `tests/test_function_tool_decorator.py` that have a Rust counterpart: Google docstring parsing,
//! schema constraints emitted from the argument type, optional parameters, decorator options and
//! the three invocation shapes.
//!
//! The `#[function_tool]` macro reads doc comments; the other options are builder calls.
#![cfg(feature = "macros")]

use std::time::Duration;

use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{AgentsError, Result, function_tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

/// Get the weather for a city.
///
/// Args:
///     city: The city to get weather for.
///     units: Temperature units to use.
#[function_tool]
fn weather_missing_blank_line(city: String, units: String) -> Result<String> {
    Ok(format!("{city} {units}"))
}

/// Get the weather for a city.
///
/// Args:
///     city: The city to get weather for.
///     units: Temperature units to use.
#[function_tool]
fn weather_blank_line(city: String, units: String) -> Result<String> {
    Ok(format!("{city} {units}"))
}

/// Args:
///     city: The city to get
///         weather for.
///     units: Temperature units to use.
#[function_tool]
fn weather_without_summary(city: String, units: String) -> Result<String> {
    Ok(format!("{city} {units}"))
}

/// Get the weather for a city.
///
/// Note:
///     Results are cached.
/// Args:
///     city: The city to get weather for.
///     units: Temperature units to use.
#[function_tool]
fn weather_after_note(city: String, units: String) -> Result<String> {
    Ok(format!("{city} {units}"))
}

/// Get the weather for a city.
///
/// Note:
///     Results are cached.
///
/// Args:
///     city: The city to get weather for.
///     units: Temperature units to use.
#[function_tool]
fn weather_after_note_blank_line(city: String, units: String) -> Result<String> {
    Ok(format!("{city} {units}"))
}

#[function_tool]
fn add_annotated_without_docstring(
    #[arg(desc = "First number to add")] a: i64,
    #[arg(desc = "Second number to add")] b: i64,
) -> Result<i64> {
    Ok(a + b)
}

/// Adds two integers.
///
/// Args:
///     a: Docstring provided description.
#[function_tool]
fn add_with_arg_desc(#[arg(desc = "Attribute description for a")] a: i64, b: i64) -> Result<i64> {
    Ok(a + b)
}

#[function_tool(name = "renamed_lookup")]
async fn lookup_without_docstring(id: String) -> Result<String> {
    Ok(id)
}

#[function_tool(strict = false)]
async fn non_strict_echo(value: String) -> Result<String> {
    Ok(value)
}

#[function_tool]
async fn all_optional(city: Option<String>, days: Option<u32>) -> Result<String> {
    Ok(format!("{:?} {:?}", city, days))
}

/// Ping with no arguments.
#[function_tool]
async fn ping() -> Result<String> {
    Ok("pong".to_string())
}

#[function_tool]
async fn echo_args(value: String) -> Result<String> {
    Ok(value)
}

#[function_tool]
async fn echo_with_context(ctx: ToolContext<()>, value: String) -> Result<String> {
    Ok(format!("{}:{value}", ctx.tool_name))
}

async fn invoke(tool: &FunctionTool<()>, args: &str) -> Result<String> {
    let ctx = ToolContext::direct(RunContext::new(()), "test", &tool.name, args);
    tool.invoke(ctx, args.to_string())
        .await
        .map(|out| out.to_model_string())
}

fn props(tool: &FunctionTool<()>, name: &str) -> Value {
    tool.params_json_schema["properties"][name].clone()
}

// ---------------------------------------------------------------------------
// Google docstrings
// ---------------------------------------------------------------------------

#[test]
fn google_docstring_missing_blank_line_before_args() {
    let tool = weather_missing_blank_line_tool::<()>();
    assert_eq!(
        props(&tool, "city")["description"],
        "The city to get weather for."
    );
    assert_eq!(
        props(&tool, "units")["description"],
        "Temperature units to use."
    );
    assert_eq!(tool.description, "Get the weather for a city.");
    assert!(!tool.description.contains("Args:"));
}

#[test]
fn google_docstring_missing_blank_line_matches_blank_line_form() {
    let fixed = weather_missing_blank_line_tool::<()>();
    let control = weather_blank_line_tool::<()>();
    assert_eq!(fixed.description, control.description);
    assert_eq!(
        fixed.params_json_schema["properties"],
        control.params_json_schema["properties"]
    );
}

#[test]
fn google_docstring_missing_blank_line_function_tool() {
    let tool = weather_missing_blank_line_tool::<()>();
    assert_eq!(
        props(&tool, "city")["description"],
        "The city to get weather for."
    );
    assert_eq!(
        props(&tool, "units")["description"],
        "Temperature units to use."
    );
    assert!(!tool.description.contains("Args:"));
}

#[test]
fn google_docstring_without_summary_keeps_wrapped_argument_lines() {
    let tool = weather_without_summary_tool::<()>();
    assert_eq!(
        props(&tool, "city")["description"],
        "The city to get\nweather for."
    );
    assert_eq!(
        props(&tool, "units")["description"],
        "Temperature units to use."
    );
    assert_eq!(tool.description, "");
}

#[test]
fn google_docstring_after_section_body_matches_blank_line_form() {
    let fixed = weather_after_note_tool::<()>();
    let control = weather_after_note_blank_line_tool::<()>();
    assert_eq!(
        fixed.params_json_schema["properties"],
        control.params_json_schema["properties"]
    );
    assert_eq!(fixed.description, control.description);
}

#[test]
fn function_uses_annotated_descriptions_without_docstring() {
    let tool = add_annotated_without_docstring_tool::<()>();
    assert_eq!(props(&tool, "a")["description"], "First number to add");
    assert_eq!(props(&tool, "b")["description"], "Second number to add");
}

#[test]
fn google_docstring_after_another_section_body_still_yields_argument_descriptions() {
    let tool = weather_after_note_tool::<()>();
    assert_eq!(
        props(&tool, "city")["description"],
        "The city to get weather for."
    );
    assert_eq!(
        props(&tool, "units")["description"],
        "Temperature units to use."
    );
    assert!(!tool.description.contains("Args:"));
}

/// Python gives the docstring precedence over `Annotated` metadata. The Rust macro gives an
/// explicit `#[arg(desc = ...)]` precedence over the doc comment, so this port asserts the Rust
/// rule. The Python order is an open design choice, see the report.
#[test]
fn explicit_arg_description_takes_precedence_over_the_docstring_in_rust() {
    let tool = add_with_arg_desc_tool::<()>();
    assert_eq!(
        props(&tool, "a")["description"],
        "Attribute description for a"
    );
    assert_eq!(tool.description, "Adds two integers.");
}

// ---------------------------------------------------------------------------
// Names, strictness, optional parameters
// ---------------------------------------------------------------------------

#[test]
fn name_override_without_docstring() {
    let tool = lookup_without_docstring_tool::<()>();
    assert_eq!(tool.name, "renamed_lookup");
    assert_eq!(tool.description, "");
}

#[tokio::test]
async fn non_strict_mode_function_keeps_the_plain_schema() {
    let tool = non_strict_echo_tool::<()>();
    assert!(!tool.strict_json_schema);
    assert!(
        tool.params_json_schema
            .get("additionalProperties")
            .is_none()
    );
    assert_eq!(
        invoke(&tool, r#"{"value": "x", "extra": 1}"#)
            .await
            .unwrap(),
        "x"
    );
}

#[tokio::test]
async fn all_optional_params_function_accepts_empty_arguments() {
    let tool = all_optional_tool::<()>();
    // Strict mode lists every property as required; a missing optional value is null.
    assert_eq!(tool.params_json_schema["required"], json!(["city", "days"]));
    assert_eq!(invoke(&tool, "{}").await.unwrap(), "None None");
    assert_eq!(
        invoke(&tool, r#"{"city": "Oslo", "days": 2}"#)
            .await
            .unwrap(),
        r#"Some("Oslo") Some(2)"#
    );
}

// ---------------------------------------------------------------------------
// Invocation shapes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_no_context_no_args_invocation() {
    let tool = ping_tool::<()>();
    assert_eq!(tool.description, "Ping with no arguments.");
    assert_eq!(invoke(&tool, "{}").await.unwrap(), "pong");
}

#[tokio::test]
async fn no_context_with_args_invocation() {
    let tool = echo_args_tool::<()>();
    assert_eq!(invoke(&tool, r#"{"value": "hi"}"#).await.unwrap(), "hi");
}

#[tokio::test]
async fn with_context_invocation_receives_the_tool_context() {
    let tool = echo_with_context_tool();
    assert_eq!(
        invoke(&tool, r#"{"value": "hi"}"#).await.unwrap(),
        "echo_with_context:hi"
    );
    let err = invoke(&tool, "[]").await.unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------------
// Decorator-style options set through the builder
// ---------------------------------------------------------------------------

#[test]
fn decorator_timeout_configuration_is_applied() {
    let tool = ping_tool::<()>().with_timeout(Duration::from_secs(3));
    assert_eq!(tool.timeout, Some(Duration::from_secs(3)));
}

#[test]
fn function_tool_defer_loading() {
    let tool = ping_tool::<()>().with_defer_loading(true);
    assert!(tool.defer_loading);
    assert!(!ping_tool::<()>().defer_loading);
}

// ---------------------------------------------------------------------------
// Schema constraints taken from the argument types
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct RequiredArgs {
    #[schemars(length(min = 2, max = 5))]
    code: String,
    #[schemars(range(min = 0, max = 10))]
    score: i64,
}

#[test]
fn function_with_field_required_constraints() {
    let tool = FunctionTool::<()>::new("required", "d", |_ctx, _args: RequiredArgs| async {
        Ok("ok")
    });
    assert_eq!(
        tool.params_json_schema["required"],
        json!(["code", "score"])
    );
    assert_eq!(props(&tool, "score")["minimum"], json!(0));
    assert_eq!(props(&tool, "score")["maximum"], json!(10));
}

#[test]
fn function_with_field_string_constraints() {
    let tool = FunctionTool::<()>::new("required", "d", |_ctx, _args: RequiredArgs| async {
        Ok("ok")
    });
    assert_eq!(props(&tool, "code")["minLength"], json!(2));
    assert_eq!(props(&tool, "code")["maxLength"], json!(5));
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct OptionalArgs {
    #[serde(default = "default_limit")]
    #[schemars(range(min = 1))]
    limit: i64,
    note: Option<String>,
}

fn default_limit() -> i64 {
    20
}

#[test]
fn function_with_field_optional_with_default() {
    let tool = FunctionTool::<()>::new("optional", "d", |_ctx, _args: OptionalArgs| async {
        Ok("ok")
    });
    // Strict mode lists every property as required, including the `Option` one.
    assert_eq!(
        tool.params_json_schema["required"],
        json!(["limit", "note"])
    );
    assert_eq!(props(&tool, "limit")["minimum"], json!(1));
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
enum Mode {
    Fast,
    Slow,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct EnumArgs {
    mode: Mode,
}

#[test]
fn enum_and_literal_function_emits_the_enum_values() {
    let tool = FunctionTool::<()>::new("enumerated", "d", |_ctx, _args: EnumArgs| async {
        Ok("ok")
    });
    let mode = serde_json::to_string(&tool.params_json_schema).unwrap();
    assert!(mode.contains("Fast") && mode.contains("Slow"), "{mode}");
    assert_eq!(
        tool.params_json_schema["additionalProperties"],
        json!(false)
    );
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct DescribedArgs {
    #[schemars(description = "Account identifier.")]
    account: String,
}

#[test]
fn function_with_field_description_only() {
    let tool = FunctionTool::<()>::new("described", "d", |_ctx, _args: DescribedArgs| async {
        Ok("ok")
    });
    assert_eq!(
        props(&tool, "account")["description"],
        "Account identifier."
    );
}
