//! Ports of the remaining cases of `tests/test_function_schema.py` and
//! `tests/test_function_tool_decorator.py`: schema shape for no-argument, nested and documented
//! tools, multiple field constraints, docstring output, and an override-named invocation.
//!
//! Python's `function_schema(...)` helper returns a pydantic model; the Rust analog is the
//! `FunctionTool` that `#[function_tool]` / `FunctionTool::new` build, whose `params_json_schema`
//! and `invoke` are asserted. The `model_extra` / `model_fields_set` shadowing cases are
//! pydantic-internal and not ported.
//!
//! Python's schemas carry pydantic `title` annotations (`<func>_args`, `City`); Rust schemas
//! deliberately omit them, so the `title` assertions are not ported.
#![cfg(feature = "macros")]

use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Result, function_tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

async fn invoke(tool: &FunctionTool<()>, args: &str) -> Result<String> {
    let ctx = ToolContext::direct(RunContext::new(()), "test", &tool.name, args);
    tool.invoke(ctx, args.to_string())
        .await
        .map(|out| out.to_model_string())
}

fn props(tool: &FunctionTool<()>, name: &str) -> Value {
    tool.params_json_schema["properties"][name].clone()
}

fn assert_no_properties(tool: &FunctionTool<()>) {
    let properties = &tool.params_json_schema["properties"];
    assert!(
        properties.is_null() || properties == &json!({}),
        "{properties}"
    );
}

// ---- no-argument functions ----

/// This function has no args.
#[function_tool]
fn no_args_function() -> Result<String> {
    Ok("ok".to_string())
}

#[tokio::test]
async fn no_args_function_schema_and_call() {
    let tool = no_args_function_tool::<()>();
    assert_eq!(tool.description, "This function has no args.");
    assert_no_properties(&tool);
    assert_eq!(invoke(&tool, "{}").await.unwrap(), "ok");
}

#[function_tool]
fn no_args_function_with_context(_ctx: ToolContext<()>) -> Result<String> {
    Ok("ok".to_string())
}

#[tokio::test]
async fn no_args_function_with_context_takes_the_context() {
    let tool = no_args_function_with_context_tool();
    // The context parameter is not part of the schema.
    assert_no_properties(&tool);
    assert_eq!(invoke(&tool, "{}").await.unwrap(), "ok");
}

// ---- simple function with a default ----

/// Args:
///     a: The first argument
///     b: The second argument
///
/// Returns:
///     The sum of a and b
#[function_tool]
fn simple_function(a: i64, b: Option<i64>) -> Result<i64> {
    Ok(a + b.unwrap_or(5))
}

#[tokio::test]
async fn simple_function_schema_defaults_and_validation() {
    let tool = simple_function_tool::<()>();
    assert_eq!(
        props(&tool, "a")["description"],
        json!("The first argument")
    );
    assert_eq!(
        props(&tool, "b")["description"],
        json!("The second argument")
    );

    assert_eq!(invoke(&tool, r#"{"a": 3}"#).await.unwrap(), "8");
    assert_eq!(invoke(&tool, r#"{"a": 3, "b": 10}"#).await.unwrap(), "13");
    // `a` must be an integer.
    let err = invoke(&tool, r#"{"a": "not an integer"}"#)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("Invalid JSON input for tool"),
        "{err}"
    );
}

// ---- nested data ----

#[derive(Deserialize, JsonSchema)]
struct Foo {
    a: i64,
    b: String,
}

#[derive(Deserialize, JsonSchema)]
struct InnerModel {
    a: i64,
    b: String,
}

#[derive(Deserialize, JsonSchema)]
struct OuterModel {
    inner: InnerModel,
    foo: Foo,
}

#[function_tool]
fn complex_args_function(model: OuterModel) -> Result<String> {
    Ok(format!(
        "{}, {}, {}, {}",
        model.inner.a, model.inner.b, model.foo.a, model.foo.b
    ))
}

#[tokio::test]
async fn nested_data_function() {
    let tool = complex_args_function_tool::<()>();
    let input = json!({"model": {"inner": {"a": 1, "b": "hello"}, "foo": {"a": 2, "b": "world"}}});
    assert_eq!(
        invoke(&tool, &input.to_string()).await.unwrap(),
        "1, hello, 2, world"
    );
}

/// This function takes a model and a flag, and returns a string.
///
/// Args:
///     model: A model with an inner and foo field
///     some_flag: An optional flag with a default of 0
///
/// Returns:
///     A string with the values of the model and flag
#[function_tool]
fn complex_args_and_docs_function(model: OuterModel, some_flag: Option<i64>) -> Result<String> {
    Ok(format!(
        "{}, {}, {}, {}, {}",
        model.inner.a,
        model.inner.b,
        model.foo.a,
        model.foo.b,
        some_flag.unwrap_or(0)
    ))
}

#[tokio::test]
async fn complex_args_and_docs_function_schema_and_call() {
    let tool = complex_args_and_docs_function_tool::<()>();
    assert_eq!(
        props(&tool, "model")["description"],
        json!("A model with an inner and foo field")
    );
    assert_eq!(
        props(&tool, "some_flag")["description"],
        json!("An optional flag with a default of 0")
    );

    let model = json!({"inner": {"a": 1, "b": "hello"}, "foo": {"a": 2, "b": "world"}});
    assert_eq!(
        invoke(&tool, &json!({"model": model}).to_string())
            .await
            .unwrap(),
        "1, hello, 2, world, 0"
    );
    let bad = json!({"model": model, "some_flag": "not an int"});
    assert!(invoke(&tool, &bad.to_string()).await.is_err());
}

// ---- multiple field constraints ----

fn default_name() -> String {
    "Unknown".to_string()
}

fn default_factor() -> f64 {
    1.0
}

#[derive(Deserialize, JsonSchema)]
struct MultipleConstraints {
    /// Score from 0 to 100
    #[schemars(range(min = 0, max = 100))]
    score: i64,
    #[serde(default = "default_name")]
    #[schemars(length(min = 1, max = 50))]
    name: String,
    /// Positive multiplier
    #[serde(default = "default_factor")]
    #[schemars(extend("exclusiveMinimum" = 0.0))]
    factor: f64,
}

fn constrained_tool() -> FunctionTool<()> {
    FunctionTool::<()>::try_new_with_strictness(
        "func_with_multiple_field_constraints",
        "d",
        false,
        |_ctx, args: MultipleConstraints| async move {
            Ok(format!(
                "{} scored {:?}",
                args.name,
                args.score as f64 * args.factor
            ))
        },
    )
    .unwrap()
}

#[tokio::test]
async fn function_with_field_multiple_constraints() {
    let tool = constrained_tool();

    let score = props(&tool, "score");
    assert_eq!(score["type"], json!("integer"));
    assert_eq!(score["minimum"], json!(0));
    assert_eq!(score["maximum"], json!(100));
    assert_eq!(score["description"], json!("Score from 0 to 100"));

    let name = props(&tool, "name");
    assert_eq!(name["type"], json!("string"));
    assert_eq!(name["minLength"], json!(1));
    assert_eq!(name["maxLength"], json!(50));
    assert_eq!(name["default"], json!("Unknown"));

    let factor = props(&tool, "factor");
    assert_eq!(factor["type"], json!("number"));
    assert_eq!(factor["exclusiveMinimum"], json!(0.0));
    assert_eq!(factor["default"], json!(1.0));
    assert_eq!(factor["description"], json!("Positive multiplier"));

    assert_eq!(
        invoke(&tool, r#"{"score": 85}"#).await.unwrap(),
        "Unknown scored 85.0"
    );
    assert_eq!(
        invoke(&tool, r#"{"score": 90, "name": "Alice", "factor": 1.5}"#)
            .await
            .unwrap(),
        "Alice scored 135.0"
    );

    // Out-of-range values are rejected as invalid tool input, as pydantic's ValidationError is.
    for bad in [
        r#"{"score": 150}"#,
        r#"{"score": 50, "name": ""}"#,
        r#"{"score": 50, "factor": 0.0}"#,
    ] {
        let err = invoke(&tool, bad).await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid JSON input for tool"),
            "{bad}: {err}"
        );
    }
}

// ---- docstring output ----

/// Get the weather for a city.
///
/// Args:
///     city: The city to get weather for.
///     units: Temperature units to use.
#[function_tool]
fn blank_line_google_function(city: String, units: String) -> Result<String> {
    Ok(format!("{city} {units}"))
}

#[test]
fn google_docstring_blank_line_form_is_unchanged() {
    let tool = blank_line_google_function_tool::<()>();
    assert_eq!(tool.description, "Get the weather for a city.");
    assert_eq!(
        props(&tool, "city")["description"],
        json!("The city to get weather for.")
    );
    assert_eq!(
        props(&tool, "units")["description"],
        json!("Temperature units to use.")
    );
}

// ---- decorator ----

#[function_tool(name = "my_custom_tool", description = "custom desc")]
fn sync_no_context_override() -> Result<String> {
    Ok("override_result".to_string())
}

#[tokio::test]
async fn sync_no_context_override_invocation() {
    let tool = sync_no_context_override_tool::<()>();
    assert_eq!(tool.name, "my_custom_tool");
    assert_eq!(tool.description, "custom desc");
    // Python invokes with an empty argument string; a no-argument tool treats it as `{}`.
    assert_eq!(invoke(&tool, "").await.unwrap(), "override_result");
}

/// Get the weather for a given city.
///
/// Args:
///     city: The city to get the weather for.
#[function_tool]
fn get_weather(city: String) -> Result<String> {
    Ok(format!("The weather in {city} is sunny."))
}

#[test]
fn extract_descriptions_from_docstring() {
    let tool = get_weather_tool::<()>();
    assert_eq!(tool.description, "Get the weather for a given city.");
    assert_eq!(
        tool.params_json_schema,
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "description": "The city to get the weather for.",
                    "type": "string"
                }
            },
            "required": ["city"],
            "additionalProperties": false
        })
    );
}

// ---- constraint enforcement beyond numbers ----

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Nested {
    #[schemars(regex(pattern = "^[A-Za-z]+$"))]
    name: String,
    #[schemars(length(min = 1, max = 2))]
    tags: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct NestedArgs {
    inner: Nested,
    maybe: Option<Vec<Nested>>,
}

#[tokio::test]
async fn pattern_and_item_bounds_are_enforced_through_nesting() {
    let tool = FunctionTool::<()>::new("nested", "d", |_ctx, _args: NestedArgs| async { Ok("ok") });
    let ok = r#"{"inner": {"name": "Alice", "tags": ["a"]}, "maybe": null}"#;
    assert_eq!(invoke(&tool, ok).await.unwrap(), "ok");

    for bad in [
        // pattern: digits are not allowed
        r#"{"inner": {"name": "Alice123", "tags": ["a"]}, "maybe": null}"#,
        // too many items
        r#"{"inner": {"name": "Alice", "tags": ["a", "b", "c"]}, "maybe": null}"#,
        // a violation inside an optional list of nested models
        r#"{"inner": {"name": "Alice", "tags": ["a"]}, "maybe": [{"name": "x1", "tags": ["a"]}]}"#,
    ] {
        let err = invoke(&tool, bad).await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid JSON input for tool"),
            "{bad}: {err}"
        );
    }
}
