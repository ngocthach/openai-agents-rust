//! Port of `test_variadic_param_descriptions_preserved` from `tests/test_function_schema.py`.
//!
//! Google and NumPy docstrings write variadic parameters with their stars (`*numbers:`,
//! `**kwargs:`). The description must still attach to the bare parameter name, as the Sphinx form
//! (`:param numbers:`) does. A Rust function has no `*args`, so the starred entries document
//! ordinary collection parameters here.

#![cfg(feature = "macros")]

use openai_agents_sdk::Result;
use openai_agents_sdk::function_tool;
use openai_agents_sdk::tool::FunctionTool;
use serde_json::json;

/// Add numbers to a base.
///
/// Args:
///     x: The base value.
///     *numbers: The numbers to add.
///     **kwargs: Extra options.
#[function_tool]
fn starred_args_google(x: i64, numbers: Vec<f64>, kwargs: Option<String>) -> Result<String> {
    Ok(format!("{x} {numbers:?} {kwargs:?}"))
}

/// Add numbers to a base.
///
/// Parameters
/// ----------
/// x : int
///     The base value.
/// *numbers : float
///     The numbers to add.
/// **kwargs : str
///     Extra options.
#[function_tool]
fn starred_args_numpy(x: i64, numbers: Vec<f64>, kwargs: Option<String>) -> Result<String> {
    Ok(format!("{x} {numbers:?} {kwargs:?}"))
}

/// Add numbers to a base.
///
/// :param x: The base value.
/// :param numbers: The numbers to add.
/// :param kwargs: Extra options.
#[function_tool]
fn starred_args_sphinx(x: i64, numbers: Vec<f64>, kwargs: Option<String>) -> Result<String> {
    Ok(format!("{x} {numbers:?} {kwargs:?}"))
}

fn assert_variadic_descriptions(tool: &FunctionTool<()>) {
    let properties = &tool.params_json_schema["properties"];
    assert_eq!(properties["x"]["description"], json!("The base value."));
    assert_eq!(
        properties["numbers"]["description"],
        json!("The numbers to add.")
    );
    assert_eq!(properties["kwargs"]["description"], json!("Extra options."));
    assert_eq!(tool.description, "Add numbers to a base.");
}

#[test]
fn google_variadic_descriptions_attach_to_the_bare_names() {
    assert_variadic_descriptions(&starred_args_google_tool::<()>());
}

#[test]
fn numpy_variadic_descriptions_attach_to_the_bare_names() {
    assert_variadic_descriptions(&starred_args_numpy_tool::<()>());
}

#[test]
fn sphinx_variadic_descriptions_attach_to_the_bare_names() {
    assert_variadic_descriptions(&starred_args_sphinx_tool::<()>());
}
