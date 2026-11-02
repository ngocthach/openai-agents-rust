//! Port of `test_functions_are_ok` of `tests/test_doc_parsing.py`: a function annotated with
//! `#[function_tool]` keeps working as written, whichever docstring style it uses and whether it
//! is a free function, a method taking `&self` or an associated function.
#![cfg(feature = "macros")]

use openai_agents_sdk::function_tool;

/// This is func_foo.
///
/// Args:
///     a: The first argument.
///     b: The second argument.
///
/// Returns:
///     A result
#[function_tool]
fn func_foo_google(a: i64, b: f64) -> String {
    let _ = (a, b);
    "ok".to_string()
}

/// This is func_foo.
///
/// Parameters
/// ----------
/// a: int
///     The first argument.
/// b: float
///     The second argument.
///
/// Returns
/// -------
/// str
///     A result
#[function_tool]
fn func_foo_numpy(a: i64, b: f64) -> String {
    let _ = (a, b);
    "ok".to_string()
}

/// This is func_foo.
///
/// :param a: The first argument.
/// :param b: The second argument.
/// :return: A result
#[function_tool]
fn func_foo_sphinx(a: i64, b: f64) -> String {
    let _ = (a, b);
    "ok".to_string()
}

#[derive(Clone, Default)]
struct Bar;

impl Bar {
    /// This is func_bar.
    ///
    /// Args:
    ///     a: The first argument.
    ///     b: The second argument.
    #[function_tool]
    fn func_bar(&self, a: i64, b: f64) -> String {
        let _ = (a, b);
        "ok".to_string()
    }

    /// This is func_baz.
    ///
    /// Args:
    ///     a: The first argument.
    ///     b: The second argument.
    #[function_tool(associated)]
    fn func_baz(a: i64, b: f64) -> String {
        let _ = (a, b);
        "ok".to_string()
    }
}

/// Python: `test_functions_are_ok`.
#[test]
fn annotated_functions_are_still_callable_directly() {
    assert_eq!(func_foo_google(1, 2.0), "ok");
    assert_eq!(func_foo_numpy(1, 2.0), "ok");
    assert_eq!(func_foo_sphinx(1, 2.0), "ok");
    assert_eq!(Bar.func_bar(1, 2.0), "ok");
    assert_eq!(Bar::func_baz(1, 2.0), "ok");

    // The generated constructors exist alongside the original functions.
    assert_eq!(func_foo_google_tool::<()>().name, "func_foo_google");
    assert_eq!(Bar.func_bar_tool::<()>().name, "func_bar");
    assert_eq!(Bar::func_baz_tool::<()>().name, "func_baz");
}
