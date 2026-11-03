//! Port of `test_debug.py`: the model-data and tool-data redaction flags default to "redact" and
//! follow `OPENAI_AGENTS_DONT_LOG_MODEL_DATA` / `OPENAI_AGENTS_DONT_LOG_TOOL_DATA`
//! (`1` / `true` redact, `0` / `false` log). Python patches `os.environ` per test; Rust edits the
//! process environment, so every case runs from one test in this single-test binary.

use openai_agents_sdk::debug::{
    dont_log_model_data, dont_log_tool_data, reset_dont_log_model_data, reset_dont_log_tool_data,
};

const MODEL: &str = "OPENAI_AGENTS_DONT_LOG_MODEL_DATA";
const TOOL: &str = "OPENAI_AGENTS_DONT_LOG_TOOL_DATA";

fn set_env(key: &str, value: Option<&str>) {
    // SAFETY: this binary holds one test, so nothing else touches the environment concurrently.
    unsafe {
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
}

#[test]
fn dont_log_flags_follow_the_environment() {
    reset_dont_log_model_data();
    reset_dont_log_tool_data();

    // `test_dont_log_model_data` / `test_dont_log_tool_data`: unset means redact.
    set_env(MODEL, None);
    set_env(TOOL, None);
    assert!(dont_log_model_data());
    assert!(dont_log_tool_data());

    // `test_dont_log_model_data_{0,1,true,false}`.
    for (value, expected) in [("0", false), ("1", true), ("true", true), ("false", false)] {
        set_env(MODEL, Some(value));
        assert_eq!(dont_log_model_data(), expected, "{MODEL}={value}");
        // The tool flag is independent of the model variable.
        assert!(
            dont_log_tool_data(),
            "{MODEL}={value} leaked into tool data"
        );
    }
    set_env(MODEL, None);

    // `test_dont_log_tool_data_{0,1,true,false}`.
    for (value, expected) in [("0", false), ("1", true), ("true", true), ("false", false)] {
        set_env(TOOL, Some(value));
        assert_eq!(dont_log_tool_data(), expected, "{TOOL}={value}");
        assert!(
            dont_log_model_data(),
            "{TOOL}={value} leaked into model data"
        );
    }
    set_env(TOOL, None);
}
