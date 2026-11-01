//! Ports of `tests/test_run_state.py` cases that the earlier audits classed as Python-only
//! although the behaviour exists in the Rust API.

use openai_agents_sdk::run_state::{
    CURRENT_SCHEMA_VERSION, SCHEMA_VERSION_SUMMARIES, SUPPORTED_SCHEMA_VERSIONS,
};

/// Python: `test_supported_schema_versions_have_non_empty_summaries`. Every supported schema
/// version has a one-line historical summary, in the same order, and the current version is one
/// of them.
#[test]
fn supported_schema_versions_have_non_empty_summaries() {
    let summarized: Vec<&str> = SCHEMA_VERSION_SUMMARIES.iter().map(|(v, _)| *v).collect();
    assert_eq!(summarized, SUPPORTED_SCHEMA_VERSIONS.to_vec());
    assert!(summarized.contains(&CURRENT_SCHEMA_VERSION));
    for (version, summary) in SCHEMA_VERSION_SUMMARIES {
        assert!(!summary.trim().is_empty(), "{version}");
        assert!(!summary.contains('\n'), "{version}: a summary is one line");
    }
}
