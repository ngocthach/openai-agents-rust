//! The released-surface half of `tests/test_released_api_contract.py`'s
//! `WebSearchToolImageSettings` cases (`test_web_search_image_settings_policy_freezes_fields`
//! and `test_new_top_level_typed_dict_requires_complete_policy`).
//!
//! Python freezes the type as a top-level export whose two fields are both optional:
//! `max_results: int` and `caption: bool`. The validator's drift detection is repository tooling;
//! what carries over is the shape itself, pinned here from the crate root.

use openai_agents_sdk::{WebSearchTool, WebSearchToolImageSettings};
use serde_json::json;

#[test]
fn image_settings_are_a_root_export_with_two_optional_fields() {
    // Both fields are optional (`total=False`): an empty object is a valid value.
    let empty: WebSearchToolImageSettings = serde_json::from_value(json!({})).unwrap();
    assert_eq!(empty, WebSearchToolImageSettings::default());
    assert_eq!(empty.to_value(), json!({}));

    // Exactly `max_results` (an integer) and `caption` (a boolean), by struct literal so a
    // renamed, removed or added field breaks the build.
    let full = WebSearchToolImageSettings {
        max_results: Some(2),
        caption: Some(true),
    };
    assert_eq!(full.to_value(), json!({"max_results": 2, "caption": true}));
    let parsed: WebSearchToolImageSettings =
        serde_json::from_value(json!({"max_results": 2, "caption": true})).unwrap();
    assert_eq!(parsed, full);

    // Changed field types are rejected.
    assert!(
        serde_json::from_value::<WebSearchToolImageSettings>(json!({"max_results": "2"})).is_err()
    );
    assert!(
        serde_json::from_value::<WebSearchToolImageSettings>(json!({"caption": "yes"})).is_err()
    );
}

#[test]
fn image_settings_reach_the_web_search_tool_payload() {
    let settings = WebSearchToolImageSettings {
        max_results: Some(3),
        caption: None,
    };
    let tool = WebSearchTool {
        image_settings: Some(settings.to_value()),
        ..Default::default()
    };
    assert_eq!(tool.image_settings, Some(json!({"max_results": 3})));
}
