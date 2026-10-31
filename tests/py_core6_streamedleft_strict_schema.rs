//! `test_strict_schema.py` cases the earlier ports did not carry: singleton `allOf` conflicts,
//! circular and over-large `$ref` chains, nested `$id` resources, and non-object schemas.

use openai_agents_sdk::strict_schema::ensure_strict_json_schema;
use serde_json::{Value, json};

fn strict(schema: Value) -> Value {
    ensure_strict_json_schema(&schema).expect("schema converts")
}

fn strict_err(schema: Value) -> String {
    ensure_strict_json_schema(&schema)
        .expect_err("schema should be rejected")
        .to_string()
}

#[test]
fn non_object_schema_errors() {
    for schema in [json!([]), json!("x"), json!(1), Value::Null] {
        assert!(ensure_strict_json_schema(&schema).is_err(), "{schema}");
    }
}

#[test]
fn all_of_single_ref_entry_rejects_overlapping_parent_constraints() {
    let t = json!({"type": "object", "properties": {"inner": {"type": "string"}}});
    let message = strict_err(json!({
        "$defs": {"T": t},
        "type": "object",
        "properties": {"outer": {"type": "string"}},
        "allOf": [{"$ref": "#/$defs/T", "description": "alias"}],
    }));
    assert!(message.contains("singleton `allOf`"), "{message}");
}

#[test]
fn nested_single_all_of_rejects_overlapping_parent_constraints() {
    let t = json!({"type": "object", "properties": {"inner": {"type": "string"}}});
    let message = strict_err(json!({
        "$defs": {"T": t},
        "type": "object",
        "properties": {"outer": {"type": "string"}},
        "allOf": [{"allOf": [{"$ref": "#/$defs/T"}]}],
    }));
    assert!(message.contains("singleton `allOf`"), "{message}");
}

/// JSON-distinct values that Python's `==` equates (`1 == True`) must still conflict.
#[test]
fn all_of_single_ref_entry_rejects_json_distinct_equal_python_values() {
    for (referenced, parent) in [
        (json!(1), json!(true)),
        (json!({"nested": [1]}), json!({"nested": [true]})),
    ] {
        let message = strict_err(json!({
            "$defs": {"T": {"const": referenced}},
            "const": parent,
            "allOf": [{"$ref": "#/$defs/T"}],
        }));
        assert!(message.contains("singleton `allOf`"), "{message}");
    }
}

#[test]
fn all_of_single_ref_entry_accepts_equal_json_numbers() {
    let result = strict(json!({
        "$defs": {"T": {"const": 1}},
        "const": 1.0,
        "allOf": [{"$ref": "#/$defs/T"}],
    }));
    assert_eq!(result["const"], json!(1));
    assert!(
        result["const"].is_i64() || result["const"].is_u64(),
        "{result}"
    );
}

#[test]
fn all_of_single_annotated_circular_ref_is_rejected() {
    let message = strict_err(json!({
        "components": {"schemas": {
            "A": {"$ref": "#/components/schemas/B", "description": "a"},
            "B": {"$ref": "#/components/schemas/A", "description": "b"},
        }},
        "type": "object",
        "allOf": [{"$ref": "#/components/schemas/A"}],
    }));
    assert!(message.contains("circular"), "{message}");
}

fn assert_nested_id(schema: Value) {
    let message = strict_err(schema);
    assert!(message.contains("nested `$id` resource"), "{message}");
}

#[test]
fn all_of_single_ref_rejects_nested_id_before_promoting_the_referenced_schema() {
    let message = strict_err(json!({
        "$defs": {"T": {"type": "string"}},
        "contentSchema": {"$id": "https://example.test/nested", "$ref": "#/$defs/T"},
        "type": "object",
        "allOf": [{"$ref": "#/contentSchema"}],
    }));
    assert!(message.contains("nested `$id`"), "{message}");
}

#[test]
fn all_of_single_ref_rejects_nested_id_owner_before_resolution() {
    assert_nested_id(json!({
        "$defs": {"T": {"type": "string"}},
        "type": "object",
        "properties": {"node": {
            "$id": "https://example.test/nested",
            "$defs": {"T": {"type": "integer"}},
            "allOf": [{"$ref": "#/$defs/T"}],
        }},
    }));
}

#[test]
fn all_of_single_ref_rejects_descendant_nested_id_owner_before_resolution() {
    assert_nested_id(json!({
        "$defs": {"T": {"type": "string"}},
        "type": "object",
        "properties": {"node": {
            "$id": "https://example.test/nested",
            "$defs": {"T": {"type": "integer"}},
            "type": "object",
            "properties": {"child": {"allOf": [{"$ref": "#/$defs/T"}]}},
        }},
    }));
}

#[test]
fn all_of_single_ref_rejects_promoted_nested_id_with_descendant_ref() {
    assert_nested_id(json!({
        "$defs": {"T": {"type": "string"}},
        "contentSchema": {
            "$id": "https://example.test/nested",
            "$defs": {"T": {"type": "integer"}},
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/T"}},
        },
        "type": "object",
        "allOf": [{"$ref": "#/contentSchema"}],
    }));
}

#[test]
fn all_of_single_ref_rejects_reference_below_nested_id_resource() {
    assert_nested_id(json!({
        "$defs": {"T": {"type": "string"}},
        "contentSchema": {
            "$id": "https://example.test/nested",
            "$defs": {"T": {"type": "integer"}},
            "inner": {"$ref": "#/$defs/T"},
        },
        "type": "object",
        "allOf": [{"$ref": "#/contentSchema/inner"}],
    }));
}

#[test]
fn all_of_single_ref_allows_id_as_schema_map_member_name() {
    for (container, reference) in [
        (
            json!({"$defs": {"$id": {"type": "object", "properties": {}}}}),
            "#/$defs/$id",
        ),
        (
            json!({"components": {"schemas": {"$id": {"type": "object", "properties": {}}}}}),
            "#/components/schemas/$id",
        ),
    ] {
        let mut schema = container;
        schema["type"] = json!("object");
        schema["allOf"] = json!([{"$ref": reference}]);
        let result = strict(schema);
        assert_eq!(result["type"], "object");
        assert_eq!(result["properties"], json!({}));
        assert_eq!(result["additionalProperties"], json!(false));
    }
}

#[test]
fn nested_ref_with_id_is_rejected_before_resolution() {
    assert_nested_id(json!({
        "$defs": {"T": {"type": "string"}},
        "type": "object",
        "properties": {"value": {
            "$id": "https://example.test/nested",
            "$defs": {"T": {"type": "integer"}},
            "$ref": "#/$defs/T",
        }},
    }));
}
