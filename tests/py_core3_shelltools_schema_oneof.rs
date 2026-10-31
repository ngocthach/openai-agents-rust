//! Port of the Python SDK's `tests/test_strict_schema_oneof.py`.
//!
//! `oneOf` is rewritten to `anyOf` in strict mode, merged with an existing `anyOf`, and kept
//! where it appears under `$defs` and inside discriminated unions. The pydantic-only case is
//! expressed with a serde internally tagged enum.

use openai_agents_sdk::strict_schema::ensure_strict_json_schema;
use openai_agents_sdk::{AgentOutputSchema, AgentOutputSchemaBase};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn strict(schema: Value) -> Value {
    ensure_strict_json_schema(&schema).expect("schema should be strict-compatible")
}

#[test]
fn oneof_converted_to_anyof() {
    let schema = json!({
        "type": "object",
        "properties": {"value": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
    });

    let result = strict(schema);

    let expected = json!({
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        "additionalProperties": false,
        "required": ["value"],
    });
    assert_eq!(result, expected);
}

#[test]
fn nested_oneof_in_array_items() {
    let schema = json!({
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "const": "buy_fruit"},
                                "color": {"type": "string"},
                            },
                            "required": ["action", "color"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "const": "buy_food"},
                                "price": {"type": "integer"},
                            },
                            "required": ["action", "price"],
                        },
                    ],
                    "discriminator": {
                        "propertyName": "action",
                        "mapping": {
                            "buy_fruit": "#/components/schemas/BuyFruitStep",
                            "buy_food": "#/components/schemas/BuyFoodStep",
                        },
                    },
                },
            }
        },
    });

    let result = strict(schema);

    let expected = json!({
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "const": "buy_fruit"},
                                "color": {"type": "string"},
                            },
                            "required": ["action", "color"],
                            "additionalProperties": false,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "const": "buy_food"},
                                "price": {"type": "integer"},
                            },
                            "required": ["action", "price"],
                            "additionalProperties": false,
                        },
                    ],
                    "discriminator": {
                        "propertyName": "action",
                        "mapping": {
                            "buy_fruit": "#/components/schemas/BuyFruitStep",
                            "buy_food": "#/components/schemas/BuyFoodStep",
                        },
                    },
                },
            }
        },
        "additionalProperties": false,
        "required": ["steps"],
    });
    assert_eq!(result, expected);
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case")]
#[allow(dead_code)]
enum Step {
    BuyFruit { args: FruitArgs },
    BuyFood { args: FoodArgs },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct FruitArgs {
    color: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct FoodArgs {
    price: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Actions {
    steps: Vec<Step>,
}

#[test]
fn discriminated_union_with_derived_schema_is_strict_anyof() {
    // Python's pydantic output emits `oneOf` plus an OpenAPI `discriminator` block; schemars
    // emits the `oneOf` without the discriminator, so only the strict-mode shape is checked.
    let output_schema = AgentOutputSchema::new::<Actions>().unwrap();
    let schema = output_schema.json_schema().unwrap();

    // schemars hoists the enum into `$defs`; resolve the reference to inspect the union.
    let items = &schema["properties"]["steps"]["items"];
    let items_schema = match items.get("$ref").and_then(Value::as_str) {
        Some(reference) => {
            let name = reference.rsplit('/').next().unwrap();
            &schema["$defs"][name]
        }
        None => items,
    };
    assert!(items_schema.get("oneOf").is_none());
    let any_of = items_schema["anyOf"].as_array().expect("anyOf");
    assert_eq!(any_of.len(), 2);
    for variant in any_of {
        assert_eq!(variant["additionalProperties"], json!(false));
    }
}

#[test]
fn oneof_merged_with_existing_anyof() {
    let schema = json!({
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [{"type": "string"}],
                "oneOf": [{"type": "integer"}, {"type": "boolean"}],
            }
        },
    });

    let result = strict(schema);

    let expected = json!({
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"type": "boolean"},
                ]
            }
        },
        "additionalProperties": false,
        "required": ["value"],
    });
    assert_eq!(result, expected);
}

#[test]
fn discriminator_preserved() {
    let schema = json!({
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [{"$ref": "#/$defs/TypeA"}, {"$ref": "#/$defs/TypeB"}],
                "discriminator": {
                    "propertyName": "type",
                    "mapping": {"a": "#/$defs/TypeA", "b": "#/$defs/TypeB"},
                },
            }
        },
        "$defs": {
            "TypeA": {
                "type": "object",
                "properties": {"type": {"const": "a"}, "value_a": {"type": "string"}},
            },
            "TypeB": {
                "type": "object",
                "properties": {"type": {"const": "b"}, "value_b": {"type": "integer"}},
            },
        },
    });

    let result = strict(schema);

    let expected = json!({
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [{"$ref": "#/$defs/TypeA"}, {"$ref": "#/$defs/TypeB"}],
                "discriminator": {
                    "propertyName": "type",
                    "mapping": {"a": "#/$defs/TypeA", "b": "#/$defs/TypeB"},
                },
            }
        },
        "$defs": {
            "TypeA": {
                "type": "object",
                "properties": {"type": {"const": "a"}, "value_a": {"type": "string"}},
                "additionalProperties": false,
                "required": ["type", "value_a"],
            },
            "TypeB": {
                "type": "object",
                "properties": {"type": {"const": "b"}, "value_b": {"type": "integer"}},
                "additionalProperties": false,
                "required": ["type", "value_b"],
            },
        },
        "additionalProperties": false,
        "required": ["value"],
    });
    assert_eq!(result, expected);
}

#[test]
fn deeply_nested_oneof() {
    let schema = json!({
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "array",
                        "items": {"oneOf": [{"type": "string"}, {"type": "number"}]},
                    }
                },
            }
        },
    });

    let result = strict(schema);

    let expected = json!({
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                    }
                },
                "additionalProperties": false,
                "required": ["level2"],
            }
        },
        "additionalProperties": false,
        "required": ["level1"],
    });
    assert_eq!(result, expected);
}

#[test]
fn oneof_with_refs() {
    let schema = json!({
        "type": "object",
        "properties": {
            "value": {"oneOf": [{"$ref": "#/$defs/StringType"}, {"$ref": "#/$defs/IntType"}]}
        },
        "$defs": {
            "StringType": {"type": "string"},
            "IntType": {"type": "integer"},
        },
    });

    let result = strict(schema);

    let expected = json!({
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"$ref": "#/$defs/StringType"}, {"$ref": "#/$defs/IntType"}]}
        },
        "$defs": {
            "StringType": {"type": "string"},
            "IntType": {"type": "integer"},
        },
        "additionalProperties": false,
        "required": ["value"],
    });
    assert_eq!(result, expected);
}
