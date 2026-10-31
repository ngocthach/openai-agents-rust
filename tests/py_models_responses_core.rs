//! Port of `tests/models/test_remove_openai_responses_api_incompatible_fields.py`,
//! `tests/models/test_trace_config.py` and the `_build_response_create_kwargs` cluster of
//! `tests/models/test_openai_responses.py`.
//!
//! Python's `_build_response_create_kwargs` is `OpenAIResponsesModel::build_request_body` here,
//! and `_remove_openai_responses_api_incompatible_fields` is `clean_input_for_openai`.

use std::collections::HashMap;

use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::model_settings::{ModelSettings, Reasoning};
use openai_agents_sdk::models::interface::{ModelRequest, ModelTracing};
use openai_agents_sdk::models::openai_responses::{OpenAIResponsesModel, clean_input_for_openai};
use openai_agents_sdk::models::trace::{
    model_config_for_trace, model_config_for_trace_with_extra, sanitize_url_for_trace,
};
use openai_agents_sdk::models::{FAKE_RESPONSES_ID, OpenAIClient};
use serde_json::{Map, Value, json};

fn model(name: &str) -> OpenAIResponsesModel {
    OpenAIResponsesModel::new(name, OpenAIClient::new("sk-test"))
}

fn items(values: Vec<Value>) -> Vec<InputItem> {
    values.into_iter().map(InputItem::from).collect()
}

fn clean(values: Vec<Value>) -> Vec<Value> {
    clean_input_for_openai(&items(values))
}

fn body(m: &OpenAIResponsesModel, settings: ModelSettings) -> Value {
    m.build_request_body(
        &ModelRequest {
            input: vec![InputItem::user("hi")],
            model_settings: settings,
            ..Default::default()
        },
        false,
    )
    .unwrap()
}

fn err(m: &OpenAIResponsesModel, settings: ModelSettings, stream: bool) -> String {
    m.build_request_body(
        &ModelRequest {
            input: vec![InputItem::user("hi")],
            model_settings: settings,
            ..Default::default()
        },
        stream,
    )
    .unwrap_err()
    .to_string()
}

fn map(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

// ------------------------------------- _remove_openai_responses_api_incompatible_fields

#[test]
fn returns_unchanged_when_no_provider_data() {
    let input = vec![
        json!({"type": "message", "content": "hello"}),
        json!({"type": "function_call", "call_id": "call_123", "name": "test"}),
    ];
    assert_eq!(clean(input.clone()), input);
}

#[test]
fn removes_reasoning_items_with_provider_data() {
    let result = clean(vec![
        json!({"type": "message", "content": "hello"}),
        json!({"type": "reasoning", "provider_data": {"model": "gemini/gemini-3"}}),
        json!({"type": "function_call", "call_id": "call_123"}),
    ]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], json!({"type": "message", "content": "hello"}));
    assert_eq!(
        result[1],
        json!({"type": "function_call", "call_id": "call_123"})
    );
}

#[test]
fn keeps_reasoning_items_without_provider_data() {
    let result = clean(vec![
        json!({"type": "reasoning", "summary": []}),
        json!({"type": "message", "content": "hello", "provider_data": {"foo": "bar"}}),
    ]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], json!({"type": "reasoning", "summary": []}));
    assert_eq!(result[1], json!({"type": "message", "content": "hello"}));
}

#[test]
fn removes_provider_data_from_all_items() {
    let result = clean(vec![
        json!({"type": "message", "content": "hello", "provider_data": {"model": "g"}}),
        json!({"type": "function_call", "call_id": "call_123", "provider_data": {"model": "g"}}),
    ]);
    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|v| v.get("provider_data").is_none()));
}

#[test]
fn removes_fake_responses_id() {
    let result = clean(vec![json!({
        "type": "message", "id": FAKE_RESPONSES_ID, "content": "hello",
        "provider_data": {"model": "gemini/gemini-3"}})]);
    assert_eq!(result.len(), 1);
    assert!(result[0].get("id").is_none());
    assert_eq!(result[0]["content"], "hello");
}

#[test]
fn removes_fake_responses_id_without_provider_data() {
    let result = clean(vec![
        json!({"role": "user", "content": "hi"}),
        json!({"type": "message", "id": FAKE_RESPONSES_ID, "content": "hello"}),
    ]);
    assert_eq!(result.len(), 2);
    assert!(result[1].get("id").is_none());
    assert_eq!(result[1]["content"], "hello");
}

#[test]
fn preserves_real_ids() {
    let result = clean(vec![json!({
        "type": "message", "id": "msg_real123", "content": "hello", "provider_data": {}})]);
    assert_eq!(result[0]["id"], "msg_real123");
}

#[test]
fn handles_empty_list() {
    assert_eq!(clean(vec![]), Vec::<Value>::new());
}

#[test]
fn combined_scenario() {
    let result = clean(vec![
        json!({"type": "message", "content": "user input"}),
        json!({"type": "reasoning", "summary": [], "provider_data": {"model": "g"}}),
        json!({"type": "function_call", "call_id": "call_abc_123", "name": "get_weather",
            "provider_data": {"model": "g"}}),
        json!({"type": "function_call_output", "call_id": "call_abc_123",
            "output": "{\"temp\": 72}"}),
        json!({"type": "message", "id": FAKE_RESPONSES_ID, "content": "The weather is 72F",
            "provider_data": {"model": "g"}}),
    ]);
    assert_eq!(result.len(), 4);
    assert_eq!(
        result[0],
        json!({"type": "message", "content": "user input"})
    );
    assert_eq!(result[1]["type"], "function_call");
    assert_eq!(result[1]["call_id"], "call_abc_123");
    assert!(result[1].get("provider_data").is_none());
    assert_eq!(result[2]["type"], "function_call_output");
    assert_eq!(result[2]["call_id"], "call_abc_123");
    assert_eq!(result[3]["type"], "message");
    assert_eq!(result[3]["content"], "The weather is 72F");
    assert!(result[3].get("id").is_none());
    assert!(result[3].get("provider_data").is_none());
}

#[test]
fn request_payload_drops_sdk_generated_placeholder_ids() {
    // The SDK builds replayed assistant messages with the placeholder id.
    let message = json!({"type": "message", "id": FAKE_RESPONSES_ID, "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "the run error handler final output",
            "annotations": []}]});
    let req = ModelRequest {
        input: items(vec![json!({"role": "user", "content": "hi"}), message]),
        ..Default::default()
    };
    let body = model("gpt-5").build_request_body(&req, false).unwrap();
    assert!(body["input"][1].get("id").is_none());
    assert_eq!(
        body["input"][1]["content"][0]["text"],
        "the run error handler final output"
    );
}

// ------------------------------------------------------------------- trace config

#[test]
fn sanitize_url_for_trace_strips_auth_query_and_fragment() {
    assert_eq!(
        sanitize_url_for_trace("https://user:pass@example.com/v1?api-key=secret#fragment"),
        "https://example.com/v1"
    );
    assert_eq!(
        sanitize_url_for_trace("https://example.com/v1?token=secret"),
        "https://example.com/v1"
    );
}

#[test]
fn model_config_for_trace_sanitizes_base_url_and_omits_request_extras() {
    let settings = ModelSettings {
        temperature: Some(0.5),
        extra_headers: Some(HashMap::from([(
            "Authorization".to_string(),
            "Bearer provider-token".to_string(),
        )])),
        extra_query: Some(HashMap::from([(
            "api-key".to_string(),
            "query-token".to_string(),
        )])),
        extra_body: Some(map(&[("secret", json!("body-token"))])),
        extra_args: Some(map(&[("api_key", json!("arg-token"))])),
        ..Default::default()
    };
    let config = model_config_for_trace_with_extra(
        &settings,
        ModelTracing::Enabled,
        Some("https://user:pass@example.com/v1?api-key=secret#fragment"),
        Some(&map(&[("model_impl", json!("test-model"))])),
    );

    assert_eq!(config["temperature"], 0.5);
    assert_eq!(config["base_url"], "https://example.com/v1");
    assert_eq!(config["model_impl"], "test-model");
    for key in ["extra_headers", "extra_query", "extra_body", "extra_args"] {
        assert!(config.get(key).is_none(), "{key} leaked into the trace");
    }
    // The no-extras entry point keeps the same sanitization.
    let plain = model_config_for_trace(&settings, ModelTracing::Enabled, Some("https://x/v1"));
    assert!(plain.get("model_impl").is_none());
    assert_eq!(plain["base_url"], "https://x/v1");
}

// --------------------------------------------------- _build_response_create_kwargs

#[test]
fn build_response_create_kwargs_rejects_duplicate_extra_args_keys() {
    let settings = ModelSettings {
        extra_args: Some(map(&[("stream", json!(false))])),
        ..Default::default()
    };
    let message = err(&model("gpt-4"), settings, true);
    assert!(message.contains("multiple values"), "{message}");
    assert!(message.contains("'stream'"), "{message}");
}

#[test]
fn build_response_create_kwargs_includes_extra_args_prompt_cache_key() {
    let settings = ModelSettings {
        extra_args: Some(map(&[("prompt_cache_key", json!("cache-key"))])),
        ..Default::default()
    };
    assert_eq!(
        body(&model("gpt-4"), settings)["prompt_cache_key"],
        "cache-key"
    );
}

#[test]
fn build_response_create_kwargs_includes_context_management() {
    let cm = vec![json!({"type": "compaction", "compact_threshold": 200000})];
    let settings = ModelSettings {
        context_management: Some(cm.clone()),
        ..Default::default()
    };
    assert_eq!(
        body(&model("gpt-4"), settings)["context_management"],
        json!(cm)
    );
}

#[test]
fn build_response_create_kwargs_includes_gpt_5_6_request_controls() {
    let reasoning = Reasoning {
        effort: Some("max".into()),
        ..Default::default()
    };
    let options = json!({"mode": "explicit", "ttl": "30m"});
    let req = ModelRequest {
        input: vec![InputItem::user("hi")],
        previous_response_id: Some("resp-previous".into()),
        model_settings: ModelSettings {
            reasoning: Some(reasoning),
            prompt_cache_retention: Some("24h".into()),
            prompt_cache_options: Some(options.clone()),
            ..Default::default()
        },
        ..Default::default()
    };
    let body = model("gpt-5.6-sol")
        .build_request_body(&req, false)
        .unwrap();
    assert_eq!(body["reasoning"]["effort"], "max");
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(body["prompt_cache_options"], options);
    assert_eq!(body["previous_response_id"], "resp-previous");
}

#[test]
fn responses_requests_normalize_agent_settings() {
    let settings = ModelSettings {
        reasoning: Some(Reasoning {
            effort: Some("low".into()),
            ..Default::default()
        }),
        context_management: Some(vec![
            json!({"type": "compaction", "compact_threshold": 200000}),
        ]),
        prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
        prompt_cache_retention: Some("24h".into()),
        store: Some(false),
        metadata: Some(HashMap::from([(
            "request".to_string(),
            "example".to_string(),
        )])),
        temperature: Some(0.0),
        top_p: Some(1.0),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        max_tokens: Some(64),
        parallel_tool_calls: Some(false),
        extra_headers: Some(HashMap::from([(
            "x-model-settings-parity".to_string(),
            "preserved".to_string(),
        )])),
        extra_query: Some(HashMap::from([(
            "model_settings_parity".to_string(),
            "verified".to_string(),
        )])),
        extra_body: Some(map(&[("prompt_cache_key", json!("extra-body-cache-key"))])),
        ..Default::default()
    };
    let body = body(&model("gpt-4"), settings);
    assert_eq!(body["reasoning"]["effort"], "low");
    assert_eq!(body["context_management"][0]["compact_threshold"], 200000);
    assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["metadata"]["request"], "example");
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["top_p"], 1.0);
    assert_eq!(body["max_output_tokens"], 64);
    assert!(body.get("max_tokens").is_none());
    // No tools and no prompt: `parallel_tool_calls` is left out entirely.
    assert!(body.get("parallel_tool_calls").is_none());
    // `extra_body` is merged into the request body; headers/query travel on the transport.
    assert_eq!(body["prompt_cache_key"], "extra-body-cache-key");
    assert!(body.get("retry").is_none());
    assert!(body.get("frequency_penalty").is_none());
    assert!(body.get("presence_penalty").is_none());
    assert!(body.get("extra_headers").is_none());
    assert!(body.get("extra_query").is_none());
    assert!(body.get("extra_body").is_none());
}

#[test]
fn build_response_create_kwargs_rejects_duplicate_prompt_cache_options_extra_args() {
    let settings = ModelSettings {
        prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
        extra_args: Some(map(&[(
            "prompt_cache_options",
            json!({"mode": "implicit"}),
        )])),
        ..Default::default()
    };
    let message = err(&model("gpt-5.6-sol"), settings, false);
    assert!(message.contains("multiple values"), "{message}");
    assert!(message.contains("'prompt_cache_options'"), "{message}");
}

#[test]
fn build_response_create_kwargs_allows_prompt_cache_options_extra_args_when_direct_omitted() {
    let options = json!({"mode": "explicit", "ttl": "30m"});
    let settings = ModelSettings {
        extra_args: Some(map(&[("prompt_cache_options", options.clone())])),
        ..Default::default()
    };
    assert_eq!(
        body(&model("gpt-5.6-sol"), settings)["prompt_cache_options"],
        options
    );
}

#[test]
fn build_response_create_kwargs_allows_extra_arg_when_explicit_arg_is_omitted() {
    let cm = json!([{"type": "compaction", "compact_threshold": 200000}]);
    let settings = ModelSettings {
        extra_args: Some(map(&[("context_management", cm.clone())])),
        ..Default::default()
    };
    assert_eq!(body(&model("gpt-4"), settings)["context_management"], cm);
}

#[test]
fn build_response_create_kwargs_rejects_duplicate_context_management_extra_args() {
    let settings = ModelSettings {
        context_management: Some(vec![
            json!({"type": "compaction", "compact_threshold": 200000}),
        ]),
        extra_args: Some(map(&[(
            "context_management",
            json!([{"type": "compaction"}]),
        )])),
        ..Default::default()
    };
    let message = err(&model("gpt-4"), settings, false);
    assert!(message.contains("multiple values"), "{message}");
    assert!(message.contains("'context_management'"), "{message}");
}

#[test]
fn build_response_create_kwargs_keeps_unset_transport_extras_absent() {
    // Python keeps `extra_query`/`extra_body` as `None` kwargs; Rust leaves them out of the
    // body and off the transport entirely.
    let body = body(&model("gpt-4"), ModelSettings::default());
    assert!(body.get("extra_query").is_none());
    assert!(body.get("extra_body").is_none());
}

#[test]
fn build_response_create_kwargs_preserves_unknown_response_include_values() {
    let settings = ModelSettings {
        response_include: Some(vec!["response.future_flag".into()]),
        ..Default::default()
    };
    assert_eq!(
        body(&model("gpt-4"), settings)["include"],
        json!(["response.future_flag"])
    );
}

#[test]
fn build_response_create_kwargs_preserves_unknown_tool_types() {
    // A hosted tool param the SDK does not know is forwarded verbatim.
    let future_tool = json!({"type": "future_beta_tool", "label": "preview"});
    let req = ModelRequest {
        input: vec![InputItem::user("hi")],
        tools: vec![openai_agents_sdk::models::interface::ToolSpec::Hosted {
            param: future_tool.clone(),
            includes: Vec::new(),
        }],
        ..Default::default()
    };
    let body = model("gpt-4").build_request_body(&req, false).unwrap();
    assert_eq!(body["tools"], json!([future_tool]));
}

#[test]
fn top_logprobs_adds_the_logprobs_include() {
    let settings = ModelSettings {
        top_logprobs: Some(2),
        ..Default::default()
    };
    let body = body(&model("gpt-4"), settings);
    assert_eq!(body["top_logprobs"], 2);
    assert_eq!(body["include"], json!(["message.output_text.logprobs"]));
}

// ----------------------------------------------- tools, prompts and computer payloads

mod tools {
    use super::*;
    use openai_agents_sdk::model_settings::ToolChoice;
    use openai_agents_sdk::models::interface::{FunctionToolSpec, ToolSpec};
    use openai_agents_sdk::prompts::Prompt;

    fn func(name: &str) -> FunctionToolSpec {
        FunctionToolSpec {
            name: name.into(),
            description: "d".into(),
            parameters: json!({"type": "object", "properties": {}}),
            strict: true,
            ..Default::default()
        }
    }

    fn hosted(param: Value) -> ToolSpec {
        ToolSpec::Hosted {
            param,
            includes: Vec::new(),
        }
    }

    fn computer() -> ToolSpec {
        hosted(json!({"type": "computer", "environment": "mac",
            "display_width": 800, "display_height": 600}))
    }

    fn settings(choice: Option<&str>, parallel: Option<bool>) -> ModelSettings {
        ModelSettings {
            tool_choice: choice.map(ToolChoice::from),
            parallel_tool_calls: parallel,
            ..Default::default()
        }
    }

    fn request(tools: Vec<ToolSpec>, settings: ModelSettings) -> ModelRequest {
        ModelRequest {
            input: vec![InputItem::user("hi")],
            tools,
            model_settings: settings,
            ..Default::default()
        }
    }

    fn prompt_request(tools: Vec<ToolSpec>, settings: ModelSettings) -> ModelRequest {
        ModelRequest {
            prompt: Some(Prompt::new("pmpt_123")),
            ..request(tools, settings)
        }
    }

    /// A model whose name came from the SDK default, not from the caller.
    fn implicit(name: &str) -> OpenAIResponsesModel {
        model(name).with_explicit_model(false)
    }

    #[test]
    fn parallel_tool_calls_follow_converted_responses_tools() {
        for parallel in [Some(true), Some(false), None] {
            for source in ["none", "function", "handoff"] {
                let mut req = request(
                    if source == "function" {
                        vec![ToolSpec::Function(func("test_tool"))]
                    } else {
                        vec![]
                    },
                    settings(None, parallel),
                );
                if source == "handoff" {
                    req.handoffs = vec![func("transfer_to_handoff")];
                }
                let body = model("gpt-4").build_request_body(&req, false).unwrap();
                let expected = if source != "none" { parallel } else { None };
                assert_eq!(
                    body.get("parallel_tool_calls").and_then(Value::as_bool),
                    expected,
                    "{source} {parallel:?}"
                );
                assert_eq!(body.get("tools").is_some(), source != "none", "{source}");
            }
        }
    }

    #[test]
    fn parallel_tool_calls_preserve_stored_prompt_overrides() {
        for parallel in [Some(true), Some(false), None] {
            let req = prompt_request(vec![], settings(None, parallel));
            let body = model("gpt-4").build_request_body(&req, false).unwrap();
            assert_eq!(
                body.get("parallel_tool_calls").and_then(Value::as_bool),
                parallel,
                "{parallel:?}"
            );
            assert!(body.get("tools").is_none());
        }
    }

    #[test]
    fn prompt_id_omits_the_model_parameter() {
        let body = implicit("gpt-4")
            .build_request_body(&prompt_request(vec![], ModelSettings::default()), false)
            .unwrap();
        assert_eq!(body["prompt"], json!({"id": "pmpt_123"}));
        assert!(body.get("model").is_none());
        // An explicitly chosen model is kept even with a prompt.
        let body = model("gpt-4")
            .build_request_body(&prompt_request(vec![], ModelSettings::default()), false)
            .unwrap();
        assert_eq!(body["model"], "gpt-4");
    }

    #[test]
    fn prompt_id_omits_tools_when_no_tools_are_configured() {
        let body = implicit("gpt-4")
            .build_request_body(&prompt_request(vec![], ModelSettings::default()), false)
            .unwrap();
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn prompt_id_omits_named_tool_choice_when_no_tools_are_configured() {
        let req = prompt_request(vec![], settings(Some("web_search_preview"), None));
        let body = implicit("gpt-4").build_request_body(&req, false).unwrap();
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn prompt_id_keeps_literal_tool_choice_without_local_tools() {
        for choice in ["none", "required"] {
            let req = prompt_request(vec![], settings(Some(choice), None));
            let body = implicit("gpt-4").build_request_body(&req, false).unwrap();
            assert!(body.get("tools").is_none(), "{choice}");
            assert_eq!(body["tool_choice"], choice);
        }
    }

    #[test]
    fn prompt_id_keeps_an_explicit_tool_search_without_a_local_surface() {
        let req = prompt_request(
            vec![hosted(json!({"type": "tool_search"}))],
            ModelSettings::default(),
        );
        let body = implicit("gpt-4").build_request_body(&req, false).unwrap();
        assert_eq!(body["prompt"], json!({"id": "pmpt_123"}));
        assert_eq!(body["tools"], json!([{"type": "tool_search"}]));
    }

    #[test]
    fn ga_computer_tool_does_not_require_preview_metadata() {
        let bare = hosted(json!({"type": "computer"}));
        let body = model("gpt-5.4")
            .build_request_body(&request(vec![bare], ModelSettings::default()), false)
            .unwrap();
        assert_eq!(body["tools"], json!([{"type": "computer"}]));
    }

    #[test]
    fn prompt_id_uses_the_preview_computer_payload_when_the_prompt_owns_the_model() {
        let req = prompt_request(vec![computer()], ModelSettings::default());
        let body = implicit("gpt-5.4").build_request_body(&req, false).unwrap();
        assert!(body.get("model").is_none());
        assert!(body.get("tool_choice").is_none());
        assert_eq!(
            body["tools"],
            json!([{"type": "computer_use_preview", "environment": "mac",
                "display_width": 800, "display_height": 600}])
        );
    }

    #[test]
    fn prompt_id_computer_without_preview_metadata_raises_a_clear_error() {
        let bare = hosted(json!({"type": "computer"}));
        let req = prompt_request(vec![bare], ModelSettings::default());
        let error = implicit("gpt-5.4")
            .build_request_body(&req, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Preview computer tool payloads require `environment` and `dimensions`"),
            "{error}"
        );
    }

    #[test]
    fn prompt_id_explicit_ga_computer_tool_choice_uses_the_ga_selector_and_tool() {
        for choice in ["computer", "computer_use"] {
            let req = prompt_request(vec![computer()], settings(Some(choice), None));
            let body = implicit("gpt-5.4").build_request_body(&req, false).unwrap();
            assert!(body.get("model").is_none(), "{choice}");
            assert_eq!(body["tool_choice"], json!({"type": "computer"}), "{choice}");
            assert_eq!(body["tools"], json!([{"type": "computer"}]), "{choice}");
        }
    }

    #[test]
    fn preview_model_forced_computer_tool_choice_uses_the_preview_selector() {
        for choice in ["computer", "computer_use"] {
            let req = request(vec![computer()], settings(Some(choice), None));
            let body = model("computer-use-preview")
                .build_request_body(&req, false)
                .unwrap();
            assert_eq!(body["model"], "computer-use-preview");
            assert_eq!(
                body["tool_choice"],
                json!({"type": "computer_use_preview"}),
                "{choice}"
            );
            assert_eq!(
                body["tools"],
                json!([{"type": "computer_use_preview", "environment": "mac",
                    "display_width": 800, "display_height": 600}]),
                "{choice}"
            );
        }
    }

    #[test]
    fn builtin_tool_choice_selects_the_builtin_among_multiple_tools() {
        for choice in ["shell", "apply_patch"] {
            let builtin = if choice == "shell" {
                hosted(json!({"type": "shell", "environment": {"type": "local"}}))
            } else {
                hosted(json!({"type": "apply_patch"}))
            };
            let mut req = request(
                vec![
                    ToolSpec::Function(func("lookup_account")),
                    builtin,
                    ToolSpec::Function(func("summarize")),
                ],
                settings(Some(choice), None),
            );
            req.handoffs = vec![func("transfer_to_target")];
            let body = model("gpt-5.4").build_request_body(&req, false).unwrap();
            assert_eq!(body["tool_choice"], json!({"type": choice}));
            let types: Vec<&str> = body["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["type"].as_str().unwrap())
                .collect();
            assert!(types.contains(&choice), "{types:?}");
        }
    }
}
