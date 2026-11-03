//! Shape checks for realtime wire events.
//!
//! Python validates every server event against the OpenAI SDK's pydantic union and every raw
//! client message against its client-event union, so an event of an unknown type, or one missing a
//! required field, is reported instead of being acted on. These tables carry the same required
//! fields and the scalar types of the nested objects the transport relies on.

use serde_json::Value;

#[derive(Clone, Copy)]
enum Kind {
    Str,
    Int,
    Num,
    List,
    /// An object, with the fields it must carry.
    Obj(&'static [(&'static str, Kind)]),
}

type Fields = &'static [(&'static str, Kind)];

use Kind::{Int, List, Num, Obj, Str};

const ANY_OBJ: Kind = Obj(&[]);

const SERVER_EVENTS: &[(&str, Fields)] = &[
    (
        "conversation.created",
        &[("conversation", ANY_OBJ), ("event_id", Str)],
    ),
    (
        "conversation.item.created",
        &[("event_id", Str), ("item", ANY_OBJ)],
    ),
    (
        "conversation.item.deleted",
        &[("event_id", Str), ("item_id", Str)],
    ),
    (
        "conversation.item.input_audio_transcription.completed",
        &[
            ("content_index", Int),
            ("event_id", Str),
            ("item_id", Str),
            ("transcript", Str),
            ("usage", ANY_OBJ),
        ],
    ),
    (
        "conversation.item.input_audio_transcription.delta",
        &[("event_id", Str), ("item_id", Str)],
    ),
    (
        "conversation.item.input_audio_transcription.failed",
        &[
            ("content_index", Int),
            ("error", ANY_OBJ),
            ("event_id", Str),
            ("item_id", Str),
        ],
    ),
    (
        "conversation.item.retrieved",
        &[("event_id", Str), ("item", ANY_OBJ)],
    ),
    (
        "conversation.item.truncated",
        &[
            ("audio_end_ms", Int),
            ("content_index", Int),
            ("event_id", Str),
            ("item_id", Str),
        ],
    ),
    (
        "error",
        &[
            ("error", Obj(&[("type", Str), ("message", Str)])),
            ("event_id", Str),
        ],
    ),
    ("input_audio_buffer.cleared", &[("event_id", Str)]),
    (
        "input_audio_buffer.committed",
        &[("event_id", Str), ("item_id", Str)],
    ),
    (
        "input_audio_buffer.dtmf_event_received",
        &[("event", Str), ("received_at", Int)],
    ),
    (
        "input_audio_buffer.speech_started",
        &[("audio_start_ms", Int), ("event_id", Str), ("item_id", Str)],
    ),
    (
        "input_audio_buffer.speech_stopped",
        &[("audio_end_ms", Int), ("event_id", Str), ("item_id", Str)],
    ),
    (
        "rate_limits.updated",
        &[("event_id", Str), ("rate_limits", List)],
    ),
    (
        "response.output_audio.delta",
        &[
            ("content_index", Int),
            ("delta", Str),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.output_audio.done",
        &[
            ("content_index", Int),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.output_audio_transcript.delta",
        &[
            ("content_index", Int),
            ("delta", Str),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.output_audio_transcript.done",
        &[
            ("content_index", Int),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
            ("transcript", Str),
        ],
    ),
    (
        "response.content_part.added",
        &[
            ("content_index", Int),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("part", ANY_OBJ),
            ("response_id", Str),
        ],
    ),
    (
        "response.content_part.done",
        &[
            ("content_index", Int),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("part", ANY_OBJ),
            ("response_id", Str),
        ],
    ),
    (
        "response.created",
        &[("event_id", Str), ("response", ANY_OBJ)],
    ),
    ("response.done", &[("event_id", Str), ("response", ANY_OBJ)]),
    (
        "response.function_call_arguments.delta",
        &[
            ("call_id", Str),
            ("delta", Str),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.function_call_arguments.done",
        &[
            ("arguments", Str),
            ("call_id", Str),
            ("event_id", Str),
            ("item_id", Str),
            ("name", Str),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.output_item.added",
        &[
            ("event_id", Str),
            ("item", ANY_OBJ),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.output_item.done",
        &[
            ("event_id", Str),
            ("item", ANY_OBJ),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.output_text.delta",
        &[
            ("content_index", Int),
            ("delta", Str),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.output_text.done",
        &[
            ("content_index", Int),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
            ("text", Str),
        ],
    ),
    (
        "session.created",
        &[("event_id", Str), ("session", ANY_OBJ)],
    ),
    (
        "session.updated",
        &[("event_id", Str), ("session", ANY_OBJ)],
    ),
    (
        "output_audio_buffer.started",
        &[("event_id", Str), ("response_id", Str)],
    ),
    (
        "output_audio_buffer.stopped",
        &[("event_id", Str), ("response_id", Str)],
    ),
    (
        "output_audio_buffer.cleared",
        &[("event_id", Str), ("response_id", Str)],
    ),
    (
        "conversation.item.added",
        &[("event_id", Str), ("item", ANY_OBJ)],
    ),
    (
        "conversation.item.done",
        &[("event_id", Str), ("item", ANY_OBJ)],
    ),
    (
        "input_audio_buffer.timeout_triggered",
        &[
            ("audio_end_ms", Int),
            ("audio_start_ms", Int),
            ("event_id", Str),
            ("item_id", Str),
        ],
    ),
    (
        "conversation.item.input_audio_transcription.segment",
        &[
            ("id", Str),
            ("content_index", Int),
            ("end", Num),
            ("event_id", Str),
            ("item_id", Str),
            ("speaker", Str),
            ("start", Num),
            ("text", Str),
        ],
    ),
    (
        "mcp_list_tools.in_progress",
        &[("event_id", Str), ("item_id", Str)],
    ),
    (
        "mcp_list_tools.completed",
        &[("event_id", Str), ("item_id", Str)],
    ),
    (
        "mcp_list_tools.failed",
        &[("event_id", Str), ("item_id", Str)],
    ),
    (
        "response.mcp_call_arguments.delta",
        &[
            ("delta", Str),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.mcp_call_arguments.done",
        &[
            ("arguments", Str),
            ("event_id", Str),
            ("item_id", Str),
            ("output_index", Int),
            ("response_id", Str),
        ],
    ),
    (
        "response.mcp_call.in_progress",
        &[("event_id", Str), ("item_id", Str), ("output_index", Int)],
    ),
    (
        "response.mcp_call.completed",
        &[("event_id", Str), ("item_id", Str), ("output_index", Int)],
    ),
    (
        "response.mcp_call.failed",
        &[("event_id", Str), ("item_id", Str), ("output_index", Int)],
    ),
];

const CLIENT_EVENTS: &[(&str, Fields)] = &[
    ("conversation.item.create", &[("item", ANY_OBJ)]),
    ("conversation.item.delete", &[("item_id", Str)]),
    ("conversation.item.retrieve", &[("item_id", Str)]),
    (
        "conversation.item.truncate",
        &[
            ("audio_end_ms", Int),
            ("content_index", Int),
            ("item_id", Str),
        ],
    ),
    ("input_audio_buffer.append", &[("audio", Str)]),
    ("input_audio_buffer.clear", &[]),
    ("output_audio_buffer.clear", &[]),
    ("input_audio_buffer.commit", &[]),
    ("response.cancel", &[]),
    ("response.create", &[]),
    ("session.update", &[("session", ANY_OBJ)]),
];

/// Check a server event against the known event types.
///
/// The error names the problem, never a field value, so it is safe to log.
pub(crate) fn validate_server_event(event: &Value) -> Result<(), String> {
    validate(event, SERVER_EVENTS, "event")
}

/// Check a raw client message against the known client event types.
pub(crate) fn validate_client_event(message: &Value) -> Result<(), String> {
    validate(message, CLIENT_EVENTS, "client event")
}

fn validate(value: &Value, table: &[(&str, Fields)], what: &str) -> Result<(), String> {
    let Some(object) = value.as_object() else {
        return Err(format!("Invalid {what}: not an object"));
    };
    let Some(event_type) = object.get("type").and_then(Value::as_str) else {
        return Err(format!("Invalid {what}: missing a string `type`"));
    };
    let Some((_, fields)) = table.iter().find(|(name, _)| *name == event_type) else {
        return Err(format!("Invalid {event_type} {what}: unknown event type"));
    };
    check_fields(object, fields)
        .map_err(|problem| format!("Invalid {event_type} {what}: {problem}"))
}

fn check_fields(
    object: &serde_json::Map<String, Value>,
    fields: &[(&str, Kind)],
) -> Result<(), String> {
    for (name, kind) in fields {
        let Some(value) = object.get(*name).filter(|value| !value.is_null()) else {
            return Err(format!("missing field `{name}`"));
        };
        let ok = match kind {
            Str => value.is_string(),
            Int => {
                value.is_i64()
                    || value.is_u64()
                    || value.as_f64().is_some_and(|number| number.fract() == 0.0)
            }
            Num => value.is_number(),
            List => value.is_array(),
            Obj(nested) => {
                let Some(inner) = value.as_object() else {
                    return Err(format!("field `{name}` must be an object"));
                };
                check_fields(inner, nested).map_err(|problem| format!("`{name}`: {problem}"))?;
                continue;
            }
        };
        if !ok {
            let expected = match kind {
                Str => "a string",
                Int => "an integer",
                Num => "a number",
                List => "an array",
                Obj(_) => "an object",
            };
            return Err(format!("field `{name}` must be {expected}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_complete_event_passes_and_a_list_transcript_does_not() {
        let mut event = json!({
            "type": "response.output_audio_transcript.done",
            "event_id": "e", "item_id": "i", "response_id": "r",
            "output_index": 0, "content_index": 0, "transcript": "ok",
        });
        assert!(validate_server_event(&event).is_ok());
        event["transcript"] = json!(["marker"]);
        let problem = validate_server_event(&event).unwrap_err();
        assert!(problem.contains("`transcript` must be a string"));
        assert!(!problem.contains("marker"));
    }

    #[test]
    fn unknown_types_and_missing_types_are_rejected() {
        assert!(validate_server_event(&json!({"type": "nope"})).is_err());
        assert!(validate_server_event(&json!({"data": 1})).is_err());
        assert!(validate_client_event(&json!({"type": "invalid.event"})).is_err());
        assert!(validate_client_event(&json!({"type": "response.cancel"})).is_ok());
    }
}
