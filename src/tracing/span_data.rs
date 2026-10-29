//! Span payloads. Mirrors `agents.tracing.span_data`.

use serde_json::{Value, json};

/// Data carried by a span.
#[derive(Debug, Clone, PartialEq)]
pub enum SpanData {
    /// An agent run.
    Agent {
        name: String,
        handoffs: Option<Vec<String>>,
        tools: Option<Vec<String>>,
        output_type: Option<String>,
    },
    /// One `Runner` invocation.
    Task { name: String, usage: Option<Value> },
    /// One model turn.
    Turn {
        turn: u64,
        agent_name: String,
        usage: Option<Value>,
    },
    /// A function tool call.
    Function {
        name: String,
        input: Option<String>,
        output: Option<String>,
        mcp_data: Option<Value>,
    },
    /// An LLM generation (Chat Completions).
    Generation {
        input: Option<Value>,
        output: Option<Value>,
        model: Option<String>,
        model_config: Option<Value>,
        usage: Option<Value>,
    },
    /// A Responses API call.
    Response {
        response_id: Option<String>,
        usage: Option<Value>,
    },
    /// A handoff between agents.
    Handoff {
        from_agent: Option<String>,
        to_agent: Option<String>,
    },
    /// A user-defined span.
    Custom { name: String, data: Value },
    /// A guardrail check.
    Guardrail { name: String, triggered: bool },
    /// Speech-to-text.
    Transcription {
        /// Base64 audio input (only when sensitive audio data is included).
        input: Option<String>,
        input_format: Option<String>,
        output: Option<String>,
        model: Option<String>,
        model_config: Option<Value>,
    },
    /// Text-to-speech.
    Speech {
        input: Option<String>,
        /// Base64 audio output (only when sensitive audio data is included).
        output: Option<String>,
        output_format: Option<String>,
        model: Option<String>,
        model_config: Option<Value>,
        first_content_at: Option<String>,
    },
    /// Groups the speech spans of one voice turn.
    SpeechGroup { input: Option<String> },
    /// Listing tools of an MCP server.
    MCPListTools {
        server: Option<String>,
        result: Option<Vec<String>>,
    },
}

impl SpanData {
    /// The span type.
    pub fn span_type(&self) -> &'static str {
        match self {
            SpanData::Agent { .. } => "agent",
            SpanData::Task { .. } => "task",
            SpanData::Turn { .. } => "turn",
            SpanData::Function { .. } => "function",
            SpanData::Generation { .. } => "generation",
            SpanData::Response { .. } => "response",
            SpanData::Handoff { .. } => "handoff",
            SpanData::Custom { .. } => "custom",
            SpanData::Guardrail { .. } => "guardrail",
            SpanData::Transcription { .. } => "transcription",
            SpanData::Speech { .. } => "speech",
            SpanData::SpeechGroup { .. } => "speech_group",
            SpanData::MCPListTools { .. } => "mcp_tools",
        }
    }

    /// Export to the OpenAI traces wire format.
    pub fn export(&self) -> Value {
        match self {
            SpanData::Agent {
                name,
                handoffs,
                tools,
                output_type,
            } => json!({
                "type": "agent",
                "name": name,
                "handoffs": handoffs,
                "tools": tools,
                "output_type": output_type,
            }),
            SpanData::Task { name, usage } => {
                let mut data = json!({"sdk_span_type": "task", "name": name});
                if let Some(u) = usage {
                    data["usage"] = u.clone();
                }
                json!({"type": "custom", "name": "task", "data": data})
            }
            SpanData::Turn {
                turn,
                agent_name,
                usage,
            } => {
                let mut data =
                    json!({"sdk_span_type": "turn", "turn": turn, "agent_name": agent_name});
                if let Some(u) = usage {
                    data["usage"] = u.clone();
                }
                json!({"type": "custom", "name": "turn", "data": data})
            }
            SpanData::Function {
                name,
                input,
                output,
                mcp_data,
            } => json!({
                "type": "function",
                "name": name,
                "input": input,
                "output": output,
                "mcp_data": mcp_data,
            }),
            SpanData::Generation {
                input,
                output,
                model,
                model_config,
                usage,
            } => json!({
                "type": "generation",
                "input": input,
                "output": output,
                "model": model,
                "model_config": model_config,
                "usage": usage,
            }),
            SpanData::Response { response_id, usage } => json!({
                "type": "response",
                "response_id": response_id,
                "usage": usage,
            }),
            SpanData::Handoff {
                from_agent,
                to_agent,
            } => json!({"type": "handoff", "from_agent": from_agent, "to_agent": to_agent}),
            SpanData::Custom { name, data } => {
                json!({"type": "custom", "name": name, "data": data})
            }
            SpanData::Guardrail { name, triggered } => {
                json!({"type": "guardrail", "name": name, "triggered": triggered})
            }
            SpanData::Transcription {
                input,
                input_format,
                output,
                model,
                model_config,
            } => json!({
                "type": "transcription",
                "input": {"data": input.clone().unwrap_or_default(), "format": input_format},
                "output": output,
                "model": model,
                "model_config": model_config,
            }),
            SpanData::Speech {
                input,
                output,
                output_format,
                model,
                model_config,
                first_content_at,
            } => json!({
                "type": "speech",
                "input": input,
                "output": {"data": output.clone().unwrap_or_default(), "format": output_format},
                "model": model,
                "model_config": model_config,
                "first_content_at": first_content_at,
            }),
            SpanData::SpeechGroup { input } => json!({"type": "speech_group", "input": input}),
            SpanData::MCPListTools { server, result } => {
                json!({"type": "mcp_tools", "server": server, "result": result})
            }
        }
    }
}
