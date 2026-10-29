//! Session decorator that compacts stored history with the OpenAI `responses.compact` API.
//! Mirrors `agents.memory.openai_responses_compaction_session`.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use super::{Session, SessionRunCompletion, SessionSettings};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;
use crate::models::OpenAIClient;
use crate::usage::Usage;

/// Number of compaction candidates that triggers compaction by default.
pub const DEFAULT_COMPACTION_THRESHOLD: usize = 10;

/// How a compaction request provides the conversation history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CompactionMode {
    /// Always send `previous_response_id`.
    PreviousResponseId,
    /// Always send the stored history as `input`.
    Input,
    /// Send `input` when the last response was not stored or has no id.
    #[default]
    Auto,
}

/// The mode a compaction request actually used, after resolving [`CompactionMode::Auto`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedCompactionMode {
    /// The request sent `previous_response_id`.
    PreviousResponseId,
    /// The request sent `input`.
    Input,
}

/// Arguments for [`OpenAIResponsesCompactionSession::run_compaction`].
#[derive(Debug, Clone, Default)]
pub struct OpenAIResponsesCompactionArgs {
    /// Id of the response the compaction continues from.
    pub response_id: Option<String>,
    /// The `store` model setting used for that response.
    pub store: Option<bool>,
    /// Override the session's compaction mode for this call.
    pub compaction_mode: Option<CompactionMode>,
    /// Compact even when the decision hook declines.
    pub force: bool,
}

/// Input of the compaction decision hook.
#[derive(Debug, Clone)]
pub struct CompactionContext {
    /// Id of the response the compaction would continue from.
    pub response_id: Option<String>,
    /// The mode the request would use.
    pub compaction_mode: ResolvedCompactionMode,
    /// Stored items eligible for compaction (no user messages, no compaction items).
    pub compaction_candidate_items: Vec<InputItem>,
    /// The complete stored history.
    pub session_items: Vec<InputItem>,
}

/// Decides whether a settled run should trigger compaction.
pub type ShouldTriggerCompaction = Arc<dyn Fn(&CompactionContext) -> bool + Send + Sync>;

/// True when `item` is a user message.
fn is_user_message(item: &InputItem) -> bool {
    match item.item_type() {
        Some("message") => item.role() == Some("user"),
        None => item.role() == Some("user") && item.as_value().get("content").is_some(),
        _ => false,
    }
}

/// Items eligible for compaction: everything but user messages and compaction items.
pub fn select_compaction_candidate_items(items: &[InputItem]) -> Vec<InputItem> {
    items
        .iter()
        .filter(|i| !is_user_message(i) && i.item_type() != Some("compaction"))
        .cloned()
        .collect()
}

/// The default decision: compact once at least [`DEFAULT_COMPACTION_THRESHOLD`] candidates exist.
pub fn default_should_trigger_compaction(context: &CompactionContext) -> bool {
    context.compaction_candidate_items.len() >= DEFAULT_COMPACTION_THRESHOLD
}

/// True when `model` follows OpenAI naming conventions (`gpt-*`, `o<digit>*`, `ft:gpt-*`).
pub fn is_openai_model_name(model: &str) -> bool {
    let trimmed = model.trim().rsplit('/').next().unwrap_or_default();
    if trimmed.is_empty() {
        return false;
    }
    let without_ft = trimmed.strip_prefix("ft:").unwrap_or(trimmed);
    let root = without_ft.split(':').next().unwrap_or_default();
    if root.starts_with("gpt-") {
        return true;
    }
    root.starts_with('o') && root[1..].starts_with(|c: char| c.is_ascii_digit())
}

/// Remove `id` from assistant messages when no reasoning item remains to pair it with.
fn strip_orphaned_assistant_ids(items: Vec<InputItem>) -> Vec<InputItem> {
    if items.iter().any(|i| i.item_type() == Some("reasoning")) {
        return items;
    }
    items
        .into_iter()
        .map(|mut i| {
            if i.role() == Some("assistant")
                && let Value::Object(m) = &mut i.0
            {
                m.remove("id");
            }
            i
        })
        .collect()
}

/// Rewrite a compacted `input_image` / `input_file` part into a replay-safe shape.
fn normalize_content_part(part: &Value) -> Value {
    let Some(obj) = part.as_object() else {
        return part.clone();
    };
    let kind = obj.get("type").and_then(Value::as_str).unwrap_or_default();
    let keep: &[&str] = match kind {
        "input_image" => &["image_url", "file_id", "detail"],
        "input_file" => &["file_data", "file_url", "file_id", "filename", "detail"],
        _ => return part.clone(),
    };
    let mut out = Map::new();
    out.insert("type".into(), json!(kind));
    for key in keep {
        if let Some(v) = obj.get(*key)
            && v.as_str().is_some_and(|s| !s.is_empty())
        {
            out.insert((*key).to_string(), v.clone());
        }
    }
    Value::Object(out)
}

/// Normalize the compaction output into items that can be replayed as input.
fn normalize_output_items(output: &[Value]) -> Vec<InputItem> {
    output
        .iter()
        .map(|item| {
            let is_user_message = item.get("type").and_then(Value::as_str) == Some("message")
                && item.get("role").and_then(Value::as_str) == Some("user");
            if !is_user_message {
                return InputItem(item.clone());
            }
            let mut out = item.clone();
            if let Some(Value::Array(parts)) = out.get("content") {
                let normalized: Vec<Value> = parts.iter().map(normalize_content_part).collect();
                out["content"] = Value::Array(normalized);
            }
            InputItem(out)
        })
        .collect()
}

#[derive(Debug, Default)]
struct CompactionState {
    response_id: Option<String>,
    deferred_response_id: Option<String>,
    last_unstored_response_id: Option<String>,
}

/// Wraps another session and compacts its history with `POST /responses/compact`.
///
/// After every run the runner calls [`Session::on_run_completed`]; this session then asks the
/// decision hook whether the stored history should be compacted and, if so, replaces it with
/// the compacted output. The usage of the compaction request is returned to the runner and
/// added to the run's usage.
///
/// # Differences from the Python SDK
///
/// The Python implementation can replace a bounded suffix of history atomically when the
/// wrapped store supports it, and matches stored items against the exact model exchange of the
/// last turn. This port always replaces the complete stored history (`clear_session` followed
/// by `add_items`, restoring the previous items when the replacement fails), which is the
/// behaviour Python uses for stores without atomic suffix replacement and for manual
/// `run_compaction` calls. `max_rollback_items` is not implemented.
pub struct OpenAIResponsesCompactionSession {
    session_id: String,
    underlying: Arc<dyn Session>,
    client: OpenAIClient,
    model: String,
    compaction_mode: CompactionMode,
    should_trigger: ShouldTriggerCompaction,
    state: Mutex<CompactionState>,
    mutation_lock: Mutex<()>,
}

impl fmt::Debug for OpenAIResponsesCompactionSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAIResponsesCompactionSession")
            .field("session_id", &self.session_id)
            .field("underlying", &self.underlying)
            .field("model", &self.model)
            .field("compaction_mode", &self.compaction_mode)
            .finish_non_exhaustive()
    }
}

impl OpenAIResponsesCompactionSession {
    /// Wrap `underlying`, compacting with `gpt-4.1` and a client built from the environment.
    pub fn new(session_id: impl Into<String>, underlying: Arc<dyn Session>) -> Self {
        OpenAIResponsesCompactionSession {
            session_id: session_id.into(),
            underlying,
            client: OpenAIClient::from_env(),
            model: "gpt-4.1".to_string(),
            compaction_mode: CompactionMode::Auto,
            should_trigger: Arc::new(default_should_trigger_compaction),
            state: Mutex::new(CompactionState::default()),
            mutation_lock: Mutex::new(()),
        }
    }

    /// Use a specific OpenAI client for the compaction requests.
    pub fn with_client(mut self, client: OpenAIClient) -> Self {
        self.client = client;
        self
    }

    /// Use another model for compaction. Must be an OpenAI model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Result<Self> {
        let model = model.into();
        if !is_openai_model_name(&model) {
            return Err(AgentsError::user(format!(
                "Unsupported model for OpenAI responses compaction: {model}"
            )));
        }
        self.model = model;
        Ok(self)
    }

    /// Set how compaction requests provide the conversation history.
    pub fn with_compaction_mode(mut self, mode: CompactionMode) -> Self {
        self.compaction_mode = mode;
        self
    }

    /// Replace the decision hook (default: [`default_should_trigger_compaction`]).
    pub fn with_should_trigger_compaction<F>(mut self, f: F) -> Self
    where
        F: Fn(&CompactionContext) -> bool + Send + Sync + 'static,
    {
        self.should_trigger = Arc::new(f);
        self
    }

    /// The wrapped session.
    pub fn underlying_session(&self) -> &Arc<dyn Session> {
        &self.underlying
    }

    /// Compact the stored history now, returning the usage of the compaction request.
    ///
    /// Returns a zero usage when the decision hook declines.
    pub async fn run_compaction(&self, args: OpenAIResponsesCompactionArgs) -> Result<Usage> {
        let _guard = self.mutation_lock.lock().await;
        self.run_compaction_locked(args).await
    }

    fn resolve_mode(
        &self,
        requested: Option<CompactionMode>,
        response_id: Option<&str>,
        store: Option<bool>,
        last_unstored: Option<&str>,
    ) -> ResolvedCompactionMode {
        let mode = requested.unwrap_or(self.compaction_mode);
        if mode == CompactionMode::Auto
            && store.is_none()
            && response_id.is_some()
            && response_id == last_unstored
        {
            return ResolvedCompactionMode::Input;
        }
        match mode {
            CompactionMode::PreviousResponseId => ResolvedCompactionMode::PreviousResponseId,
            CompactionMode::Input => ResolvedCompactionMode::Input,
            CompactionMode::Auto => {
                if store == Some(false) || response_id.is_none() {
                    ResolvedCompactionMode::Input
                } else {
                    ResolvedCompactionMode::PreviousResponseId
                }
            }
        }
    }

    async fn run_compaction_locked(&self, args: OpenAIResponsesCompactionArgs) -> Result<Usage> {
        let (response_id, resolved_mode) = {
            let mut state = self.state.lock().await;
            if let Some(id) = args.response_id.clone() {
                state.response_id = Some(id);
            }
            if let Some(store) = args.store {
                let current = state.response_id.clone();
                if !store && current.is_some() {
                    state.last_unstored_response_id = current;
                } else if store && state.response_id == state.last_unstored_response_id {
                    state.last_unstored_response_id = None;
                }
            }
            let mode = self.resolve_mode(
                args.compaction_mode,
                state.response_id.as_deref(),
                args.store,
                state.last_unstored_response_id.as_deref(),
            );
            (state.response_id.clone(), mode)
        };

        if resolved_mode == ResolvedCompactionMode::PreviousResponseId && response_id.is_none() {
            return Err(AgentsError::user(
                "OpenAIResponsesCompactionSession::run_compaction requires a response_id when \
                 using previous_response_id compaction",
            ));
        }

        let session_items = self.underlying.get_items(None).await?;
        let candidates = select_compaction_candidate_items(&session_items);
        let context = CompactionContext {
            response_id: response_id.clone(),
            compaction_mode: resolved_mode,
            compaction_candidate_items: candidates,
            session_items,
        };
        if !args.force && !(self.should_trigger)(&context) {
            log::debug!("skip: decision hook declined compaction for {response_id:?}");
            return Ok(Usage::default());
        }

        let mut body = json!({"model": self.model});
        match resolved_mode {
            ResolvedCompactionMode::PreviousResponseId => {
                body["previous_response_id"] = json!(response_id);
            }
            ResolvedCompactionMode::Input => {
                let input: Vec<&Value> = context
                    .session_items
                    .iter()
                    .map(InputItem::as_value)
                    .collect();
                body["input"] = json!(input);
            }
        }

        let (compacted, _) = self
            .client
            .post_json("responses/compact", &body, None, None)
            .await?;
        let usage = compacted
            .get("usage")
            .map(Usage::from_responses_usage)
            .unwrap_or_default();
        let output = compacted
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let output_items = strip_orphaned_assistant_ids(normalize_output_items(&output));

        self.replace_history(output_items, context.session_items)
            .await?;
        self.state.lock().await.deferred_response_id = None;
        Ok(usage)
    }

    /// Replace the stored history, restoring `previous` when the replacement fails.
    async fn replace_history(
        &self,
        output_items: Vec<InputItem>,
        previous: Vec<InputItem>,
    ) -> Result<()> {
        let replace = async {
            self.underlying.clear_session().await?;
            if !output_items.is_empty() {
                self.underlying.add_items(output_items).await?;
            }
            Ok::<(), AgentsError>(())
        };
        match replace.await {
            Ok(()) => Ok(()),
            Err(error) => {
                if let Err(restore_error) = self.restore(previous).await {
                    log::warn!(
                        "Failed to restore session history after compaction replacement failed: \
                         {restore_error}"
                    );
                } else {
                    log::warn!(
                        "Restored previous session history after compaction replacement failed: \
                         {error}"
                    );
                }
                Err(error)
            }
        }
    }

    async fn restore(&self, previous: Vec<InputItem>) -> Result<()> {
        self.underlying.clear_session().await?;
        if !previous.is_empty() {
            self.underlying.add_items(previous).await?;
        }
        Ok(())
    }

    /// Remember that `response_id` still owes a compaction once its tool outputs settle.
    async fn defer_compaction(&self, response_id: &str, store: Option<bool>) -> Result<()> {
        let state = self.state.lock().await;
        if state.deferred_response_id.is_some() {
            return Ok(());
        }
        let mode = self.resolve_mode(
            None,
            Some(response_id),
            store,
            state.last_unstored_response_id.as_deref(),
        );
        drop(state);
        let session_items = self.underlying.get_items(None).await?;
        let context = CompactionContext {
            response_id: Some(response_id.to_string()),
            compaction_mode: mode,
            compaction_candidate_items: select_compaction_candidate_items(&session_items),
            session_items,
        };
        if (self.should_trigger)(&context) {
            self.state.lock().await.deferred_response_id = Some(response_id.to_string());
        }
        Ok(())
    }
}

#[async_trait]
impl Session for OpenAIResponsesCompactionSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.underlying.session_settings()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.underlying.get_items(limit).await
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        self.underlying.add_items(items).await
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        let _guard = self.mutation_lock.lock().await;
        let popped = self.underlying.pop_item().await?;
        if popped.is_some() {
            let mut state = self.state.lock().await;
            *state = CompactionState::default();
        }
        Ok(popped)
    }

    async fn clear_session(&self) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        self.underlying.clear_session().await?;
        *self.state.lock().await = CompactionState::default();
        Ok(())
    }

    async fn on_run_completed(&self, info: &SessionRunCompletion) -> Result<Usage> {
        let Some(response_id) = info.response_id.clone() else {
            return Ok(Usage::default());
        };
        let _guard = self.mutation_lock.lock().await;
        if info.has_local_tool_outputs {
            self.defer_compaction(&response_id, info.store).await?;
            log::debug!("skip: deferring compaction for response {response_id} (tool outputs)");
            return Ok(Usage::default());
        }
        let force = self.state.lock().await.deferred_response_id.is_some();
        self.run_compaction_locked(OpenAIResponsesCompactionArgs {
            response_id: Some(response_id),
            store: info.store,
            compaction_mode: None,
            force,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_selection_skips_user_and_compaction_items() {
        let items = vec![
            InputItem::user("hi"),
            InputItem::assistant("hello"),
            InputItem(json!({"type": "compaction", "summary": "..."})),
            InputItem(json!({"type": "function_call", "name": "f", "arguments": "{}"})),
        ];
        let candidates = select_compaction_candidate_items(&items);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].role(), Some("assistant"));
    }

    #[test]
    fn model_names_follow_openai_conventions() {
        for ok in ["gpt-4.1", "o3", "ft:gpt-4.1:acme:proj:abc", "openai/gpt-5"] {
            assert!(is_openai_model_name(ok), "{ok}");
        }
        for bad in ["", "claude-3", "llama", "o"] {
            assert!(!is_openai_model_name(bad), "{bad}");
        }
    }

    #[test]
    fn orphaned_assistant_ids_are_stripped() {
        let items = vec![InputItem(
            json!({"id": "msg_1", "role": "assistant", "content": "x"}),
        )];
        let stripped = strip_orphaned_assistant_ids(items);
        assert!(stripped[0].as_value().get("id").is_none());

        let with_reasoning = vec![
            InputItem(json!({"type": "reasoning", "id": "rs_1"})),
            InputItem(json!({"id": "msg_1", "role": "assistant", "content": "x"})),
        ];
        let kept = strip_orphaned_assistant_ids(with_reasoning);
        assert!(kept[1].as_value().get("id").is_some());
    }

    #[test]
    fn output_user_messages_are_normalized_for_replay() {
        let output = vec![json!({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://x/y.png", "file_id": null},
                {"type": "input_text", "text": "hi"}
            ]
        })];
        let items = normalize_output_items(&output);
        let parts = items[0].as_value()["content"].as_array().unwrap();
        assert_eq!(
            parts[0],
            json!({"type": "input_image", "image_url": "https://x/y.png"})
        );
        assert_eq!(parts[1]["text"], "hi");
    }

    #[test]
    fn auto_mode_resolves_from_store_and_response_id() {
        let s = OpenAIResponsesCompactionSession::new(
            "s",
            Arc::new(super::super::InMemorySession::new("s")),
        );
        assert_eq!(
            s.resolve_mode(None, Some("resp_1"), None, None),
            ResolvedCompactionMode::PreviousResponseId
        );
        assert_eq!(
            s.resolve_mode(None, Some("resp_1"), Some(false), None),
            ResolvedCompactionMode::Input
        );
        assert_eq!(
            s.resolve_mode(None, None, None, None),
            ResolvedCompactionMode::Input
        );
        assert_eq!(
            s.resolve_mode(None, Some("resp_1"), None, Some("resp_1")),
            ResolvedCompactionMode::Input
        );
        assert_eq!(
            s.resolve_mode(
                Some(CompactionMode::PreviousResponseId),
                Some("resp_1"),
                Some(false),
                None
            ),
            ResolvedCompactionMode::PreviousResponseId
        );
    }
}
