//! Session decorator that compacts stored history with the OpenAI `responses.compact` API.
//! Mirrors `agents.memory.openai_responses_compaction_session`.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use super::{
    CompactionSnapshot, Session, SessionRunCompletion, SessionSettings, compaction_item_digest,
};
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
fn normalize_content_part(part: &Value) -> Result<Value> {
    let Some(obj) = part.as_object() else {
        return Ok(part.clone());
    };
    let non_empty = |key: &str| {
        obj.get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let kind = obj.get("type").and_then(Value::as_str).unwrap_or_default();
    // The Responses API accepts exactly one source per part, so the first source that is
    // present wins and the rest are dropped even when the stored item carried several.
    let (sources, extras): (&[&str], &[&str]) = match kind {
        "input_image" => (&["image_url", "file_id"], &["detail"]),
        "input_file" => (
            &["file_data", "file_url", "file_id"],
            &["filename", "detail"],
        ),
        _ => return Ok(part.clone()),
    };
    let mut out = Map::new();
    out.insert("type".into(), json!(kind));
    let source = sources
        .iter()
        .find_map(|key| non_empty(key).map(|v| ((*key).to_string(), v)));
    let Some((key, value)) = source else {
        return Err(AgentsError::user(format!(
            "Compaction {kind} item missing {}.",
            sources.join(" or ")
        )));
    };
    out.insert(key, json!(value));
    for key in extras {
        if let Some(v) = non_empty(key) {
            out.insert((*key).to_string(), json!(v));
        }
    }
    Ok(Value::Object(out))
}

/// The limit compaction reads history with: large enough to mean "everything", explicit
/// enough that a wrapped store's own `SessionSettings::limit` cannot truncate the read.
///
/// Compaction replaces the history it read. A truncated read would make the coverage check
/// pass against a partial view and then delete the items it never saw, so the full-history
/// path must never pass `None` here. Mirrors Python's `_ALL_SESSION_ITEMS_LIMIT`.
const ALL_SESSION_ITEMS_LIMIT: usize = 2_147_483_647;

/// The error raised for a wrapped store that owns its history remotely.
fn unwrappable_session_error() -> AgentsError {
    AgentsError::user(
        "OpenAIResponsesCompactionSession cannot wrap a session that manages its own history \
         on the server, such as OpenAIConversationsSession.",
    )
}

/// Normalize the compaction output into items that can be replayed as input.
fn normalize_output_items(output: &[Value]) -> Result<Vec<InputItem>> {
    output
        .iter()
        .map(|item| {
            let is_user_message = item.get("type").and_then(Value::as_str) == Some("message")
                && item.get("role").and_then(Value::as_str) == Some("user");
            if !is_user_message {
                return Ok(InputItem(item.clone()));
            }
            let mut out = item.clone();
            if let Some(Value::Array(parts)) = out.get("content") {
                let normalized: Vec<Value> = parts
                    .iter()
                    .map(normalize_content_part)
                    .collect::<Result<_>>()?;
                out["content"] = Value::Array(normalized);
            }
            Ok(InputItem(out))
        })
        .collect()
}

/// The index at which the longest suffix of `session_items` that occurs, in order, inside
/// `model_exchange` begins.
///
/// Walks both backwards, consuming exchange digests as it goes: items the exchange contains but
/// storage does not are skipped, while a stored item that is missing or out of order ends the
/// match. `session_items.len()` means nothing matched.
fn match_model_exchange_suffix(
    session_items: &[InputItem],
    model_exchange: &[String],
    ignore_ids_for_matching: bool,
) -> usize {
    let mut remaining = model_exchange.len();
    let mut matched = 0usize;
    for item in session_items.iter().rev() {
        let digest = if ignore_ids_for_matching {
            compaction_item_digest(&crate::run_internal::strip_item_id(item.clone()))
        } else {
            compaction_item_digest(item)
        };
        let mut found = false;
        while remaining > 0 {
            remaining -= 1;
            if model_exchange[remaining] == digest {
                found = true;
                break;
            }
        }
        if !found {
            break;
        }
        matched += 1;
    }
    session_items.len() - matched
}

/// The `call_id` of a tool or program call item.
fn tool_call_id(item: &InputItem) -> Option<&str> {
    let kind = item.item_type()?;
    if !kind.ends_with("_call") {
        return None;
    }
    item.as_value().get("call_id").and_then(Value::as_str)
}

/// The `call_id` of a tool call output item. Program outputs are checked separately.
fn tool_output_call_id(item: &InputItem) -> Option<&str> {
    let kind = item.item_type()?;
    if !kind.ends_with("_output") || kind == "program_output" {
        return None;
    }
    item.as_value().get("call_id").and_then(Value::as_str)
}

/// The `call_id` of the program that owns `item`, for a program-owned item.
///
/// Port of `_get_program_caller_id`.
fn program_caller_id(item: &InputItem) -> Option<&str> {
    let caller = item.as_value().get("caller")?;
    if caller.get("type").and_then(Value::as_str) != Some("program") {
        return None;
    }
    caller
        .get("caller_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// The `call_id` of an item of exactly `kind`.
fn call_id_of_type<'a>(item: &'a InputItem, kind: &str) -> Option<&'a str> {
    if item.item_type()? != kind {
        return None;
    }
    item.as_value().get("call_id").and_then(Value::as_str)
}

/// True when every tool call in `items` has its output and vice versa.
///
/// Compaction must not break a call/output pair apart: sending half a group to the compaction
/// API is rejected, and compacting a call away while its output is still pending would strand
/// the output. Hosted MCP approvals pair on the request id rather than a call id, so they are
/// checked separately.
fn has_complete_tool_groups(items: &[InputItem]) -> bool {
    let calls: BTreeSet<&str> = items.iter().filter_map(tool_call_id).collect();
    let outputs: BTreeSet<&str> = items.iter().filter_map(tool_output_call_id).collect();
    if calls != outputs {
        return false;
    }

    // Programs are a group of their own: `drop_orphan_function_calls` keeps a program that has
    // no output while items it owns are retained, and drops it — together with those items and
    // any reasoning item in front of it — once nothing is left. Either way the suffix would
    // come back changed, which is what Python refuses to compact.
    let programs: BTreeSet<&str> = items
        .iter()
        .filter_map(|i| call_id_of_type(i, "program"))
        .collect();
    let program_outputs: BTreeSet<&str> = items
        .iter()
        .filter_map(|i| call_id_of_type(i, "program_output"))
        .collect();
    if !program_outputs.is_subset(&programs) {
        return false;
    }
    if programs.iter().any(|call_id| {
        !program_outputs.contains(call_id)
            && !items.iter().any(|i| program_caller_id(i) == Some(*call_id))
    }) {
        return false;
    }
    // A program-owned item whose program is outside the suffix would be dropped with it.
    if items
        .iter()
        .filter_map(program_caller_id)
        .any(|call_id| !programs.contains(call_id))
    {
        return false;
    }
    let requests: BTreeSet<&str> = items
        .iter()
        .filter(|i| i.item_type() == Some("mcp_approval_request"))
        .filter_map(|i| i.as_value().get("id").and_then(Value::as_str))
        .collect();
    let responses: BTreeSet<&str> = items
        .iter()
        .filter(|i| i.item_type() == Some("mcp_approval_response"))
        .filter_map(|i| {
            i.as_value()
                .get("approval_request_id")
                .and_then(Value::as_str)
        })
        .collect();
    requests == responses
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
/// Python memoizes the candidate list it hands the decision hook and keeps it up to date
/// through its own mutations (`_ensure_compaction_candidates`, `openai_responses_compaction
/// _session.py` L814-850, maintained at L765-812). This port re-reads the store instead. The
/// cache is consulted only for an unlimited read through a non-encrypted store, and every
/// wrapper mutation either extends it with exactly what was written or clears it, so for
/// wrapper-only traffic it holds what a fresh read would return. It can only diverge when
/// another writer changes the wrapped store directly, and then it is stale — so re-reading is
/// never less accurate. Mutations through this session are serialized against compaction by
/// an internal lock, and the stored history is re-read after the compaction request so a
/// concurrent append by another writer makes the replacement be skipped rather than lost.
pub struct OpenAIResponsesCompactionSession {
    session_id: String,
    underlying: Arc<dyn Session>,
    client: OpenAIClient,
    model: String,
    compaction_mode: CompactionMode,
    should_trigger: ShouldTriggerCompaction,
    max_rollback_items: Option<usize>,
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
    /// Whether item ids must be ignored when matching stored history against the model
    /// exchange, inherited from the wrapped store. Port of `_ignore_ids_for_matching`.
    fn ignore_ids_for_matching(&self) -> bool {
        self.underlying.manages_history_remotely()
    }

    /// Fail when the wrapped store manages its own history on the provider's servers.
    fn ensure_wrappable(&self) -> Result<()> {
        if self.underlying.manages_history_remotely() {
            return Err(unwrappable_session_error());
        }
        Ok(())
    }

    /// Wrap `underlying`, compacting with `gpt-4.1` and a client built from the environment.
    ///
    /// Rejects a store that manages its own history on the provider's servers, such as
    /// [`crate::memory::OpenAIConversationsSession`]: compaction rewrites stored items, which
    /// such a store does not own. Port of Python's `__init__`, which raises for the same
    /// reason.
    pub fn new(session_id: impl Into<String>, underlying: Arc<dyn Session>) -> Result<Self> {
        if underlying.manages_history_remotely() {
            return Err(unwrappable_session_error());
        }
        Ok(OpenAIResponsesCompactionSession {
            session_id: session_id.into(),
            underlying,
            client: OpenAIClient::from_env(),
            model: "gpt-4.1".to_string(),
            compaction_mode: CompactionMode::Auto,
            should_trigger: Arc::new(default_should_trigger_compaction),
            max_rollback_items: None,
            state: Mutex::new(CompactionState::default()),
            mutation_lock: Mutex::new(()),
        })
    }

    /// Bound the size of the complete-history snapshots manual compaction and whole-history
    /// replacement take.
    ///
    /// A history longer than `max_rollback_items` fails the compaction before the API call and
    /// before any replacement, so the stored history is retained. This is a count of items, not
    /// of bytes, and it does not bound stored history, model input or
    /// [`SessionSettings::limit`]. Native automatic suffix replacement needs no full-history
    /// snapshot and is unaffected. Port of Python's `max_rollback_items`.
    pub fn with_max_rollback_items(mut self, max_rollback_items: usize) -> Result<Self> {
        if max_rollback_items == 0 {
            return Err(AgentsError::user(
                "max_rollback_items must be positive or None",
            ));
        }
        self.max_rollback_items = Some(max_rollback_items);
        Ok(self)
    }

    /// Read the complete stored history, refusing one longer than `max_rollback_items`.
    ///
    /// Reads one item beyond the budget so an oversized history is detected rather than
    /// silently truncated. Port of `_get_all_underlying_session_items`.
    async fn all_underlying_items(&self) -> Result<Vec<InputItem>> {
        let limit = match self.max_rollback_items {
            None => ALL_SESSION_ITEMS_LIMIT,
            Some(budget) => budget.saturating_add(1),
        };
        let items = self.underlying.get_items(Some(limit)).await?;
        if self
            .max_rollback_items
            .is_some_and(|budget| items.len() > budget)
        {
            return Err(AgentsError::user(
                "Compaction history exceeds max_rollback_items; history was retained",
            ));
        }
        Ok(items)
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
        self.run_compaction_inner(args, None).await
    }

    /// The compaction body.
    ///
    /// `model_exchange` is `Some` only on the automatic path, where it authorizes replacing a
    /// bounded suffix of history: the stored tail is matched against the items the last model
    /// exchange produced, and only that tail is compacted away. Manual calls pass `None` and
    /// replace the complete stored history.
    async fn run_compaction_inner(
        &self,
        args: OpenAIResponsesCompactionArgs,
        model_exchange: Option<&[String]>,
    ) -> Result<Usage> {
        let (response_id, mut resolved_mode) = {
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

        // A manual call always replaces the whole history, so its budget is checked up front,
        // before the compaction API is asked to do any work.
        let session_items = if model_exchange.is_none() {
            self.all_underlying_items().await?
        } else {
            self.underlying
                .get_items(Some(ALL_SESSION_ITEMS_LIMIT))
                .await?
        };
        let candidates = select_compaction_candidate_items(&session_items);
        let approved_mode = resolved_mode;
        let approved_items = session_items.clone();
        let context = CompactionContext {
            response_id: response_id.clone(),
            compaction_mode: resolved_mode,
            compaction_candidate_items: candidates,
            session_items,
        };
        if !args.force && !(self.should_trigger)(&context) {
            log::debug!(target: crate::logger::LOG_TARGET, "skip: decision hook declined compaction for {response_id:?}");
            return Ok(Usage::default());
        }

        // The automatic path asks the store for a bounded window it can replace atomically.
        // Without one (or on a manual call) the whole stored history is replaced instead.
        let mut snapshot: Option<CompactionSnapshot> = None;
        let mut suffix_start = 0usize;
        let mut items_to_compact = context.session_items.clone();

        if let Some(model_exchange) = model_exchange {
            let window = model_exchange.len().saturating_add(1);
            let Some(found) = self.underlying.compaction_snapshot(window).await? else {
                // A store without atomic suffix replacement still has to prove that the model
                // exchange covers its complete history before the whole of it is replaced.
                // Without this a store whose history is longer than the run loaded would have
                // the older part summarized away by a chain that never contained it.
                let bounded = self.underlying.get_items(Some(window)).await?;
                let complete = bounded.len() < window;
                if !complete
                    || match_model_exchange_suffix(
                        &bounded,
                        model_exchange,
                        self.ignore_ids_for_matching(),
                    ) > 0
                {
                    log::warn!(target: crate::logger::LOG_TARGET,
                        "Skipped automatic compaction because the stored history could not be \
                         fully matched to the latest model exchange and this store cannot \
                         replace a suffix; history was retained"
                    );
                    return Ok(Usage::default());
                }
                return self
                    .compact_whole_history(response_id, resolved_mode, context.session_items)
                    .await;
            };
            if found.items.len() != found.handles.len() {
                return Err(AgentsError::other(
                    "Session::compaction_snapshot returned a different number of handles than \
                     items",
                ));
            }

            suffix_start = match_model_exchange_suffix(
                &found.items,
                model_exchange,
                self.ignore_ids_for_matching(),
            );
            if suffix_start == found.items.len() {
                log::warn!(target: crate::logger::LOG_TARGET,
                    "Skipped automatic compaction because the stored history could not be \
                     matched to the latest model exchange; history was retained"
                );
                return Ok(Usage::default());
            }

            if !found.complete || suffix_start > 0 {
                // A response chain may summarize earlier items that will be retained, so only
                // an explicit input can scope the summary to exactly this suffix.
                if args.compaction_mode.unwrap_or(self.compaction_mode)
                    == CompactionMode::PreviousResponseId
                {
                    log::warn!(target: crate::logger::LOG_TARGET,
                        "Skipped automatic compaction because previous_response_id mode cannot \
                         summarize a partial history suffix"
                    );
                    return Ok(Usage::default());
                }
                resolved_mode = ResolvedCompactionMode::Input;

                // Start at a caller message so a retained reasoning item stays together with
                // the model item that must follow it.
                let Some(boundary) = found.items[suffix_start..].iter().position(is_user_message)
                else {
                    log::warn!(target: crate::logger::LOG_TARGET,
                        "Skipped automatic compaction because the replaceable suffix contains no \
                         user message to start from"
                    );
                    return Ok(Usage::default());
                };
                suffix_start += boundary;
            }

            items_to_compact = found.items[suffix_start..].to_vec();

            // Keep tool and approval groups intact, including calls whose outputs are still
            // pending: wait for them rather than compacting the call away.
            if !has_complete_tool_groups(&items_to_compact) {
                log::warn!(target: crate::logger::LOG_TARGET,
                    "Skipped automatic compaction because the replaceable suffix splits a tool \
                     call group; history was retained"
                );
                return Ok(Usage::default());
            }

            // The bounded reload can expose history the hook's first view did not include, so
            // the actual snapshot has to be approved too.
            if (items_to_compact != approved_items || resolved_mode != approved_mode)
                && !(self.should_trigger)(&CompactionContext {
                    response_id: response_id.clone(),
                    compaction_mode: resolved_mode,
                    compaction_candidate_items: select_compaction_candidate_items(
                        &items_to_compact,
                    ),
                    session_items: items_to_compact.clone(),
                })
            {
                log::debug!(target: crate::logger::LOG_TARGET, "skip: decision hook declined the reloaded compaction snapshot");
                return Ok(Usage::default());
            }

            snapshot = Some(found);
        }

        let Some(snapshot) = snapshot else {
            return self
                .compact_whole_history(response_id, resolved_mode, context.session_items)
                .await;
        };

        let (usage, output_items) = self
            .request_compaction(&response_id, resolved_mode, &items_to_compact)
            .await?;

        let replaced = self
            .underlying
            .replace_compaction_suffix(&snapshot.handles[suffix_start..], output_items)
            .await?;
        if !replaced {
            log::warn!(target: crate::logger::LOG_TARGET,
                "Skipped compaction replacement because the stored suffix changed while the \
                 compaction request was in flight; history was retained"
            );
            return Ok(usage);
        }
        self.state.lock().await.deferred_response_id = None;
        Ok(usage)
    }

    /// Replace the complete stored history, for manual calls and stores without an atomic
    /// suffix replacement.
    async fn compact_whole_history(
        &self,
        response_id: Option<String>,
        resolved_mode: ResolvedCompactionMode,
        session_items: Vec<InputItem>,
    ) -> Result<Usage> {
        // Check the rollback budget before the request, so an oversized history costs nothing.
        // The automatic path reaches here only when the store has no atomic suffix
        // replacement, which is exactly when Python checks it too.
        if self.max_rollback_items.is_some() {
            self.all_underlying_items().await?;
        }
        let (usage, output_items) = self
            .request_compaction(&response_id, resolved_mode, &session_items)
            .await?;

        // Re-read after the request: another writer sharing this store may have appended while
        // the compaction API call was in flight, and replacing the whole history would drop
        // those items. This also refreshes the rollback snapshot, so a failed replacement
        // cannot revive items that expired meanwhile.
        let current_items = self.all_underlying_items().await?;
        if current_items != session_items {
            log::warn!(target: crate::logger::LOG_TARGET,
                "Skipped compaction replacement because the stored history changed while the \
                 compaction request was in flight; history was retained"
            );
            return Ok(usage);
        }

        self.replace_history(output_items, current_items).await?;
        self.state.lock().await.deferred_response_id = None;
        Ok(usage)
    }

    /// Issue the `responses/compact` request and normalize its output.
    async fn request_compaction(
        &self,
        response_id: &Option<String>,
        resolved_mode: ResolvedCompactionMode,
        items: &[InputItem],
    ) -> Result<(Usage, Vec<InputItem>)> {
        let mut body = json!({"model": self.model});
        match resolved_mode {
            ResolvedCompactionMode::PreviousResponseId => {
                body["previous_response_id"] = json!(response_id);
            }
            ResolvedCompactionMode::Input => {
                let input: Vec<&Value> = items.iter().map(InputItem::as_value).collect();
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
        Ok((
            usage,
            strip_orphaned_assistant_ids(normalize_output_items(&output)?),
        ))
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
                    log::warn!(target: crate::logger::LOG_TARGET,
                        "Failed to restore session history after compaction replacement failed: \
                         {restore_error}"
                    );
                } else {
                    log::warn!(target: crate::logger::LOG_TARGET,
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
        let session_items = self
            .underlying
            .get_items(Some(ALL_SESSION_ITEMS_LIMIT))
            .await?;
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

    /// Automatic suffix replacement needs the exchange, so the runner records it.
    fn wants_model_exchange(&self) -> bool {
        true
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        self.ensure_wrappable()?;
        self.underlying.get_items(limit).await
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        self.ensure_wrappable()?;
        let _guard = self.mutation_lock.lock().await;
        self.underlying.add_items(items).await
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.ensure_wrappable()?;
        let _guard = self.mutation_lock.lock().await;
        let popped = self.underlying.pop_item().await?;
        if popped.is_some() {
            let mut state = self.state.lock().await;
            *state = CompactionState::default();
        }
        Ok(popped)
    }

    async fn clear_session(&self) -> Result<()> {
        self.ensure_wrappable()?;
        let _guard = self.mutation_lock.lock().await;
        self.underlying.clear_session().await?;
        *self.state.lock().await = CompactionState::default();
        Ok(())
    }

    async fn on_run_completed(&self, info: &SessionRunCompletion) -> Result<Usage> {
        self.ensure_wrappable()?;
        let Some(response_id) = info.response_id.clone() else {
            return Ok(Usage::default());
        };
        let _guard = self.mutation_lock.lock().await;
        if info.has_local_tool_outputs {
            self.defer_compaction(&response_id, info.store).await?;
            log::debug!(target: crate::logger::LOG_TARGET, "skip: deferring compaction for response {response_id} (tool outputs)");
            return Ok(Usage::default());
        }
        let force = self.state.lock().await.deferred_response_id.is_some();
        // An empty exchange means the runner recorded none, so there is nothing to authorize a
        // suffix replacement; fall back to replacing the whole history.
        let model_exchange = (!info.model_exchange.is_empty()).then_some(&info.model_exchange[..]);
        self.run_compaction_inner(
            OpenAIResponsesCompactionArgs {
                response_id: Some(response_id),
                store: info.store,
                compaction_mode: None,
                force,
            },
            model_exchange,
        )
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
        let items = normalize_output_items(&output).unwrap();
        let parts = items[0].as_value()["content"].as_array().unwrap();
        assert_eq!(
            parts[0],
            json!({"type": "input_image", "image_url": "https://x/y.png"})
        );
        assert_eq!(parts[1]["text"], "hi");
    }

    #[test]
    fn content_sources_are_priority_exclusive() {
        // Both sources set: only the higher-priority one survives.
        let both = json!({
            "type": "input_file",
            "file_data": "ZGF0YQ==",
            "file_url": "https://x/y.pdf",
            "file_id": "file_1",
            "filename": "y.pdf"
        });
        assert_eq!(
            normalize_content_part(&both).unwrap(),
            json!({"type": "input_file", "file_data": "ZGF0YQ==", "filename": "y.pdf"})
        );
        // No source at all is an error rather than a bare, unusable part.
        let empty = json!({"type": "input_image", "detail": "high"});
        let err = normalize_content_part(&empty).unwrap_err().to_string();
        assert!(err.contains("image_url or file_id"), "{err}");
    }

    fn exchange(items: &[InputItem]) -> Vec<String> {
        items.iter().map(compaction_item_digest).collect()
    }

    #[test]
    fn suffix_matching_finds_the_longest_tail_present_in_the_exchange() {
        let stored = vec![
            InputItem::user("old question"),
            InputItem::assistant("old answer"),
            InputItem::user("new question"),
            InputItem::assistant("new answer"),
        ];

        // The whole history was sent to the model, so everything is replaceable.
        assert_eq!(
            match_model_exchange_suffix(&stored, &exchange(&stored), false),
            0
        );

        // Only the last turn was part of the exchange: the older turn is retained.
        assert_eq!(
            match_model_exchange_suffix(&stored, &exchange(&stored[2..]), false),
            2
        );

        // Items the exchange holds but storage does not are skipped over.
        let mut padded = exchange(&stored[2..]);
        padded.insert(1, compaction_item_digest(&InputItem::assistant("dropped")));
        assert_eq!(match_model_exchange_suffix(&stored, &padded, false), 2);

        // Nothing in common means nothing is replaceable.
        assert_eq!(
            match_model_exchange_suffix(&stored, &exchange(&[InputItem::user("unrelated")]), false),
            stored.len()
        );
        assert_eq!(
            match_model_exchange_suffix(&stored, &[], false),
            stored.len()
        );

        // A reordered tail is rejected rather than matched out of order.
        let reordered = exchange(&[stored[3].clone(), stored[2].clone()]);
        assert_eq!(match_model_exchange_suffix(&stored, &reordered, false), 3);
    }

    #[test]
    fn digests_normalize_cosmetic_differences() {
        // Stored history keeps the API's content list and status; a replayed item may not.
        let stored = InputItem(json!({
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi"}],
        }));
        assert_eq!(
            compaction_item_digest(&stored),
            compaction_item_digest(&InputItem::assistant("hi"))
        );
        // A non-terminal status is a real difference and is kept.
        let in_progress = InputItem(json!({
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": "hi"}],
        }));
        assert_ne!(
            compaction_item_digest(&in_progress),
            compaction_item_digest(&stored)
        );
        // `created_by` is Conversations API bookkeeping, never part of a replayed item.
        let with_created_by =
            InputItem(json!({"role": "user", "content": "hi", "created_by": "x"}));
        assert_eq!(
            compaction_item_digest(&with_created_by),
            compaction_item_digest(&InputItem::user("hi"))
        );
    }

    #[test]
    fn program_groups_must_survive_orphan_pruning() {
        let program = InputItem(json!({"type": "program", "call_id": "p1"}));
        let output = InputItem(json!({"type": "program_output", "call_id": "p1"}));
        let owned = InputItem(json!({
            "type": "function_call",
            "call_id": "c1",
            "name": "f",
            "caller": {"type": "program", "caller_id": "p1"},
        }));
        let owned_output = InputItem::function_call_output("c1", "done");

        // A program with its output is a complete group.
        assert!(has_complete_tool_groups(&[program.clone(), output.clone()]));
        // So is a program still running whose owned items are present: Python keeps it.
        assert!(has_complete_tool_groups(&[
            program.clone(),
            owned.clone(),
            owned_output.clone()
        ]));
        // An orphan program would be pruned along with the reasoning before it, so the suffix
        // is refused rather than compacted.
        assert!(!has_complete_tool_groups(std::slice::from_ref(&program)));
        // A program output without its program, and an owned item without its program, would
        // both be pruned too.
        assert!(!has_complete_tool_groups(&[output]));
        assert!(!has_complete_tool_groups(&[owned, owned_output]));
    }

    #[test]
    fn tool_groups_must_be_complete() {
        let call = InputItem(json!({"type": "function_call", "call_id": "c1", "name": "f"}));
        let output = InputItem::function_call_output("c1", "done");
        assert!(has_complete_tool_groups(&[call.clone(), output.clone()]));
        assert!(has_complete_tool_groups(&[InputItem::user("hi")]));
        // A call still waiting for its output must not be compacted away.
        assert!(!has_complete_tool_groups(std::slice::from_ref(&call)));
        // Nor may an output be sent without the call it answers.
        assert!(!has_complete_tool_groups(&[output]));

        // Hosted MCP approvals pair on the request id instead of a call id.
        let request = InputItem(json!({"type": "mcp_approval_request", "id": "r1"}));
        let response =
            InputItem(json!({"type": "mcp_approval_response", "approval_request_id": "r1"}));
        assert!(has_complete_tool_groups(&[
            request.clone(),
            response.clone()
        ]));
        assert!(!has_complete_tool_groups(std::slice::from_ref(&request)));
        assert!(!has_complete_tool_groups(&[response]));
    }

    #[test]
    fn auto_mode_resolves_from_store_and_response_id() {
        let s = OpenAIResponsesCompactionSession::new(
            "s",
            Arc::new(super::super::InMemorySession::new("s")),
        )
        .unwrap();
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
