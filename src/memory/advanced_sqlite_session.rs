//! SQLite session with conversation branching, turn tracking and usage analytics
//! (feature `sqlite`). Mirrors `agents.extensions.memory.advanced_sqlite_session`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

use super::sqlite_session::{SqliteSession, snapshot_from_rows, window_limit};
use super::{CompactionSnapshot, Session, SessionSettings, SnapshotHandle};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;
use crate::usage::Usage;

/// Name of the branch a session starts on.
pub const MAIN_BRANCH: &str = "main";

fn sql_err(e: rusqlite::Error) -> AgentsError {
    AgentsError::other(format!("SQLite error: {e}"))
}

/// One branch of a session's conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchInfo {
    /// The branch identifier.
    pub branch_id: String,
    /// Number of messages on the branch.
    pub message_count: u64,
    /// Number of user turns on the branch.
    pub user_turns: u64,
    /// Whether this is the branch the session currently reads and writes.
    pub is_current: bool,
    /// When the branch's first structure row was written.
    pub created_at: Option<String>,
}

/// One user turn, for browsing a conversation and picking a branch point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationTurn {
    /// The branch-local turn number.
    pub turn: i64,
    /// The user message content, truncated to 100 characters for display.
    pub content: String,
    /// The untruncated user message content.
    pub full_content: String,
    /// When the turn was recorded.
    pub timestamp: Option<String>,
    /// Whether a branch can be created from this turn. Every user turn is a valid branch point.
    pub can_branch: bool,
}

/// Truncate `text` to `max_length` characters, marking a cut with `...`.
fn content_preview(text: &str, max_length: usize) -> String {
    if text.chars().count() <= max_length {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_length).collect();
    format!("{cut}...")
}

/// One message of a branch, as [`AdvancedSqliteSession::get_turn_structure`] reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnMessage {
    /// The structural message type (`user`, `assistant`, `tool_call`, ...).
    pub message_type: String,
    /// The tool name, for tool calls and outputs.
    pub tool_name: Option<String>,
}

/// How often a tool was used in one turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolUsage {
    /// The tool name, or the call type when the item carries no name.
    pub tool_name: String,
    /// Number of calls in that turn.
    pub count: u64,
    /// The turn the calls belong to.
    pub user_turn_number: i64,
}

/// Usage recorded for one turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnUsage {
    /// The turn the usage belongs to.
    pub user_turn_number: i64,
    /// Model requests made in that turn.
    pub requests: u64,
    /// Input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Breakdown of the input tokens (cached and cache-write tokens).
    pub input_tokens_details: crate::usage::InputTokensDetails,
    /// Breakdown of the output tokens (reasoning tokens).
    pub output_tokens_details: crate::usage::OutputTokensDetails,
}

/// Usage aggregated over a session or one of its branches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionUsage {
    /// Model requests.
    pub requests: u64,
    /// Input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Number of turns with recorded usage.
    pub total_turns: u64,
}

/// Aggregated usage columns as SQLite returns them: `SUM` is `NULL` over an empty set.
type UsageTotals = (Option<i64>, Option<i64>, Option<i64>, Option<i64>, i64);

/// Everything a blocking database closure needs about the session.
#[derive(Debug, Clone)]
struct Ctx {
    session_id: String,
    sessions_table: String,
    messages_table: String,
    branch: Arc<Mutex<String>>,
}

impl Ctx {
    fn branch_id(&self) -> String {
        self.branch.lock().expect("branch poisoned").clone()
    }
}

/// The message type recorded for an item in `message_structure`.
fn classify_message_type(item: &InputItem) -> String {
    match item.role() {
        Some("user") => return "user".to_string(),
        Some("assistant") => return "assistant".to_string(),
        _ => {}
    }
    match item.item_type() {
        Some(t) => t.to_string(),
        None => "other".to_string(),
    }
}

/// The tool a call item targets, if any.
fn extract_tool_name(item: &InputItem) -> Option<String> {
    let value = item.as_value();
    let item_type = item.item_type();
    let name = value.get("name").and_then(Value::as_str);
    match item_type {
        Some("mcp_call" | "mcp_approval_request") => {
            let server = value.get("server_label").and_then(Value::as_str);
            match (server, name) {
                (Some(s), Some(n)) => Some(format!("{s}.{n}")),
                (Some(s), None) => Some(s.to_string()),
                (None, Some(n)) => Some(n.to_string()),
                (None, None) => None,
            }
        }
        Some("tool_search_call" | "tool_search_output") => Some("tool_search".to_string()),
        Some(
            t @ ("computer_call"
            | "file_search_call"
            | "web_search_call"
            | "code_interpreter_call"
            | "shell_call"
            | "apply_patch_call"),
        ) => Some(t.to_string()),
        _ => name.map(str::to_string),
    }
}

/// True when the item is a user message.
fn is_user_message(item: &InputItem) -> bool {
    item.role() == Some("user")
}

/// SQLite session that also records conversation structure, turns and usage.
///
/// On top of the [`SqliteSession`] schema this adds two tables with the same shape as the
/// Python SDK:
///
/// - `message_structure`: one row per stored item with its branch, classified type, global
///   sequence number, global and branch-local turn numbers, and tool name.
/// - `turn_usage`: token usage per `(session_id, branch_id, user_turn_number)`.
///
/// History reads and writes are scoped to the current branch, so branching off an earlier turn
/// forks the conversation while the underlying message rows stay shared.
///
/// # Differences from the Python SDK
///
/// The table-ownership claims, cross-process branch-id reservations and clear-generation
/// bookkeeping of the Python implementation are not ported: this session assumes it is the only
/// writer of the structure tables for its database. `store_run_usage` is called explicitly with
/// a run's [`Usage`] rather than hooked into the runner.
#[derive(Debug, Clone)]
pub struct AdvancedSqliteSession {
    base: SqliteSession,
    branch: Arc<Mutex<String>>,
}

impl AdvancedSqliteSession {
    /// An in-memory session (lost when the last clone is dropped).
    pub fn in_memory(session_id: impl Into<String>) -> Result<Self> {
        Self::from_base(SqliteSession::in_memory(session_id)?)
    }

    /// A file-backed session.
    pub fn open(
        session_id: impl Into<String>,
        db_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        Self::from_base(SqliteSession::open(session_id, db_path)?)
    }

    /// Extend an existing [`SqliteSession`] with the structure and usage tables.
    pub fn from_session(base: SqliteSession) -> Result<Self> {
        Self::from_base(base)
    }

    fn from_base(base: SqliteSession) -> Result<Self> {
        let session = AdvancedSqliteSession {
            base,
            branch: Arc::new(Mutex::new(MAIN_BRANCH.to_string())),
        };
        session.init_structure_tables()?;
        Ok(session)
    }

    /// Set default session settings.
    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.base = self.base.with_settings(settings);
        self
    }

    fn ctx(&self) -> Ctx {
        Ctx {
            session_id: self.base.session_id().to_string(),
            sessions_table: self.base.sessions_table().to_string(),
            messages_table: self.base.messages_table().to_string(),
            branch: self.branch.clone(),
        }
    }

    fn init_structure_tables(&self) -> Result<()> {
        let ctx = self.ctx();
        let conn = self.base.connection().lock().expect("sqlite poisoned");
        conn.execute_batch(&format!(
            "CREATE TABLE IF NOT EXISTS message_structure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                branch_id TEXT NOT NULL DEFAULT 'main',
                message_type TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                user_turn_number INTEGER,
                branch_turn_number INTEGER,
                tool_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES {s}(session_id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES {m}(id) ON DELETE CASCADE
             );
             CREATE TABLE IF NOT EXISTS turn_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                branch_id TEXT NOT NULL DEFAULT 'main',
                user_turn_number INTEGER NOT NULL,
                requests INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                input_tokens_details JSON,
                output_tokens_details JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES {s}(session_id) ON DELETE CASCADE,
                UNIQUE(session_id, branch_id, user_turn_number)
             );
             CREATE INDEX IF NOT EXISTS idx_structure_session_seq
                ON message_structure(session_id, sequence_number);
             CREATE INDEX IF NOT EXISTS idx_structure_branch
                ON message_structure(session_id, branch_id);
             CREATE INDEX IF NOT EXISTS idx_structure_turn
                ON message_structure(session_id, branch_id, user_turn_number);
             CREATE INDEX IF NOT EXISTS idx_structure_branch_seq
                ON message_structure(session_id, branch_id, sequence_number);
             CREATE INDEX IF NOT EXISTS idx_turn_usage_session_turn
                ON turn_usage(session_id, branch_id, user_turn_number);",
            s = ctx.sessions_table,
            m = ctx.messages_table
        ))
        .map_err(sql_err)
    }

    async fn blocking<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&Connection, &Ctx) -> Result<T> + Send + 'static,
    {
        let conn = self.base.connection().clone();
        let ctx = self.ctx();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("sqlite poisoned");
            f(&conn, &ctx)
        })
        .await
        .map_err(|e| AgentsError::other(format!("SQLite task failed: {e}")))?
    }

    /// The branch the session currently reads and writes.
    pub fn current_branch(&self) -> String {
        self.branch.lock().expect("branch poisoned").clone()
    }

    /// Switch to an existing branch.
    pub async fn switch_to_branch(&self, branch_id: impl Into<String>) -> Result<()> {
        let branch_id = branch_id.into();
        let target = branch_id.clone();
        let exists = self
            .blocking(move |conn, ctx| {
                let count: i64 = conn
                    .query_row(
                        "SELECT COUNT(*) FROM message_structure WHERE session_id = ? AND branch_id = ?",
                        params![ctx.session_id, target],
                        |r| r.get(0),
                    )
                    .map_err(sql_err)?;
                Ok(count > 0)
            })
            .await?;
        if !exists && branch_id != MAIN_BRANCH {
            return Err(AgentsError::user(format!(
                "Branch '{branch_id}' does not exist in this session"
            )));
        }
        *self.branch.lock().expect("branch poisoned") = branch_id;
        Ok(())
    }

    /// All branches of this session, oldest first.
    pub async fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        let current = self.current_branch();
        self.blocking(move |conn, ctx| {
            let mut stmt = conn
                .prepare(
                    "SELECT branch_id, COUNT(*),
                            COUNT(CASE WHEN message_type = 'user' THEN 1 END),
                            MIN(created_at)
                     FROM message_structure WHERE session_id = ?
                     GROUP BY branch_id ORDER BY MIN(created_at)",
                )
                .map_err(sql_err)?;
            let rows = stmt
                .query_map(params![ctx.session_id], |r| {
                    Ok(BranchInfo {
                        branch_id: r.get(0)?,
                        message_count: r.get::<_, i64>(1)? as u64,
                        user_turns: r.get::<_, i64>(2)? as u64,
                        is_current: false,
                        created_at: r.get(3)?,
                    })
                })
                .map_err(sql_err)?;
            let mut branches: Vec<BranchInfo> = rows.filter_map(std::result::Result::ok).collect();
            for b in &mut branches {
                b.is_current = b.branch_id == current;
            }
            Ok(branches)
        })
        .await
    }

    /// Fork the conversation before `turn_number` into a new branch and switch to it.
    ///
    /// `turn_number` is a branch-local turn number that must hold a user message on the current
    /// branch. Items of earlier turns are shared with the source branch: only their structure
    /// rows are copied, not the stored messages.
    pub async fn create_branch_from_turn(
        &self,
        turn_number: i64,
        branch_name: Option<&str>,
    ) -> Result<String> {
        let requested = branch_name.map(str::to_string);
        let branch_id = self
            .blocking(move |conn, ctx| {
                let source = ctx.branch_id();
                let tx = conn.unchecked_transaction().map_err(sql_err)?;
                let has_user_message: i64 = tx
                    .query_row(
                        "SELECT COUNT(*) FROM message_structure
                         WHERE session_id = ? AND branch_id = ? AND branch_turn_number = ?
                           AND message_type = 'user'",
                        params![ctx.session_id, source, turn_number],
                        |r| r.get(0),
                    )
                    .map_err(sql_err)?;
                if has_user_message == 0 {
                    return Err(AgentsError::user(format!(
                        "Turn {turn_number} does not contain a user message in branch '{source}'"
                    )));
                }
                let branch_count: i64 = tx
                    .query_row(
                        "SELECT COUNT(DISTINCT branch_id) FROM message_structure WHERE session_id = ?",
                        params![ctx.session_id],
                        |r| r.get(0),
                    )
                    .map_err(sql_err)?;
                let branch_id = requested.unwrap_or_else(|| {
                    format!("branch_{}_from_turn_{turn_number}", branch_count + 1)
                });
                let taken: i64 = tx
                    .query_row(
                        "SELECT COUNT(*) FROM message_structure WHERE session_id = ? AND branch_id = ?",
                        params![ctx.session_id, branch_id],
                        |r| r.get(0),
                    )
                    .map_err(sql_err)?;
                if taken > 0 {
                    return Err(AgentsError::user(format!(
                        "Branch '{branch_id}' already exists in this session"
                    )));
                }
                let seq_start: i64 = tx
                    .query_row(
                        "SELECT COALESCE(MAX(sequence_number), 0) FROM message_structure
                         WHERE session_id = ?",
                        params![ctx.session_id],
                        |r| r.get(0),
                    )
                    .map_err(sql_err)?;
                tx.execute(
                    "INSERT INTO message_structure
                        (session_id, message_id, branch_id, message_type, sequence_number,
                         user_turn_number, branch_turn_number, tool_name)
                     SELECT session_id, message_id, ?, message_type,
                            ? + ROW_NUMBER() OVER (ORDER BY sequence_number),
                            user_turn_number, branch_turn_number, tool_name
                     FROM message_structure
                     WHERE session_id = ? AND branch_id = ? AND branch_turn_number < ?",
                    params![branch_id, seq_start, ctx.session_id, source, turn_number],
                )
                .map_err(sql_err)?;
                tx.commit().map_err(sql_err)?;
                Ok(branch_id)
            })
            .await?;
        *self.branch.lock().expect("branch poisoned") = branch_id.clone();
        Ok(branch_id)
    }

    /// Fork the conversation before the first turn whose user message contains `search`.
    pub async fn create_branch_from_content(
        &self,
        search: &str,
        branch_name: Option<&str>,
    ) -> Result<String> {
        let turns = self.get_conversation_turns(None).await?;
        let needle = search.to_lowercase();
        let turn = turns
            .iter()
            .find(|t| t.content.to_lowercase().contains(&needle))
            .ok_or_else(|| {
                AgentsError::user(format!("No user turn contains the text {search:?}"))
            })?;
        self.create_branch_from_turn(turn.turn, branch_name).await
    }

    /// Delete a branch and the messages only it referenced.
    ///
    /// The `main` branch cannot be deleted. Deleting the current branch requires `force` and
    /// switches the session back to `main`.
    pub async fn delete_branch(&self, branch_id: &str, force: bool) -> Result<()> {
        if branch_id == MAIN_BRANCH {
            return Err(AgentsError::user("Cannot delete the 'main' branch"));
        }
        let is_current = self.current_branch() == branch_id;
        if is_current && !force {
            return Err(AgentsError::user(format!(
                "Branch '{branch_id}' is the current branch; pass force to delete it"
            )));
        }
        let target = branch_id.to_string();
        self.blocking(move |conn, ctx| {
            let tx = conn.unchecked_transaction().map_err(sql_err)?;
            tx.execute(
                "DELETE FROM message_structure WHERE session_id = ? AND branch_id = ?",
                params![ctx.session_id, target],
            )
            .map_err(sql_err)?;
            tx.execute(
                "DELETE FROM turn_usage WHERE session_id = ? AND branch_id = ?",
                params![ctx.session_id, target],
            )
            .map_err(sql_err)?;
            tx.execute(
                &format!(
                    "DELETE FROM {m} WHERE session_id = ?
                     AND id NOT IN (SELECT message_id FROM message_structure WHERE session_id = ?)",
                    m = ctx.messages_table
                ),
                params![ctx.session_id, ctx.session_id],
            )
            .map_err(sql_err)?;
            tx.commit().map_err(sql_err)
        })
        .await?;
        if is_current {
            *self.branch.lock().expect("branch poisoned") = MAIN_BRANCH.to_string();
        }
        Ok(())
    }

    /// The user turns of a branch (the current one when `branch_id` is `None`).
    pub async fn get_conversation_turns(
        &self,
        branch_id: Option<&str>,
    ) -> Result<Vec<ConversationTurn>> {
        let branch = branch_id.map(str::to_string);
        self.blocking(move |conn, ctx| {
            let branch = branch.unwrap_or_else(|| ctx.branch_id());
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT s.branch_turn_number, m.message_data, s.created_at
                     FROM message_structure s
                     JOIN {m} m ON m.id = s.message_id
                     WHERE s.session_id = ? AND s.branch_id = ? AND s.message_type = 'user'
                     ORDER BY s.branch_turn_number",
                    m = ctx.messages_table
                ))
                .map_err(sql_err)?;
            let rows = stmt
                .query_map(params![ctx.session_id, branch], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                    ))
                })
                .map_err(sql_err)?;
            Ok(rows
                .filter_map(std::result::Result::ok)
                .map(|(turn, data, timestamp)| {
                    let full_content = serde_json::from_str::<Value>(&data)
                        .ok()
                        .map(InputItem)
                        .and_then(|i| i.text_content())
                        .unwrap_or_default();
                    ConversationTurn {
                        turn,
                        content: content_preview(&full_content, 100),
                        full_content,
                        timestamp,
                        can_branch: true,
                    }
                })
                .collect())
        })
        .await
    }

    /// The items of specific turns on a branch, in conversation order.
    pub async fn get_conversation_by_turns(
        &self,
        turns: Vec<i64>,
        branch_id: Option<&str>,
    ) -> Result<Vec<InputItem>> {
        let branch = branch_id.map(str::to_string);
        self.blocking(move |conn, ctx| {
            let branch = branch.unwrap_or_else(|| ctx.branch_id());
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT m.message_data, s.branch_turn_number
                     FROM message_structure s
                     JOIN {m} m ON m.id = s.message_id
                     WHERE s.session_id = ? AND s.branch_id = ?
                     ORDER BY s.sequence_number",
                    m = ctx.messages_table
                ))
                .map_err(sql_err)?;
            let rows = stmt
                .query_map(params![ctx.session_id, branch], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
                })
                .map_err(sql_err)?;
            Ok(rows
                .filter_map(std::result::Result::ok)
                .filter(|(_, turn)| turns.contains(turn))
                .filter_map(|(data, _)| serde_json::from_str(&data).ok().map(InputItem))
                .collect())
        })
        .await
    }

    /// The branch's messages grouped by the user turn they belong to, in conversation order.
    ///
    /// This is the structure view Python's `get_conversation_by_turns` returns: message types
    /// and tool names rather than the items themselves. Use
    /// [`AdvancedSqliteSession::get_conversation_by_turns`] for the items.
    pub async fn get_turn_structure(
        &self,
        branch_id: Option<&str>,
    ) -> Result<BTreeMap<i64, Vec<TurnMessage>>> {
        let branch = branch_id.map(str::to_string);
        self.blocking(move |conn, ctx| {
            let branch = branch.unwrap_or_else(|| ctx.branch_id());
            let mut stmt = conn
                .prepare(
                    "SELECT user_turn_number, message_type, tool_name
                     FROM message_structure
                     WHERE session_id = ? AND branch_id = ?
                     ORDER BY sequence_number",
                )
                .map_err(sql_err)?;
            let rows = stmt
                .query_map(params![ctx.session_id, branch], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                    ))
                })
                .map_err(sql_err)?;
            let mut out: BTreeMap<i64, Vec<TurnMessage>> = BTreeMap::new();
            for (turn, message_type, tool_name) in rows.filter_map(std::result::Result::ok) {
                out.entry(turn).or_default().push(TurnMessage {
                    message_type,
                    tool_name,
                });
            }
            Ok(out)
        })
        .await
    }

    /// Turns on a branch whose user message contains `search`.
    pub async fn find_turns_by_content(
        &self,
        search: &str,
        branch_id: Option<&str>,
    ) -> Result<Vec<ConversationTurn>> {
        let needle = search.to_lowercase();
        let turns = self.get_conversation_turns(branch_id).await?;
        Ok(turns
            .into_iter()
            .filter(|t| t.full_content.to_lowercase().contains(&needle))
            .collect())
    }

    /// Tool calls per turn on a branch.
    pub async fn get_tool_usage(&self, branch_id: Option<&str>) -> Result<Vec<ToolUsage>> {
        let branch = branch_id.map(str::to_string);
        self.blocking(move |conn, ctx| {
            let branch = branch.unwrap_or_else(|| ctx.branch_id());
            let mut stmt = conn
                .prepare(
                    "SELECT COALESCE(tool_name, message_type) AS name,
                            COUNT(*), user_turn_number
                     FROM message_structure
                     WHERE session_id = ? AND branch_id = ? AND message_type IN (
                        'tool_call', 'function_call', 'computer_call', 'file_search_call',
                        'web_search_call', 'code_interpreter_call', 'tool_search_call',
                        'custom_tool_call', 'mcp_call', 'mcp_approval_request',
                        'shell_call', 'apply_patch_call'
                     )
                     GROUP BY name, user_turn_number
                     ORDER BY user_turn_number",
                )
                .map_err(sql_err)?;
            let rows = stmt
                .query_map(params![ctx.session_id, branch], |r| {
                    Ok(ToolUsage {
                        tool_name: r.get(0)?,
                        count: r.get::<_, i64>(1)? as u64,
                        user_turn_number: r.get(2)?,
                    })
                })
                .map_err(sql_err)?;
            Ok(rows.filter_map(std::result::Result::ok).collect())
        })
        .await
    }

    /// Record a run's usage against the current turn of the current branch.
    ///
    /// Call this with [`RunResult::usage`](crate::result::RunResult::usage) after a run. Usage
    /// of several runs in the same turn is summed.
    pub async fn store_run_usage(&self, usage: &Usage) -> Result<()> {
        let usage = usage.clone();
        self.blocking(move |conn, ctx| {
            let branch = ctx.branch_id();
            let turn: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(user_turn_number), 0) FROM message_structure
                     WHERE session_id = ? AND branch_id = ?",
                    params![ctx.session_id, branch],
                    |r| r.get(0),
                )
                .map_err(sql_err)?;
            if turn == 0 {
                return Ok(());
            }
            conn.execute(
                "INSERT INTO turn_usage
                    (session_id, branch_id, user_turn_number, requests, input_tokens,
                     output_tokens, total_tokens, input_tokens_details, output_tokens_details)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(session_id, branch_id, user_turn_number) DO UPDATE SET
                    requests = requests + excluded.requests,
                    input_tokens = input_tokens + excluded.input_tokens,
                    output_tokens = output_tokens + excluded.output_tokens,
                    total_tokens = total_tokens + excluded.total_tokens,
                    input_tokens_details = excluded.input_tokens_details,
                    output_tokens_details = excluded.output_tokens_details",
                params![
                    ctx.session_id,
                    branch,
                    turn,
                    usage.requests as i64,
                    usage.input_tokens as i64,
                    usage.output_tokens as i64,
                    usage.total_tokens as i64,
                    serde_json::to_string(&usage.input_tokens_details)?,
                    serde_json::to_string(&usage.output_tokens_details)?,
                ],
            )
            .map_err(sql_err)?;
            Ok(())
        })
        .await
    }

    /// Usage per turn on a branch, or for one turn when `user_turn_number` is given.
    pub async fn get_turn_usage(
        &self,
        user_turn_number: Option<i64>,
        branch_id: Option<&str>,
    ) -> Result<Vec<TurnUsage>> {
        let branch = branch_id.map(str::to_string);
        self.blocking(move |conn, ctx| {
            let branch = branch.unwrap_or_else(|| ctx.branch_id());
            let mut stmt = conn
                .prepare(
                    "SELECT user_turn_number, requests, input_tokens, output_tokens, total_tokens,
                            input_tokens_details, output_tokens_details
                     FROM turn_usage
                     WHERE session_id = ? AND branch_id = ?
                       AND (? IS NULL OR user_turn_number = ?)
                     ORDER BY user_turn_number",
                )
                .map_err(sql_err)?;
            let rows = stmt
                .query_map(
                    params![ctx.session_id, branch, user_turn_number, user_turn_number],
                    |r| {
                        fn details<T: Default + serde::de::DeserializeOwned>(
                            raw: Option<String>,
                        ) -> T {
                            raw.and_then(|s| serde_json::from_str(&s).ok())
                                .unwrap_or_default()
                        }
                        Ok(TurnUsage {
                            user_turn_number: r.get(0)?,
                            requests: r.get::<_, i64>(1)? as u64,
                            input_tokens: r.get::<_, i64>(2)? as u64,
                            output_tokens: r.get::<_, i64>(3)? as u64,
                            total_tokens: r.get::<_, i64>(4)? as u64,
                            input_tokens_details: details(r.get(5)?),
                            output_tokens_details: details(r.get(6)?),
                        })
                    },
                )
                .map_err(sql_err)?;
            Ok(rows.filter_map(std::result::Result::ok).collect())
        })
        .await
    }

    /// Usage aggregated over one branch, or the whole session when `branch_id` is `None`.
    ///
    /// Returns `None` when no usage was recorded.
    pub async fn get_session_usage(&self, branch_id: Option<&str>) -> Result<Option<SessionUsage>> {
        let branch = branch_id.map(str::to_string);
        self.blocking(move |conn, ctx| {
            let row: Option<UsageTotals> = conn
                .query_row(
                    "SELECT SUM(requests), SUM(input_tokens), SUM(output_tokens),
                            SUM(total_tokens), COUNT(*)
                     FROM turn_usage
                     WHERE session_id = ? AND (? IS NULL OR branch_id = ?)",
                    params![ctx.session_id, branch, branch],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                )
                .optional()
                .map_err(sql_err)?;
            Ok(row.and_then(|(req, inp, out, total, turns)| {
                req.map(|requests| SessionUsage {
                    requests: requests as u64,
                    input_tokens: inp.unwrap_or(0) as u64,
                    output_tokens: out.unwrap_or(0) as u64,
                    total_tokens: total.unwrap_or(0) as u64,
                    total_turns: turns as u64,
                })
            }))
        })
        .await
    }

    /// Insert items and their structure metadata in one transaction.
    fn insert_with_structure(conn: &Connection, ctx: &Ctx, items: &[InputItem]) -> Result<()> {
        let tx = conn.unchecked_transaction().map_err(sql_err)?;
        Self::insert_with_structure_in(&tx, ctx, items)?;
        tx.commit().map_err(sql_err)
    }

    /// Insert items and their structure metadata inside an existing transaction.
    fn insert_with_structure_in(
        tx: &rusqlite::Transaction<'_>,
        ctx: &Ctx,
        items: &[InputItem],
    ) -> Result<()> {
        let branch = ctx.branch_id();
        tx.execute(
            &format!(
                "INSERT OR IGNORE INTO {s} (session_id) VALUES (?)",
                s = ctx.sessions_table
            ),
            params![ctx.session_id],
        )
        .map_err(sql_err)?;

        let seq_start: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(sequence_number), 0) FROM message_structure
                 WHERE session_id = ?",
                params![ctx.session_id],
                |r| r.get(0),
            )
            .map_err(sql_err)?;
        let (mut turn, mut branch_turn): (i64, i64) = tx
            .query_row(
                "SELECT COALESCE(MAX(user_turn_number), 0), COALESCE(MAX(branch_turn_number), 0)
                 FROM message_structure WHERE session_id = ? AND branch_id = ?",
                params![ctx.session_id, branch],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(sql_err)?;

        for (offset, item) in items.iter().enumerate() {
            tx.execute(
                &format!(
                    "INSERT INTO {m} (session_id, message_data) VALUES (?, ?)",
                    m = ctx.messages_table
                ),
                params![ctx.session_id, serde_json::to_string(item.as_value())?],
            )
            .map_err(sql_err)?;
            let message_id = tx.last_insert_rowid();
            if is_user_message(item) {
                turn += 1;
                branch_turn += 1;
            }
            tx.execute(
                "INSERT INTO message_structure
                    (session_id, message_id, branch_id, message_type, sequence_number,
                     user_turn_number, branch_turn_number, tool_name)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    ctx.session_id,
                    message_id,
                    branch,
                    classify_message_type(item),
                    seq_start + offset as i64 + 1,
                    turn,
                    branch_turn,
                    extract_tool_name(item),
                ],
            )
            .map_err(sql_err)?;
        }

        tx.execute(
            &format!(
                "UPDATE {s} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                s = ctx.sessions_table
            ),
            params![ctx.session_id],
        )
        .map_err(sql_err)?;
        Ok(())
    }
}

#[async_trait]
impl Session for AdvancedSqliteSession {
    fn session_id(&self) -> &str {
        self.base.session_id()
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.base.session_settings()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let limit = super::resolve_session_limit(limit, self.session_settings().as_ref());
        self.blocking(move |conn, ctx| {
            let branch = ctx.branch_id();
            let rows: Vec<String> = match limit {
                Some(0) => Vec::new(),
                Some(n) => {
                    let mut stmt = conn
                        .prepare(&format!(
                            "SELECT m.message_data FROM {m} m
                             JOIN message_structure s ON m.id = s.message_id
                             WHERE m.session_id = ? AND s.branch_id = ?
                             ORDER BY s.sequence_number DESC LIMIT ?",
                            m = ctx.messages_table
                        ))
                        .map_err(sql_err)?;
                    let mut v: Vec<String> = stmt
                        .query_map(params![ctx.session_id, branch, n as i64], |r| r.get(0))
                        .map_err(sql_err)?
                        .filter_map(std::result::Result::ok)
                        .collect();
                    v.reverse();
                    v
                }
                None => {
                    let mut stmt = conn
                        .prepare(&format!(
                            "SELECT m.message_data FROM {m} m
                             JOIN message_structure s ON m.id = s.message_id
                             WHERE m.session_id = ? AND s.branch_id = ?
                             ORDER BY s.sequence_number ASC",
                            m = ctx.messages_table
                        ))
                        .map_err(sql_err)?;
                    stmt.query_map(params![ctx.session_id, branch], |r| r.get(0))
                        .map_err(sql_err)?
                        .filter_map(std::result::Result::ok)
                        .collect()
                }
            };
            Ok(rows
                .into_iter()
                .filter_map(|r| serde_json::from_str(&r).ok().map(InputItem))
                .collect())
        })
        .await
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        self.blocking(move |conn, ctx| Self::insert_with_structure(conn, ctx, &items))
            .await
    }

    /// Remove the most recent item of the current branch.
    ///
    /// The structure row is always removed; the stored message is removed only when no other
    /// branch still references it. A `turn_usage` row is removed when the pop empties its turn.
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.blocking(move |conn, ctx| {
            let branch = ctx.branch_id();
            let tx = conn.unchecked_transaction().map_err(sql_err)?;
            let row: Option<(i64, i64, i64, String)> = tx
                .query_row(
                    &format!(
                        "SELECT s.id, s.message_id, s.branch_turn_number, m.message_data
                         FROM message_structure s
                         JOIN {m} m ON m.id = s.message_id
                         WHERE s.session_id = ? AND s.branch_id = ?
                         ORDER BY s.sequence_number DESC LIMIT 1",
                        m = ctx.messages_table
                    ),
                    params![ctx.session_id, branch],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )
                .optional()
                .map_err(sql_err)?;
            let Some((structure_id, message_id, turn, data)) = row else {
                return Ok(None);
            };
            tx.execute(
                "DELETE FROM message_structure WHERE id = ?",
                params![structure_id],
            )
            .map_err(sql_err)?;
            let still_referenced: i64 = tx
                .query_row(
                    "SELECT COUNT(*) FROM message_structure WHERE message_id = ?",
                    params![message_id],
                    |r| r.get(0),
                )
                .map_err(sql_err)?;
            if still_referenced == 0 {
                tx.execute(
                    &format!("DELETE FROM {m} WHERE id = ?", m = ctx.messages_table),
                    params![message_id],
                )
                .map_err(sql_err)?;
            }
            let turn_remaining: i64 = tx
                .query_row(
                    "SELECT COUNT(*) FROM message_structure
                     WHERE session_id = ? AND branch_id = ? AND branch_turn_number = ?",
                    params![ctx.session_id, branch, turn],
                    |r| r.get(0),
                )
                .map_err(sql_err)?;
            if turn_remaining == 0 {
                tx.execute(
                    "DELETE FROM turn_usage
                     WHERE session_id = ? AND branch_id = ? AND user_turn_number = ?",
                    params![ctx.session_id, branch, turn],
                )
                .map_err(sql_err)?;
            }
            tx.commit().map_err(sql_err)?;
            Ok(serde_json::from_str(&data).ok().map(InputItem))
        })
        .await
    }

    /// Snapshot the end of the current branch; handles are `message_structure` row ids.
    ///
    /// Branch rows are what a compaction replacement must act on: the underlying message may be
    /// shared with another branch, so it is only removed when nothing else references it.
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        self.blocking(move |conn, ctx| {
            let branch = ctx.branch_id();
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT s.id, m.message_data FROM message_structure s
                     JOIN {m} m ON m.id = s.message_id
                     WHERE s.session_id = ? AND s.branch_id = ?
                     ORDER BY s.sequence_number DESC LIMIT ?",
                    m = ctx.messages_table
                ))
                .map_err(sql_err)?;
            let mut rows: Vec<(i64, String)> = stmt
                .query_map(params![ctx.session_id, branch, window_limit(limit)], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .map_err(sql_err)?
                .filter_map(std::result::Result::ok)
                .collect();
            let complete = rows.len() <= limit;
            rows.truncate(limit);
            rows.reverse();
            Ok(Some(snapshot_from_rows(rows, complete)))
        })
        .await
    }

    async fn replace_compaction_suffix(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
    ) -> Result<bool> {
        let handles: Vec<String> = handles.iter().map(|h| h.0.clone()).collect();
        self.blocking(move |conn, ctx| {
            let branch = ctx.branch_id();
            let tx = conn.unchecked_transaction().map_err(sql_err)?;
            let mut stmt = tx
                .prepare(
                    "SELECT id, message_id FROM message_structure
                     WHERE session_id = ? AND branch_id = ?
                     ORDER BY sequence_number DESC LIMIT ?",
                )
                .map_err(sql_err)?;
            let mut tail: Vec<(i64, i64)> = stmt
                .query_map(params![ctx.session_id, branch, handles.len() as i64], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .map_err(sql_err)?
                .filter_map(std::result::Result::ok)
                .collect();
            drop(stmt);
            tail.reverse();
            if tail.len() != handles.len()
                || tail
                    .iter()
                    .zip(&handles)
                    .any(|((id, _), handle)| id.to_string() != *handle)
            {
                return Ok(false);
            }

            for (structure_id, message_id) in &tail {
                tx.execute(
                    "DELETE FROM message_structure WHERE id = ?",
                    params![structure_id],
                )
                .map_err(sql_err)?;
                let still_referenced: i64 = tx
                    .query_row(
                        "SELECT COUNT(*) FROM message_structure WHERE message_id = ?",
                        params![message_id],
                        |r| r.get(0),
                    )
                    .map_err(sql_err)?;
                if still_referenced == 0 {
                    tx.execute(
                        &format!("DELETE FROM {m} WHERE id = ?", m = ctx.messages_table),
                        params![message_id],
                    )
                    .map_err(sql_err)?;
                }
            }
            Self::insert_with_structure_in(&tx, ctx, &items)?;
            tx.commit().map_err(sql_err)?;
            Ok(true)
        })
        .await
    }

    /// Remove every branch, structure row and usage row of this session.
    async fn clear_session(&self) -> Result<()> {
        self.blocking(move |conn, ctx| {
            let tx = conn.unchecked_transaction().map_err(sql_err)?;
            for sql in [
                "DELETE FROM message_structure WHERE session_id = ?".to_string(),
                "DELETE FROM turn_usage WHERE session_id = ?".to_string(),
                format!(
                    "DELETE FROM {m} WHERE session_id = ?",
                    m = ctx.messages_table
                ),
                format!(
                    "DELETE FROM {s} WHERE session_id = ?",
                    s = ctx.sessions_table
                ),
            ] {
                tx.execute(&sql, params![ctx.session_id]).map_err(sql_err)?;
            }
            tx.commit().map_err(sql_err)
        })
        .await?;
        *self.branch.lock().expect("branch poisoned") = MAIN_BRANCH.to_string();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn message_types_are_classified_like_the_python_backend() {
        assert_eq!(classify_message_type(&InputItem::user("x")), "user");
        assert_eq!(
            classify_message_type(&InputItem::assistant("x")),
            "assistant"
        );
        assert_eq!(
            classify_message_type(&InputItem(json!({"type": "function_call", "name": "f"}))),
            "function_call"
        );
        assert_eq!(classify_message_type(&InputItem(json!({}))), "other");
    }

    #[test]
    fn tool_names_cover_named_and_typed_calls() {
        assert_eq!(
            extract_tool_name(&InputItem(json!({"type": "function_call", "name": "f"}))).as_deref(),
            Some("f")
        );
        assert_eq!(
            extract_tool_name(&InputItem(
                json!({"type": "mcp_call", "server_label": "srv", "name": "t"})
            ))
            .as_deref(),
            Some("srv.t")
        );
        assert_eq!(
            extract_tool_name(&InputItem(json!({"type": "web_search_call"}))).as_deref(),
            Some("web_search_call")
        );
        assert_eq!(
            extract_tool_name(&InputItem(json!({"type": "tool_search_output"}))).as_deref(),
            Some("tool_search")
        );
        assert_eq!(extract_tool_name(&InputItem::user("x")), None);
    }

    #[tokio::test]
    async fn turns_and_usage_are_tracked() {
        let s = AdvancedSqliteSession::in_memory("u1").unwrap();
        s.add_items(vec![
            InputItem::user("hello"),
            InputItem(json!({"type": "function_call", "name": "weather", "arguments": "{}"})),
            InputItem::assistant("sunny"),
        ])
        .await
        .unwrap();
        s.store_run_usage(&Usage::single(100, 20)).await.unwrap();
        s.add_items(vec![InputItem::user("thanks")]).await.unwrap();
        s.store_run_usage(&Usage::single(10, 2)).await.unwrap();

        let turns = s.get_conversation_turns(None).await.unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].turn, 1);
        assert_eq!(turns[0].content, "hello");

        let tools = s.get_tool_usage(None).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_name, "weather");
        assert_eq!(tools[0].user_turn_number, 1);

        let per_turn = s.get_turn_usage(None, None).await.unwrap();
        assert_eq!(per_turn.len(), 2);
        assert_eq!(per_turn[0].input_tokens, 100);
        assert_eq!(per_turn[1].input_tokens, 10);
        let total = s.get_session_usage(None).await.unwrap().unwrap();
        assert_eq!(total.total_tokens, 132);
        assert_eq!(total.total_turns, 2);
    }

    #[tokio::test]
    async fn branching_forks_history_and_shares_earlier_messages() {
        let s = AdvancedSqliteSession::in_memory("u1").unwrap();
        s.add_items(vec![InputItem::user("q1"), InputItem::assistant("a1")])
            .await
            .unwrap();
        s.add_items(vec![InputItem::user("q2"), InputItem::assistant("a2")])
            .await
            .unwrap();
        assert_eq!(s.get_items(None).await.unwrap().len(), 4);

        let branch = s.create_branch_from_turn(2, None).await.unwrap();
        assert_eq!(s.current_branch(), branch);
        let forked = s.get_items(None).await.unwrap();
        assert_eq!(forked.len(), 2);
        assert_eq!(forked[0].text_content().as_deref(), Some("q1"));

        s.add_items(vec![InputItem::user("q2 again")])
            .await
            .unwrap();
        assert_eq!(s.get_items(None).await.unwrap().len(), 3);

        s.switch_to_branch(MAIN_BRANCH).await.unwrap();
        assert_eq!(s.get_items(None).await.unwrap().len(), 4);

        let branches = s.list_branches().await.unwrap();
        assert_eq!(branches.len(), 2);
        assert!(
            branches
                .iter()
                .any(|b| b.branch_id == MAIN_BRANCH && b.is_current)
        );

        // The shared `q1` message survives deleting the fork.
        s.delete_branch(&branch, false).await.unwrap();
        assert_eq!(s.get_items(None).await.unwrap().len(), 4);
        assert_eq!(s.list_branches().await.unwrap().len(), 1);
        assert!(s.switch_to_branch(&branch).await.is_err());
    }

    #[tokio::test]
    async fn popping_and_clearing_remove_structure_rows() {
        let s = AdvancedSqliteSession::in_memory("u1").unwrap();
        s.add_items(vec![InputItem::user("a"), InputItem::assistant("b")])
            .await
            .unwrap();
        s.store_run_usage(&Usage::single(1, 1)).await.unwrap();
        assert_eq!(
            s.pop_item()
                .await
                .unwrap()
                .unwrap()
                .text_content()
                .as_deref(),
            Some("b")
        );
        assert_eq!(s.get_items(None).await.unwrap().len(), 1);
        assert!(s.pop_item().await.unwrap().is_some());
        assert!(s.pop_item().await.unwrap().is_none());
        assert!(s.get_session_usage(None).await.unwrap().is_none());

        s.add_items(vec![InputItem::user("c")]).await.unwrap();
        s.clear_session().await.unwrap();
        assert!(s.get_items(None).await.unwrap().is_empty());
        assert!(s.list_branches().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn branch_from_content_finds_the_turn() {
        let s = AdvancedSqliteSession::in_memory("u1").unwrap();
        s.add_items(vec![
            InputItem::user("about rust"),
            InputItem::assistant("ok"),
        ])
        .await
        .unwrap();
        s.add_items(vec![InputItem::user("about python")])
            .await
            .unwrap();
        let branch = s
            .create_branch_from_content("PYTHON", Some("py-fork"))
            .await
            .unwrap();
        assert_eq!(branch, "py-fork");
        assert_eq!(s.get_items(None).await.unwrap().len(), 2);
        assert!(
            s.create_branch_from_content("nothing here", None)
                .await
                .is_err()
        );
    }
}
