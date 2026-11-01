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

/// The branch this instance reads and writes, and the clear generation it was chosen under.
///
/// Another process (or another instance on the same file) can clear this session, which deletes
/// every branch. The generation stored in `session_clear_generations` is bumped by each clear,
/// so an instance whose captured generation is stale knows its branch pointer names a branch
/// that no longer exists and falls back to `main`. Port of Python's `_branch_state`.
#[derive(Debug, Clone)]
struct BranchState {
    branch: String,
    generation: i64,
}

/// Everything a blocking database closure needs about the session.
#[derive(Debug, Clone)]
struct Ctx {
    session_id: String,
    sessions_table: String,
    messages_table: String,
    branch: Arc<Mutex<BranchState>>,
}

impl Ctx {
    fn branch_id(&self) -> String {
        self.branch.lock().expect("branch poisoned").branch.clone()
    }

    fn generation(&self) -> i64 {
        self.branch.lock().expect("branch poisoned").generation
    }

    fn set_branch(&self, branch: String, generation: i64) {
        *self.branch.lock().expect("branch poisoned") = BranchState { branch, generation };
    }
}

/// Create the cross-process branch-id reservation table and backfill this session's branches.
///
/// Reservations outlive the rows of a branch, so a branch id that was used once cannot be
/// handed out again — not even after `clear_session`, and not to a second process racing for
/// the same name. Port of `_ensure_branch_reservations_table`.
fn ensure_branch_reservations(conn: &Connection, ctx: &Ctx) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS branch_reservations (
            session_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            PRIMARY KEY (session_id, branch_id)
         );",
    )
    .map_err(sql_err)?;
    let missing: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM message_structure ms
             WHERE ms.session_id = ?
               AND NOT EXISTS (
                   SELECT 1 FROM branch_reservations br
                   WHERE br.session_id = ms.session_id AND br.branch_id = ms.branch_id
               )
             LIMIT 1",
            params![ctx.session_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql_err)?;
    if missing.is_some() {
        conn.execute(
            "INSERT OR IGNORE INTO branch_reservations (session_id, branch_id)
             SELECT DISTINCT session_id, branch_id FROM message_structure WHERE session_id = ?",
            params![ctx.session_id],
        )
        .map_err(sql_err)?;
    }
    Ok(())
}

/// Create the durable clear-generation table and seed this session's row.
///
/// Port of `_ensure_session_clear_generations_table`.
fn ensure_clear_generations(conn: &Connection, ctx: &Ctx) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS session_clear_generations (
            session_id TEXT PRIMARY KEY,
            generation INTEGER NOT NULL DEFAULT 0
         );",
    )
    .map_err(sql_err)?;
    conn.execute(
        "INSERT OR IGNORE INTO session_clear_generations (session_id, generation) VALUES (?, 0)",
        params![ctx.session_id],
    )
    .map_err(sql_err)?;
    Ok(())
}

/// The durable clear generation of this session, or 0 when nothing recorded one yet.
fn stored_generation(conn: &Connection, ctx: &Ctx) -> Result<i64> {
    conn.query_row(
        "SELECT generation FROM session_clear_generations WHERE session_id = ?",
        params![ctx.session_id],
        |r| r.get(0),
    )
    .optional()
    .map_err(sql_err)
    .map(|g| g.unwrap_or(0))
}

/// Reset a stale branch pointer after another instance cleared this session.
///
/// Port of `_refresh_branch_after_external_clear`.
fn refresh_branch_after_external_clear(
    conn: &Connection,
    ctx: &Ctx,
    initialize: bool,
) -> Result<()> {
    if initialize {
        ensure_clear_generations(conn, ctx)?;
    } else {
        let exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM sqlite_master
                 WHERE type = 'table' AND name = 'session_clear_generations'",
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql_err)?;
        if exists.is_none() {
            return Ok(());
        }
    }
    let generation = stored_generation(conn, ctx)?;
    if generation != ctx.generation() {
        ctx.set_branch(MAIN_BRANCH.to_string(), generation);
    }
    Ok(())
}

/// The branch a read should use: the requested one, or the current one after synchronizing an
/// external clear. Port of `_resolve_read_branch`.
fn resolve_read_branch(conn: &Connection, ctx: &Ctx, branch_id: Option<String>) -> Result<String> {
    if let Some(branch_id) = branch_id {
        return Ok(branch_id);
    }
    refresh_branch_after_external_clear(conn, ctx, false)?;
    Ok(ctx.branch_id())
}

/// Claim `branch_id`, or generate an unused one. Port of `_reserve_branch_id`.
///
/// The insert is the claim: `INSERT OR IGNORE` on the reservation table's primary key succeeds
/// for exactly one writer, so two processes picking the same name cannot both proceed.
fn reserve_branch_id(
    tx: &rusqlite::Transaction<'_>,
    ctx: &Ctx,
    requested: Option<String>,
    from_turn_number: i64,
) -> Result<String> {
    let claim = |branch_id: &str| -> Result<bool> {
        let inserted = tx
            .execute(
                "INSERT OR IGNORE INTO branch_reservations (session_id, branch_id) VALUES (?, ?)",
                params![ctx.session_id, branch_id],
            )
            .map_err(sql_err)?;
        Ok(inserted == 1)
    };
    if let Some(branch_id) = requested {
        if !claim(&branch_id)? {
            return Err(AgentsError::user(format!(
                "Branch ID '{branch_id}' has already been used. Choose a new branch ID."
            )));
        }
        return Ok(branch_id);
    }
    let base = format!(
        "branch_from_turn_{from_turn_number}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let mut branch_id = base.clone();
    let mut suffix = 1u32;
    loop {
        if claim(&branch_id)? {
            return Ok(branch_id);
        }
        suffix += 1;
        branch_id = format!("{base}_{suffix}");
    }
}

/// Point the session at `branch_id` unless a clear committed since `generation` was captured.
///
/// Returns `false` when it did; the clear's reset to `main` wins, because the branch this call
/// selected no longer exists. Port of `_commit_branch_pointer`.
fn commit_branch_pointer(
    conn: &Connection,
    ctx: &Ctx,
    branch_id: String,
    generation: i64,
) -> Result<bool> {
    let durable = stored_generation(conn, ctx)?;
    if durable != generation {
        ctx.set_branch(MAIN_BRANCH.to_string(), durable);
        return Ok(false);
    }
    ctx.set_branch(branch_id, durable);
    Ok(true)
}

/// Compare two table names the way SQLite compares identifiers.
///
/// SQLite folds identifiers with ASCII rules only, so it is asked directly rather than having
/// the rule reimplemented: Rust's `to_lowercase` would equate names SQLite keeps distinct.
/// Port of `_identifiers_equal`.
fn identifiers_equal(conn: &Connection, left: &str, right: &str) -> Result<bool> {
    conn.query_row("SELECT ? = ? COLLATE NOCASE", params![left, right], |r| {
        r.get::<_, i64>(0)
    })
    .map(|v| v != 0)
    .map_err(sql_err)
}

/// Ask SQLite which table object a configured identifier selects.
///
/// A SQLite authorizer reports the table every read touches, so running `SELECT * FROM <ident>
/// LIMIT 0` names the table the identifier actually resolves to, after quoting and
/// case-folding. Port of `_resolve_table_identifier`.
fn resolve_table_identifier(conn: &Connection, identifier: &str) -> Result<String> {
    let seen: Arc<Mutex<std::collections::BTreeSet<String>>> = Arc::default();
    let collector = Arc::clone(&seen);
    conn.authorizer(Some(
        move |ctx: rusqlite::hooks::AuthContext<'_>| -> rusqlite::hooks::Authorization {
            if let rusqlite::hooks::AuthAction::Read { table_name, .. } = ctx.action {
                collector
                    .lock()
                    .expect("authorizer poisoned")
                    .insert(table_name.to_string());
            }
            rusqlite::hooks::Authorization::Allow
        },
    ))
    .map_err(sql_err)?;
    let probe = conn.execute_batch(&format!("SELECT * FROM {identifier} LIMIT 0"));
    conn.authorizer::<fn(rusqlite::hooks::AuthContext<'_>) -> rusqlite::hooks::Authorization>(None)
        .map_err(sql_err)?;
    probe.map_err(sql_err)?;
    let mut tables = Arc::try_unwrap(seen)
        .unwrap_or_else(|shared| Mutex::new(shared.lock().expect("authorizer poisoned").clone()))
        .into_inner()
        .expect("authorizer poisoned");
    if tables.len() != 1 {
        return Err(AgentsError::user(format!(
            "The configured table identifier '{identifier}' is ambiguous."
        )));
    }
    Ok(tables.pop_first().expect("exactly one table"))
}

/// The base tables the structure tables must belong to, after SQLite resolved the configured
/// identifiers. Port of `_resolve_base_table_owners`.
fn resolve_base_table_owners(conn: &Connection, ctx: &Ctx) -> Result<(String, String)> {
    let missing = || {
        AgentsError::user(
            "The configured base tables are missing. Create an AdvancedSqliteSession with \
             create_tables set to initialize and claim the database before opening it without \
             table creation.",
        )
    };
    let sessions_table =
        resolve_table_identifier(conn, &ctx.sessions_table).map_err(|_| missing())?;
    let messages_table =
        resolve_table_identifier(conn, &ctx.messages_table).map_err(|_| missing())?;

    let mut stmt = conn
        .prepare(&format!(
            "PRAGMA foreign_key_list({m})",
            m = ctx.messages_table
        ))
        .map_err(sql_err)?;
    // `table`, `from`, `to` of each foreign key of the messages table.
    let keys: Vec<(String, String, Option<String>)> = stmt
        .query_map([], |r| Ok((r.get(2)?, r.get(3)?, r.get(4)?)))
        .map_err(sql_err)?
        .filter_map(std::result::Result::ok)
        .collect();
    drop(stmt);
    let mut owners = Vec::new();
    for (table, from, to) in keys {
        if identifiers_equal(conn, &from, "session_id")?
            && to
                .as_deref()
                .is_some_and(|to| to.eq_ignore_ascii_case("session_id"))
        {
            owners.push(table);
        }
    }
    let found = match owners.as_slice() {
        [one] if identifiers_equal(conn, one, &sessions_table)? => {
            return Ok((sessions_table, messages_table));
        }
        [one] => one.clone(),
        _ => "ambiguous ownership".to_string(),
    };
    Err(AgentsError::user(format!(
        "The configured messages table '{messages_table}' belongs to sessions table '{found}', \
         not to the configured '{sessions_table}'. Give each sessions_table/messages_table pair \
         its own db_path."
    )))
}

/// Require that the structure tables in this database file belong to the configured base-table
/// pair. Port of `_claim_structure_tables`.
///
/// `message_structure` and `turn_usage` are not named after the base tables, so one database
/// file can only hold the structure rows of one pair; `CREATE TABLE IF NOT EXISTS` keeps the
/// first pair's foreign keys. Without this check a second pair would join `message_structure`
/// against the wrong messages table and read back the other pair's rows.
fn claim_structure_tables(conn: &Connection, ctx: &Ctx) -> Result<()> {
    let structure_exists: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'message_structure'",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql_err)?;
    if structure_exists.is_none() {
        return Err(unclaimed_database_error());
    }
    let (sessions_table, messages_table) = resolve_base_table_owners(conn, ctx)?;
    check_structure_owners(conn, &sessions_table, &messages_table)
}

fn unclaimed_database_error() -> AgentsError {
    AgentsError::user(
        "The database has no claimed structure tables. Create an AdvancedSqliteSession with \
         create_tables=True to create and claim them before opening the database without table \
         creation.",
    )
}

/// Verify that the structure tables' foreign keys name exactly this base-table pair.
fn check_structure_owners(
    conn: &Connection,
    sessions_table: &str,
    messages_table: &str,
) -> Result<()> {
    for (table, columns) in [
        ("message_structure", &["session_id", "message_id"][..]),
        ("turn_usage", &["session_id"][..]),
    ] {
        let mut stmt = conn
            .prepare(&format!("PRAGMA foreign_key_list({table})"))
            .map_err(sql_err)?;
        let keys: Vec<(String, String, Option<String>)> = stmt
            .query_map([], |r| Ok((r.get(2)?, r.get(3)?, r.get(4)?)))
            .map_err(sql_err)?
            .filter_map(std::result::Result::ok)
            .collect();
        drop(stmt);
        for column in columns {
            let (configured_table, expected_column) = match *column {
                "session_id" => (sessions_table, "session_id"),
                _ => (messages_table, "id"),
            };
            // A quoted configured name (`"my sessions"`) names the table without its quotes.
            // Ask SQLite when the table exists; before creation, unquote it ourselves.
            let resolved_table = resolve_table_identifier(conn, configured_table)
                .unwrap_or_else(|_| super::sqlite_session::unquoted_identifier(configured_table));
            let expected_table = resolved_table.as_str();
            let matching: Vec<&(String, String, Option<String>)> = keys
                .iter()
                .filter(|(_, from, _)| from.eq_ignore_ascii_case(column))
                .collect();
            let [key] = matching.as_slice() else {
                return Err(AgentsError::user(format!(
                    "The `{table}` table does not record exactly one owner foreign key for each \
                     required base-table column. Create an AdvancedSqliteSession with \
                     create_tables set to create and claim the structure tables before opening \
                     the database without them."
                )));
            };
            let target = key.2.as_deref().unwrap_or_default();
            if !identifiers_equal(conn, &key.0, expected_table)?
                || !target.eq_ignore_ascii_case(expected_column)
            {
                return Err(AgentsError::user(format!(
                    "The `{table}` table already belongs to '{}({target})', not to the \
                     configured '{expected_table}({expected_column})'. Structure tables are \
                     shared per database file, so give each sessions_table/messages_table pair \
                     its own db_path.",
                    key.0
                )));
            }
        }
    }
    Ok(())
}

/// Check who owns the structure tables of `path` before any base table is created.
///
/// A file that already holds structure tables belongs to one base-table pair; opening it with
/// another pair, or without table creation when nothing claimed it, must fail without leaving
/// new tables behind.
fn preflight_structure_claim(
    path: &std::path::Path,
    sessions_table: &str,
    messages_table: &str,
    create_tables: bool,
) -> Result<()> {
    if !path.exists() {
        return if create_tables {
            Ok(())
        } else {
            Err(unclaimed_database_error())
        };
    }
    let conn = Connection::open(path).map_err(sql_err)?;
    let structure_exists: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'message_structure'",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql_err)?;
    if structure_exists.is_none() {
        return if create_tables {
            Ok(())
        } else {
            Err(unclaimed_database_error())
        };
    }
    if !create_tables {
        let ctx = Ctx {
            session_id: String::new(),
            sessions_table: sessions_table.to_string(),
            messages_table: messages_table.to_string(),
            branch: Arc::new(Mutex::new(BranchState {
                branch: MAIN_BRANCH.to_string(),
                generation: 0,
            })),
        };
        // Names the missing base tables before any of them is created.
        resolve_base_table_owners(&conn, &ctx)?;
    }
    check_structure_owners(&conn, sessions_table, messages_table)
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
        _ => {
            let name = name?;
            // A namespace equal to the tool name is the reserved deferred top-level wire
            // shape, so it collapses to the bare name; any other namespace qualifies it.
            match value.get("namespace").and_then(Value::as_str) {
                Some(ns) if !ns.is_empty() && ns != name && !name.is_empty() => {
                    Some(format!("{ns}.{name}"))
                }
                _ => Some(name.to_string()),
            }
        }
    }
}

/// True when the item is a user message.
fn is_user_message(item: &InputItem) -> bool {
    item.role() == Some("user")
}

/// Start a write transaction that takes SQLite's write lock before reading anything.
///
/// A transaction that reads first and writes later has to upgrade a stale read snapshot, which
/// fails at once with "database is locked" when another connection wrote in between, without
/// waiting for the busy timeout. Taking the lock up front makes concurrent writers queue.
fn write_transaction(conn: &Connection) -> Result<rusqlite::Transaction<'_>> {
    rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)
        .map_err(sql_err)
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
/// Three further tables keep the session safe to share:
///
/// - `branch_reservations`: every branch id ever used, so two processes cannot create the same
///   branch and a cleared id is never handed out again.
/// - `session_clear_generations`: a counter bumped by every `clear_session`, so an instance
///   whose branch was deleted by another process falls back to `main` instead of writing into a
///   branch that no longer exists.
/// - the foreign keys of `message_structure` and `turn_usage` record which base-table pair owns
///   the file; a session configured with a different pair is rejected rather than silently
///   reading the other pair's rows.
///
/// # Differences from the Python SDK
///
/// `store_run_usage` is called explicitly with a run's [`Usage`] rather than hooked into the
/// runner.
#[derive(Debug, Clone)]
pub struct AdvancedSqliteSession {
    base: SqliteSession,
    branch: Arc<Mutex<BranchState>>,
}

impl AdvancedSqliteSession {
    /// An in-memory session (lost when the last clone is dropped).
    pub fn in_memory(session_id: impl Into<String>) -> Result<Self> {
        Self::from_base(SqliteSession::in_memory(session_id)?, true)
    }

    /// A file-backed session, creating the structure tables if they are missing.
    pub fn open(
        session_id: impl Into<String>,
        db_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        Self::from_base(SqliteSession::open(session_id, db_path)?, true)
    }

    /// A file-backed session that only creates the structure tables when `create_tables` is set.
    ///
    /// With `create_tables` cleared the database must already hold structure tables claimed by
    /// this session's base-table pair; an unclaimed or foreign file is rejected, so a pair
    /// cannot open a file and have another pair claim it underneath. Port of Python's
    /// `create_tables` constructor parameter, whose default is the same `false`.
    pub fn open_with(
        session_id: impl Into<String>,
        db_path: impl AsRef<std::path::Path>,
        create_tables: bool,
    ) -> Result<Self> {
        let path = db_path.as_ref();
        preflight_structure_claim(path, "agent_sessions", "agent_messages", create_tables)?;
        Self::from_base(SqliteSession::open(session_id, path)?, create_tables)
    }

    /// Like [`AdvancedSqliteSession::open_with`] with custom base-table names.
    ///
    /// The structure tables' owner is checked before any base table is created, so a rejected
    /// pair leaves the database untouched, as Python's constructor does.
    pub fn open_with_tables(
        session_id: impl Into<String>,
        db_path: impl AsRef<std::path::Path>,
        sessions_table: &str,
        messages_table: &str,
        create_tables: bool,
    ) -> Result<Self> {
        let path = db_path.as_ref();
        preflight_structure_claim(path, sessions_table, messages_table, create_tables)?;
        if create_tables {
            let base = SqliteSession::open_uninitialized(
                session_id,
                path,
                sessions_table,
                messages_table,
            )?;
            let session = AdvancedSqliteSession {
                base,
                branch: Arc::new(Mutex::new(BranchState {
                    branch: MAIN_BRANCH.to_string(),
                    generation: 0,
                })),
            };
            session.init_all_tables_atomically()?;
            return Ok(session);
        }
        let base = SqliteSession::with_tables(session_id, path, sessions_table, messages_table)?;
        Self::from_base(base, create_tables)
    }

    /// Extend an existing [`SqliteSession`] with the structure and usage tables.
    pub fn from_session(base: SqliteSession) -> Result<Self> {
        Self::from_base(base, true)
    }

    /// Extend an existing [`SqliteSession`], creating the structure tables only when asked.
    ///
    /// See [`AdvancedSqliteSession::open_with`].
    pub fn from_session_with(base: SqliteSession, create_tables: bool) -> Result<Self> {
        Self::from_base(base, create_tables)
    }

    fn from_base(base: SqliteSession, create_tables: bool) -> Result<Self> {
        let session = AdvancedSqliteSession {
            base,
            branch: Arc::new(Mutex::new(BranchState {
                branch: MAIN_BRANCH.to_string(),
                generation: 0,
            })),
        };
        session.init_structure_tables(create_tables)?;
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

    fn init_structure_tables(&self, create_tables: bool) -> Result<()> {
        let ctx = self.ctx();
        let conn = self.base.connection().lock().expect("sqlite poisoned");
        Self::init_structure_tables_on(&conn, &ctx, create_tables)
    }

    /// Create the base and structure tables in one write transaction, so a pair rejected by the
    /// file's owner leaves nothing behind even when another connection is claiming the same
    /// fresh file at the same moment.
    fn init_all_tables_atomically(&self) -> Result<()> {
        let ctx = self.ctx();
        let conn = self.base.connection().lock().expect("sqlite poisoned");
        conn.execute_batch("BEGIN IMMEDIATE").map_err(sql_err)?;
        let created = self
            .base
            .create_tables_on(&conn)
            .and_then(|()| Self::init_structure_tables_on(&conn, &ctx, true));
        match created {
            Ok(()) => conn.execute_batch("COMMIT").map_err(sql_err),
            Err(error) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(error)
            }
        }
    }

    fn init_structure_tables_on(conn: &Connection, ctx: &Ctx, create_tables: bool) -> Result<()> {
        if !create_tables {
            claim_structure_tables(conn, ctx)?;
            refresh_branch_after_external_clear(conn, ctx, false)?;
            return Ok(());
        }
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
        .map_err(sql_err)?;
        // Validate the owner-bearing tables before anything else consumes them.
        claim_structure_tables(conn, ctx)?;
        ensure_branch_reservations(conn, ctx)?;
        ensure_clear_generations(conn, ctx)?;
        refresh_branch_after_external_clear(conn, ctx, false)
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

    /// History of one branch, the current branch when `branch_id` is `None`.
    ///
    /// Port of Python's `get_items(limit, branch_id=...)`, which can read a branch other than
    /// the one the session is on. Reading the current branch first synchronizes an external
    /// clear, so a branch another process deleted falls back to `main`.
    pub async fn get_items_for_branch(
        &self,
        limit: Option<usize>,
        branch_id: Option<&str>,
    ) -> Result<Vec<InputItem>> {
        let limit = super::resolve_session_limit(limit, self.session_settings().as_ref());
        let requested = branch_id.map(str::to_string);
        self.blocking(move |conn, ctx| {
            let branch = resolve_read_branch(conn, ctx, requested)?;
            let rows: Vec<String> = match limit {
                Some(0) => Vec::new(),
                Some(n) => {
                    // Widen the fetch window while corrupt rows sit among the newest entries, so
                    // `limit` counts readable items rather than stored rows. Port of the loop
                    // in Python's `AdvancedSQLiteSession.get_items`.
                    let mut window = n;
                    loop {
                        let mut stmt = conn
                            .prepare(&format!(
                                "SELECT m.message_data FROM {m} m
                                 JOIN message_structure s ON m.id = s.message_id
                                 WHERE m.session_id = ? AND s.branch_id = ?
                                 ORDER BY s.sequence_number DESC LIMIT ?",
                                m = ctx.messages_table
                            ))
                            .map_err(sql_err)?;
                        let mut fetched: Vec<String> = stmt
                            .query_map(params![ctx.session_id, branch, window as i64], |r| r.get(0))
                            .map_err(sql_err)?
                            .filter_map(std::result::Result::ok)
                            .collect();
                        let fetched_rows = fetched.len();
                        fetched.reverse();
                        let valid: Vec<String> = fetched
                            .into_iter()
                            .filter(|r| serde_json::from_str::<Value>(r).is_ok())
                            .collect();
                        if valid.len() >= n {
                            break valid[valid.len() - n..].to_vec();
                        }
                        if fetched_rows < window {
                            break valid;
                        }
                        window = window.saturating_mul(2);
                    }
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

    /// The branch the session currently reads and writes.
    pub fn current_branch(&self) -> String {
        self.branch.lock().expect("branch poisoned").branch.clone()
    }

    /// Switch to an existing branch.
    pub async fn switch_to_branch(&self, branch_id: impl Into<String>) -> Result<()> {
        let branch_id = branch_id.into();
        let target = branch_id.clone();
        self.blocking(move |conn, ctx| {
            refresh_branch_after_external_clear(conn, ctx, true)?;
            let generation = ctx.generation();
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM message_structure WHERE session_id = ? AND branch_id = ?",
                    params![ctx.session_id, target],
                    |r| r.get(0),
                )
                .map_err(sql_err)?;
            if count == 0 && target != MAIN_BRANCH {
                return Err(AgentsError::user(format!(
                    "Branch '{target}' does not exist in this session"
                )));
            }
            commit_branch_pointer(conn, ctx, target, generation)?;
            Ok(())
        })
        .await
    }

    /// All branches of this session, oldest first.
    pub async fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        self.blocking(move |conn, ctx| {
            let current = resolve_read_branch(conn, ctx, None)?;
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
        if branch_name.is_some_and(|name| name.trim().is_empty()) {
            return Err(AgentsError::user("Branch name cannot be empty"));
        }
        let requested = branch_name.map(str::to_string);
        let branch_id = self
            .blocking(move |conn, ctx| {
                // Take SQLite's write reservation before claiming the branch id, so a session
                // in another process cannot pass the same check concurrently.
                let tx = rusqlite::Transaction::new_unchecked(
                    conn,
                    rusqlite::TransactionBehavior::Immediate,
                )
                .map_err(sql_err)?;
                ensure_branch_reservations(&tx, ctx)?;
                refresh_branch_after_external_clear(&tx, ctx, true)?;
                let source = ctx.branch_id();
                let generation = ctx.generation();
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
                // The reservation is the claim: it outlives the branch's rows, so a cleared
                // id is never reused and two processes cannot both take the same name.
                let branch_id = reserve_branch_id(&tx, ctx, requested, turn_number)?;
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
                commit_branch_pointer(conn, ctx, branch_id.clone(), generation)?;
                Ok(branch_id)
            })
            .await?;
        Ok(branch_id)
    }

    /// Fork the conversation before the first turn whose user message contains `search`.
    pub async fn create_branch_from_content(
        &self,
        search: &str,
        branch_name: Option<&str>,
    ) -> Result<String> {
        if branch_name.is_some_and(|name| name.trim().is_empty()) {
            return Err(AgentsError::user("Branch name cannot be empty"));
        }
        let turns = self.find_turns_by_content(search, None).await?;
        let turn = turns.first().ok_or_else(|| {
            AgentsError::user(format!("No user turns found containing '{search}'"))
        })?;
        self.create_branch_from_turn(turn.turn, branch_name).await
    }

    /// Delete a branch and the messages only it referenced.
    ///
    /// The `main` branch cannot be deleted. Deleting the current branch requires `force` and
    /// switches the session back to `main`.
    pub async fn delete_branch(&self, branch_id: &str, force: bool) -> Result<()> {
        if branch_id.trim().is_empty() {
            return Err(AgentsError::user("Branch ID cannot be empty"));
        }
        if branch_id == MAIN_BRANCH {
            return Err(AgentsError::user("Cannot delete the 'main' branch"));
        }
        let is_current = self.current_branch() == branch_id;
        if is_current && !force {
            return Err(AgentsError::user(format!(
                "Cannot delete current branch '{branch_id}'. Use force=True or switch branches \
                 first"
            )));
        }
        let target = branch_id.to_string();
        self.blocking(move |conn, ctx| {
            let tx = write_transaction(conn)?;
            // Back-fill the reservations before the branch's rows, their only durable evidence
            // in a database written before reservations existed, are deleted.
            ensure_branch_reservations(&tx, ctx)?;
            let known: i64 = tx
                .query_row(
                    "SELECT COUNT(*) FROM message_structure WHERE session_id = ? AND branch_id = ?",
                    params![ctx.session_id, target],
                    |r| r.get(0),
                )
                .map_err(sql_err)?;
            if known == 0 {
                return Err(AgentsError::user(format!(
                    "Branch '{target}' does not exist"
                )));
            }
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
            let generation = self.branch.lock().expect("branch poisoned").generation;
            self.ctx().set_branch(MAIN_BRANCH.to_string(), generation);
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
            let branch = resolve_read_branch(conn, ctx, branch)?;
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
            let branch = resolve_read_branch(conn, ctx, branch)?;
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
            let branch = resolve_read_branch(conn, ctx, branch)?;
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
        // SQLite's LIKE folds ASCII case only, so `é` does not match `É`.
        let needle = search.to_ascii_lowercase();
        let turns = self.get_conversation_turns(branch_id).await?;
        Ok(turns
            .into_iter()
            .filter(|t| t.full_content.to_ascii_lowercase().contains(&needle))
            .collect())
    }

    /// Tool calls per turn on a branch.
    pub async fn get_tool_usage(&self, branch_id: Option<&str>) -> Result<Vec<ToolUsage>> {
        let branch = branch_id.map(str::to_string);
        self.blocking(move |conn, ctx| {
            let branch = resolve_read_branch(conn, ctx, branch)?;
            let mut stmt = conn
                .prepare(
                    "SELECT tool_name, SUM(usage_count), user_turn_number
                     FROM (
                        SELECT COALESCE(
                                tool_name,
                                CASE WHEN message_type IN ('shell_call', 'apply_patch_call')
                                     THEN message_type END
                               ) AS tool_name, 1 AS usage_count, user_turn_number
                        FROM message_structure
                        WHERE session_id = ?1 AND branch_id = ?2 AND message_type IN (
                            'tool_call', 'function_call', 'computer_call', 'file_search_call',
                            'web_search_call', 'code_interpreter_call', 'tool_search_call',
                            'custom_tool_call', 'mcp_call', 'mcp_approval_request',
                            'shell_call', 'apply_patch_call'
                        )
                        UNION ALL
                        SELECT ms.tool_name, 1 AS usage_count, ms.user_turn_number
                        FROM message_structure ms
                        WHERE ms.session_id = ?1 AND ms.branch_id = ?2
                          AND ms.message_type = 'tool_search_output'
                          AND NOT EXISTS (
                              SELECT 1 FROM message_structure calls
                              WHERE calls.session_id = ms.session_id
                                AND calls.branch_id = ms.branch_id
                                AND calls.user_turn_number = ms.user_turn_number
                                AND calls.tool_name = ms.tool_name
                                AND calls.message_type = 'tool_search_call'
                          )
                     )
                     WHERE tool_name IS NOT NULL
                     GROUP BY tool_name, user_turn_number
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
            let branch = resolve_read_branch(conn, ctx, branch)?;
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
        let tx = write_transaction(conn)?;
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
        self.get_items_for_branch(limit, None).await
    }
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        self.blocking(move |conn, ctx| {
            // Another instance may have cleared the session since this one picked its branch.
            refresh_branch_after_external_clear(conn, ctx, true)?;
            Self::insert_with_structure(conn, ctx, &items)
        })
        .await
    }

    /// Remove the most recent item of the current branch.
    ///
    /// The structure row is always removed; the stored message is removed only when no other
    /// branch still references it. A `turn_usage` row is removed when the pop empties its turn.
    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.blocking(move |conn, ctx| {
            refresh_branch_after_external_clear(conn, ctx, true)?;
            let branch = ctx.branch_id();
            let tx = write_transaction(conn)?;
            // Back-fill the reservations before deleting the rows that are their only durable
            // evidence in a database written before reservations existed.
            ensure_branch_reservations(&tx, ctx)?;
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
                tx.commit().map_err(sql_err)?;
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
            let tx = write_transaction(conn)?;
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
            let tx = write_transaction(conn)?;
            // Back-fill the reservations before deleting their only durable evidence, so the
            // branch ids this session used stay reserved after the clear.
            ensure_branch_reservations(&tx, ctx)?;
            ensure_clear_generations(&tx, ctx)?;
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
            // Bumping the generation invalidates any in-flight switch or branch creation that
            // captured the pre-clear value, in this process or another one.
            tx.execute(
                "UPDATE session_clear_generations SET generation = generation + 1
                 WHERE session_id = ?",
                params![ctx.session_id],
            )
            .map_err(sql_err)?;
            let generation = stored_generation(&tx, ctx)?;
            tx.commit().map_err(sql_err)?;
            // Still under the connection lock, so no other operation sees a cleared session
            // whose pointer still names a deleted branch.
            ctx.set_branch(MAIN_BRANCH.to_string(), generation);
            Ok(())
        })
        .await
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
