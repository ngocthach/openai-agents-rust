//! SQLite-backed session (feature `sqlite`). Uses the same schema as the Python
//! `SQLiteSession`, so databases are interchangeable.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

use super::{CompactionSnapshot, Session, SessionSettings, SnapshotHandle};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;

/// Conversation history stored in SQLite (file-backed or in-memory).
///
/// The database is trusted storage: rows are not authenticated, and the session id selects
/// history but does not authorize access to it.
#[derive(Debug, Clone)]
pub struct SqliteSession {
    session_id: String,
    conn: SharedConnection,
    sessions_table: String,
    messages_table: String,
    settings: Option<SessionSettings>,
}

/// The connection a session and its clones share; `None` once [`SqliteSession::close`] ran.
pub(super) type SharedConnection = Arc<Mutex<Option<Connection>>>;

/// Run `f` on the shared connection, or fail with `SQLiteSession is closed` after a close.
pub(super) fn with_open_connection<T>(
    conn: &Mutex<Option<Connection>>,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    let guard = conn.lock().expect("sqlite poisoned");
    match guard.as_ref() {
        Some(conn) => f(conn),
        None => Err(closed_error()),
    }
}

fn closed_error() -> AgentsError {
    AgentsError::other("SQLiteSession is closed")
}

fn sql_err(e: rusqlite::Error) -> AgentsError {
    AgentsError::other(format!("SQLite error: {e}"))
}

/// A table name is a bare identifier (Unicode letters, digits and `_`, as SQLite and Python accept)
/// or a SQL double-quoted one (`"my table"`, with `""`
/// for an embedded quote).
///
/// Python interpolates the configured names into SQL verbatim, so its released `SQLiteSession`
/// accepts SQL-quoted identifiers (and, by accident, any SQL fragment). Rust accepts the same
/// quoted identifiers but nothing else, so a table name can never inject SQL.
fn valid_identifier(name: &str) -> bool {
    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return true;
    }
    let Some(inner) = name
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    else {
        return false;
    };
    !inner.is_empty() && inner.replace("\"\"", "").find('"').is_none()
}

/// The text a quoted identifier names (`"a""b"` is `a"b`); a bare identifier is returned as is.
pub(super) fn unquoted_identifier(name: &str) -> String {
    match name
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    {
        Some(inner) => inner.replace("\"\"", "\""),
        None => name.to_string(),
    }
}

/// The name of the `(session_id, id)` index of `messages_table`, always safely quoted.
fn messages_index_name(messages_table: &str) -> String {
    let name = format!("idx_{}_session_id", unquoted_identifier(messages_table));
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Switch the connection to WAL mode and enable foreign keys.
///
/// Concurrent first connections to one new file can fail the WAL switch with "database is
/// locked" before the busy timeout applies, so a busy or locked answer is retried for a few
/// seconds; any other failure is returned at once.
fn enable_wal(conn: &Connection) -> Result<()> {
    const RETRY_FOR: std::time::Duration = std::time::Duration::from_secs(5);
    let started = std::time::Instant::now();
    loop {
        match conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;") {
            Ok(()) => return Ok(()),
            Err(rusqlite::Error::SqliteFailure(failure, _))
                if matches!(
                    failure.code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                ) && started.elapsed() < RETRY_FOR =>
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(error) => return Err(sql_err(error)),
        }
    }
}

impl SqliteSession {
    /// An in-memory session (lost when dropped).
    pub fn in_memory(session_id: impl Into<String>) -> Result<Self> {
        Self::from_connection(session_id, Connection::open_in_memory().map_err(sql_err)?)
    }

    /// A file-backed session.
    pub fn open(session_id: impl Into<String>, db_path: impl AsRef<Path>) -> Result<Self> {
        Self::from_connection(session_id, Connection::open(db_path).map_err(sql_err)?)
    }

    /// Use custom table names (ASCII letters, digits and underscores only).
    pub fn with_tables(
        session_id: impl Into<String>,
        db_path: impl AsRef<Path>,
        sessions_table: &str,
        messages_table: &str,
    ) -> Result<Self> {
        if !valid_identifier(sessions_table) || !valid_identifier(messages_table) {
            return Err(AgentsError::user("Invalid SQLite table name"));
        }
        let conn = Connection::open(db_path).map_err(sql_err)?;
        let s = SqliteSession {
            session_id: session_id.into(),
            conn: Arc::new(Mutex::new(Some(conn))),
            sessions_table: sessions_table.to_string(),
            messages_table: messages_table.to_string(),
            settings: None,
        };
        s.init()?;
        Ok(s)
    }

    fn from_connection(session_id: impl Into<String>, conn: Connection) -> Result<Self> {
        let s = SqliteSession {
            session_id: session_id.into(),
            conn: Arc::new(Mutex::new(Some(conn))),
            sessions_table: "agent_sessions".into(),
            messages_table: "agent_messages".into(),
            settings: None,
        };
        s.init()?;
        Ok(s)
    }

    /// Set default session settings.
    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// A session with another id sharing this database connection.
    pub fn for_session(&self, session_id: impl Into<String>) -> Self {
        SqliteSession {
            session_id: session_id.into(),
            ..self.clone()
        }
    }

    /// A file-backed session whose tables are not created yet; the caller creates them with
    /// [`SqliteSession::create_tables_on`] inside a transaction of its own.
    pub(super) fn open_uninitialized(
        session_id: impl Into<String>,
        db_path: impl AsRef<Path>,
        sessions_table: &str,
        messages_table: &str,
    ) -> Result<Self> {
        if !valid_identifier(sessions_table) || !valid_identifier(messages_table) {
            return Err(AgentsError::user("Invalid SQLite table name"));
        }
        let conn = Connection::open(db_path).map_err(sql_err)?;
        enable_wal(&conn)?;
        Ok(SqliteSession {
            session_id: session_id.into(),
            conn: Arc::new(Mutex::new(Some(conn))),
            sessions_table: sessions_table.to_string(),
            messages_table: messages_table.to_string(),
            settings: None,
        })
    }

    fn init(&self) -> Result<()> {
        with_open_connection(&self.conn, |conn| {
            enable_wal(conn)?;
            self.create_tables_on(conn)
        })
    }

    /// Create the session and message tables on `conn`.
    pub(super) fn create_tables_on(&self, conn: &Connection) -> Result<()> {
        conn.execute_batch(&format!(
            "CREATE TABLE IF NOT EXISTS {s} (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             );
             CREATE TABLE IF NOT EXISTS {m} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES {s} (session_id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS {i} ON {m} (session_id, id);",
            s = self.sessions_table,
            m = self.messages_table,
            i = messages_index_name(&self.messages_table)
        ))
        .map_err(sql_err)
    }

    /// The shared connection, for sessions in this module that extend the base schema.
    pub(super) fn connection(&self) -> &SharedConnection {
        &self.conn
    }

    /// Close the database connection. Port of `SQLiteSession.close`.
    ///
    /// The session is terminal from the first call: every later operation, including
    /// `add_items` with no items, fails with `SQLiteSession is closed`, and repeated calls are
    /// safe. Clones and the sessions [`SqliteSession::for_session`] made share the connection,
    /// so they are closed with it. An in-memory database is lost.
    pub fn close(&self) {
        // Dropping the connection closes it; a failed close is only possible while a statement
        // is still open, which cannot happen outside the lock.
        drop(self.conn.lock().expect("sqlite poisoned").take());
    }

    /// True once [`SqliteSession::close`] closed the connection.
    pub fn is_closed(&self) -> bool {
        self.conn.lock().expect("sqlite poisoned").is_none()
    }

    /// Fail with `SQLiteSession is closed` after [`SqliteSession::close`].
    pub(super) fn check_not_closed(&self) -> Result<()> {
        if self.is_closed() {
            return Err(closed_error());
        }
        Ok(())
    }

    /// The name of the session metadata table.
    pub(super) fn sessions_table(&self) -> &str {
        &self.sessions_table
    }

    /// The name of the message table.
    pub(super) fn messages_table(&self) -> &str {
        &self.messages_table
    }

    async fn blocking<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&Connection, &SqliteSession) -> Result<T> + Send + 'static,
    {
        let this = self.clone();
        tokio::task::spawn_blocking(move || with_open_connection(&this.conn, |conn| f(conn, &this)))
            .await
            .map_err(|e| AgentsError::other(format!("SQLite task failed: {e}")))?
    }
}

#[async_trait]
impl Session for SqliteSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let limit = super::resolve_session_limit(limit, self.session_settings().as_ref());
        self.blocking(move |conn, s| {
            let rows: Vec<String> = match limit {
                Some(0) => Vec::new(),
                Some(n) => {
                    // Expand the fetch window while corrupt rows sit among the newest entries, so
                    // `limit` counts readable conversation items rather than stored rows. Port of
                    // the widening loop in Python's `SQLiteSession.get_items`.
                    let mut window = n;
                    loop {
                        let mut stmt = conn
                            .prepare(&format!(
                                "SELECT message_data FROM {} WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                                s.messages_table
                            ))
                            .map_err(sql_err)?;
                        let mut fetched: Vec<String> = stmt
                            .query_map(params![s.session_id, window as i64], |r| r.get(0))
                            .map_err(sql_err)?
                            .filter_map(|r| r.ok())
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
                            "SELECT message_data FROM {} WHERE session_id = ? ORDER BY id ASC",
                            s.messages_table
                        ))
                        .map_err(sql_err)?;
                    stmt.query_map(params![s.session_id], |r| r.get(0))
                        .map_err(sql_err)?
                        .filter_map(|r| r.ok())
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
        // Checked before the empty-list fast path, which would otherwise succeed on a closed
        // session.
        self.check_not_closed()?;
        if items.is_empty() {
            return Ok(());
        }
        self.blocking(move |conn, s| {
            let tx = conn.unchecked_transaction().map_err(sql_err)?;
            tx.execute(
                &format!(
                    "INSERT OR IGNORE INTO {} (session_id) VALUES (?)",
                    s.sessions_table
                ),
                params![s.session_id],
            )
            .map_err(sql_err)?;
            for item in &items {
                tx.execute(
                    &format!(
                        "INSERT INTO {} (session_id, message_data) VALUES (?, ?)",
                        s.messages_table
                    ),
                    params![s.session_id, serde_json::to_string(&item.0)?],
                )
                .map_err(sql_err)?;
            }
            tx.execute(
                &format!(
                    "UPDATE {} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    s.sessions_table
                ),
                params![s.session_id],
            )
            .map_err(sql_err)?;
            tx.commit().map_err(sql_err)
        })
        .await
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        self.pop_item_validated(Box::new(|_| Ok(()))).await
    }

    /// The store keeps one history per session id and ignores the run's context, so a run's
    /// validated pop is the same atomic claim.
    async fn pop_item_validated_with_context(
        &self,
        _ctx: super::SessionRunContext<'_>,
        validate: super::PopValidator,
    ) -> Result<Option<InputItem>> {
        self.pop_item_validated(validate).await
    }

    /// Claim the newest row and run `validate` on it inside the transaction that owns its
    /// removal, so a rejected item keeps its row id and its position.
    ///
    /// A row whose JSON no longer parses is dropped and the search continues, as in Python.
    async fn pop_item_validated(&self, validate: super::PopValidator) -> Result<Option<InputItem>> {
        self.blocking(move |conn, s| {
            loop {
                // Immediate: the claim reads the newest row and then deletes it, so a deferred
                // transaction would fail instead of waiting when another connection wrote in
                // between.
                let tx = rusqlite::Transaction::new_unchecked(
                    conn,
                    rusqlite::TransactionBehavior::Immediate,
                )
                .map_err(sql_err)?;
                let row: Option<(i64, String)> = tx
                    .query_row(
                        &format!(
                            "SELECT id, message_data FROM {} WHERE session_id = ? \
                             ORDER BY id DESC LIMIT 1",
                            s.messages_table
                        ),
                        params![s.session_id],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()
                    .map_err(sql_err)?;
                let Some((id, data)) = row else {
                    return Ok(None);
                };
                tx.execute(
                    &format!("DELETE FROM {} WHERE id = ?", s.messages_table),
                    params![id],
                )
                .map_err(sql_err)?;
                let Ok(value) = serde_json::from_str(&data) else {
                    // Corrupted row: commit its removal and keep looking for a readable item.
                    tx.commit().map_err(sql_err)?;
                    continue;
                };
                let item = InputItem(value);
                // An `Err` here drops the transaction, which rolls the deletion back.
                validate(&item)?;
                tx.commit().map_err(sql_err)?;
                return Ok(Some(item));
            }
        })
        .await
    }

    async fn clear_session(&self) -> Result<()> {
        self.blocking(move |conn, s| {
            // One transaction: a failure on the sessions row restores the messages deleted
            // before it, and no write lock is left behind.
            let tx = conn.unchecked_transaction().map_err(sql_err)?;
            tx.execute(
                &format!("DELETE FROM {} WHERE session_id = ?", s.messages_table),
                params![s.session_id],
            )
            .map_err(sql_err)?;
            tx.execute(
                &format!("DELETE FROM {} WHERE session_id = ?", s.sessions_table),
                params![s.session_id],
            )
            .map_err(sql_err)?;
            tx.commit().map_err(sql_err)
        })
        .await
    }

    /// Row ids make stable handles: SQLite never reuses an `AUTOINCREMENT` id.
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        self.blocking(move |conn, s| {
            // One row beyond the window tells us whether it reached the start of the history.
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT id, message_data FROM {} WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    s.messages_table
                ))
                .map_err(sql_err)?;
            let mut rows: Vec<(i64, String)> = stmt
                .query_map(params![s.session_id, window_limit(limit)], |r| {
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
        self.replace_compaction_suffix_pruning(handles, items, Arc::new(|_: &InputItem| false))
            .await
    }

    /// Replace the tail and, in the same transaction, delete the leading run of items
    /// `prune_prefix` accepts, in bounded batches.
    async fn replace_compaction_suffix_pruning(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
        prune_prefix: super::PrunePrefix,
    ) -> Result<bool> {
        let handles: Vec<String> = handles.iter().map(|h| h.0.clone()).collect();
        self.blocking(move |conn, s| {
            let tx = rusqlite::Transaction::new_unchecked(
                conn,
                rusqlite::TransactionBehavior::Immediate,
            )
            .map_err(sql_err)?;
            let mut stmt = tx
                .prepare(&format!(
                    "SELECT id FROM {} WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    s.messages_table
                ))
                .map_err(sql_err)?;
            let mut tail: Vec<i64> = stmt
                .query_map(params![s.session_id, handles.len() as i64], |r| r.get(0))
                .map_err(sql_err)?
                .filter_map(std::result::Result::ok)
                .collect();
            drop(stmt);
            tail.reverse();
            if tail.len() != handles.len()
                || tail
                    .iter()
                    .zip(&handles)
                    .any(|(id, handle)| id.to_string() != *handle)
            {
                return Ok(false);
            }
            if let Some(&first) = tail.first() {
                prune_prefix_before(&tx, s, first, prune_prefix.as_ref())?;
            }
            for id in &tail {
                tx.execute(
                    &format!("DELETE FROM {} WHERE id = ?", s.messages_table),
                    params![id],
                )
                .map_err(sql_err)?;
            }
            for item in &items {
                tx.execute(
                    &format!(
                        "INSERT INTO {} (session_id, message_data) VALUES (?, ?)",
                        s.messages_table
                    ),
                    params![s.session_id, serde_json::to_string(&item.0)?],
                )
                .map_err(sql_err)?;
            }
            tx.execute(
                &format!(
                    "UPDATE {} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    s.sessions_table
                ),
                params![s.session_id],
            )
            .map_err(sql_err)?;
            tx.commit().map_err(sql_err)?;
            Ok(true)
        })
        .await
    }
}

use super::PRUNE_PREFIX_BATCH;

/// Delete the leading run of rows before `before_id` that `prune_prefix` accepts.
///
/// Stops at the first row the predicate rejects or whose JSON does not parse, so a retained row
/// always keeps everything after it. Port of the `prune_prefix` loop in Python's
/// `SQLiteSession._get_compaction_snapshot.replace_suffix`.
pub(super) fn prune_prefix_before(
    tx: &rusqlite::Transaction<'_>,
    s: &SqliteSession,
    before_id: i64,
    prune_prefix: &(dyn Fn(&InputItem) -> bool + Send + Sync),
) -> Result<()> {
    loop {
        let mut stmt = tx
            .prepare(&format!(
                "SELECT id, message_data FROM {} WHERE session_id = ? AND id < ? \
                 ORDER BY id LIMIT ?",
                s.messages_table
            ))
            .map_err(sql_err)?;
        let batch: Vec<(i64, String)> = stmt
            .query_map(params![s.session_id, before_id, PRUNE_PREFIX_BATCH], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .map_err(sql_err)?
            .filter_map(std::result::Result::ok)
            .collect();
        drop(stmt);
        let mut expired_end: Option<i64> = None;
        for (id, data) in &batch {
            let Ok(value) = serde_json::from_str(data) else {
                break;
            };
            if !prune_prefix(&InputItem(value)) {
                break;
            }
            expired_end = Some(*id);
        }
        let Some(expired_end) = expired_end else {
            return Ok(());
        };
        tx.execute(
            &format!(
                "DELETE FROM {} WHERE session_id = ? AND id <= ?",
                s.messages_table
            ),
            params![s.session_id, expired_end],
        )
        .map_err(sql_err)?;
        // A partially reclaimed batch, or one that did not fill, means the run ended here.
        if batch.last().map(|(id, _)| *id) != Some(expired_end)
            || (batch.len() as i64) < PRUNE_PREFIX_BATCH
        {
            return Ok(());
        }
    }
}

/// One row more than the window, so a snapshot can tell whether it reached the start of the
/// history. Saturates rather than wrapping, since `usize::MAX` is a natural "everything".
pub(super) fn window_limit(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(i64::MAX).saturating_add(1)
}

/// Build a snapshot from `(row id, message JSON)` pairs, oldest first.
///
/// A row whose JSON no longer parses cannot be handed to the compaction API, so it and
/// everything before it are excluded from the replaceable window.
pub(super) fn snapshot_from_rows(rows: Vec<(i64, String)>, complete: bool) -> CompactionSnapshot {
    let mut items = Vec::new();
    let mut handles = Vec::new();
    let mut intact = complete;
    for (id, data) in rows {
        match serde_json::from_str(&data) {
            Ok(value) => {
                items.push(InputItem(value));
                handles.push(SnapshotHandle(id.to_string()));
            }
            Err(_) => {
                items.clear();
                handles.clear();
                intact = false;
            }
        }
    }
    CompactionSnapshot {
        items,
        handles,
        complete: intact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sqlite_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.db");
        let s = SqliteSession::open("u1", &path).unwrap();
        s.add_items(vec![
            InputItem::user("a"),
            InputItem::assistant("b"),
            InputItem::user("c"),
        ])
        .await
        .unwrap();
        let other = s.for_session("u2");
        other.add_items(vec![InputItem::user("x")]).await.unwrap();
        assert_eq!(s.get_items(None).await.unwrap().len(), 3);
        assert_eq!(
            s.get_items(Some(2)).await.unwrap()[0]
                .text_content()
                .as_deref(),
            Some("b")
        );
        assert_eq!(
            s.pop_item()
                .await
                .unwrap()
                .unwrap()
                .text_content()
                .as_deref(),
            Some("c")
        );
        drop(s);
        let reopened = SqliteSession::open("u1", &path).unwrap();
        assert_eq!(reopened.get_items(None).await.unwrap().len(), 2);
        reopened.clear_session().await.unwrap();
        assert!(reopened.get_items(None).await.unwrap().is_empty());
        assert_eq!(other.get_items(None).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_rejected_pop_leaves_the_row_in_place() {
        let s = SqliteSession::in_memory("u1").unwrap();
        s.add_items(vec![InputItem::user("a"), InputItem::user("b")])
            .await
            .unwrap();

        let err = s
            .pop_item_validated(Box::new(|_| Err(AgentsError::other("unauthenticated"))))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unauthenticated"), "{err}");
        // The claim was rolled back, so both rows and their order survive.
        let left = s.get_items(None).await.unwrap();
        assert_eq!(left.len(), 2);
        assert_eq!(left[1].text_content().as_deref(), Some("b"));

        let popped = s
            .pop_item_validated(Box::new(|_| Ok(())))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(popped.text_content().as_deref(), Some("b"));
        assert_eq!(s.get_items(None).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_successful_replacement_reclaims_the_accepted_prefix() {
        let s = SqliteSession::in_memory("u1").unwrap();
        s.add_items(vec![
            InputItem::user("a"),
            InputItem::user("b"),
            InputItem::user("c"),
            InputItem::user("d"),
        ])
        .await
        .unwrap();
        let snapshot = s.compaction_snapshot(10).await.unwrap().unwrap();
        assert_eq!(snapshot.handles.len(), 4);

        // Accepts "a" and "b"; "c" stops the scan, so "a" and "b" go and "c" stays.
        let prune: super::super::PrunePrefix = Arc::new(|item: &InputItem| {
            matches!(item.text_content().as_deref(), Some("a" | "b" | "c"))
                && item.text_content().as_deref() != Some("c")
        });
        let replaced = s
            .replace_compaction_suffix_pruning(
                &snapshot.handles[3..],
                vec![InputItem::user("summary")],
                prune,
            )
            .await
            .unwrap();
        assert!(replaced);

        let left: Vec<String> = s
            .get_items(None)
            .await
            .unwrap()
            .iter()
            .filter_map(|i| i.text_content())
            .collect();
        assert_eq!(left, ["c", "summary"]);
    }

    #[tokio::test]
    async fn a_refused_replacement_reclaims_nothing() {
        let s = SqliteSession::in_memory("u1").unwrap();
        s.add_items(vec![InputItem::user("a"), InputItem::user("b")])
            .await
            .unwrap();
        let snapshot = s.compaction_snapshot(10).await.unwrap().unwrap();
        s.add_items(vec![InputItem::user("c")]).await.unwrap();

        let prune: super::super::PrunePrefix = Arc::new(|_: &InputItem| true);
        let replaced = s
            .replace_compaction_suffix_pruning(&snapshot.handles, Vec::new(), prune)
            .await
            .unwrap();
        assert!(!replaced);
        assert_eq!(s.get_items(None).await.unwrap().len(), 3);
    }

    #[test]
    fn identifiers_are_bare_or_sql_quoted() {
        assert!(valid_identifier("agent_messages"));
        assert!(valid_identifier("\"quoted_sessions\""));
        assert!(valid_identifier("\"two words\""));
        assert!(valid_identifier("\"a\"\"b\""));
        for bad in [
            "",
            "bad table",
            "t; DROP TABLE x",
            "\"\"",
            "\"unterminated",
            "\"a\"b\"",
            "[bracket]",
        ] {
            assert!(!valid_identifier(bad), "{bad}");
        }
        assert_eq!(unquoted_identifier("\"a\"\"b\""), "a\"b");
        assert_eq!(unquoted_identifier("plain"), "plain");
        assert_eq!(messages_index_name("plain"), "\"idx_plain_session_id\"");
        assert_eq!(
            messages_index_name("\"two words\""),
            "\"idx_two words_session_id\""
        );
    }

    #[tokio::test]
    async fn quoted_table_names_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quoted.db");
        let session =
            SqliteSession::with_tables("u1", &path, "\"my sessions\"", "\"my messages\"").unwrap();
        session
            .add_items(vec![InputItem::user("hi")])
            .await
            .unwrap();
        assert_eq!(session.get_items(None).await.unwrap().len(), 1);
        // A second open reuses the same tables and index without error.
        let again =
            SqliteSession::with_tables("u1", &path, "\"my sessions\"", "\"my messages\"").unwrap();
        assert_eq!(again.get_items(None).await.unwrap().len(), 1);
    }
}
