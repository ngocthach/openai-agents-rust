//! SQLite-backed session (feature `sqlite`). Uses the same schema as the Python
//! `SQLiteSession`, so databases are interchangeable.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, params};

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
    conn: Arc<Mutex<Connection>>,
    sessions_table: String,
    messages_table: String,
    settings: Option<SessionSettings>,
}

fn sql_err(e: rusqlite::Error) -> AgentsError {
    AgentsError::other(format!("SQLite error: {e}"))
}

fn valid_identifier(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
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
            conn: Arc::new(Mutex::new(conn)),
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
            conn: Arc::new(Mutex::new(conn)),
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

    fn init(&self) -> Result<()> {
        let conn = self.conn.lock().expect("sqlite poisoned");
        conn.execute_batch(&format!(
            "PRAGMA journal_mode=WAL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS {s} (
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
             CREATE INDEX IF NOT EXISTS idx_{m}_session_id ON {m} (session_id, id);",
            s = self.sessions_table,
            m = self.messages_table
        ))
        .map_err(sql_err)
    }

    /// The shared connection, for sessions in this module that extend the base schema.
    pub(super) fn connection(&self) -> &Arc<Mutex<Connection>> {
        &self.conn
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
        tokio::task::spawn_blocking(move || {
            let conn = this.conn.lock().expect("sqlite poisoned");
            f(&conn, &this)
        })
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
        self.blocking(move |conn, s| {
            let rows: Vec<String> = match limit {
                Some(0) => Vec::new(),
                Some(n) => {
                    let mut stmt = conn
                        .prepare(&format!(
                            "SELECT message_data FROM {} WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                            s.messages_table
                        ))
                        .map_err(sql_err)?;
                    let mut v: Vec<String> = stmt
                        .query_map(params![s.session_id, n as i64], |r| r.get(0))
                        .map_err(sql_err)?
                        .filter_map(|r| r.ok())
                        .collect();
                    v.reverse();
                    v
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
        self.blocking(move |conn, s| {
            let row: Option<(i64, String)> = conn
                .query_row(
                    &format!(
                        "SELECT id, message_data FROM {} WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                        s.messages_table
                    ),
                    params![s.session_id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .ok();
            let Some((id, data)) = row else {
                return Ok(None);
            };
            conn.execute(
                &format!("DELETE FROM {} WHERE id = ?", s.messages_table),
                params![id],
            )
            .map_err(sql_err)?;
            Ok(serde_json::from_str(&data).ok().map(InputItem))
        })
        .await
    }

    async fn clear_session(&self) -> Result<()> {
        self.blocking(move |conn, s| {
            conn.execute(
                &format!("DELETE FROM {} WHERE session_id = ?", s.messages_table),
                params![s.session_id],
            )
            .map_err(sql_err)?;
            conn.execute(
                &format!("DELETE FROM {} WHERE session_id = ?", s.sessions_table),
                params![s.session_id],
            )
            .map_err(sql_err)?;
            Ok(())
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
        let handles: Vec<String> = handles.iter().map(|h| h.0.clone()).collect();
        self.blocking(move |conn, s| {
            let tx = conn.unchecked_transaction().map_err(sql_err)?;
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
}
