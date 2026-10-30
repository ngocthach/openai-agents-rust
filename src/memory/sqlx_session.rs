//! PostgreSQL / MySQL session on top of `sqlx` (feature `sqlx`).
//! Mirrors `agents.extensions.memory.sqlalchemy_session`.

use std::fmt;

use async_trait::async_trait;
use sqlx::AnyPool;
use sqlx::{AssertSqlSafe, Row};

use super::{CompactionSnapshot, Session, SessionSettings, SnapshotHandle};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;

/// Default name of the table holding session metadata.
pub const DEFAULT_SESSIONS_TABLE: &str = "agent_sessions";
/// Default name of the table holding conversation history.
pub const DEFAULT_MESSAGES_TABLE: &str = "agent_messages";

fn sqlx_err(e: sqlx::Error) -> AgentsError {
    AgentsError::other(format!("SQL error: {e}"))
}

/// One row more than the window, so a snapshot can tell whether it reached the start of the
/// history. Saturates rather than wrapping, since `usize::MAX` is a natural "everything".
fn window_limit(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(i64::MAX).saturating_add(1)
}

fn valid_identifier(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The SQL dialect a [`SqlxSession`] generates statements for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlxDialect {
    /// PostgreSQL, with `$n` placeholders.
    Postgres,
    /// MySQL or MariaDB, with `?` placeholders.
    MySql,
    /// SQLite, with `?` placeholders.
    ///
    /// Shares the `libsqlite3-sys` build with [`SqliteSession`](super::SqliteSession), so both
    /// sessions can be used in one process. `SqliteSession` is the lighter choice when SQLite is
    /// the only backend; this variant exists so one `sqlx` code path covers the three databases
    /// Python's `SQLAlchemySession` covers.
    Sqlite,
}

impl SqlxDialect {
    /// The dialect a connection URL selects.
    pub fn from_url(url: &str) -> Result<Self> {
        let scheme = url.split(':').next().unwrap_or_default();
        match scheme {
            "postgres" | "postgresql" => Ok(SqlxDialect::Postgres),
            "mysql" | "mariadb" => Ok(SqlxDialect::MySql),
            "sqlite" => Ok(SqlxDialect::Sqlite),
            other => Err(AgentsError::user(format!(
                "Unsupported database URL scheme for SqlxSession: {other}"
            ))),
        }
    }

    /// The placeholder for bind parameter `n` (one-based).
    pub fn placeholder(self, n: usize) -> String {
        match self {
            SqlxDialect::Postgres => format!("${n}"),
            SqlxDialect::MySql | SqlxDialect::Sqlite => "?".to_string(),
        }
    }
}

/// Statement templates for one dialect and table pair.
#[derive(Debug, Clone)]
struct Schema {
    dialect: SqlxDialect,
    sessions_table: String,
    messages_table: String,
}

impl Schema {
    /// `CREATE TABLE` / index statements, run in order.
    fn create_tables(&self) -> Vec<String> {
        let (s, m) = (&self.sessions_table, &self.messages_table);
        match self.dialect {
            SqlxDialect::Postgres => vec![
                format!(
                    "CREATE TABLE IF NOT EXISTS {s} (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                     )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {m} (
                        id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES {s} (session_id) ON DELETE CASCADE,
                        message_data TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                     )"
                ),
                format!(
                    "CREATE INDEX IF NOT EXISTS idx_{m}_session_time ON {m} (session_id, created_at)"
                ),
            ],
            SqlxDialect::MySql => vec![
                format!(
                    "CREATE TABLE IF NOT EXISTS {s} (
                        session_id VARCHAR(255) PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                     )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {m} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        message_data LONGTEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_{m}_session_time (session_id, created_at),
                        FOREIGN KEY (session_id) REFERENCES {s} (session_id) ON DELETE CASCADE
                     )"
                ),
            ],
            // Identical to the schema `SqliteSession` creates, so the two are interchangeable
            // on the same file.
            SqlxDialect::Sqlite => vec![
                format!(
                    "CREATE TABLE IF NOT EXISTS {s} (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     )"
                ),
                format!(
                    "CREATE TABLE IF NOT EXISTS {m} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        message_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES {s} (session_id) ON DELETE CASCADE
                     )"
                ),
                format!("CREATE INDEX IF NOT EXISTS idx_{m}_session_id ON {m} (session_id, id)"),
            ],
        }
    }

    /// Insert the session row, ignoring an existing one.
    fn insert_session(&self) -> String {
        let (s, p1) = (&self.sessions_table, self.dialect.placeholder(1));
        match self.dialect {
            SqlxDialect::Postgres => format!(
                "INSERT INTO {s} (session_id) VALUES ({p1}) ON CONFLICT (session_id) DO NOTHING"
            ),
            SqlxDialect::MySql => format!("INSERT IGNORE INTO {s} (session_id) VALUES ({p1})"),
            SqlxDialect::Sqlite => format!("INSERT OR IGNORE INTO {s} (session_id) VALUES ({p1})"),
        }
    }

    fn insert_message(&self) -> String {
        format!(
            "INSERT INTO {} (session_id, message_data) VALUES ({}, {})",
            self.messages_table,
            self.dialect.placeholder(1),
            self.dialect.placeholder(2)
        )
    }

    fn touch_session(&self) -> String {
        format!(
            "UPDATE {} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = {}",
            self.sessions_table,
            self.dialect.placeholder(1)
        )
    }

    fn select_all(&self) -> String {
        format!(
            "SELECT message_data FROM {} WHERE session_id = {} ORDER BY created_at ASC, id ASC",
            self.messages_table,
            self.dialect.placeholder(1)
        )
    }

    fn select_latest(&self) -> String {
        format!(
            "SELECT message_data FROM {} WHERE session_id = {} \
             ORDER BY created_at DESC, id DESC LIMIT {}",
            self.messages_table,
            self.dialect.placeholder(1),
            self.dialect.placeholder(2)
        )
    }

    /// The newest `n` rows with their ids, newest first.
    fn select_window(&self) -> String {
        format!(
            "SELECT id, message_data FROM {} WHERE session_id = {} \
             ORDER BY created_at DESC, id DESC LIMIT {}",
            self.messages_table,
            self.dialect.placeholder(1),
            self.dialect.placeholder(2)
        )
    }

    /// The ids of the newest `n` rows, newest first.
    fn select_window_ids(&self) -> String {
        format!(
            "SELECT id FROM {} WHERE session_id = {} \
             ORDER BY created_at DESC, id DESC LIMIT {}",
            self.messages_table,
            self.dialect.placeholder(1),
            self.dialect.placeholder(2)
        )
    }

    fn select_tail(&self) -> String {
        format!(
            "SELECT id, message_data FROM {} WHERE session_id = {} \
             ORDER BY created_at DESC, id DESC LIMIT 1",
            self.messages_table,
            self.dialect.placeholder(1)
        )
    }

    fn delete_message(&self) -> String {
        format!(
            "DELETE FROM {} WHERE id = {}",
            self.messages_table,
            self.dialect.placeholder(1)
        )
    }

    /// The oldest rows before a given id, oldest first, bounded by a batch size.
    fn select_prefix(&self) -> String {
        format!(
            "SELECT id, message_data FROM {} WHERE session_id = {} AND id < {} \
             ORDER BY id ASC LIMIT {}",
            self.messages_table,
            self.dialect.placeholder(1),
            self.dialect.placeholder(2),
            self.dialect.placeholder(3)
        )
    }

    /// Delete every row of this session up to and including an id.
    fn delete_prefix(&self) -> String {
        format!(
            "DELETE FROM {} WHERE session_id = {} AND id <= {}",
            self.messages_table,
            self.dialect.placeholder(1),
            self.dialect.placeholder(2)
        )
    }

    fn delete_messages(&self) -> String {
        format!(
            "DELETE FROM {} WHERE session_id = {}",
            self.messages_table,
            self.dialect.placeholder(1)
        )
    }

    fn delete_session(&self) -> String {
        format!(
            "DELETE FROM {} WHERE session_id = {}",
            self.sessions_table,
            self.dialect.placeholder(1)
        )
    }
}

/// Conversation history stored in PostgreSQL, MySQL or SQLite through `sqlx`.
///
/// The schema matches the Python `SQLAlchemySession`: an `agent_sessions` table keyed by
/// `session_id` with `created_at`/`updated_at`, and an `agent_messages` table holding one JSON
/// item per row with an `idx_agent_messages_session_time` index on `(session_id, created_at)`.
/// Databases are therefore interchangeable with the Python backend.
///
/// # SQLite
///
/// `sqlite:` URLs are supported. `sqlx-sqlite` and the `rusqlite` behind
/// [`SqliteSession`](super::SqliteSession) resolve to the same `libsqlite3-sys`, so both link
/// one copy of the C library and can be used in the same process. `SqliteSession` remains the
/// lighter choice when SQLite is the only backend.
pub struct SqlxSession {
    session_id: String,
    pool: AnyPool,
    schema: Schema,
    settings: Option<SessionSettings>,
}

impl fmt::Debug for SqlxSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SqlxSession")
            .field("session_id", &self.session_id)
            .field("dialect", &self.schema.dialect)
            .field("sessions_table", &self.schema.sessions_table)
            .field("messages_table", &self.schema.messages_table)
            .finish_non_exhaustive()
    }
}

impl SqlxSession {
    /// Connect to `url` and, when `create_tables` is set, create the schema if it is missing.
    ///
    /// `url` must be a PostgreSQL, MySQL or SQLite URL.
    pub async fn from_url(
        session_id: impl Into<String>,
        url: &str,
        create_tables: bool,
    ) -> Result<Self> {
        let dialect = SqlxDialect::from_url(url)?;
        sqlx::any::install_default_drivers();
        let pool = AnyPool::connect(url).await.map_err(sqlx_err)?;
        Self::with_pool(session_id, pool, dialect, create_tables).await
    }

    /// Use a pool the application already manages.
    pub async fn with_pool(
        session_id: impl Into<String>,
        pool: AnyPool,
        dialect: SqlxDialect,
        create_tables: bool,
    ) -> Result<Self> {
        let session = SqlxSession {
            session_id: session_id.into(),
            pool,
            schema: Schema {
                dialect,
                sessions_table: DEFAULT_SESSIONS_TABLE.to_string(),
                messages_table: DEFAULT_MESSAGES_TABLE.to_string(),
            },
            settings: None,
        };
        if create_tables {
            session.create_tables().await?;
        }
        Ok(session)
    }

    /// Use custom table names (ASCII letters, digits and underscores only).
    ///
    /// Call this before [`SqlxSession::create_tables`]; the constructors create the default
    /// tables when asked to.
    pub fn with_tables(mut self, sessions_table: &str, messages_table: &str) -> Result<Self> {
        if !valid_identifier(sessions_table) || !valid_identifier(messages_table) {
            return Err(AgentsError::user("Invalid SQL table name"));
        }
        self.schema.sessions_table = sessions_table.to_string();
        self.schema.messages_table = messages_table.to_string();
        Ok(self)
    }

    /// Set default session settings.
    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// The connection pool.
    pub fn pool(&self) -> &AnyPool {
        &self.pool
    }

    /// The dialect statements are generated for.
    pub fn dialect(&self) -> SqlxDialect {
        self.schema.dialect
    }

    /// Create the session and message tables if they do not exist.
    pub async fn create_tables(&self) -> Result<()> {
        for statement in self.schema.create_tables() {
            sqlx::query(AssertSqlSafe(statement))
                .execute(&self.pool)
                .await
                .map_err(sqlx_err)?;
        }
        Ok(())
    }

    /// Decode serialized rows, skipping corrupted ones.
    fn decode(raw: &[String]) -> Vec<InputItem> {
        raw.iter()
            .filter_map(|r| serde_json::from_str(r).ok().map(InputItem))
            .collect()
    }
}

#[async_trait]
impl Session for SqlxSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let limit = super::resolve_session_limit(limit, self.session_settings().as_ref());
        let Some(limit) = limit else {
            let rows = sqlx::query(AssertSqlSafe(self.schema.select_all()))
                .bind(self.session_id.clone())
                .fetch_all(&self.pool)
                .await
                .map_err(sqlx_err)?;
            let raw: Vec<String> = rows.iter().filter_map(|r| r.try_get(0).ok()).collect();
            return Ok(Self::decode(&raw));
        };
        if limit == 0 {
            return Ok(Vec::new());
        }
        // Corrupted rows must not consume the caller's budget, so widen the window until
        // `limit` valid items are found or the history is exhausted.
        let mut window = limit;
        loop {
            let rows = sqlx::query(AssertSqlSafe(self.schema.select_latest()))
                .bind(self.session_id.clone())
                .bind(window as i64)
                .fetch_all(&self.pool)
                .await
                .map_err(sqlx_err)?;
            let mut raw: Vec<String> = rows.iter().filter_map(|r| r.try_get(0).ok()).collect();
            raw.reverse();
            let items = Self::decode(&raw);
            if items.len() >= limit {
                return Ok(items[items.len() - limit..].to_vec());
            }
            if rows.len() < window {
                return Ok(items);
            }
            window = window.saturating_mul(2);
        }
    }

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let mut tx = self.pool.begin().await.map_err(sqlx_err)?;
        sqlx::query(AssertSqlSafe(self.schema.insert_session()))
            .bind(self.session_id.clone())
            .execute(&mut *tx)
            .await
            .map_err(sqlx_err)?;
        for item in &items {
            sqlx::query(AssertSqlSafe(self.schema.insert_message()))
                .bind(self.session_id.clone())
                .bind(serde_json::to_string(item.as_value())?)
                .execute(&mut *tx)
                .await
                .map_err(sqlx_err)?;
        }
        sqlx::query(AssertSqlSafe(self.schema.touch_session()))
            .bind(self.session_id.clone())
            .execute(&mut *tx)
            .await
            .map_err(sqlx_err)?;
        tx.commit().await.map_err(sqlx_err)
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        let mut tx = self.pool.begin().await.map_err(sqlx_err)?;
        let row = sqlx::query(AssertSqlSafe(self.schema.select_tail()))
            .bind(self.session_id.clone())
            .fetch_optional(&mut *tx)
            .await
            .map_err(sqlx_err)?;
        let Some(row) = row else {
            tx.commit().await.map_err(sqlx_err)?;
            return Ok(None);
        };
        let id: i64 = row.try_get(0).map_err(sqlx_err)?;
        let data: String = row.try_get(1).map_err(sqlx_err)?;
        sqlx::query(AssertSqlSafe(self.schema.delete_message()))
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(sqlx_err)?;
        tx.commit().await.map_err(sqlx_err)?;
        Ok(serde_json::from_str(&data).ok().map(InputItem))
    }

    /// Row ids make stable handles on both PostgreSQL and MySQL.
    async fn compaction_snapshot(&self, limit: usize) -> Result<Option<CompactionSnapshot>> {
        // One row beyond the window tells us whether it reached the start of the history.
        let rows = sqlx::query(AssertSqlSafe(self.schema.select_window()))
            .bind(self.session_id.clone())
            .bind(window_limit(limit))
            .fetch_all(&self.pool)
            .await
            .map_err(sqlx_err)?;
        let mut window: Vec<(i64, String)> = rows
            .iter()
            .filter_map(|r| Some((r.try_get(0).ok()?, r.try_get(1).ok()?)))
            .collect();
        let complete = window.len() <= limit;
        window.truncate(limit);
        window.reverse();

        let mut items = Vec::new();
        let mut handles = Vec::new();
        let mut intact = complete;
        for (id, data) in window {
            match serde_json::from_str(&data) {
                Ok(value) => {
                    items.push(InputItem(value));
                    handles.push(SnapshotHandle(id.to_string()));
                }
                Err(_) => {
                    // A corrupted row cannot be sent to the compaction API, so it and
                    // everything before it stay outside the replaceable window.
                    items.clear();
                    handles.clear();
                    intact = false;
                }
            }
        }
        Ok(Some(CompactionSnapshot {
            items,
            handles,
            complete: intact,
        }))
    }

    async fn replace_compaction_suffix(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
    ) -> Result<bool> {
        self.replace_compaction_suffix_pruning(
            handles,
            items,
            std::sync::Arc::new(|_: &InputItem| false),
        )
        .await
    }

    /// Replace the tail and, in the same transaction, delete the leading run of rows
    /// `prune_prefix` accepts, in bounded batches.
    async fn replace_compaction_suffix_pruning(
        &self,
        handles: &[SnapshotHandle],
        items: Vec<InputItem>,
        prune_prefix: super::PrunePrefix,
    ) -> Result<bool> {
        let mut tx = self.pool.begin().await.map_err(sqlx_err)?;
        let rows = sqlx::query(AssertSqlSafe(self.schema.select_window_ids()))
            .bind(self.session_id.clone())
            .bind(handles.len() as i64)
            .fetch_all(&mut *tx)
            .await
            .map_err(sqlx_err)?;
        let mut tail: Vec<i64> = rows.iter().filter_map(|r| r.try_get(0).ok()).collect();
        tail.reverse();
        if tail.len() != handles.len()
            || tail
                .iter()
                .zip(handles)
                .any(|(id, handle)| id.to_string() != handle.0)
        {
            return Ok(false);
        }
        if let Some(&first) = tail.first() {
            // Stops at the first row the predicate rejects or that no longer parses, so a
            // retained row always keeps everything after it.
            loop {
                let rows = sqlx::query(AssertSqlSafe(self.schema.select_prefix()))
                    .bind(self.session_id.clone())
                    .bind(first)
                    .bind(super::PRUNE_PREFIX_BATCH)
                    .fetch_all(&mut *tx)
                    .await
                    .map_err(sqlx_err)?;
                let batch: Vec<(i64, String)> = rows
                    .iter()
                    .filter_map(|r| Some((r.try_get(0).ok()?, r.try_get(1).ok()?)))
                    .collect();
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
                    break;
                };
                sqlx::query(AssertSqlSafe(self.schema.delete_prefix()))
                    .bind(self.session_id.clone())
                    .bind(expired_end)
                    .execute(&mut *tx)
                    .await
                    .map_err(sqlx_err)?;
                if batch.last().map(|(id, _)| *id) != Some(expired_end)
                    || (batch.len() as i64) < super::PRUNE_PREFIX_BATCH
                {
                    break;
                }
            }
        }
        for id in &tail {
            sqlx::query(AssertSqlSafe(self.schema.delete_message()))
                .bind(*id)
                .execute(&mut *tx)
                .await
                .map_err(sqlx_err)?;
        }
        for item in &items {
            sqlx::query(AssertSqlSafe(self.schema.insert_message()))
                .bind(self.session_id.clone())
                .bind(serde_json::to_string(item.as_value())?)
                .execute(&mut *tx)
                .await
                .map_err(sqlx_err)?;
        }
        sqlx::query(AssertSqlSafe(self.schema.touch_session()))
            .bind(self.session_id.clone())
            .execute(&mut *tx)
            .await
            .map_err(sqlx_err)?;
        tx.commit().await.map_err(sqlx_err)?;
        Ok(true)
    }

    async fn clear_session(&self) -> Result<()> {
        let mut tx = self.pool.begin().await.map_err(sqlx_err)?;
        sqlx::query(AssertSqlSafe(self.schema.delete_messages()))
            .bind(self.session_id.clone())
            .execute(&mut *tx)
            .await
            .map_err(sqlx_err)?;
        sqlx::query(AssertSqlSafe(self.schema.delete_session()))
            .bind(self.session_id.clone())
            .execute(&mut *tx)
            .await
            .map_err(sqlx_err)?;
        tx.commit().await.map_err(sqlx_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(dialect: SqlxDialect) -> Schema {
        Schema {
            dialect,
            sessions_table: DEFAULT_SESSIONS_TABLE.to_string(),
            messages_table: DEFAULT_MESSAGES_TABLE.to_string(),
        }
    }

    #[test]
    fn dialects_come_from_the_url_scheme() {
        assert_eq!(
            SqlxDialect::from_url("postgres://u:p@h/db").unwrap(),
            SqlxDialect::Postgres
        );
        assert_eq!(
            SqlxDialect::from_url("postgresql://h/db").unwrap(),
            SqlxDialect::Postgres
        );
        assert_eq!(
            SqlxDialect::from_url("mysql://h/db").unwrap(),
            SqlxDialect::MySql
        );
        assert_eq!(
            SqlxDialect::from_url("sqlite://x.db").unwrap(),
            SqlxDialect::Sqlite
        );
        assert!(SqlxDialect::from_url("mssql://h/db").is_err());
    }

    #[test]
    fn placeholders_follow_the_dialect() {
        assert_eq!(SqlxDialect::Postgres.placeholder(2), "$2");
        assert_eq!(SqlxDialect::MySql.placeholder(2), "?");
        assert_eq!(SqlxDialect::Sqlite.placeholder(2), "?");
    }

    #[test]
    fn postgres_statements_use_numbered_placeholders_and_on_conflict() {
        let s = schema(SqlxDialect::Postgres);
        assert_eq!(
            s.insert_message(),
            "INSERT INTO agent_messages (session_id, message_data) VALUES ($1, $2)"
        );
        assert!(
            s.insert_session()
                .contains("ON CONFLICT (session_id) DO NOTHING")
        );
        assert!(s.select_latest().ends_with("LIMIT $2"));
        let ddl = s.create_tables();
        assert_eq!(ddl.len(), 3);
        assert!(ddl[1].contains("BIGSERIAL"));
        assert!(ddl[2].contains("CREATE INDEX IF NOT EXISTS idx_agent_messages_session_time"));
    }

    #[test]
    fn mysql_statements_use_question_marks_and_inline_index() {
        let s = schema(SqlxDialect::MySql);
        assert_eq!(
            s.insert_message(),
            "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)"
        );
        assert!(s.insert_session().starts_with("INSERT IGNORE INTO"));
        assert!(s.select_latest().ends_with("LIMIT ?"));
        let ddl = s.create_tables();
        assert_eq!(ddl.len(), 2);
        assert!(ddl[0].contains("VARCHAR(255) PRIMARY KEY"));
        assert!(ddl[1].contains("AUTO_INCREMENT"));
        assert!(ddl[1].contains("INDEX idx_agent_messages_session_time (session_id, created_at)"));
    }

    #[test]
    fn table_names_are_validated() {
        let s = Schema {
            dialect: SqlxDialect::Postgres,
            sessions_table: "s".into(),
            messages_table: "m".into(),
        };
        assert!(valid_identifier(&s.sessions_table));
        assert!(!valid_identifier("drop table; --"));
        assert!(!valid_identifier(""));
    }

    /// Integration coverage against a real server; skipped unless `DATABASE_URL` is set.
    #[tokio::test]
    async fn sqlx_round_trip() {
        let Ok(url) = std::env::var("DATABASE_URL") else {
            return;
        };
        let s = SqlxSession::from_url("rust-test-session", &url, true)
            .await
            .unwrap();
        s.clear_session().await.unwrap();
        s.add_items(vec![InputItem::user("a"), InputItem::assistant("b")])
            .await
            .unwrap();
        assert_eq!(s.get_items(None).await.unwrap().len(), 2);
        assert_eq!(
            s.get_items(Some(1)).await.unwrap()[0].role(),
            Some("assistant")
        );
        assert_eq!(
            s.pop_item().await.unwrap().unwrap().role(),
            Some("assistant")
        );
        s.clear_session().await.unwrap();
        assert!(s.get_items(None).await.unwrap().is_empty());
    }
}
