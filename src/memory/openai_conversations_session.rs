//! Session backed by the OpenAI Conversations API. Mirrors
//! `agents.memory.openai_conversations_session`.

use std::sync::RwLock;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::{Session, SessionSettings, resolve_session_limit};
use crate::error::{AgentsError, Result};
use crate::items::InputItem;
use crate::models::OpenAIClient;

/// The Conversations items-create endpoint accepts at most this many items per request.
const MAX_ITEMS_PER_REQUEST: usize = 20;

/// Stores history server-side in an OpenAI conversation (`conv_...`).
///
/// The conversation is created lazily on first use. Reads and mutations through one instance are
/// serialized; separate instances or external writers need application-level coordination.
#[derive(Debug)]
pub struct OpenAIConversationsSession {
    client: OpenAIClient,
    /// The live conversation id. `None` before the conversation is created and again after
    /// [`Session::clear_session`] deleted it, so a later call creates a fresh one.
    conversation_id: Mutex<Option<String>>,
    /// Borrowable view of `conversation_id` for [`Session::session_id`], which returns `&str`.
    ///
    /// Each distinct id is leaked once, so the view stays valid for the process. A session leaks
    /// one small string per conversation it creates.
    id_view: RwLock<&'static str>,
    settings: Option<SessionSettings>,
    mutation_lock: Mutex<()>,
}

impl OpenAIConversationsSession {
    /// A session that creates a new conversation on first use.
    pub fn new() -> Self {
        Self::with_client(OpenAIClient::from_env())
    }

    /// A session that creates a new conversation with a specific client.
    pub fn with_client(client: OpenAIClient) -> Self {
        OpenAIConversationsSession {
            client,
            conversation_id: Mutex::new(None),
            id_view: RwLock::new(""),
            settings: None,
            mutation_lock: Mutex::new(()),
        }
    }

    /// Resume an existing conversation.
    pub fn from_conversation_id(conversation_id: impl Into<String>, client: OpenAIClient) -> Self {
        let id = conversation_id.into();
        OpenAIConversationsSession {
            client,
            id_view: RwLock::new(Box::leak(id.clone().into_boxed_str())),
            conversation_id: Mutex::new(Some(id)),
            settings: None,
            mutation_lock: Mutex::new(()),
        }
    }

    /// Set default session settings.
    pub fn with_settings(mut self, settings: SessionSettings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// Update the borrowable view returned by [`Session::session_id`].
    fn publish_id(&self, id: Option<&str>) {
        let view: &'static str = match id {
            None => "",
            Some(id) => Box::leak(id.to_string().into_boxed_str()),
        };
        *self.id_view.write().expect("session poisoned") = view;
    }

    /// The conversation id, creating the conversation if needed.
    pub async fn conversation_id(&self) -> Result<String> {
        let mut guard = self.conversation_id.lock().await;
        if let Some(id) = guard.as_ref() {
            return Ok(id.clone());
        }
        let (v, _) = self
            .client
            .post_json("conversations", &json!({"items": []}), None, None)
            .await?;
        let id = v
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| AgentsError::other("Conversations API returned no id"))?;
        self.publish_id(Some(&id));
        *guard = Some(id.clone());
        Ok(id)
    }

    async fn list_items(&self, order: &str, limit: Option<usize>) -> Result<Vec<Value>> {
        let id = self.conversation_id().await?;
        let mut out = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let mut query = vec![("order", order.to_string()), ("limit", "100".to_string())];
            if let Some(a) = &after {
                query.push(("after", a.clone()));
            }
            let page = self
                .client
                .get_json(&format!("conversations/{id}/items"), &query)
                .await?;
            let data = page
                .get("data")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            out.extend(data);
            if let Some(l) = limit
                && out.len() >= l
            {
                out.truncate(l);
                break;
            }
            let has_more = page
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            after = page
                .get("last_id")
                .and_then(Value::as_str)
                .map(str::to_string);
            if !has_more || after.is_none() {
                break;
            }
        }
        Ok(out)
    }

    /// Read history while the caller already holds the mutation lock.
    async fn read_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let limit = resolve_session_limit(limit, self.settings.as_ref());
        if limit == Some(0) {
            return Ok(Vec::new());
        }
        let items = match limit {
            None => self.list_items("asc", None).await?,
            Some(n) => {
                let mut v = self.list_items("desc", Some(n)).await?;
                v.reverse();
                v
            }
        };
        Ok(items
            .into_iter()
            .map(|v| InputItem(strip_server_fields(v)))
            .collect())
    }
}

impl Default for OpenAIConversationsSession {
    fn default() -> Self {
        Self::new()
    }
}

fn strip_server_fields(mut v: Value) -> Value {
    if let Value::Object(m) = &mut v {
        m.remove("id");
        m.remove("status");
        m.remove("created_by");
    }
    v
}

#[async_trait]
impl Session for OpenAIConversationsSession {
    /// The conversation id, or `""` while the conversation has not been created yet.
    fn session_id(&self) -> &str {
        *self.id_view.read().expect("session poisoned")
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        self.settings.clone()
    }

    fn manages_history_remotely(&self) -> bool {
        true
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
        let _guard = self.mutation_lock.lock().await;
        self.read_items(limit).await
    }

    /// Append items in order, in requests of at most 20 items each.
    ///
    /// Writes spanning several requests are not atomic: if one fails, earlier batches stay
    /// saved and later ones are never sent. Retrying the whole list can duplicate what was
    /// already stored, so a caller must reconcile the remote history first.
    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let _guard = self.mutation_lock.lock().await;
        let id = self.conversation_id().await?;
        for chunk in items.chunks(MAX_ITEMS_PER_REQUEST) {
            let body = json!({"items": chunk.iter().map(|i| i.0.clone()).collect::<Vec<_>>()});
            self.client
                .post_json(&format!("conversations/{id}/items"), &body, None, None)
                .await?;
        }
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
        let _guard = self.mutation_lock.lock().await;
        let id = self.conversation_id().await?;
        let latest = self.list_items("desc", Some(1)).await?;
        let Some(item) = latest.into_iter().next() else {
            return Ok(None);
        };
        if let Some(item_id) = item.get("id").and_then(Value::as_str) {
            self.client
                .delete(&format!("conversations/{id}/items/{item_id}"))
                .await?;
        }
        Ok(Some(InputItem(strip_server_fields(item))))
    }

    /// Delete the conversation and forget its id, so the next call creates a fresh one.
    async fn clear_session(&self) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        let mut id_guard = self.conversation_id.lock().await;
        let Some(id) = id_guard.clone() else {
            return Ok(());
        };
        self.client.delete(&format!("conversations/{id}")).await?;
        *id_guard = None;
        self.publish_id(None);
        Ok(())
    }
}
