//! Session backed by the OpenAI Conversations API. Mirrors
//! `agents.memory.openai_conversations_session`.

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::OnceCell;

use super::Session;
use crate::error::{AgentsError, Result};
use crate::items::InputItem;
use crate::models::OpenAIClient;

/// Stores history server-side in an OpenAI conversation (`conv_...`).
#[derive(Debug)]
pub struct OpenAIConversationsSession {
    client: OpenAIClient,
    conversation_id: OnceCell<String>,
    requested_id: Option<String>,
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
            conversation_id: OnceCell::new(),
            requested_id: None,
        }
    }

    /// Resume an existing conversation.
    pub fn from_conversation_id(conversation_id: impl Into<String>, client: OpenAIClient) -> Self {
        let id = conversation_id.into();
        OpenAIConversationsSession {
            client,
            conversation_id: OnceCell::new_with(Some(id.clone())),
            requested_id: Some(id),
        }
    }

    /// The conversation id, creating the conversation if needed.
    pub async fn conversation_id(&self) -> Result<String> {
        self.conversation_id
            .get_or_try_init(|| async {
                let (v, _) = self
                    .client
                    .post_json("conversations", &json!({"items": []}), None, None)
                    .await?;
                v.get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| AgentsError::other("Conversations API returned no id"))
            })
            .await
            .cloned()
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
    fn session_id(&self) -> &str {
        self.conversation_id
            .get()
            .map(String::as_str)
            .or(self.requested_id.as_deref())
            .unwrap_or("")
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<InputItem>> {
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

    async fn add_items(&self, items: Vec<InputItem>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let id = self.conversation_id().await?;
        let body = json!({"items": items.into_iter().map(|i| i.0).collect::<Vec<_>>()});
        self.client
            .post_json(&format!("conversations/{id}/items"), &body, None, None)
            .await?;
        Ok(())
    }

    async fn pop_item(&self) -> Result<Option<InputItem>> {
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

    async fn clear_session(&self) -> Result<()> {
        if let Some(id) = self.conversation_id.get() {
            self.client.delete(&format!("conversations/{id}")).await?;
        }
        Ok(())
    }
}
