//! Conversation items tracked by a realtime session. Mirrors `agents.realtime.items`.

use serde::{Deserialize, Serialize};

/// A single content part of a [`RealtimeMessageItem`].
///
/// Input parts (`input_text`, `input_audio`, `input_image`) come from the user, output parts
/// (`text`, `audio`) from the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeContent {
    /// Text typed by the user.
    InputText {
        /// The text, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        /// Any other fields the transport reported, kept so they survive a round trip.
        #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
        extra: serde_json::Map<String, serde_json::Value>,
    },
    /// Audio sent by the user.
    InputAudio {
        /// Base64-encoded audio, when the transport includes it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio: Option<String>,
        /// Transcript of the audio, when available.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
        /// Any other fields the transport reported, kept so they survive a round trip.
        #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
        extra: serde_json::Map<String, serde_json::Value>,
    },
    /// An image sent by the user.
    InputImage {
        /// Data or remote URL.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        /// Detail hint (`auto`, `low` or `high`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        /// Any other fields the transport reported, kept so they survive a round trip.
        #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
        extra: serde_json::Map<String, serde_json::Value>,
    },
    /// Text produced by the assistant.
    Text {
        /// The text, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        /// Any other fields the transport reported, kept so they survive a round trip.
        #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
        extra: serde_json::Map<String, serde_json::Value>,
    },
    /// Audio produced by the assistant.
    Audio {
        /// Base64-encoded audio, when the transport includes it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio: Option<String>,
        /// Transcript of the audio, when available.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
        /// Any other fields the transport reported, kept so they survive a round trip.
        #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
        extra: serde_json::Map<String, serde_json::Value>,
    },
}

impl RealtimeContent {
    /// Input text content.
    pub fn input_text(text: impl Into<String>) -> Self {
        RealtimeContent::InputText {
            text: Some(text.into()),
            extra: serde_json::Map::new(),
        }
    }

    /// Assistant text content.
    pub fn text(text: impl Into<String>) -> Self {
        RealtimeContent::Text {
            text: Some(text.into()),
            extra: serde_json::Map::new(),
        }
    }

    /// Assistant audio content carrying only a transcript.
    pub fn audio_transcript(transcript: impl Into<String>) -> Self {
        RealtimeContent::Audio {
            audio: None,
            transcript: Some(transcript.into()),
            extra: serde_json::Map::new(),
        }
    }

    /// The text of a text part, if this is one.
    pub fn text_value(&self) -> Option<&str> {
        match self {
            RealtimeContent::InputText { text, .. } | RealtimeContent::Text { text, .. } => {
                text.as_deref()
            }
            _ => None,
        }
    }

    /// The transcript of an audio part, if this is one.
    pub fn transcript(&self) -> Option<&str> {
        match self {
            RealtimeContent::InputAudio { transcript, .. }
            | RealtimeContent::Audio { transcript, .. } => transcript.as_deref(),
            _ => None,
        }
    }

    /// Whether this part carries audio.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            RealtimeContent::InputAudio { .. } | RealtimeContent::Audio { .. }
        )
    }

    fn with_transcript(&self, transcript: String) -> Self {
        match self {
            RealtimeContent::InputAudio { audio, extra, .. } => RealtimeContent::InputAudio {
                audio: audio.clone(),
                transcript: Some(transcript),
                extra: extra.clone(),
            },
            RealtimeContent::Audio { audio, extra, .. } => RealtimeContent::Audio {
                audio: audio.clone(),
                transcript: Some(transcript),
                extra: extra.clone(),
            },
            other => other.clone(),
        }
    }
}

/// Who produced a realtime message item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeMessageRole {
    /// A system message.
    System,
    /// A user message.
    User,
    /// An assistant message.
    Assistant,
}

/// A message item in a realtime conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeMessageItem {
    /// Unique identifier of the item.
    pub item_id: String,
    /// Identifier of the item this one follows, when the transport reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_item_id: Option<String>,
    /// Who produced the message.
    pub role: RealtimeMessageRole,
    /// `in_progress`, `completed` or `incomplete`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// The content parts of the message.
    #[serde(default)]
    pub content: Vec<RealtimeContent>,
    /// Any other fields the transport reported, kept so they survive a round trip.
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl RealtimeMessageItem {
    /// A message item with the given id, role and content.
    pub fn new(
        item_id: impl Into<String>,
        role: RealtimeMessageRole,
        content: Vec<RealtimeContent>,
    ) -> Self {
        RealtimeMessageItem {
            item_id: item_id.into(),
            previous_item_id: None,
            role,
            status: None,
            content,
            extra: serde_json::Map::new(),
        }
    }

    /// Concatenated text and transcripts of the message.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| c.text_value().or_else(|| c.transcript()))
            .collect::<Vec<_>>()
            .join("")
    }
}

/// A tool call item in a realtime conversation. The same item carries the call and its output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeToolCallItem {
    /// Unique identifier of the item.
    pub item_id: String,
    /// Identifier of the item this one follows, when the transport reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_item_id: Option<String>,
    /// The call id used to correlate the output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// `in_progress` or `completed`.
    pub status: String,
    /// JSON-encoded arguments.
    pub arguments: String,
    /// Name of the tool being called.
    pub name: String,
    /// Output of the call, once it is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Any other fields the transport reported, kept so they survive a round trip.
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// An item of the session history: a message or a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeItem {
    /// A message item.
    Message(RealtimeMessageItem),
    /// A tool call item.
    FunctionCall(RealtimeToolCallItem),
}

impl RealtimeItem {
    /// The item id.
    pub fn item_id(&self) -> &str {
        match self {
            RealtimeItem::Message(m) => &m.item_id,
            RealtimeItem::FunctionCall(c) => &c.item_id,
        }
    }

    /// The id of the item this one follows, when known.
    pub fn previous_item_id(&self) -> Option<&str> {
        match self {
            RealtimeItem::Message(m) => m.previous_item_id.as_deref(),
            RealtimeItem::FunctionCall(c) => c.previous_item_id.as_deref(),
        }
    }

    /// The message item, if this is one.
    pub fn as_message(&self) -> Option<&RealtimeMessageItem> {
        match self {
            RealtimeItem::Message(m) => Some(m),
            RealtimeItem::FunctionCall(_) => None,
        }
    }

    /// The tool call item, if this is one.
    pub fn as_function_call(&self) -> Option<&RealtimeToolCallItem> {
        match self {
            RealtimeItem::FunctionCall(c) => Some(c),
            RealtimeItem::Message(_) => None,
        }
    }
}

impl From<RealtimeMessageItem> for RealtimeItem {
    fn from(value: RealtimeMessageItem) -> Self {
        RealtimeItem::Message(value)
    }
}

impl From<RealtimeToolCallItem> for RealtimeItem {
    fn from(value: RealtimeToolCallItem) -> Self {
        RealtimeItem::FunctionCall(value)
    }
}

/// A completed response from the realtime model.
///
/// Mirrors `agents.realtime.items.RealtimeResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeResponse {
    /// Unique identifier of the response.
    pub id: String,
    /// The message items the response produced.
    #[serde(default)]
    pub output: Vec<RealtimeMessageItem>,
}

/// Drop the base64 audio from a message item, keeping transcripts and every other field.
pub(crate) fn strip_audio(item: &mut RealtimeItem) {
    let RealtimeItem::Message(message) = item else {
        return;
    };
    for part in message.content.iter_mut() {
        match part {
            RealtimeContent::Audio { audio, .. } | RealtimeContent::InputAudio { audio, .. } => {
                *audio = None;
            }
            _ => {}
        }
    }
}

/// Insert or merge `incoming` into `history`, mirroring `RealtimeSession._get_new_history`.
///
/// An item with a known id replaces the existing one, preserving transcripts and text that the
/// incoming item leaves empty. Otherwise the item is inserted after `previous_item_id`, or
/// appended.
pub(crate) fn merge_item(history: &mut Vec<RealtimeItem>, incoming: RealtimeItem) {
    let existing = history
        .iter()
        .position(|item| item.item_id() == incoming.item_id());

    if let Some(index) = existing {
        history[index] = match (&history[index], &incoming) {
            (RealtimeItem::Message(old), RealtimeItem::Message(new)) if !new.content.is_empty() => {
                RealtimeItem::Message(merge_message(old, new))
            }
            _ => incoming,
        };
        return;
    }

    if let Some(previous) = incoming.previous_item_id()
        && let Some(index) = history.iter().position(|item| item.item_id() == previous)
    {
        history.insert(index + 1, incoming);
        return;
    }

    history.push(incoming);
}

fn merge_message(old: &RealtimeMessageItem, new: &RealtimeMessageItem) -> RealtimeMessageItem {
    if old.role != new.role {
        return new.clone();
    }
    let mut merged = new.clone();
    for (index, part) in merged.content.iter_mut().enumerate() {
        let Some(previous) = old.content.get(index) else {
            continue;
        };
        match part {
            RealtimeContent::Audio { transcript, .. }
            | RealtimeContent::InputAudio { transcript, .. }
                if transcript.is_none() =>
            {
                if let Some(kept) = previous.transcript() {
                    *part = part.with_transcript(kept.to_string());
                }
            }
            RealtimeContent::Text { text, .. } | RealtimeContent::InputText { text, .. }
                if text.as_deref().unwrap_or_default().is_empty() =>
            {
                if let Some(kept) = previous.text_value() {
                    *text = Some(kept.to_string());
                }
            }
            _ => {}
        }
    }
    merged
}

/// Apply a completed input-audio transcription to the history.
pub(crate) fn apply_input_transcription(
    history: &mut Vec<RealtimeItem>,
    item_id: &str,
    transcript: &str,
) -> bool {
    for item in history.iter_mut() {
        if let RealtimeItem::Message(message) = item
            && message.item_id == item_id
            && message.role == RealtimeMessageRole::User
        {
            for part in message.content.iter_mut() {
                if matches!(part, RealtimeContent::InputAudio { .. }) {
                    *part = part.with_transcript(transcript.to_string());
                }
            }
            message.status = Some("completed".to_string());
            return false;
        }
    }

    history.push(RealtimeItem::Message(RealtimeMessageItem::new(
        item_id,
        RealtimeMessageRole::User,
        vec![RealtimeContent::input_text(transcript)],
    )));
    true
}

/// Record an assistant transcript delta for `item_id` in the history.
pub(crate) fn apply_assistant_transcript(
    history: &mut Vec<RealtimeItem>,
    item_id: &str,
    transcript: &str,
) {
    merge_item(
        history,
        RealtimeItem::Message(RealtimeMessageItem::new(
            item_id,
            RealtimeMessageRole::Assistant,
            vec![RealtimeContent::audio_transcript(transcript)],
        )),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_audio_keeps_transcripts() {
        let mut item = RealtimeItem::Message(RealtimeMessageItem::new(
            "item_1",
            RealtimeMessageRole::Assistant,
            vec![
                RealtimeContent::Audio {
                    audio: Some("AAAA".to_string()),
                    transcript: Some("hello".to_string()),
                    extra: Default::default(),
                },
                RealtimeContent::text("hi"),
            ],
        ));
        strip_audio(&mut item);
        let content = &item.as_message().unwrap().content;
        assert_eq!(
            content[0],
            RealtimeContent::Audio {
                audio: None,
                transcript: Some("hello".to_string()),
                extra: Default::default(),
            }
        );
        assert_eq!(content[1].text_value(), Some("hi"));
    }

    #[test]
    fn merges_message_preserving_transcript() {
        let mut history = vec![RealtimeItem::Message(RealtimeMessageItem::new(
            "item_1",
            RealtimeMessageRole::Assistant,
            vec![RealtimeContent::audio_transcript("hello")],
        ))];
        merge_item(
            &mut history,
            RealtimeItem::Message(RealtimeMessageItem::new(
                "item_1",
                RealtimeMessageRole::Assistant,
                vec![RealtimeContent::Audio {
                    audio: None,
                    transcript: None,
                    extra: Default::default(),
                }],
            )),
        );
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].as_message().unwrap().text(), "hello");
    }

    #[test]
    fn inserts_after_previous_item() {
        let mut history = vec![
            RealtimeItem::Message(RealtimeMessageItem::new(
                "a",
                RealtimeMessageRole::User,
                vec![],
            )),
            RealtimeItem::Message(RealtimeMessageItem::new(
                "c",
                RealtimeMessageRole::User,
                vec![],
            )),
        ];
        let mut item = RealtimeMessageItem::new("b", RealtimeMessageRole::Assistant, vec![]);
        item.previous_item_id = Some("a".to_string());
        merge_item(&mut history, RealtimeItem::Message(item));
        let ids: Vec<&str> = history.iter().map(|i| i.item_id()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn transcription_updates_existing_user_item() {
        let mut history = vec![RealtimeItem::Message(RealtimeMessageItem::new(
            "u1",
            RealtimeMessageRole::User,
            vec![RealtimeContent::InputAudio {
                audio: None,
                transcript: None,
                extra: Default::default(),
            }],
        ))];
        let added = apply_input_transcription(&mut history, "u1", "hi there");
        assert!(!added);
        assert_eq!(history[0].as_message().unwrap().text(), "hi there");

        let added = apply_input_transcription(&mut history, "u2", "second");
        assert!(added);
        assert_eq!(history.len(), 2);
    }
}
