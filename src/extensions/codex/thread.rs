//! A Codex thread and its turns. Mirrors `agents.extensions.experimental.codex.thread`.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures::stream::{BoxStream, Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::error::{AgentsError, Result};
use crate::extensions::codex::events::{ThreadError, ThreadEvent, Usage};
use crate::extensions::codex::exec::{CodexExec, CodexExecArgs, CodexLineStream};
use crate::extensions::codex::items::ThreadItem;
use crate::extensions::codex::options::CodexOptions;
use crate::extensions::codex::output_schema::OutputSchemaFile;
use crate::extensions::codex::thread_options::ThreadOptions;
use crate::extensions::codex::turn_options::TurnOptions;

/// One piece of user input for a turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserInput {
    /// Text appended to the prompt.
    Text {
        /// The text.
        text: String,
    },
    /// A local image attached to the turn (`--image`).
    LocalImage {
        /// Path of the image on disk.
        path: String,
    },
}

impl UserInput {
    /// A text input.
    pub fn text(text: impl Into<String>) -> Self {
        UserInput::Text { text: text.into() }
    }

    /// A local image input.
    pub fn local_image(path: impl Into<String>) -> Self {
        UserInput::LocalImage { path: path.into() }
    }
}

/// The input of a turn: a prompt, or a list of structured inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Input {
    /// A plain prompt.
    Text(String),
    /// Structured inputs; text parts are joined with a blank line and images are attached.
    Items(Vec<UserInput>),
}

impl Input {
    /// Split the input into the prompt and the attached image paths.
    pub fn normalize(&self) -> (String, Vec<String>) {
        match self {
            Input::Text(text) => (text.clone(), Vec::new()),
            Input::Items(items) => {
                let mut parts: Vec<String> = Vec::new();
                let mut images: Vec<String> = Vec::new();
                for item in items {
                    match item {
                        UserInput::Text { text } => parts.push(text.clone()),
                        UserInput::LocalImage { path } => {
                            if !path.is_empty() {
                                images.push(path.clone());
                            }
                        }
                    }
                }
                (parts.join("\n\n"), images)
            }
        }
    }
}

impl From<&str> for Input {
    fn from(value: &str) -> Self {
        Input::Text(value.to_string())
    }
}

impl From<String> for Input {
    fn from(value: String) -> Self {
        Input::Text(value)
    }
}

impl From<Vec<UserInput>> for Input {
    fn from(value: Vec<UserInput>) -> Self {
        Input::Items(value)
    }
}

/// The aggregated result of a turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    /// Items completed during the turn.
    pub items: Vec<ThreadItem>,
    /// Text of the last agent message, or an empty string.
    pub final_response: String,
    /// Token usage reported by the turn.
    pub usage: Option<Usage>,
}

/// Alias matching the Python SDK name.
pub type RunResult = Turn;

/// A streaming turn: a stream of [`ThreadEvent`]s.
///
/// Dropping the stream kills the Codex CLI process.
pub struct StreamedTurn {
    events: BoxStream<'static, Result<ThreadEvent>>,
}

impl std::fmt::Debug for StreamedTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamedTurn").finish_non_exhaustive()
    }
}

impl StreamedTurn {
    /// The next event, or `None` once the turn ended.
    pub async fn next_event(&mut self) -> Option<Result<ThreadEvent>> {
        self.events.next().await
    }
}

impl Stream for StreamedTurn {
    type Item = Result<ThreadEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.events.poll_next_unpin(cx)
    }
}

/// Alias matching the Python SDK name.
pub type RunStreamedResult = StreamedTurn;

/// A Codex conversation. Clones share the thread id, so a resumed id captured by one clone is
/// visible to all of them.
#[derive(Debug, Clone)]
pub struct Thread {
    exec: Arc<CodexExec>,
    options: CodexOptions,
    thread_options: ThreadOptions,
    id: Arc<Mutex<Option<String>>>,
}

struct EventStreamState {
    lines: CodexLineStream,
    id: Arc<Mutex<Option<String>>>,
    // Kept alive so the Codex CLI can read the schema for the whole turn.
    _schema_file: Option<OutputSchemaFile>,
}

impl Thread {
    /// Create a thread; `thread_id` resumes an existing conversation.
    pub(crate) fn new(
        exec: Arc<CodexExec>,
        options: CodexOptions,
        thread_options: ThreadOptions,
        thread_id: Option<String>,
    ) -> Self {
        Thread {
            exec,
            options,
            thread_options,
            id: Arc::new(Mutex::new(thread_id)),
        }
    }

    /// The thread id, once Codex reported it (or the id this thread resumes).
    pub fn id(&self) -> Option<String> {
        self.id.lock().expect("codex thread poisoned").clone()
    }

    /// The thread options used for every turn.
    pub fn options(&self) -> &ThreadOptions {
        &self.thread_options
    }

    /// Start a turn and stream its events.
    pub async fn run_streamed(
        &self,
        input: impl Into<Input>,
        turn_options: Option<TurnOptions>,
    ) -> Result<StreamedTurn> {
        let turn_options = turn_options.unwrap_or_default();
        let schema_file = match &turn_options.output_schema {
            Some(schema) => Some(OutputSchemaFile::create(schema)?),
            None => None,
        };
        let (prompt, images) = input.into().normalize();
        let options = &self.thread_options;
        let args = CodexExecArgs {
            input: prompt,
            base_url: self.options.base_url.clone(),
            api_key: self.options.api_key.clone(),
            thread_id: self.id(),
            images,
            model: options.model.clone(),
            sandbox_mode: options.sandbox_mode,
            working_directory: options.working_directory.clone(),
            additional_directories: options.additional_directories.clone().unwrap_or_default(),
            skip_git_repo_check: options.skip_git_repo_check,
            output_schema_file: schema_file
                .as_ref()
                .map(|f| f.path().to_string_lossy().into_owned()),
            model_reasoning_effort: options.model_reasoning_effort,
            idle_timeout: turn_options.idle_timeout,
            network_access_enabled: options.network_access_enabled,
            web_search_mode: options.web_search_mode,
            web_search_enabled: options.web_search_enabled,
            approval_policy: options.approval_policy,
        };

        let lines = self.exec.run(args).await?;
        let state = EventStreamState {
            lines,
            id: self.id.clone(),
            _schema_file: schema_file,
        };
        let events = futures::stream::unfold(state, |mut state| async move {
            let line = match state.lines.next().await {
                Some(Ok(line)) => line,
                Some(Err(e)) => return Some((Err(e), state)),
                None => return None,
            };
            let event = match ThreadEvent::from_json_line(&line) {
                Ok(event) => event,
                Err(e) => return Some((Err(e), state)),
            };
            if let ThreadEvent::ThreadStarted { thread_id } = &event {
                // Capture the thread id so callers can resume later.
                *state.id.lock().expect("codex thread poisoned") = Some(thread_id.clone());
            }
            Some((Ok(event), state))
        })
        .boxed();
        Ok(StreamedTurn { events })
    }

    /// Run a turn and aggregate its events into a [`Turn`].
    pub async fn run(
        &self,
        input: impl Into<Input>,
        turn_options: Option<TurnOptions>,
    ) -> Result<Turn> {
        let mut stream = self.run_streamed(input, turn_options).await?;
        let mut items: Vec<ThreadItem> = Vec::new();
        let mut final_response = String::new();
        let mut usage: Option<Usage> = None;
        let mut turn_failure: Option<ThreadError> = None;

        while let Some(event) = stream.next_event().await {
            match event? {
                ThreadEvent::ItemCompleted { item } => {
                    if let Some(text) = item.as_agent_message() {
                        final_response = text.to_string();
                    }
                    items.push(item);
                }
                ThreadEvent::TurnCompleted { usage: turn_usage } => usage = turn_usage,
                ThreadEvent::TurnFailed { error } => {
                    turn_failure = Some(error);
                    break;
                }
                ThreadEvent::Error { message } => {
                    return Err(AgentsError::other(format!("Codex stream error: {message}")));
                }
                _ => {}
            }
        }

        if let Some(failure) = turn_failure {
            return Err(AgentsError::other(failure.message));
        }

        Ok(Turn {
            items,
            final_response,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_text_input() {
        assert_eq!(
            Input::from("do the thing").normalize(),
            ("do the thing".to_string(), Vec::new())
        );
    }

    #[test]
    fn normalizes_structured_input() {
        let input = Input::Items(vec![
            UserInput::text("first"),
            UserInput::local_image("/a.png"),
            UserInput::text("second"),
            UserInput::local_image(""),
        ]);
        assert_eq!(
            input.normalize(),
            ("first\n\nsecond".to_string(), vec!["/a.png".to_string()])
        );
    }
}
