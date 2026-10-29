//! Serializable snapshot of a paused run. Mirrors `agents.run_state`.

use serde::{Deserialize, Serialize};

use crate::items::InputItem;

/// A resumable snapshot of a run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    /// The original input of the run.
    pub original_input: Vec<InputItem>,
}
