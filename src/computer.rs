//! The computer interface driven by [`ComputerTool`](crate::tool::ComputerTool).
//!
//! Port of `agents.computer`. Implement [`Computer`] for a browser automation driver, a VM or
//! a desktop controller; the runner executes the model's `computer_call` actions against it and
//! returns a screenshot after every batch.

use async_trait::async_trait;

use crate::error::Result;

/// The environment a computer runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Environment {
    /// macOS.
    Mac,
    /// Windows.
    Windows,
    /// Ubuntu Linux.
    Ubuntu,
    /// A web browser.
    Browser,
}

impl Environment {
    /// The wire name of the environment.
    pub fn as_str(&self) -> &'static str {
        match self {
            Environment::Mac => "mac",
            Environment::Windows => "windows",
            Environment::Ubuntu => "ubuntu",
            Environment::Browser => "browser",
        }
    }
}

/// A computer the model can control. All methods are async; blocking drivers should use
/// `tokio::task::spawn_blocking` internally.
#[async_trait]
pub trait Computer: Send + Sync {
    /// The environment, required only for the preview `computer_use_preview` tool payload.
    fn environment(&self) -> Option<Environment> {
        None
    }

    /// Display `(width, height)`, required only for the preview tool payload.
    fn dimensions(&self) -> Option<(u32, u32)> {
        None
    }

    /// Take a screenshot and return it as base64-encoded PNG.
    async fn screenshot(&self) -> Result<String>;

    /// Click at a position. `button` is `left`, `right`, `wheel`, `back` or `forward`.
    /// `keys` lists modifier keys held during the click.
    async fn click(&self, x: i64, y: i64, button: &str, keys: Option<&[String]>) -> Result<()>;

    /// Double click at a position.
    async fn double_click(&self, x: i64, y: i64, keys: Option<&[String]>) -> Result<()>;

    /// Scroll at a position by `(scroll_x, scroll_y)`.
    async fn scroll(
        &self,
        x: i64,
        y: i64,
        scroll_x: i64,
        scroll_y: i64,
        keys: Option<&[String]>,
    ) -> Result<()>;

    /// Type text.
    async fn type_text(&self, text: &str) -> Result<()>;

    /// Wait for a short while.
    async fn wait(&self) -> Result<()>;

    /// Move the mouse.
    async fn move_mouse(&self, x: i64, y: i64, keys: Option<&[String]>) -> Result<()>;

    /// Press a key combination.
    async fn keypress(&self, keys: &[String]) -> Result<()>;

    /// Drag along a path of points.
    async fn drag(&self, path: &[(i64, i64)], keys: Option<&[String]>) -> Result<()>;
}
