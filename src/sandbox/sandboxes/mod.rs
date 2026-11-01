//! Sandbox backends. Mirrors `agents.sandbox.sandboxes`.
//!
//! [`unix_local`] runs commands on the host inside a temporary workspace directory, and
//! [`docker`] runs them inside a container through the `docker` CLI. Cloud providers
//! (`agents.extensions.sandbox.*`) are not ported.

pub mod docker;
#[cfg(unix)]
mod pinned_ops;
pub mod unix_local;
