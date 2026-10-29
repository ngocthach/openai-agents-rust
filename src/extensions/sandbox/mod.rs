//! Cloud sandbox providers (feature `sandbox`). Mirrors `agents.extensions.sandbox`: each module
//! implements [`crate::sandbox::session::SandboxClient`] for a hosted sandbox service.

pub mod blaxel;
pub mod cloudflare;
pub mod daytona;
pub mod e2b;
pub mod modal;
pub mod vercel;
