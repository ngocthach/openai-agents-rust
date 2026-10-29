//! Model interfaces and OpenAI implementations.

pub mod default_models;
pub mod interface;

pub use interface::{Model, ModelProvider, ModelRequest, ModelTracing};
