//! Conversation memory encrypted at rest with a TTL (feature `encrypt`).
//!
//! Items are stored as authenticated envelopes in the wrapped session, so the underlying store
//! never sees plaintext, and items older than the TTL are silently skipped when history is read.
//!
//! Run: `cargo run --example sessions_encrypted --features encrypt`
//!
//! The envelope format is not compatible with the Python SDK's Fernet payloads; see
//! `EncryptedSession` for details.

#[cfg(feature = "encrypt")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use openai_agents_sdk::memory::{EncryptedSession, InMemorySession};
    use openai_agents_sdk::{Agent, RunOptions, Runner, Session, flush_traces_async};

    // Load a high-entropy key from your secret manager. Reuse the same key and session id to
    // read stored history again after a restart.
    let encryption_key = std::env::var("SESSION_ENCRYPTION_KEY")
        .unwrap_or_else(|_| "replace-me-with-32-random-bytes!".to_string());

    // Any `Session` can be wrapped. Swap `InMemorySession` for `SqliteSession` (feature
    // `sqlite`) or one of the other backends to encrypt history that outlives the process.
    let store = Arc::new(InMemorySession::new("conversation_123"));
    let session: Arc<dyn Session> = Arc::new(EncryptedSession::new(
        "conversation_123",
        store.clone(),
        &encryption_key,
        600, // ten minutes
    )?);

    let agent = Agent::<()>::new("Assistant").with_instructions("Reply very concisely.");
    for question in [
        "What city is the Golden Gate Bridge in?",
        "What state is it in?",
    ] {
        println!("User: {question}");
        let result = Runner::run_with_options(
            &agent,
            question,
            RunOptions::new(()).session(session.clone()),
        )
        .await?;
        println!(
            "Assistant: {}\n",
            result.final_output_text().unwrap_or_default()
        );
    }

    println!(
        "Plaintext history ({} items):",
        session.get_items(None).await?.len()
    );
    for item in session.get_items(None).await? {
        println!(
            "  {}: {}",
            item.role().unwrap_or("?"),
            item.text_content().unwrap_or_default()
        );
    }
    println!("\nAs stored in the underlying session:");
    for item in store.get_items(None).await? {
        println!("  {}", item.as_value());
    }

    flush_traces_async().await;
    Ok(())
}

#[cfg(not(feature = "encrypt"))]
fn main() {
    eprintln!(
        "This example needs the `encrypt` feature: cargo run --example sessions_encrypted --features encrypt"
    );
}
