//! Run a voice pipeline over one static audio buffer (feature `voice`).
//! Run: cargo run --example voice_static --features voice
//!
//! The speech models are scripted so the example runs offline; swap them for
//! `OpenAIVoiceModelProvider` (or drop both `with_*_model` calls) to speak to the real API.
#[cfg(feature = "voice")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use futures::StreamExt;
    use openai_agents_sdk::testing::{FakeModel, assistant_message};
    use openai_agents_sdk::voice::testing::{FakeSTTModel, FakeTTSModel, pcm16_bytes};
    use openai_agents_sdk::voice::{
        AudioInput, SingleAgentVoiceWorkflow, VoicePipeline, VoicePipelineConfig, VoiceStreamEvent,
    };
    use openai_agents_sdk::{Agent, flush_traces_async};

    // A scripted agent keeps the example deterministic; use a real model for live audio.
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message(
        "The Golden Gate Bridge is in San Francisco. It opened in 1937.",
    )]);
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("Answer in one or two short sentences.")
        .with_model_instance(model);

    let stt =
        Arc::new(FakeSTTModel::new().with_transcription("What city is the Golden Gate Bridge in?"));
    let tts = Arc::new(
        FakeTTSModel::new()
            .with_result(vec![pcm16_bytes(&[0; 240])])
            .with_result(vec![pcm16_bytes(&[0; 120])]),
    );

    let pipeline = VoicePipeline::new(Arc::new(SingleAgentVoiceWorkflow::new(agent)))
        .with_stt_model(stt)
        .with_tts_model(tts)
        .with_config(VoicePipelineConfig::default().with_workflow_name("Static voice demo"));

    // One second of silence at 24 kHz stands in for recorded microphone audio.
    let audio = AudioInput::new(vec![0i16; 24_000]);
    let mut result = pipeline.run(audio).await?;

    let mut samples = 0usize;
    {
        let mut events = std::pin::pin!(result.stream());
        while let Some(event) = events.next().await {
            match event {
                VoiceStreamEvent::Audio(audio) => {
                    // Write these samples to your audio output device.
                    samples += audio.len();
                }
                VoiceStreamEvent::Lifecycle(event) => println!("[{}]", event.as_str()),
                VoiceStreamEvent::Error(error) => eprintln!("voice pipeline failed: {error}"),
            }
        }
    }
    println!("Spoken text: {}", result.total_output_text());
    println!("Received {samples} audio samples");
    flush_traces_async().await;
    Ok(())
}

#[cfg(not(feature = "voice"))]
fn main() {
    eprintln!(
        "This example needs the `voice` feature: cargo run --example voice_static --features voice"
    );
}
