//! The streamed output of a [`VoicePipeline`](crate::voice::VoicePipeline). Mirrors
//! `agents.voice.result`.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::json;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::error::{AgentsError, Result};
use crate::tracing::span_data::SpanData;
use crate::tracing::spans::Span;
use crate::tracing::{create_span, speech_group_span};
use crate::util::time_iso;
use crate::voice::events::{VoiceLifecycleEvent, VoiceStreamEvent};
use crate::voice::input::{AudioData, AudioFormat};
use crate::voice::model::{TTSModel, TTSModelSettings};
use crate::voice::pipeline_config::VoicePipelineConfig;

/// The trace-safe rendering of an error message. Port of `util._error_tracing.get_trace_error`.
///
/// Delegates to the crate-wide redaction layer so voice spans carry exactly the same
/// [`crate::tracing::REDACTED_TRACE_ERROR_MESSAGE`] text as model and tool spans.
pub(crate) fn get_trace_error(trace_include_sensitive_data: bool, error: &AgentsError) -> String {
    crate::tracing::model_error_text(error, trace_include_sensitive_data)
}

/// Events of one text segment, consumed by the dispatcher in the order segments were created.
type SegmentReceiver = mpsc::UnboundedReceiver<VoiceStreamEvent>;

/// The output of a [`VoicePipeline`](crate::voice::VoicePipeline).
///
/// Consume [`StreamedAudioResult::stream`] until it ends; the last event is either
/// [`VoiceLifecycleEvent::SessionEnded`] or an error. Dropping the result stops the pipeline.
pub struct StreamedAudioResult {
    events: mpsc::UnboundedReceiver<VoiceStreamEvent>,
    writer: Arc<AudioResultWriter>,
    dispatcher: Option<JoinHandle<()>>,
    producer: Option<JoinHandle<()>>,
    done: bool,
}

impl fmt::Debug for StreamedAudioResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamedAudioResult")
            .field("tts_model", &self.writer.tts_model.model_name())
            .field("is_complete", &self.done)
            .finish_non_exhaustive()
    }
}

impl StreamedAudioResult {
    /// Create the result and the writer the pipeline's producer task feeds.
    pub(crate) fn new(
        tts_model: Arc<dyn TTSModel>,
        tts_settings: Arc<TTSModelSettings>,
        config: Arc<VoicePipelineConfig>,
    ) -> (Self, Arc<AudioResultWriter>) {
        let (events_tx, events) = mpsc::unbounded_channel();
        let (segments_tx, segments) = mpsc::unbounded_channel();
        let writer = Arc::new(AudioResultWriter {
            tts_model,
            tts_settings,
            config,
            events_tx: events_tx.clone(),
            segments_tx: Mutex::new(Some(segments_tx)),
            state: Mutex::new(TurnState::default()),
            tasks: Mutex::new(Vec::new()),
            failed: AtomicBool::new(false),
        });
        let dispatcher = tokio::spawn(dispatch_audio(segments, events_tx));
        let result = StreamedAudioResult {
            events,
            writer: writer.clone(),
            dispatcher: Some(dispatcher),
            producer: None,
            done: false,
        };
        (result, writer)
    }

    /// Attach the task that drives transcription and the workflow.
    pub(crate) fn set_producer(&mut self, producer: JoinHandle<()>) {
        self.producer = Some(producer);
    }

    /// Stream events and audio as they are generated.
    ///
    /// The stream ends after [`VoiceLifecycleEvent::SessionEnded`] or an error event.
    pub fn stream(&mut self) -> impl Stream<Item = VoiceStreamEvent> + Send + '_ {
        futures::stream::unfold(self, |this| async move {
            if this.done {
                return None;
            }
            let event = this.events.recv().await?;
            if matches!(
                event,
                VoiceStreamEvent::Error(_)
                    | VoiceStreamEvent::Lifecycle(VoiceLifecycleEvent::SessionEnded)
            ) {
                this.done = true;
            }
            Some((event, this))
        })
    }

    /// Whether the terminal event has been delivered.
    pub fn is_complete(&self) -> bool {
        self.done
    }

    /// All text produced by the workflow so far.
    pub fn total_output_text(&self) -> String {
        self.writer
            .state
            .lock()
            .expect("poisoned")
            .total_output_text
            .clone()
    }
}

impl Drop for StreamedAudioResult {
    fn drop(&mut self) {
        if let Some(producer) = self.producer.take() {
            producer.abort();
        }
        if let Some(dispatcher) = self.dispatcher.take() {
            dispatcher.abort();
        }
        self.writer.abort_tasks();
    }
}

/// Forward each segment's events in the order the segments were created.
async fn dispatch_audio(
    mut segments: mpsc::UnboundedReceiver<SegmentReceiver>,
    events_tx: mpsc::UnboundedSender<VoiceStreamEvent>,
) {
    while let Some(mut segment) = segments.recv().await {
        while let Some(event) = segment.recv().await {
            let terminal = matches!(event, VoiceStreamEvent::Error(_));
            if events_tx.send(event).is_err() {
                return;
            }
            if terminal {
                return;
            }
        }
    }
    let _ = events_tx.send(VoiceStreamEvent::Lifecycle(
        VoiceLifecycleEvent::SessionEnded,
    ));
}

/// Text buffers and the tracing span of the turn being synthesized.
#[derive(Default)]
struct TurnState {
    text_buffer: String,
    turn_text_buffer: String,
    total_output_text: String,
    started_turn: bool,
    group_span: Option<Span>,
}

/// The producer half of a [`StreamedAudioResult`]: it turns workflow text into audio segments.
pub(crate) struct AudioResultWriter {
    tts_model: Arc<dyn TTSModel>,
    tts_settings: Arc<TTSModelSettings>,
    config: Arc<VoicePipelineConfig>,
    events_tx: mpsc::UnboundedSender<VoiceStreamEvent>,
    segments_tx: Mutex<Option<mpsc::UnboundedSender<SegmentReceiver>>>,
    state: Mutex<TurnState>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    failed: AtomicBool,
}

impl fmt::Debug for AudioResultWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioResultWriter")
            .field("tts_model", &self.tts_model.model_name())
            .finish_non_exhaustive()
    }
}

impl AudioResultWriter {
    /// Add workflow text, speaking every complete sentence it yields.
    pub(crate) fn add_text(self: &Arc<Self>, text: &str) {
        self.start_turn();
        let combined = {
            let mut state = self.state.lock().expect("poisoned");
            state.text_buffer.push_str(text);
            state.total_output_text.push_str(text);
            state.turn_text_buffer.push_str(text);
            let (combined, remaining) = (self.tts_settings.text_splitter)(&state.text_buffer);
            state.text_buffer = remaining;
            combined
        };
        if !combined.is_empty() {
            self.spawn_segment(combined, false);
        }
    }

    /// Whether a segment of this session failed to synthesize.
    ///
    /// The failure was already reported as the session's terminal error event.
    pub(crate) fn has_failed(&self) -> bool {
        self.failed.load(Ordering::SeqCst)
    }

    /// Finish the current turn: speak the held-back text and wait for all of its audio.
    ///
    /// Fails once a segment of the turn could not be synthesized, so the session stops instead
    /// of running further turns whose audio could never be delivered.
    pub(crate) async fn turn_done(self: &Arc<Self>) -> Result<()> {
        let (remaining, started) = {
            let mut state = self.state.lock().expect("poisoned");
            (std::mem::take(&mut state.text_buffer), state.started_turn)
        };
        if !remaining.is_empty() {
            self.spawn_segment(remaining, true);
        } else if started {
            self.emit_ordered(VoiceStreamEvent::Lifecycle(VoiceLifecycleEvent::TurnEnded));
        }
        self.join_tasks().await;
        self.finish_turn();
        if self.has_failed() {
            return Err(AgentsError::other("Voice audio synthesis failed"));
        }
        Ok(())
    }

    /// Report a failure. This is the terminal event of the session.
    pub(crate) fn add_error(&self, error: AgentsError) {
        let _ = self
            .events_tx
            .send(VoiceStreamEvent::Error(Arc::new(error)));
    }

    /// End the session: no further segments are created, so `session_ended` follows the last one.
    pub(crate) async fn finish(self: &Arc<Self>) {
        self.close_segments();
        self.join_tasks().await;
    }

    /// Stop accepting segments, which lets the dispatcher emit `session_ended`.
    pub(crate) fn close_segments(&self) {
        self.segments_tx.lock().expect("poisoned").take();
    }

    fn start_turn(self: &Arc<Self>) {
        {
            let mut state = self.state.lock().expect("poisoned");
            if state.started_turn {
                return;
            }
            let span = speech_group_span(None, self.config.tracing_disabled);
            span.start();
            state.group_span = Some(span);
            state.started_turn = true;
        }
        self.emit_ordered(VoiceStreamEvent::Lifecycle(
            VoiceLifecycleEvent::TurnStarted,
        ));
    }

    fn finish_turn(&self) {
        let mut state = self.state.lock().expect("poisoned");
        if let Some(span) = state.group_span.take() {
            let input = if self.config.trace_include_sensitive_data {
                state.turn_text_buffer.clone()
            } else {
                String::new()
            };
            span.update_data(|data| {
                if let SpanData::SpeechGroup { input: field } = data {
                    *field = Some(input);
                }
            });
            span.finish();
        }
        state.turn_text_buffer.clear();
        state.started_turn = false;
    }

    /// Queue one event as its own ordered segment.
    fn emit_ordered(&self, event: VoiceStreamEvent) {
        let (tx, rx) = mpsc::unbounded_channel();
        if self.send_segment(rx) {
            let _ = tx.send(event);
        }
    }

    fn send_segment(&self, segment: SegmentReceiver) -> bool {
        match self.segments_tx.lock().expect("poisoned").as_ref() {
            Some(tx) => tx.send(segment).is_ok(),
            None => false,
        }
    }

    /// Synthesize `text` in the background, keeping its audio ordered after earlier segments.
    fn spawn_segment(self: &Arc<Self>, text: String, finish_turn: bool) {
        let (tx, rx) = mpsc::unbounded_channel();
        if !self.send_segment(rx) {
            return;
        }
        let parent = self.state.lock().expect("poisoned").group_span.clone();
        let writer = self.clone();
        let task = tokio::spawn(async move {
            writer.stream_audio(text, tx, finish_turn, parent).await;
        });
        self.tasks.lock().expect("poisoned").push(task);
    }

    async fn stream_audio(
        &self,
        text: String,
        events: mpsc::UnboundedSender<VoiceStreamEvent>,
        finish_turn: bool,
        parent: Option<Span>,
    ) {
        let include_text = self.config.trace_include_sensitive_data;
        let span = create_span(
            SpanData::Speech {
                input: Some(if include_text {
                    text.clone()
                } else {
                    String::new()
                }),
                output: None,
                output_format: Some("pcm".to_string()),
                model: Some(self.tts_model.model_name().to_string()),
                model_config: Some(json!({
                    "voice": self.tts_settings.voice.as_ref().map(crate::voice::model::TTSVoice::to_json),
                    "instructions": include_text.then(|| self.tts_settings.instructions.clone()),
                    "speed": self.tts_settings.speed,
                })),
                first_content_at: None,
            },
            parent.as_ref(),
            self.config.tracing_disabled,
        );
        span.start();
        match self.synthesize(&text, &events, &span).await {
            Ok(()) => {
                if finish_turn {
                    let _ =
                        events.send(VoiceStreamEvent::Lifecycle(VoiceLifecycleEvent::TurnEnded));
                }
            }
            Err(error) => {
                self.failed.store(true, Ordering::SeqCst);
                span.set_error(
                    get_trace_error(include_text, &error),
                    Some(json!({"text": if include_text { text } else { String::new() }})),
                );
                log::error!("Error streaming voice audio: {error}");
                let _ = events.send(VoiceStreamEvent::Error(Arc::new(error)));
            }
        }
        span.finish();
    }

    /// Run the TTS model and emit audio buffers of at least `buffer_size` chunks.
    async fn synthesize(
        &self,
        text: &str,
        events: &mpsc::UnboundedSender<VoiceStreamEvent>,
        span: &Span,
    ) -> Result<()> {
        let mut stream = self.tts_model.run(text, &self.tts_settings).await?;
        let trace_audio = self.config.trace_include_sensitive_audio_data;
        let mut buffer: Vec<Bytes> = Vec::new();
        let mut full_audio: Vec<u8> = Vec::new();
        let mut pending_byte: Option<u8> = None;
        let mut first_byte_received = false;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if !first_byte_received {
                first_byte_received = true;
                let at = time_iso();
                span.update_data(|data| {
                    if let SpanData::Speech {
                        first_content_at, ..
                    } = data
                    {
                        *first_content_at = Some(at);
                    }
                });
            }
            if chunk.is_empty() {
                continue;
            }
            if trace_audio {
                full_audio.extend_from_slice(&chunk);
            }
            buffer.push(chunk);
            if buffer.len() >= self.tts_settings.buffer_size {
                let combined = join_chunks(&mut buffer, &mut pending_byte, false);
                if !combined.is_empty() {
                    self.emit_audio(events, &combined);
                }
            }
        }

        let combined = join_chunks(&mut buffer, &mut pending_byte, true);
        if !combined.is_empty() {
            self.emit_audio(events, &combined);
        }

        let output = if trace_audio {
            BASE64.encode(&full_audio)
        } else {
            String::new()
        };
        span.update_data(|data| {
            if let SpanData::Speech { output: field, .. } = data {
                *field = Some(output);
            }
        });
        Ok(())
    }

    fn emit_audio(&self, events: &mpsc::UnboundedSender<VoiceStreamEvent>, pcm: &[u8]) {
        let mut audio = transform_audio_buffer(pcm, self.tts_settings.dtype);
        if let Some(transform) = &self.tts_settings.transform_data {
            audio = transform(audio);
        }
        let _ = events.send(VoiceStreamEvent::Audio(audio));
    }

    async fn join_tasks(&self) {
        loop {
            let tasks: Vec<JoinHandle<()>> =
                std::mem::take(&mut *self.tasks.lock().expect("poisoned"));
            if tasks.is_empty() {
                return;
            }
            for task in tasks {
                let _ = task.await;
            }
        }
    }

    fn abort_tasks(&self) {
        for task in std::mem::take(&mut *self.tasks.lock().expect("poisoned")) {
            task.abort();
        }
    }
}

/// Concatenate buffered chunks, carrying a trailing odd byte over to the next flush.
fn join_chunks(
    buffer: &mut Vec<Bytes>,
    pending_byte: &mut Option<u8>,
    final_flush: bool,
) -> Vec<u8> {
    let mut combined: Vec<u8> = Vec::new();
    if let Some(byte) = pending_byte.take() {
        combined.push(byte);
    }
    for chunk in buffer.drain(..) {
        combined.extend_from_slice(&chunk);
    }
    if !combined.len().is_multiple_of(2) {
        if final_flush {
            // Pad the remaining half sample so it still decodes as one 16-bit frame.
            combined.push(0);
        } else {
            *pending_byte = combined.pop();
        }
    }
    combined
}

/// Decode little-endian 16-bit PCM into the configured sample format.
pub(crate) fn transform_audio_buffer(pcm: &[u8], format: AudioFormat) -> AudioData {
    let mut samples: Vec<i16> = pcm
        .chunks_exact(2)
        .map(|frame| i16::from_le_bytes([frame[0], frame[1]]))
        .collect();
    if !pcm.len().is_multiple_of(2) {
        // 16-bit samples need 2-byte alignment; pad an odd-length chunk safely.
        samples.push(i16::from_le_bytes([pcm[pcm.len() - 1], 0]));
    }
    match format {
        AudioFormat::Int16 => AudioData::Int16(samples),
        AudioFormat::Float32 => AudioData::Float32(crate::voice::input::i16_to_f32(&samples)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_odd_length_buffers() {
        assert_eq!(
            transform_audio_buffer(&[1], AudioFormat::Int16),
            AudioData::Int16(vec![1])
        );
        assert_eq!(
            transform_audio_buffer(&[0, 1, 2], AudioFormat::Int16),
            AudioData::Int16(vec![256, 2])
        );
    }

    #[test]
    fn carries_an_odd_byte_to_the_next_flush() {
        let mut buffer = vec![Bytes::from_static(&[1, 2, 3])];
        let mut pending = None;
        assert_eq!(join_chunks(&mut buffer, &mut pending, false), vec![1, 2]);
        assert_eq!(pending, Some(3));

        buffer.push(Bytes::from_static(&[4]));
        assert_eq!(join_chunks(&mut buffer, &mut pending, true), vec![3, 4]);
        assert_eq!(pending, None);
    }

    #[test]
    fn pads_a_trailing_half_sample_on_the_final_flush() {
        let mut buffer = vec![Bytes::from_static(&[7])];
        let mut pending = None;
        assert_eq!(join_chunks(&mut buffer, &mut pending, true), vec![7, 0]);
    }
}
