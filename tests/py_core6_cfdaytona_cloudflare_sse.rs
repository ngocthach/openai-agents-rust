//! SSE decoder cases of `tests/extensions/sandbox/test_cloudflare.py`: line splitting across every
//! chunk boundary, CRLF straddling chunks, multi-byte UTF-8 split across chunks and flushing.
#![cfg(feature = "sandbox")]

use openai_agents_sdk::extensions::sandbox::cloudflare::{SseDecoder, SseLineDecoder};

/// `str.splitlines()` for a stream that only contains CR, LF and CRLF terminators.
fn splitlines(stream: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut chars = stream.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                lines.push(std::mem::take(&mut current));
            }
            '\n' => lines.push(std::mem::take(&mut current)),
            other => current.push(other),
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn decode_in_chunks(stream: &[u8], size: usize) -> Vec<String> {
    let mut decoder = SseLineDecoder::new();
    let mut lines = Vec::new();
    for chunk in stream.chunks(size) {
        lines.extend(decoder.decode(chunk));
    }
    lines.extend(decoder.flush());
    lines
}

fn events(lines: &[String]) -> Vec<(String, String)> {
    let mut decoder = SseDecoder::new();
    lines
        .iter()
        .filter_map(|line| decoder.decode(line))
        .map(|event| (event.event, event.data))
        .collect()
}

/// `test_sse_line_decoder_matches_splitlines_for_every_chunk_boundary` (all seven streams).
#[test]
fn sse_line_decoder_matches_splitlines_for_every_chunk_boundary() {
    for (name, stream) in [
        ("lf", "data: a\ndata: b\n\n"),
        ("crlf", "data: a\r\ndata: b\r\n\r\n"),
        ("cr", "data: a\rdata: b\r\r"),
        ("mixed", "data: a\r\ndata: b\n\r\n"),
        ("empty_data", "data:\r\n\r\n"),
        ("trailing_cr", "data: a\r"),
        ("bare_crlf", "\r\n"),
    ] {
        let expected = splitlines(stream);
        for size in 1..=stream.len() {
            assert_eq!(
                decode_in_chunks(stream.as_bytes(), size),
                expected,
                "{name}, chunk size {size}"
            );
        }
    }
}

/// `test_sse_event_is_not_split_when_crlf_straddles_a_chunk_boundary`
#[test]
fn sse_event_is_not_split_when_crlf_straddles_a_chunk_boundary() {
    let stream = "data: a\r\ndata: b\r\n\r\n";

    let whole = events(&decode_in_chunks(stream.as_bytes(), stream.len()));

    assert_eq!(whole, vec![("message".to_string(), "a\nb".to_string())]);
    for size in 1..=stream.len() {
        assert_eq!(
            events(&decode_in_chunks(stream.as_bytes(), size)),
            whole,
            "chunk size {size}"
        );
    }
}

/// `test_sse_line_decoder_delivers_a_cr_terminated_line_immediately`
#[test]
fn sse_line_decoder_delivers_a_cr_terminated_line_immediately() {
    let mut decoder = SseLineDecoder::new();

    assert_eq!(decoder.decode(b"data: a\r"), vec!["data: a"]);
    // The LF is the second half of that CRLF, not a blank line.
    assert!(decoder.decode(b"\n").is_empty());
    assert!(decoder.flush().is_empty());
}

/// `test_sse_line_decoder_handles_utf8_split_across_byte_chunks`
#[test]
fn sse_line_decoder_handles_utf8_split_across_byte_chunks() {
    let mut decoder = SseLineDecoder::new();
    let stream = "data: 失败\n\n".as_bytes();
    let split = stream
        .windows("失败".len())
        .position(|w| w == "失败".as_bytes())
        .expect("payload")
        + 1;

    assert!(decoder.decode(&stream[..split]).is_empty());
    assert_eq!(decoder.decode(&stream[split..]), vec!["data: 失败", ""]);
    assert!(decoder.flush().is_empty());
}

/// `test_sse_line_decoder_dispatches_a_cr_only_blank_line_without_more_input`
#[test]
fn sse_line_decoder_dispatches_a_cr_only_blank_line_without_more_input() {
    let mut decoder = SseLineDecoder::new();

    let lines = decoder.decode(b"data: a\r\r");

    assert_eq!(lines, vec!["data: a", ""]);
    assert_eq!(
        events(&lines),
        vec![("message".to_string(), "a".to_string())]
    );
}

/// `test_sse_line_decoder_flush_emits_only_an_unterminated_line`
#[test]
fn sse_line_decoder_flush_emits_only_an_unterminated_line() {
    let mut decoder = SseLineDecoder::new();
    assert_eq!(decoder.decode(b"data: a\r"), vec!["data: a"]);
    assert!(decoder.flush().is_empty());

    let mut partial = SseLineDecoder::new();
    assert!(partial.decode(b"data: b").is_empty());
    assert_eq!(partial.flush(), vec!["data: b"]);
}
