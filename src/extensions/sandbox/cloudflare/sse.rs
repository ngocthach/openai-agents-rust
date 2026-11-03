//! The Server-Sent Events decoders the Cloudflare sandbox worker speaks. Mirrors the
//! `_SSELineDecoder` / `_SSEDecoder` pair in `agents.extensions.sandbox.cloudflare.sandbox`.

/// One decoded Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSentEvent {
    /// The event name, `message` when the stream does not set one.
    pub event: String,
    /// The joined `data:` lines.
    pub data: String,
    /// The last `id:` seen, or an empty string.
    pub id: String,
    /// The `retry:` hint in milliseconds.
    pub retry: Option<i64>,
}

/// Splits a byte stream into SSE lines, tolerating `CR`, `LF` and `CRLF` across chunks.
#[derive(Debug, Default)]
pub struct SseLineDecoder {
    buffer: Vec<u8>,
    skip_leading_lf: bool,
}

impl SseLineDecoder {
    /// A decoder with an empty buffer.
    pub fn new() -> Self {
        SseLineDecoder::default()
    }

    /// Feed one chunk and return the lines it completes.
    pub fn decode(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut data = chunk;
        if self.skip_leading_lf && !data.is_empty() {
            // The previous chunk ended on a CR, which already terminated its line. A LF opening
            // this chunk is the second half of that CRLF, so consume it instead of reading it as
            // a blank line, which SSE treats as an event dispatch.
            self.skip_leading_lf = false;
            if data[0] == b'\n' {
                data = &data[1..];
            }
        }
        let mut raw = std::mem::take(&mut self.buffer);
        raw.extend_from_slice(data);

        let mut lines = Vec::new();
        let length = raw.len();
        let mut index = 0usize;
        while index < length {
            let cr = raw[index..]
                .iter()
                .position(|b| *b == b'\r')
                .map(|p| p + index);
            let lf = raw[index..]
                .iter()
                .position(|b| *b == b'\n')
                .map(|p| p + index);
            match (cr, lf) {
                (None, None) => {
                    self.buffer = raw[index..].to_vec();
                    return lines;
                }
                (Some(cr), lf) if lf.is_none_or(|lf| cr < lf) => {
                    let line = String::from_utf8_lossy(&raw[index..cr]).into_owned();
                    if cr + 1 < length && raw[cr + 1] == b'\n' {
                        index = cr + 2;
                    } else if cr + 1 == length {
                        // A CR is a complete line ending on its own, so deliver the line now
                        // rather than waiting to learn whether a LF follows. Remember only that a
                        // LF opening the next chunk belongs to this CRLF; without that, the LF
                        // reads as a blank line and dispatches the event early, splitting one
                        // multi-line event into several.
                        self.skip_leading_lf = true;
                        lines.push(line);
                        return lines;
                    } else {
                        index = cr + 1;
                    }
                    lines.push(line);
                }
                (_, Some(lf)) => {
                    lines.push(String::from_utf8_lossy(&raw[index..lf]).into_owned());
                    index = lf + 1;
                }
                (Some(_), None) => unreachable!("the CR arm above covers a CR without a LF"),
            }
        }
        lines
    }

    /// Return whatever is buffered as a final line.
    pub fn flush(&mut self) -> Vec<String> {
        let buffer = std::mem::take(&mut self.buffer);
        // A trailing CR already delivered its line, so there is nothing pending for it here.
        self.skip_leading_lf = false;
        if buffer.is_empty() {
            return Vec::new();
        }
        vec![String::from_utf8_lossy(&buffer).into_owned()]
    }
}

/// Assembles SSE lines into events.
#[derive(Debug, Default)]
pub struct SseDecoder {
    event: Option<String>,
    data: Vec<String>,
    last_event_id: Option<String>,
    retry: Option<i64>,
}

impl SseDecoder {
    /// A decoder with no pending event.
    pub fn new() -> Self {
        SseDecoder::default()
    }

    /// Feed one line, returning the event a blank line dispatches.
    pub fn decode(&mut self, line: &str) -> Option<ServerSentEvent> {
        if line.is_empty() {
            if self.event.is_none()
                && self.data.is_empty()
                && self.last_event_id.is_none()
                && self.retry.is_none()
            {
                return None;
            }
            let event = ServerSentEvent {
                event: self.event.take().unwrap_or_else(|| "message".to_string()),
                data: self.data.join("\n"),
                id: self.last_event_id.clone().unwrap_or_default(),
                retry: self.retry,
            };
            self.data.clear();
            self.retry = None;
            return Some(event);
        }

        if line.starts_with(':') {
            return None;
        }

        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => self.data.push(value.to_string()),
            "id" => {
                if !value.contains('\0') {
                    self.last_event_id = Some(value.to_string());
                }
            }
            "retry" => {
                if let Ok(retry) = value.parse::<i64>() {
                    self.retry = Some(retry);
                }
            }
            _ => {}
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_lines_across_a_crlf_boundary() {
        let mut decoder = SseLineDecoder::new();
        assert_eq!(decoder.decode(b"event: stdout\r"), vec!["event: stdout"]);
        // The LF opening the next chunk completes the previous CRLF and must not dispatch.
        assert_eq!(decoder.decode(b"\ndata: aGk=\r\n"), vec!["data: aGk="]);
        assert_eq!(decoder.decode(b"\r\n"), vec![""]);
    }

    #[test]
    fn assembles_a_multi_line_event() {
        let mut lines = SseLineDecoder::new();
        let mut events = SseDecoder::new();
        let mut decoded = Vec::new();
        for chunk in [
            &b"event: exit\nda"[..],
            &b"ta: {\"exit_"[..],
            &b"code\": 3}\n\nevent: done\n\n"[..],
        ] {
            for line in lines.decode(chunk) {
                if let Some(event) = events.decode(&line) {
                    decoded.push(event);
                }
            }
        }
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].event, "exit");
        assert_eq!(decoded[0].data, "{\"exit_code\": 3}");
        assert_eq!(decoded[1].event, "done");
        assert_eq!(decoded[1].data, "");
    }

    #[test]
    fn ignores_comments_and_keeps_the_last_id() {
        let mut decoder = SseDecoder::new();
        assert!(decoder.decode(": keep-alive").is_none());
        assert!(decoder.decode("id: 7").is_none());
        assert!(decoder.decode("retry: 1500").is_none());
        assert!(decoder.decode("data: x").is_none());
        let event = decoder.decode("").unwrap();
        assert_eq!(event.id, "7");
        assert_eq!(event.retry, Some(1500));
        assert_eq!(event.event, "message");
    }

    #[test]
    fn flushes_a_trailing_partial_line() {
        let mut decoder = SseLineDecoder::new();
        assert!(decoder.decode(b"data: tail").is_empty());
        assert_eq!(decoder.flush(), vec!["data: tail"]);
        assert!(decoder.flush().is_empty());
    }
}
