//! Decoding policy for a stdio MCP server's output.
//!
//! The MCP stdio transport mandates UTF-8, and a Rust `str` is UTF-8, so Python's `encoding`
//! parameter has no transcoding counterpart here — anything but `utf-8` is rejected at connect
//! time. Python's `encoding_error_handler` does have one: it decides what happens when the
//! server writes bytes that are not valid UTF-8, and [`Utf8Sanitizer`] implements the `ignore`
//! and `replace` policies. `strict` needs no wrapper: the line codec already fails.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, ReadBuf};

use super::EncodingErrorHandler;

/// The encodings that mean "UTF-8", spelled the way Python's codec aliases accept them.
const UTF8_ALIASES: [&str; 4] = ["utf-8", "utf8", "utf_8", "u8"];

/// Whether `encoding` names UTF-8.
pub(crate) fn is_utf8_encoding(encoding: &str) -> bool {
    let normalized = encoding.trim().to_ascii_lowercase().replace('_', "-");
    UTF8_ALIASES
        .iter()
        .any(|a| a.replace('_', "-") == normalized)
}

/// Rewrite `bytes` under the error policy, returning how many bytes were consumed.
///
/// Trailing bytes that could still become a valid sequence are left unconsumed so the caller
/// can carry them into the next read.
fn sanitize(bytes: &[u8], replace: bool, out: &mut VecDeque<u8>) -> usize {
    let mut i = 0usize;
    loop {
        match std::str::from_utf8(&bytes[i..]) {
            Ok(valid) => {
                out.extend(valid.as_bytes());
                return bytes.len();
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                out.extend(&bytes[i..i + valid_up_to]);
                match e.error_len() {
                    Some(len) => {
                        if replace {
                            out.extend("\u{FFFD}".as_bytes());
                        }
                        i += valid_up_to + len;
                    }
                    // An incomplete sequence at the very end: the next read may finish it.
                    None => return i + valid_up_to,
                }
            }
        }
    }
}

/// An [`AsyncRead`] that drops or replaces invalid UTF-8 from the wrapped reader.
pub(crate) struct Utf8Sanitizer<R> {
    inner: R,
    replace: bool,
    carry: Vec<u8>,
    pending: VecDeque<u8>,
    eof: bool,
}

impl<R> Utf8Sanitizer<R> {
    /// Wrap `inner`; `handler` must not be [`EncodingErrorHandler::Strict`].
    pub(crate) fn new(inner: R, handler: EncodingErrorHandler) -> Self {
        Utf8Sanitizer {
            inner,
            replace: handler == EncodingErrorHandler::Replace,
            carry: Vec::new(),
            pending: VecDeque::new(),
            eof: false,
        }
    }

    fn drain(&mut self, buf: &mut ReadBuf<'_>) -> bool {
        let n = buf.remaining().min(self.pending.len());
        if n == 0 {
            return false;
        }
        let bytes: Vec<u8> = self.pending.drain(..n).collect();
        buf.put_slice(&bytes);
        true
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for Utf8Sanitizer<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        loop {
            if this.drain(buf) {
                return Poll::Ready(Ok(()));
            }
            if this.eof {
                return Poll::Ready(Ok(()));
            }
            let mut scratch = [0u8; 8192];
            let mut inner_buf = ReadBuf::new(&mut scratch);
            match Pin::new(&mut this.inner).poll_read(cx, &mut inner_buf) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(())) => {
                    let filled = inner_buf.filled();
                    if filled.is_empty() {
                        this.eof = true;
                        // Flush whatever the carry holds: at EOF it can never complete.
                        let carry = std::mem::take(&mut this.carry);
                        if !carry.is_empty() {
                            let consumed = sanitize(&carry, this.replace, &mut this.pending);
                            if consumed < carry.len() && this.replace {
                                this.pending.extend("\u{FFFD}".as_bytes());
                            }
                        }
                        continue;
                    }
                    this.carry.extend_from_slice(filled);
                    let carry = std::mem::take(&mut this.carry);
                    let consumed = sanitize(&carry, this.replace, &mut this.pending);
                    this.carry = carry[consumed..].to_vec();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    async fn run(chunks: Vec<Vec<u8>>, handler: EncodingErrorHandler) -> String {
        let joined: Vec<u8> = chunks.concat();
        let reader = std::io::Cursor::new(joined);
        let mut sanitizer = Utf8Sanitizer::new(reader, handler);
        let mut out = Vec::new();
        sanitizer.read_to_end(&mut out).await.unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn recognizes_utf8_aliases() {
        assert!(is_utf8_encoding("utf-8"));
        assert!(is_utf8_encoding("UTF_8"));
        assert!(is_utf8_encoding(" utf8 "));
        assert!(!is_utf8_encoding("latin-1"));
        assert!(!is_utf8_encoding("cp1252"));
    }

    #[tokio::test]
    async fn drops_or_replaces_invalid_sequences() {
        let chunks = vec![b"ok \xff tail".to_vec()];
        assert_eq!(
            run(chunks.clone(), EncodingErrorHandler::Ignore).await,
            "ok  tail"
        );
        assert_eq!(
            run(chunks, EncodingErrorHandler::Replace).await,
            "ok \u{FFFD} tail"
        );
    }

    #[tokio::test]
    async fn carries_a_split_multibyte_character_across_reads() {
        // "é" is 0xC3 0xA9; splitting it must not produce a replacement character.
        let chunks = vec![vec![b'a', 0xC3], vec![0xA9, b'b']];
        assert_eq!(run(chunks, EncodingErrorHandler::Replace).await, "a\u{e9}b");
    }

    #[tokio::test]
    async fn flushes_an_incomplete_tail_at_eof() {
        let chunks = vec![vec![b'a', 0xC3]];
        assert_eq!(run(chunks.clone(), EncodingErrorHandler::Ignore).await, "a");
        assert_eq!(
            run(chunks, EncodingErrorHandler::Replace).await,
            "a\u{FFFD}"
        );
    }
}
