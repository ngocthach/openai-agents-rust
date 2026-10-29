//! Text splitting helpers for streamed speech. Mirrors `agents.voice.utils`.

use std::sync::Arc;

/// Splits a text buffer into `(text to speak, text to hold back)`.
pub type TextSplitter = Arc<dyn Fn(&str) -> (String, String) + Send + Sync>;

/// The minimum number of characters a chunk must have before it is spoken.
pub const DEFAULT_MIN_SENTENCE_LENGTH: usize = 20;

/// A splitter that flushes complete sentences once they reach `min_sentence_length` characters.
///
/// The last (possibly unfinished) sentence is held back together with the trailing whitespace
/// that separates it from the next streamed delta, so that `"He "` followed by `"arrived"` is
/// not spoken as `"Hearrived"`.
///
/// ```
/// use openai_agents_sdk::voice::get_sentence_based_splitter;
///
/// let split = get_sentence_based_splitter(20);
/// let (spoken, remaining) = split("This sentence is long enough to flush. He ");
/// assert_eq!(spoken, "This sentence is long enough to flush.");
/// assert_eq!(remaining, "He ");
/// ```
pub fn get_sentence_based_splitter(min_sentence_length: usize) -> TextSplitter {
    Arc::new(move |text_buffer: &str| {
        let trimmed = text_buffer.trim();
        let sentences = split_after_sentence_end(trimmed);
        let (last, head) = sentences.split_last().expect("at least one sentence");
        let combined = head.join(" ");
        if combined.chars().count() >= min_sentence_length {
            let trailing = &text_buffer[text_buffer.trim_end().len()..];
            return (combined, format!("{last}{trailing}"));
        }
        (String::new(), text_buffer.to_string())
    })
}

/// Split on whitespace runs that follow a sentence-final `.`, `!` or `?`.
///
/// Port of Python's `re.split(r"(?<=[.!?])\s+", text)`; always returns at least one element.
fn split_after_sentence_end(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut start = 0;
    let mut i = 0;
    let mut previous: Option<char> = None;
    while i < chars.len() {
        let (index, c) = chars[i];
        if c.is_whitespace() && matches!(previous, Some('.' | '!' | '?')) {
            out.push(&text[start..index]);
            while i < chars.len() && chars[i].1.is_whitespace() {
                i += 1;
            }
            start = chars.get(i).map_or(text.len(), |(index, _)| *index);
            previous = None;
            continue;
        }
        previous = Some(c);
        i += 1;
    }
    out.push(&text[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed text through the splitter the way `StreamedAudioResult::add_text` does.
    fn stream(text: &str, chunk_size: usize) -> Vec<String> {
        let split = get_sentence_based_splitter(DEFAULT_MIN_SENTENCE_LENGTH);
        let mut buffer = String::new();
        let mut spoken = Vec::new();
        for chunk in text.as_bytes().chunks(chunk_size) {
            buffer.push_str(std::str::from_utf8(chunk).expect("ascii test input"));
            let (chunk, rest) = split(&buffer);
            buffer = rest;
            if !chunk.is_empty() {
                spoken.push(chunk);
            }
        }
        if !buffer.trim().is_empty() {
            spoken.push(buffer.trim().to_string());
        }
        spoken
    }

    #[test]
    fn streamed_text_is_spoken_without_losing_or_gluing_words() {
        let texts = [
            "Dr. Smith went to Washington. He arrived at 3 p.m. sharp.",
            "Hello there friend. How are you doing today? I am fine. Goodbye now.",
            "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.",
            "A short one. Then a much longer sentence that exceeds the minimum easily.",
        ];
        for text in texts {
            for chunk_size in [1, 2, 3, 5, 7, 11, 15] {
                let spoken = stream(text, chunk_size);
                let words: Vec<&str> = spoken.iter().flat_map(|s| s.split_whitespace()).collect();
                let expected: Vec<&str> = text.split_whitespace().collect();
                assert_eq!(words, expected, "text={text:?} chunk_size={chunk_size}");
            }
        }
    }

    #[test]
    fn keeps_the_separator_before_the_next_delta() {
        let split = get_sentence_based_splitter(20);
        let (spoken, remaining) = split("This sentence is long enough to flush. He ");
        assert_eq!(spoken, "This sentence is long enough to flush.");
        assert_eq!(remaining, "He ");
    }

    #[test]
    fn leaves_the_buffer_untouched_when_nothing_is_flushed() {
        let split = get_sentence_based_splitter(20);
        assert_eq!(
            split("Too short. "),
            (String::new(), "Too short. ".to_string())
        );
        assert_eq!(split("   "), (String::new(), "   ".to_string()));
        assert_eq!(split(""), (String::new(), String::new()));
    }

    #[test]
    fn splits_without_trailing_whitespace() {
        let split = get_sentence_based_splitter(20);
        assert_eq!(
            split("This sentence is long enough to flush. He"),
            (
                "This sentence is long enough to flush.".to_string(),
                "He".to_string()
            )
        );
    }
}
