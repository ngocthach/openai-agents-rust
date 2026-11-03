//! The log assertions of `test_transforms.py`: renaming a tool for invalid characters warns
//! with both names, a case-only rename does not.

use std::sync::{Mutex, OnceLock};

use openai_agents_sdk::transform_string_function_style;

#[derive(Default)]
struct Collect(Mutex<Vec<(log::Level, String)>>);

impl log::Log for Collect {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == openai_agents_sdk::logger::LOG_TARGET {
            self.0
                .lock()
                .unwrap()
                .push((record.level(), record.args().to_string()));
        }
    }
    fn flush(&self) {}
}

static SINK: OnceLock<Collect> = OnceLock::new();

/// Records emitted while `f` runs (one test function drives every case, so records of
/// concurrent tests cannot interleave).
fn records_of(f: impl FnOnce()) -> Vec<(log::Level, String)> {
    let sink: &'static Collect = SINK.get_or_init(Collect::default);
    log::set_logger(sink).ok();
    log::set_max_level(log::LevelFilter::Trace);
    sink.0.lock().unwrap().clear();
    f();
    std::mem::take(&mut *sink.0.lock().unwrap())
}

#[test]
fn renaming_for_invalid_characters_warns_and_a_case_only_rename_does_not() {
    for (name, transformed) in [("My Tool", "my_tool"), ("My-Tool", "my_tool")] {
        let records = records_of(|| assert_eq!(transform_string_function_style(name), transformed));
        let text: String = records
            .iter()
            .filter(|(level, _)| *level == log::Level::Warn)
            .map(|(_, message)| message.as_str())
            .collect();
        assert!(
            text.contains(&format!("Tool name {name:?} contains invalid characters")),
            "{text}"
        );
        assert!(
            text.contains(&format!("transformed to {transformed:?}")),
            "{text}"
        );
    }
    for (name, transformed) in [
        ("MyTool", "mytool"),
        ("transfer_to_Agent", "transfer_to_agent"),
        ("snake_case", "snake_case"),
    ] {
        let records = records_of(|| assert_eq!(transform_string_function_style(name), transformed));
        assert!(records.is_empty(), "{name}: {records:?}");
    }
}
