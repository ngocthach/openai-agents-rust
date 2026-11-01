//! Python `tests/tracing/test_trace_context.py::test_started_trace_id_cache_is_bounded`: the
//! process remembers a bounded number of started trace ids and forgets the oldest first.
//! Python patches the limit down to two; the Rust limit is a constant, so the test starts more
//! traces than it.

use openai_agents_sdk::tracing::{
    clear_started_trace_ids, gen_trace_id, reset_trace_provider, trace_id_was_started,
};

#[test]
fn the_started_trace_id_cache_is_bounded() {
    reset_trace_provider();
    clear_started_trace_ids();

    let ids: Vec<String> = (0..4_200)
        .map(|_| {
            let id = gen_trace_id();
            let trace = openai_agents_sdk::tracing::trace("bounded")
                .trace_id(id.clone())
                .build();
            trace.start();
            trace.finish();
            id
        })
        .collect();

    assert!(
        !trace_id_was_started(Some(&ids[0])),
        "the oldest id is forgotten"
    );
    assert!(!trace_id_was_started(Some(&ids[50])));
    assert!(
        trace_id_was_started(Some(ids.last().unwrap())),
        "the newest id is kept"
    );
    let remembered = ids
        .iter()
        .filter(|id| trace_id_was_started(Some(id.as_str())))
        .count();
    assert!(remembered < ids.len(), "{remembered} ids remembered");
    assert!(remembered >= 1000, "{remembered} ids remembered");
    // What is kept is the most recent suffix, so no gap precedes a remembered id.
    let first_kept = ids
        .iter()
        .position(|id| trace_id_was_started(Some(id.as_str())))
        .unwrap();
    assert!(
        ids[first_kept..]
            .iter()
            .all(|id| trace_id_was_started(Some(id.as_str())))
    );
}
