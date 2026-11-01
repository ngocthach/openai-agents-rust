//! Port of `test_clone_does_not_mutate_original_lists` from `tests/realtime/test_agent.py`:
//! giving a clone its own tool list leaves the original's lists untouched.
#![cfg(feature = "realtime")]

use openai_agents_sdk::realtime::RealtimeAgent;
use openai_agents_sdk::tool::FunctionTool;

#[test]
fn a_clone_with_its_own_tools_leaves_the_original_untouched() {
    let original = RealtimeAgent::<()>::new("orig");
    let mut cloned = original.clone();
    cloned.tools =
        vec![FunctionTool::<()>::new_no_args("t1", "tool", |_ctx| async { Ok("ok") }).into()];
    assert!(original.tools.is_empty());
    assert_eq!(cloned.tools.len(), 1);
    assert!(original.handoffs.is_empty());
    assert!(cloned.handoffs.is_empty());
}
