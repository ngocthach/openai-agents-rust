//! Input-item hygiene for replay: orphan pruning, dedupe and tool-output ordering.
//! Port of `agents.run_internal.items`.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::items::{InputItem, RunItem, RunItemKind};

/// Call item type -> the output item type that completes it.
const TOOL_CALL_TO_OUTPUT_TYPE: [(&str, &str); 8] = [
    ("program", "program_output"),
    ("function_call", "function_call_output"),
    ("custom_tool_call", "custom_tool_call_output"),
    ("shell_call", "shell_call_output"),
    ("apply_patch_call", "apply_patch_call_output"),
    ("computer_call", "computer_call_output"),
    ("local_shell_call", "local_shell_call_output"),
    ("tool_search_call", "tool_search_output"),
];

fn output_type_for(call_type: &str) -> Option<&'static str> {
    TOOL_CALL_TO_OUTPUT_TYPE
        .iter()
        .find(|(call, _)| *call == call_type)
        .map(|(_, out)| *out)
}

/// The output item type that completes a call of this type.
pub(crate) fn output_type_of_call(call_type: &str) -> Option<&'static str> {
    output_type_for(call_type)
}

fn is_call_or_output_type(ty: &str) -> bool {
    TOOL_CALL_TO_OUTPUT_TYPE
        .iter()
        .any(|(call, out)| *call == ty || *out == ty)
}

/// These items must keep their original position relative to required follower items.
fn is_earliest_anchor_type(ty: &str) -> bool {
    TOOL_CALL_TO_OUTPUT_TYPE.iter().any(|(call, _)| *call == ty)
        || ty == "mcp_approval_request"
        || ty == "reasoning"
}

/// Placeholder id used by the Chat Completions converter; never a real identity.
const FAKE_RESPONSES_ID: &str = "__fake_id__";

/// A stable identity key, when the item carries an explicit identifier.
/// Port of `_dedupe_key`.
fn dedupe_key(item: &Value) -> Option<String> {
    let role = item.get("role").and_then(Value::as_str);
    let item_type = item
        .get("type")
        .and_then(Value::as_str)
        .or(role)
        .unwrap_or_default();
    if role.is_some() || item_type == "message" {
        return None;
    }
    let call_id = item.get("call_id").and_then(Value::as_str);
    if let Some(cid) = call_id
        && is_call_or_output_type(item_type)
    {
        return Some(format!("call_id:{item_type}:{cid}"));
    }
    let item_id = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| *id != FAKE_RESPONSES_ID);
    if let Some(id) = item_id {
        return Some(format!("id:{item_type}:{id}"));
    }
    if let Some(cid) = call_id {
        return Some(format!("call_id:{item_type}:{cid}"));
    }
    item.get("approval_request_id")
        .and_then(Value::as_str)
        .map(|rid| format!("approval_request_id:{item_type}:{rid}"))
}

/// Remove duplicate items that share stable identifiers, keeping the first.
/// Port of `deduplicate_input_items`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn deduplicate_input_items(items: Vec<InputItem>) -> Vec<InputItem> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match dedupe_key(&item.0) {
            None => out.push(item),
            Some(key) => {
                if seen.insert(key) {
                    out.push(item);
                }
            }
        }
    }
    out
}

/// Deduplicate by stable identity while keeping the latest value. Causal precursor items stay
/// at their earliest position; other identified items move to their latest.
/// Port of `deduplicate_input_items_preferring_latest`.
pub(crate) fn deduplicate_input_items_preferring_latest(items: Vec<InputItem>) -> Vec<InputItem> {
    let mut latest_by_key: HashMap<String, InputItem> = HashMap::new();
    let mut anchor_index_by_key: HashMap<String, usize> = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let Some(key) = dedupe_key(&item.0) else {
            continue;
        };
        latest_by_key.insert(key.clone(), item.clone());
        let item_type = item.0.get("type").and_then(Value::as_str).unwrap_or("");
        if !anchor_index_by_key.contains_key(&key) || !is_earliest_anchor_type(item_type) {
            anchor_index_by_key.insert(key, index);
        }
    }
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match dedupe_key(&item.0) {
            None => out.push(item.clone()),
            Some(key) => {
                if anchor_index_by_key.get(&key) == Some(&index) {
                    out.push(latest_by_key[&key].clone());
                }
            }
        }
    }
    out
}

fn program_caller_id(item: &Value) -> Option<&str> {
    let caller = item.get("caller")?;
    if caller.get("type").and_then(Value::as_str) != Some("program") {
        return None;
    }
    caller
        .get("caller_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// Call ids that already have a matching output, grouped by output type.
fn completed_call_ids(items: &[InputItem]) -> HashMap<&'static str, HashSet<String>> {
    let mut map: HashMap<&'static str, HashSet<String>> = HashMap::new();
    for item in items {
        let Some(ty) = item.0.get("type").and_then(Value::as_str) else {
            continue;
        };
        let Some(out_type) = TOOL_CALL_TO_OUTPUT_TYPE
            .iter()
            .find(|(_, out)| *out == ty)
            .map(|(_, out)| *out)
        else {
            continue;
        };
        if let Some(cid) = item.0.get("call_id").and_then(Value::as_str) {
            map.entry(out_type).or_default().insert(cid.to_string());
        }
    }
    map
}

/// Hosted item types a program can own; they keep their program active. Mirrors
/// `_PROGRAM_OWNED_HOSTED_ITEM_TYPES`.
const PROGRAM_OWNED_HOSTED_ITEM_TYPES: [&str; 9] = [
    "hosted_tool_call",
    "file_search_call",
    "web_search_call",
    "code_interpreter_call",
    "image_generation_call",
    "mcp_list_tools",
    "mcp_call",
    "mcp_approval_request",
    "mcp_approval_response",
];

/// Whether a program-owned item keeps its parent program active. With a pruning scope only an
/// item outside it is known to be retained; otherwise only hosted items, tool outputs and
/// pending hosted shell calls count (a bare call without its output does not). Port of
/// `_is_retained_program_owned_item`.
fn is_retained_program_owned_item(
    item: &Value,
    index: usize,
    pruning_indexes: Option<&HashSet<usize>>,
) -> bool {
    if let Some(set) = pruning_indexes {
        return !set.contains(&index);
    }
    let ty = item.get("type").and_then(Value::as_str).unwrap_or_default();
    let pending_hosted_shell = ty == "shell_call"
        && item
            .get("status")
            .is_none_or(|s| s.is_null() || s == "in_progress");
    pending_hosted_shell
        || PROGRAM_OWNED_HOSTED_ITEM_TYPES.contains(&ty)
        || TOOL_CALL_TO_OUTPUT_TYPE.iter().any(|(_, out)| *out == ty)
}

/// Remove tool and program call items that have no matching output, so a resumption or retry
/// does not replay stale calls, together with the reasoning items tied to them.
/// Port of `drop_orphan_function_calls`.
pub(crate) fn drop_orphan_function_calls(items: Vec<InputItem>) -> Vec<InputItem> {
    drop_orphan_function_calls_in(items, None, None)
}

/// [`drop_orphan_function_calls`] restricted to the stored-history positions in
/// `pruning_indexes` (items elsewhere are caller input and stay verbatim). When
/// `output_pruning_indexes` names unambiguous stored history, tool outputs whose call is no
/// longer present after pruning, such as one a session limit cut off, are dropped too.
pub(crate) fn drop_orphan_function_calls_in(
    items: Vec<InputItem>,
    pruning_indexes: Option<&HashSet<usize>>,
    output_pruning_indexes: Option<&HashSet<usize>>,
) -> Vec<InputItem> {
    let completed = completed_call_ids(&items);
    let in_scope = |index: usize| pruning_indexes.is_none_or(|set| set.contains(&index));

    // A program stays only while one of the items it owns is retained.
    let mut active_program_call_ids: HashSet<String> = HashSet::new();
    let mut orphan_program_call_ids: HashSet<String> = HashSet::new();
    for (index, item) in items.iter().enumerate() {
        if !in_scope(index) || item.0.get("type").and_then(Value::as_str) != Some("program") {
            continue;
        }
        let Some(call_id) = item.0.get("call_id").and_then(Value::as_str) else {
            continue;
        };
        if completed
            .get("program_output")
            .is_some_and(|ids| ids.contains(call_id))
        {
            continue;
        }
        let has_owned = items
            .iter()
            .enumerate()
            .any(|(candidate_index, candidate)| {
                program_caller_id(&candidate.0) == Some(call_id)
                    && is_retained_program_owned_item(
                        &candidate.0,
                        candidate_index,
                        pruning_indexes,
                    )
            });
        if has_owned {
            active_program_call_ids.insert(call_id.to_string());
        } else {
            orphan_program_call_ids.insert(call_id.to_string());
        }
    }

    // Anonymous `tool_search_call`s paired with a later anonymous output, latest call first.
    // Port of `_matched_anonymous_tool_search_call_indexes`.
    let mut matched_anonymous_searches: HashSet<usize> = HashSet::new();
    let mut pending_anonymous_outputs = 0usize;
    for (index, item) in items.iter().enumerate().rev() {
        let anonymous = !item.0.get("call_id").is_some_and(Value::is_string);
        match item.0.get("type").and_then(Value::as_str) {
            Some("tool_search_output") if anonymous => pending_anonymous_outputs += 1,
            Some("tool_search_call") if anonymous && pending_anonymous_outputs > 0 => {
                matched_anonymous_searches.insert(index);
                pending_anonymous_outputs -= 1;
            }
            _ => {}
        }
    }

    let mut dropped: HashSet<usize> = HashSet::new();
    for (index, item) in items.iter().enumerate() {
        let Some(ty) = item.0.get("type").and_then(Value::as_str) else {
            continue;
        };
        if !in_scope(index) {
            continue;
        }
        if let Some(caller_id) = program_caller_id(&item.0)
            && orphan_program_call_ids.contains(caller_id)
        {
            dropped.insert(index);
            continue;
        }
        let Some(out_type) = output_type_for(ty) else {
            continue;
        };
        let call_id = item.0.get("call_id").and_then(Value::as_str);
        if ty == "program" && call_id.is_some_and(|id| active_program_call_ids.contains(id)) {
            continue;
        }
        // A program-owned hosted shell call may stay pending without an output.
        if ty == "shell_call"
            && program_caller_id(&item.0).is_some()
            && item
                .0
                .get("status")
                .is_none_or(|s| s.is_null() || s == "in_progress")
        {
            continue;
        }
        match call_id {
            Some(cid) if completed.get(out_type).is_some_and(|ids| ids.contains(cid)) => continue,
            // An anonymous `tool_search_call` is matched positionally, not by id: it survives
            // only when a later anonymous output pairs with it.
            None if ty == "tool_search_call" && matched_anonymous_searches.contains(&index) => {
                continue;
            }
            _ => {}
        }
        dropped.insert(index);
    }

    // Only a dropped call or program-owned item strands the reasoning item before it.
    let reasoning_triggers: HashSet<usize> = dropped.clone();

    // Outputs whose call did not survive: `available` is every call id still retained.
    if let Some(output_indexes) = output_pruning_indexes {
        let mut available: HashMap<&'static str, HashSet<String>> = HashMap::new();
        for (index, item) in items.iter().enumerate() {
            if dropped.contains(&index) {
                continue;
            }
            let Some(ty) = item.0.get("type").and_then(Value::as_str) else {
                continue;
            };
            let Some(out_type) = output_type_for(ty) else {
                continue;
            };
            if let Some(cid) = item.0.get("call_id").and_then(Value::as_str) {
                available
                    .entry(out_type)
                    .or_default()
                    .insert(cid.to_string());
            }
        }
        let mut indexes: Vec<usize> = output_indexes.iter().copied().collect();
        indexes.sort_unstable();
        for index in indexes {
            if dropped.contains(&index) || index >= items.len() {
                continue;
            }
            let item = &items[index];
            let Some(ty) = item.0.get("type").and_then(Value::as_str) else {
                continue;
            };
            if !TOOL_CALL_TO_OUTPUT_TYPE.iter().any(|(_, out)| *out == ty) {
                continue;
            }
            if let Some(cid) = item.0.get("call_id").and_then(Value::as_str)
                && !available.get(ty).is_some_and(|ids| ids.contains(cid))
            {
                dropped.insert(index);
            }
        }
    }

    if dropped.is_empty() {
        return items;
    }

    // A reasoning item belongs to the next non-reasoning item; drop it when that item went.
    let mut drop_reasoning: HashSet<usize> = HashSet::new();
    for index in (0..items.len()).rev() {
        if items[index].0.get("type").and_then(Value::as_str) != Some("reasoning")
            || dropped.contains(&index)
        {
            continue;
        }
        for (next, item) in items.iter().enumerate().skip(index + 1) {
            if drop_reasoning.contains(&next) {
                continue;
            }
            if item.0.get("type").and_then(Value::as_str) == Some("reasoning") {
                continue;
            }
            if reasoning_triggers.contains(&next) {
                drop_reasoning.insert(index);
            }
            break;
        }
    }

    items
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !dropped.contains(index) && !drop_reasoning.contains(index))
        .map(|(_, item)| item)
        .collect()
}

/// Reorder the tool outputs appended from `start` so they follow the order of their calls.
/// Port of `order_current_turn_tool_outputs`.
pub(crate) fn order_current_turn_tool_outputs(
    items: &mut [RunItem],
    start: usize,
    call_positions: &HashMap<String, usize>,
) {
    let slots: Vec<usize> = (start.min(items.len())..items.len())
        .filter(|index| {
            matches!(items[*index].kind, RunItemKind::ToolCallOutput { .. })
                && output_call_id(&items[*index])
                    .is_some_and(|cid| call_positions.contains_key(&cid))
        })
        .collect();
    if slots.len() < 2 {
        return;
    }
    let mut outputs: Vec<RunItem> = slots.iter().map(|index| items[*index].clone()).collect();
    outputs.sort_by_key(|item| {
        output_call_id(item)
            .and_then(|cid| call_positions.get(&cid).copied())
            .unwrap_or(usize::MAX)
    });
    for (slot, output) in slots.into_iter().zip(outputs) {
        items[slot] = output;
    }
}

fn output_call_id(item: &RunItem) -> Option<String> {
    match &item.kind {
        RunItemKind::ToolCallOutput { raw_item, .. } => raw_item
            .get("call_id")
            .or_else(|| raw_item.get("id"))
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

/// Provider-executed call types that finished server side once their status is `completed`.
const PROVIDER_COMPLETED_CALL_TYPES: [&str; 5] = [
    "file_search_call",
    "web_search_call",
    "code_interpreter_call",
    "image_generation_call",
    "mcp_call",
];

/// The part of a failed tool turn that stays in history: accepted call/output pairs in model
/// order with the reasoning right before them, completed provider-side items, then the
/// committed outputs. Calls that never got an output, the turn's messages and stranded
/// reasoning are dropped, so a retry neither replays nor repeats them.
/// Port of `_completed_tool_step_items`.
pub(crate) fn completed_tool_step_items(
    prior: &[RunItem],
    model: &[RunItem],
    outputs: &[RunItem],
) -> Vec<RunItem> {
    // The raw item keeps fields (such as `status`) the replayed input item may drop.
    let raw = |item: &RunItem| match &item.kind {
        RunItemKind::ToolCall { raw_item } | RunItemKind::ToolCallOutput { raw_item, .. } => {
            raw_item.clone()
        }
        _ => item.to_input_item().0,
    };
    let str_of = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).map(str::to_string);
    let is_output_type = |ty: &str| TOOL_CALL_TO_OUTPUT_TYPE.iter().any(|(_, out)| *out == ty);
    // Outputs the model returned itself count as accepted once they completed.
    let mut accepted_call_ids: HashSet<String> = HashSet::new();
    for item in model.iter().chain(outputs) {
        let v = raw(item);
        let ty = str_of(&v, "type").unwrap_or_default();
        let from_model = !outputs.iter().any(|o| std::ptr::eq(o, item));
        if is_output_type(&ty)
            && (!from_model || str_of(&v, "status").as_deref() == Some("completed"))
            && let Some(call_id) = str_of(&v, "call_id")
        {
            accepted_call_ids.insert(call_id);
        }
    }
    let mut candidates: Vec<RunItem> = Vec::new();
    let mut reasoning: Vec<RunItem> = Vec::new();
    for item in model {
        let v = raw(item);
        let ty = str_of(&v, "type").unwrap_or_default();
        if ty == "reasoning" {
            reasoning.push(item.clone());
            continue;
        }
        let completed = str_of(&v, "status").as_deref() == Some("completed");
        let call_id = str_of(&v, "call_id");
        let keep = if ty == "mcp_approval_request" {
            // An answered hosted approval request is a settled decision point; its response is
            // kept below. A request left unanswered (its approval callback failed) is dropped,
            // as Python's `_completed_tool_step_items` drops every approval request.
            let id = str_of(&v, "id");
            outputs.iter().any(|o| {
                matches!(o.kind, RunItemKind::MCPApprovalResponse { .. })
                    && str_of(&o.to_input_item().0, "approval_request_id") == id
            })
        } else if ty == "tool_search_call" || ty == "tool_search_output" {
            str_of(&v, "execution").as_deref() == Some("server") && completed
        } else if is_output_type(&ty) {
            completed
        } else if output_type_for(&ty).is_some()
            || PROVIDER_COMPLETED_CALL_TYPES.contains(&ty.as_str())
        {
            call_id.is_some_and(|id| accepted_call_ids.contains(&id))
                || (PROVIDER_COMPLETED_CALL_TYPES.contains(&ty.as_str()) && completed)
                || ty == "program"
        } else {
            false
        };
        if keep {
            candidates.append(&mut reasoning);
            candidates.push(item.clone());
        }
        // Reasoning belongs to the next model item, even when that item is omitted.
        reasoning.clear();
    }
    candidates.extend(outputs.iter().cloned());

    // Prune what would still be an orphan next to the prior history; the index marker maps the
    // surviving payloads back to their run items.
    const MARKER: &str = "__completed_tool_step_index";
    let mut inputs: Vec<InputItem> = prior.iter().map(RunItem::to_input_item).collect();
    let offset = inputs.len();
    for (index, item) in candidates.iter().enumerate() {
        let mut v = raw(item);
        if let Value::Object(m) = &mut v {
            m.insert(MARKER.into(), Value::from(index));
        }
        inputs.push(InputItem(v));
    }
    let scope: HashSet<usize> = (offset..inputs.len()).collect();
    let kept: HashSet<usize> = drop_orphan_function_calls_in(inputs, None, Some(&scope))
        .iter()
        .filter_map(|i| i.0.get(MARKER).and_then(Value::as_u64))
        .map(|i| i as usize)
        .collect();
    candidates
        .into_iter()
        .enumerate()
        .filter(|(index, _)| kept.contains(index))
        .map(|(_, item)| item)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(v: Value) -> InputItem {
        InputItem(v)
    }

    #[test]
    fn drops_calls_without_outputs_and_their_reasoning() {
        let items = vec![
            item(json!({"type": "reasoning", "id": "rs_1"})),
            item(json!({"type": "function_call", "call_id": "c1", "name": "t"})),
            item(json!({"type": "reasoning", "id": "rs_2"})),
            item(json!({"type": "function_call", "call_id": "c2", "name": "t"})),
            item(json!({"type": "function_call_output", "call_id": "c2", "output": "ok"})),
        ];
        let kept = drop_orphan_function_calls(items);
        let types: Vec<&str> = kept.iter().map(|i| i.0["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            vec!["reasoning", "function_call", "function_call_output",]
        );
        assert_eq!(kept[0].0["id"], "rs_2");
    }

    #[test]
    fn keeps_everything_when_all_calls_are_answered() {
        let items = vec![
            item(json!({"role": "user", "content": "hi"})),
            item(json!({"type": "function_call", "call_id": "c1", "name": "t"})),
            item(json!({"type": "function_call_output", "call_id": "c1", "output": "ok"})),
        ];
        assert_eq!(drop_orphan_function_calls(items.clone()), items);
    }

    #[test]
    fn program_owned_items_follow_their_program() {
        let items = vec![
            item(json!({"type": "program", "call_id": "p1"})),
            item(json!({
                "type": "function_call",
                "call_id": "c1",
                "caller": {"type": "program", "caller_id": "p1"},
            })),
        ];
        // A bare call without its output does not keep its program active, so the whole
        // orphan chain goes (Python `test_replay_pruning_drops_orphan_program_call_chain`).
        assert!(drop_orphan_function_calls(items).is_empty());
    }

    #[test]
    fn dedupes_repeated_call_ids() {
        let items = vec![
            item(json!({"type": "function_call", "call_id": "c1", "name": "a"})),
            item(json!({"type": "function_call", "call_id": "c1", "name": "a"})),
            item(json!({"role": "user", "content": "hi"})),
            item(json!({"role": "user", "content": "hi"})),
        ];
        let kept = deduplicate_input_items(items);
        assert_eq!(kept.len(), 3, "messages are never deduped");
    }

    #[test]
    fn preferring_latest_keeps_call_position_but_latest_value() {
        let items = vec![
            item(json!({"type": "function_call", "call_id": "c1", "name": "a"})),
            item(json!({"role": "user", "content": "hi"})),
            item(json!({"type": "function_call", "call_id": "c1", "name": "b"})),
        ];
        let kept = deduplicate_input_items_preferring_latest(items);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].0["name"], "b", "latest value at earliest anchor");
        assert_eq!(kept[1].0["role"], "user");
    }

    #[test]
    fn orders_outputs_by_call_order() {
        let mut items = vec![
            RunItem::new(
                "A",
                RunItemKind::ToolCallOutput {
                    custom_data: None,
                    raw_item: json!({"type": "function_call_output", "call_id": "c2"}),
                    output: json!("2"),
                },
            ),
            RunItem::new(
                "A",
                RunItemKind::ToolCallOutput {
                    custom_data: None,
                    raw_item: json!({"type": "function_call_output", "call_id": "c1"}),
                    output: json!("1"),
                },
            ),
        ];
        let positions: HashMap<String, usize> =
            [("c1".to_string(), 0), ("c2".to_string(), 1)].into();
        order_current_turn_tool_outputs(&mut items, 0, &positions);
        assert_eq!(items[0].tool_output().unwrap(), &json!("1"));
        assert_eq!(items[1].tool_output().unwrap(), &json!("2"));
    }
    fn settle(items: Vec<Value>, pending: &[&str]) -> Vec<Value> {
        let pending: HashSet<String> = pending.iter().map(|id| id.to_string()).collect();
        crate::run_internal::settle_safe_items(items.into_iter().map(item).collect(), &pending)
            .into_iter()
            .map(|item| item.0)
            .collect()
    }

    #[test]
    fn a_held_batch_settles_only_requests_that_can_pair() {
        let unpaired_mcp = json!({"type": "mcp_approval_request", "id": "mcpr_1",
            "name": "do_it", "server_label": "srv", "arguments": "{}"});
        let paired_mcp = json!({"type": "mcp_approval_request", "id": "mcpr_2",
            "name": "do_it", "server_label": "srv", "arguments": "{}"});
        let mcp_response = json!({"type": "mcp_approval_response",
            "approval_request_id": "mcpr_2", "approve": true});
        let unpaired_custom = json!({"type": "custom_tool_call", "call_id": "cust_1",
            "name": "custom", "input": ""});
        let preamble = json!({"role": "assistant", "content": "hi", "type": "message"});
        let batch = vec![
            unpaired_mcp.clone(),
            paired_mcp.clone(),
            mcp_response.clone(),
            unpaired_custom,
            preamble.clone(),
        ];
        assert_eq!(settle(batch, &[]), vec![paired_mcp, mcp_response, preamble]);
        // A request whose approval is still open settles now and pairs up at a later exit.
        assert_eq!(
            settle(vec![unpaired_mcp.clone()], &["mcpr_1"]),
            vec![unpaired_mcp]
        );
    }

    #[test]
    fn a_held_batch_prunes_with_the_canonical_rule() {
        let reasoning = json!({"type": "reasoning", "id": "rs_1", "summary": []});
        let unpaired_shell = json!({"type": "shell_call", "call_id": "sh_1", "id": "sh_item_1",
            "status": "completed", "action": {"type": "exec", "command": "ls"}});
        let paired_call = json!({"type": "function_call", "call_id": "fn_1",
            "name": "write_thing", "arguments": "{}"});
        let paired_output = json!({"type": "function_call_output", "call_id": "fn_1",
            "output": "ok"});
        // The reasoning item riding before a dropped call goes with it.
        assert_eq!(
            settle(
                vec![
                    reasoning.clone(),
                    unpaired_shell.clone(),
                    paired_call.clone(),
                    paired_output.clone()
                ],
                &[]
            ),
            vec![paired_call, paired_output]
        );
        assert_eq!(
            settle(vec![reasoning.clone(), unpaired_shell.clone()], &["sh_1"]),
            vec![reasoning, unpaired_shell]
        );
    }
}
