//! Ports of `tests/sandbox/test_pty_types.py`, `test_pty_output.py` and `test_parse_utils.py`.
//!
//! Python's `collect_pty_output` takes injectable asyncio primitives; the Rust counterpart is
//! [`RemotePtyProcess::collect`], which owns its buffer and notification, so the same behaviours
//! are driven by pushing output and closing the process.
#![cfg(feature = "sandbox")]

use std::collections::BTreeSet;
use std::time::Duration;

use openai_agents_sdk::sandbox::pty::{
    PTY_EMPTY_YIELD_TIME_MS_MIN, PTY_PROCESS_ID_MAX_EXCLUSIVE, PTY_PROCESS_ID_MIN,
    PTY_PROCESSES_MAX, PTY_YIELD_TIME_MS_MIN, RemotePtyManager, clamp_pty_yield_time_ms,
    incomplete_utf8_suffix_length, resolve_pty_write_yield_time_ms,
};
use openai_agents_sdk::sandbox::types::{ListingKind, file_mode, try_parse_ls_la};

// ----------------------------------------------------------------------------------------------
// test_pty_types.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_clamp_pty_yield_time_ms_enforces_minimum`.
#[test]
fn clamp_pty_yield_time_ms_enforces_minimum() {
    assert_eq!(clamp_pty_yield_time_ms(0), PTY_YIELD_TIME_MS_MIN);
}

/// Python: `test_resolve_pty_write_yield_time_ms_uses_longer_poll_for_empty_input`.
#[test]
fn resolve_pty_write_yield_time_ms_uses_a_longer_poll_for_empty_input() {
    assert_eq!(
        resolve_pty_write_yield_time_ms(PTY_YIELD_TIME_MS_MIN, true),
        PTY_EMPTY_YIELD_TIME_MS_MIN
    );
    assert_eq!(
        resolve_pty_write_yield_time_ms(PTY_YIELD_TIME_MS_MIN, false),
        PTY_YIELD_TIME_MS_MIN
    );
}

/// Python: `test_allocate_pty_process_id_avoids_used_ids`. Ids are allocated by the manager, so
/// the check is that live sessions never share an id and stay inside the id range.
#[test]
fn allocated_pty_process_ids_are_distinct_and_in_range() {
    let manager = RemotePtyManager::new();
    let mut seen = BTreeSet::new();
    for _ in 0..40 {
        let (process, _) = manager.register(true).unwrap();
        assert!((PTY_PROCESS_ID_MIN..PTY_PROCESS_ID_MAX_EXCLUSIVE).contains(&process.id()));
        assert!(seen.insert(process.id()), "id {} was reused", process.id());
    }
}

/// Python: `test_process_id_to_prune_from_meta_prefers_exited_unprotected_sessions`. Python picks
/// one id from usage metadata; the manager prunes while registering, finished sessions first.
#[test]
fn pruning_prefers_exited_sessions() {
    let manager = RemotePtyManager::new();
    let mut processes = Vec::new();
    for _ in 0..PTY_PROCESSES_MAX {
        processes.push(manager.register(true).unwrap().0);
    }
    let exited = processes[3].clone();
    exited.close(Some(0));

    let (_, pruned) = manager.register(true).unwrap();

    let pruned_ids: Vec<u32> = pruned.iter().map(|process| process.id()).collect();
    assert!(pruned_ids.contains(&exited.id()), "{pruned_ids:?}");
    assert_eq!(
        pruned_ids.len(),
        1,
        "only the exited session made room: {pruned_ids:?}"
    );
}

// ----------------------------------------------------------------------------------------------
// test_pty_output.py
// ----------------------------------------------------------------------------------------------

/// Python: `test_collect_pty_output_waits_for_notification`.
#[tokio::test]
async fn collect_waits_for_notification() {
    let manager = RemotePtyManager::new();
    let (process, _) = manager.register(true).unwrap();
    let producer = process.clone();
    let task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        producer.push_output(b"notified output");
        producer.close(Some(0));
    });

    let update = process.collect(500, None).await;
    task.await.unwrap();

    assert_eq!(update.output, b"notified output");
    assert_eq!(update.original_token_count, None);
    assert_eq!(
        update.process_id, None,
        "a finished process is not resumable"
    );
}

/// Python: `test_collect_pty_output_drains_chunks_added_when_done`.
#[tokio::test]
async fn collect_drains_chunks_added_when_done() {
    let manager = RemotePtyManager::new();
    let (process, _) = manager.register(true).unwrap();
    process.push_output(b"before done");
    process.push_output(b" after done");
    process.close(Some(0));

    let update = process.collect(500, None).await;

    assert_eq!(update.output, b"before done after done");
    assert_eq!(update.process_id, None);
}

/// Python: `test_collect_pty_output_drains_chunks_queued_when_wait_times_out`.
#[tokio::test]
async fn collect_returns_output_queued_before_the_window_ends() {
    let manager = RemotePtyManager::new();
    let (process, _) = manager.register(true).unwrap();
    process.push_output(b"queued at timeout");

    let update = process.collect(250, None).await;

    assert_eq!(update.output, b"queued at timeout");
    assert_eq!(update.process_id, Some(process.id()), "still running");
    // Nothing is left behind for the next window.
    let next = process.collect(250, None).await;
    assert!(next.output.is_empty());
}

/// Python: `test_collect_pty_output_preserves_valid_utf8_at_every_split` (6 parameter sets).
#[tokio::test]
async fn collect_preserves_valid_utf8_at_every_split() {
    let cases: [(&str, usize); 6] = [
        ("\u{e9}", 1),
        ("\u{20ac}", 1),
        ("\u{20ac}", 2),
        ("\u{1f600}", 1),
        ("\u{1f600}", 2),
        ("\u{1f600}", 3),
    ];
    for (character, split) in cases {
        let encoded = character.as_bytes();
        let manager = RemotePtyManager::new();
        let (process, _) = manager.register(true).unwrap();
        process.push_output(&[b"a".as_slice(), &encoded[..split]].concat());

        let first = process.collect(250, None).await;
        process.push_output(&[&encoded[split..], b"b"].concat());
        process.close(Some(0));
        let second = process.collect(250, None).await;

        assert_eq!(first.output, b"a", "{character:?} split at {split}");
        assert!(first.process_id.is_some());
        let mut joined = first.output.clone();
        joined.extend(&second.output);
        assert_eq!(
            joined,
            format!("a{character}b").into_bytes(),
            "{character:?} split at {split}"
        );
        assert_eq!(second.process_id, None);
    }
}

/// Python: `test_collect_pty_output_replaces_restricted_utf8_prefixes_without_carry` (4 sets).
#[tokio::test]
async fn collect_replaces_restricted_utf8_prefixes_without_carry() {
    for invalid_prefix in [
        &b"\xe0\x80"[..],
        &b"\xed\xa0"[..],
        &b"\xf0\x80"[..],
        &b"\xf4\x90"[..],
    ] {
        let manager = RemotePtyManager::new();
        let (process, _) = manager.register(true).unwrap();
        process.push_output(&[b"prompt".as_slice(), invalid_prefix].concat());

        let update = process.collect(250, None).await;

        assert_eq!(
            String::from_utf8(update.output).unwrap(),
            "prompt\u{fffd}\u{fffd}",
            "{invalid_prefix:?}"
        );
        assert!(update.process_id.is_some(), "the process is still running");
        let next = process.collect(250, None).await;
        assert!(
            next.output.is_empty(),
            "nothing was carried: {invalid_prefix:?}"
        );
    }
}

/// Python: `test_incomplete_utf8_suffix_length_accepts_only_completable_sequences` (14 sets).
#[test]
fn incomplete_utf8_suffix_length_accepts_only_completable_sequences() {
    let cases: [(&[u8], usize); 14] = [
        (b"", 0),
        (b"abc", 0),
        (b"\xc3", 1),
        (b"\xe2\x82", 2),
        (b"\xf0\x9f\x98", 3),
        (b"\xe0\x80", 0),
        (b"\xe0\xa0", 2),
        (b"\xed\xa0", 0),
        (b"\xed\x9f", 2),
        (b"\xf0\x80", 0),
        (b"\xf0\x90", 2),
        (b"\xf4\x90", 0),
        (b"\xf4\x8f", 2),
        (b"\x80\x80\x80", 0),
    ];
    for (data, expected) in cases {
        assert_eq!(incomplete_utf8_suffix_length(data), expected, "{data:?}");
    }
}

/// Python: `test_collect_pty_output_settles_terminal_carry_once_across_repeated_reads`.
#[tokio::test]
async fn collect_settles_a_terminal_carry_once_across_repeated_reads() {
    let manager = RemotePtyManager::new();
    let (process, _) = manager.register(true).unwrap();
    process.push_output(b"tail\xe2\x82");

    let first = process.collect(250, None).await;
    process.close(Some(0));
    let terminal = process.collect(250, None).await;
    let repeated = process.collect(250, None).await;

    assert_eq!(first.output, b"tail");
    assert!(first.process_id.is_some());
    assert_eq!(String::from_utf8(terminal.output).unwrap(), "\u{fffd}");
    assert_eq!(terminal.process_id, None);
    assert!(repeated.output.is_empty());
    assert_eq!(repeated.process_id, None);
}

/// Python: `test_collect_pty_output_restores_carry_when_next_collection_is_cancelled`.
#[tokio::test]
async fn collect_restores_the_carry_when_the_next_collection_is_cancelled() {
    let manager = RemotePtyManager::new();
    let (process, _) = manager.register(true).unwrap();
    process.push_output(b"\xc3");

    let first = process.collect(250, None).await;
    assert!(first.output.is_empty());
    // The next window is dropped mid-wait, as a cancelled task would be.
    let cancelled =
        tokio::time::timeout(Duration::from_millis(60), process.collect(5_000, None)).await;
    assert!(cancelled.is_err());

    process.push_output(b"\xa9");
    process.close(Some(0));
    let terminal = process.collect(250, None).await;

    assert_eq!(terminal.output, "\u{e9}".as_bytes());
    assert_eq!(terminal.process_id, None);
}

// ----------------------------------------------------------------------------------------------
// test_parse_utils.py
// ----------------------------------------------------------------------------------------------

fn paths(output: &str, base: &str) -> Vec<String> {
    try_parse_ls_la(output, base)
        .unwrap()
        .into_iter()
        .map(|listing| listing.entry.path)
        .collect()
}

/// Python: `test_parse_ls_la_preserves_absolute_file_paths`.
#[test]
fn parse_ls_la_preserves_absolute_file_paths() {
    let output = "-rwxr-xr-x 1 root root 48915747 Jan 1 00:00 /workspace/bin/tool\n";

    let entries = try_parse_ls_la(output, "/workspace/bin/tool").unwrap();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].entry.path, "/workspace/bin/tool");
    assert_eq!(entries[0].kind, ListingKind::File);
}

/// Python: `test_parse_ls_la_prefixes_directory_entries_with_base`.
#[test]
fn parse_ls_la_prefixes_directory_entries_with_base() {
    let output = "drwxr-xr-x 2 root root     4096 Jan  1 00:00 .\n\
                  drwxr-xr-x 3 root root     4096 Jan  1 00:00 ..\n\
                  -rw-r--r-- 1 root root      123 Jan  1 00:00 notes.md\n";

    let entries = try_parse_ls_la(output, "/workspace/docs").unwrap();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].entry.path, "/workspace/docs/notes.md");
    assert_eq!(entries[0].kind, ListingKind::File);
}

/// Python: `test_parse_ls_la_keeps_arrow_in_regular_file_names`.
#[test]
fn parse_ls_la_keeps_an_arrow_in_regular_file_names() {
    let output = "-rw-r--r-- 1 root root 123 Jan 1 00:00 notes -> final.txt\n";

    let entries = try_parse_ls_la(output, "/workspace/docs").unwrap();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].entry.path, "/workspace/docs/notes -> final.txt");
    assert_eq!(entries[0].kind, ListingKind::File);
}

/// Python: `test_parse_ls_la_accepts_special_permission_bits`.
#[test]
fn parse_ls_la_accepts_special_permission_bits() {
    let output = "drwxrwxrwt 2 root root 4096 Jan 1 00:00 tmp\n\
                  -rwsr-sr-t 1 root root 123 Jan 1 00:00 setuid-tool\n\
                  -rwSr-Sr-T 1 root root 456 Jan 1 00:00 special-no-exec\n";

    let entries = try_parse_ls_la(output, "/").unwrap();

    assert_eq!(
        entries
            .iter()
            .map(|e| e.entry.path.as_str())
            .collect::<Vec<_>>(),
        ["/tmp", "/setuid-tool", "/special-no-exec"]
    );
    let permissions = |index: usize| entries[index].entry.permissions.unwrap();
    assert!(permissions(0).directory);
    assert_ne!(permissions(0).other & file_mode::EXEC, 0);
    assert_ne!(permissions(1).owner & file_mode::EXEC, 0);
    assert_ne!(permissions(1).group & file_mode::EXEC, 0);
    assert_ne!(permissions(1).other & file_mode::EXEC, 0);
    assert_eq!(permissions(2).owner & file_mode::EXEC, 0);
    assert_eq!(permissions(2).group & file_mode::EXEC, 0);
    assert_eq!(permissions(2).other & file_mode::EXEC, 0);
}

/// Python: `test_parse_ls_la_strips_trailing_alternate_access_markers`.
#[test]
fn parse_ls_la_strips_trailing_alternate_access_markers() {
    let output = "drwxr-xr-x. 2 root root 4096 Jan 1 00:00 selinux-dir\n\
                  -rw-r--r--+ 1 root root  123 Jan 1 00:00 acl-file\n\
                  -rw-r--r--@ 1 root root  456 Jan 1 00:00 xattr-file\n";

    let entries = try_parse_ls_la(output, "/").unwrap();

    assert_eq!(
        entries
            .iter()
            .map(|e| e.entry.path.as_str())
            .collect::<Vec<_>>(),
        ["/selinux-dir", "/acl-file", "/xattr-file"]
    );
    assert!(entries[0].entry.permissions.unwrap().directory);
    assert_ne!(
        entries[0].entry.permissions.unwrap().owner & file_mode::READ,
        0
    );
    assert_ne!(
        entries[1].entry.permissions.unwrap().owner & file_mode::WRITE,
        0
    );
    assert_ne!(
        entries[2].entry.permissions.unwrap().owner & file_mode::READ,
        0
    );
}

/// Python: `test_parse_ls_la_includes_gnu_device_nodes`.
#[test]
fn parse_ls_la_includes_gnu_device_nodes() {
    let output = "-rw-r--r-- 1 root root      123 Jan  1 00:00 regular.txt\n\
                  crw-rw-rw- 1 root root     1, 3 Jan  1 00:00 null\n\
                  brw-rw---- 1 root disk     8, 0 Jan  1 00:00 sda\n";

    let entries = try_parse_ls_la(output, "/dev").unwrap();

    assert_eq!(
        paths(output, "/dev"),
        ["/dev/regular.txt", "/dev/null", "/dev/sda"]
    );
    assert_eq!(entries[1].kind, ListingKind::Other);
    assert_eq!(entries[1].entry.size, Some(0));
    assert_eq!(entries[2].owner, "root");
    assert_eq!(entries[2].group, "disk");
}

/// Python: `test_parse_ls_la_includes_bsd_device_nodes`.
#[test]
fn parse_ls_la_includes_bsd_device_nodes() {
    let output = "-rw-r--r--  1 root  wheel           123 Jan  1 00:00 regular.txt\n\
                  crw-rw-rw-  1 root  wheel     0x3000002 Jan  1 00:00 null\n\
                  brw-r-----  1 root  operator  0x1000000 Jan  1 00:00 disk0\n";

    let entries = try_parse_ls_la(output, "/dev").unwrap();

    assert_eq!(
        paths(output, "/dev"),
        ["/dev/regular.txt", "/dev/null", "/dev/disk0"]
    );
    assert_eq!(entries[1].kind, ListingKind::Other);
    assert_eq!(entries[1].entry.size, Some(0));
    assert_eq!(entries[2].owner, "root");
    assert_eq!(entries[2].group, "operator");
}

/// Python: `test_parse_ls_la_rejects_special_permission_bits_in_wrong_position` (6 sets).
#[test]
fn parse_ls_la_rejects_special_permission_bits_in_the_wrong_position() {
    for permissions in [
        "-rwTr--r--",
        "-rwxrwTr--",
        "-rwxrwxr-S",
        "-rwtr--r--",
        "-rwxrwtr--",
        "-rwxrwxr-s",
    ] {
        let output = format!("{permissions} 1 root root 123 Jan 1 00:00 invalid\n");

        let error = try_parse_ls_la(&output, "/").expect_err(permissions);

        assert!(
            error.contains("invalid exec flag"),
            "{permissions}: {error}"
        );
    }
}
