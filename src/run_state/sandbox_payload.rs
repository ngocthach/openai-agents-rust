//! The `sandbox` resume payload of a Python `RunState` snapshot.
//!
//! Python stores the serialized sandbox session state next to the run state. Rust keeps sandbox
//! state in its own [`SandboxSessionState`](crate::sandbox::state::SandboxSessionState), so this
//! payload stays opaque JSON: it is read from a snapshot and written back so that re-saving a
//! Python snapshot loses nothing. What is *not* kept is mount authority. Port of
//! `agents.sandbox._mount_security.sanitize_run_state_sandbox_mount_authority`: credentials,
//! credential-file sources, unsafe endpoint URLs and opaque provider configuration are removed
//! from the raw manifest, and a session state that lost any is marked so it is never resumed as
//! if it still carried them. Errors never include any payload value.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::error::{AgentsError, Result};

const REDACTED_MOUNT_AUTHORITY_KEY: &str = "__openai_agents_redacted_mount_authority";
const CREDENTIALLESS_MOUNT_AUTHORITY_KEY: &str = "__openai_agents_credentialless_mount_authority";

const INVALID_ENVELOPE: &str = "RunState sandbox resume state has an invalid envelope";
const INVALID_MANIFEST: &str = "RunState sandbox resume state contains an invalid manifest";

/// Registered manifest entry types: `(type, is_dir, serialized fields)`.
const ENTRY_TYPES: &[(&str, bool, &[&str])] = &[
    (
        "dir",
        true,
        &[
            "type",
            "description",
            "ephemeral",
            "group",
            "is_dir",
            "permissions",
            "children",
        ],
    ),
    (
        "file",
        false,
        &[
            "type",
            "description",
            "ephemeral",
            "group",
            "is_dir",
            "permissions",
            "content",
        ],
    ),
    (
        "local_file",
        false,
        &[
            "type",
            "description",
            "ephemeral",
            "group",
            "is_dir",
            "permissions",
            "src",
        ],
    ),
    (
        "local_dir",
        false,
        &[
            "type",
            "description",
            "ephemeral",
            "group",
            "is_dir",
            "permissions",
            "src",
        ],
    ),
    (
        "git_repo",
        false,
        &[
            "type",
            "description",
            "ephemeral",
            "group",
            "is_dir",
            "permissions",
            "host",
            "repo",
            "ref",
            "subpath",
        ],
    ),
];

/// Fields every mount entry serializes.
const MOUNT_BASE_FIELDS: &[&str] = &[
    "type",
    "description",
    "ephemeral",
    "group",
    "is_dir",
    "permissions",
    "mount_path",
    "read_only",
    "mount_strategy",
];

/// Mount entry types and the provider fields they serialize beyond [`MOUNT_BASE_FIELDS`].
const MOUNT_TYPES: &[(&str, &[&str])] = &[
    (
        "azure_blob_mount",
        &[
            "account",
            "container",
            "endpoint",
            "identity_client_id",
            "account_key",
        ],
    ),
    (
        "box_mount",
        &[
            "path",
            "client_id",
            "client_secret",
            "access_token",
            "token",
            "box_config_file",
            "config_credentials",
            "box_sub_type",
            "root_folder_id",
            "impersonate",
            "owned_by",
        ],
    ),
    (
        "gcs_mount",
        &[
            "bucket",
            "access_id",
            "secret_access_key",
            "prefix",
            "region",
            "endpoint_url",
            "service_account_file",
            "service_account_credentials",
            "access_token",
        ],
    ),
    (
        "r2_mount",
        &[
            "bucket",
            "account_id",
            "access_key_id",
            "secret_access_key",
            "custom_domain",
        ],
    ),
    (
        "s3_mount",
        &[
            "bucket",
            "access_key_id",
            "secret_access_key",
            "session_token",
            "prefix",
            "region",
            "endpoint_url",
            "s3_provider",
        ],
    ),
    (
        "s3_files_mount",
        &[
            "file_system_id",
            "subpath",
            "mount_target_ip",
            "access_point",
            "region",
            "extra_options",
        ],
    ),
];

/// SDK extension mounts that may not be registered when the snapshot is read.
const EXTENSION_MOUNT_TYPES: &[(&str, &[&str])] = &[(
    "blaxel_drive_mount",
    &[
        "type",
        "description",
        "ephemeral",
        "group",
        "is_dir",
        "permissions",
        "mount_path",
        "read_only",
        "mount_strategy",
        "drive_name",
        "drive_mount_path",
        "drive_path",
        "drive_read_only",
    ],
)];

const AUTHORITY_FIELDS: &[(&str, &[&str])] = &[
    ("azure_blob_mount", &["identity_client_id", "account_key"]),
    (
        "box_mount",
        &[
            "client_secret",
            "access_token",
            "token",
            "box_config_file",
            "config_credentials",
        ],
    ),
    (
        "gcs_mount",
        &[
            "access_id",
            "secret_access_key",
            "service_account_file",
            "service_account_credentials",
            "access_token",
        ],
    ),
    ("r2_mount", &["access_key_id", "secret_access_key"]),
    (
        "s3_mount",
        &["access_key_id", "secret_access_key", "session_token"],
    ),
];
const AUTHORITY_FILE_FIELDS: &[(&str, &[&str])] = &[
    ("box_mount", &["box_config_file"]),
    ("gcs_mount", &["service_account_file"]),
];
const URL_FIELDS: &[(&str, &[&str])] = &[
    ("azure_blob_mount", &["endpoint"]),
    ("gcs_mount", &["endpoint_url"]),
    ("r2_mount", &["custom_domain"]),
    ("s3_mount", &["endpoint_url"]),
];
const BLAXEL_S3FS_OPTION_FIELDS: &[(&str, &[&str])] = &[
    ("r2_mount", &["custom_domain", "account_id"]),
    ("s3_mount", &["endpoint_url", "region"]),
];
const RCLONE_CONFIG_VALUE_FIELDS: &[(&str, &[&str])] = &[
    (
        "azure_blob_mount",
        &["account", "endpoint", "identity_client_id", "account_key"],
    ),
    (
        "box_mount",
        &[
            "client_id",
            "client_secret",
            "access_token",
            "token",
            "box_config_file",
            "config_credentials",
            "root_folder_id",
            "impersonate",
            "owned_by",
        ],
    ),
    (
        "gcs_mount",
        &[
            "access_id",
            "secret_access_key",
            "region",
            "endpoint_url",
            "service_account_file",
            "service_account_credentials",
            "access_token",
        ],
    ),
    (
        "r2_mount",
        &[
            "account_id",
            "access_key_id",
            "secret_access_key",
            "custom_domain",
        ],
    ),
    (
        "s3_mount",
        &[
            "s3_provider",
            "endpoint_url",
            "region",
            "access_key_id",
            "secret_access_key",
            "session_token",
        ],
    ),
];

/// Serialized fields of each known mount strategy type.
const STRATEGY_FIELDS: &[(&str, &[&str])] = &[
    ("in_container", &["type", "pattern"]),
    ("docker_volume", &["type", "driver", "driver_options"]),
    ("blaxel_cloud_bucket", &["type"]),
    ("blaxel_drive", &["type"]),
    ("cloudflare_bucket_mount", &["type"]),
    ("daytona_cloud_bucket", &["type", "pattern"]),
    ("e2b_cloud_bucket", &["type", "pattern"]),
    (
        "modal_cloud_bucket",
        &["type", "secret_name", "secret_environment_name"],
    ),
    ("vercel_cloud_bucket", &["type"]),
];
/// Strategy fields holding opaque third-party configuration: removed as a unit.
const OPAQUE_STRATEGY_FIELDS: &[(&str, &[&str])] = &[
    ("docker_volume", &["driver_options"]),
    (
        "modal_cloud_bucket",
        &["secret_name", "secret_environment_name"],
    ),
];
const PATTERN_FIELDS: &[(&str, &[&str])] = &[
    (
        "fuse",
        &[
            "type",
            "allow_other",
            "log_type",
            "log_level",
            "cache_type",
            "cache_path",
            "cache_size_mb",
            "block_cache_block_size_mb",
            "block_cache_disk_timeout_sec",
            "file_cache_timeout_sec",
            "file_cache_max_size_mb",
            "attr_cache_timeout_sec",
            "entry_cache_timeout_sec",
            "negative_entry_cache_timeout_sec",
        ],
    ),
    ("mountpoint", &["type", "options"]),
    (
        "rclone",
        &[
            "type",
            "mode",
            "remote_name",
            "extra_args",
            "nfs_addr",
            "nfs_mount_options",
            "config_file_path",
        ],
    ),
    ("s3files", &["type", "options"]),
];
const OPTIONS_FIELDS: &[(&str, &[&str])] = &[
    ("mountpoint", &["prefix", "region", "endpoint_url"]),
    (
        "s3files",
        &["mount_target_ip", "access_point", "region", "extra_options"],
    ),
];

const RCLONE_SAFE_FLAG_ARGS: &[&str] = &["allow-other"];
const RCLONE_SAFE_VALUE_ARGS: &[&str] = &["buffer-size", "gid", "uid"];

fn lookup<'a, T: ?Sized>(table: &'a [(&str, &'a T)], key: &str) -> Option<&'a T> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn invalid_manifest() -> AgentsError {
    AgentsError::user(INVALID_MANIFEST)
}

fn invalid_envelope() -> AgentsError {
    AgentsError::user(INVALID_ENVELOPE)
}

/// Whether `value` is a URL carrying inline authority: userinfo, a query string, or a value
/// that is not a string at all. Port of `_url_contains_inline_authority`.
fn url_contains_inline_authority(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(s) => {
            if s.contains('@') {
                return true;
            }
            // `urlsplit`: the query is whatever follows the first `?` before a fragment.
            let before_fragment = s.split('#').next().unwrap_or_default();
            before_fragment.contains('?')
        }
        _ => true,
    }
}

fn contains_line_break(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.contains('\r') || s.contains('\n'))
}

fn contains_s3fs_delimiter(value: &Value) -> bool {
    value.as_str().is_some_and(|s| s.contains(','))
}

fn rclone_remote_name_is_safe(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) if s.is_empty() => true,
        Value::String(s) => {
            let mut chars = s.chars();
            let first_ok = chars
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
            s == s.trim()
                && first_ok
                && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ' ' | '-'))
        }
        _ => false,
    }
}

/// `(safe, config paths, invalid config path)`. Port of `_rclone_extra_args_analysis`.
fn rclone_extra_args_analysis(args: &[Value]) -> (bool, Vec<String>, bool) {
    let mut config_paths = Vec::new();
    let mut safe = true;
    let mut invalid = false;
    let mut index = 0;
    while index < args.len() {
        let Some(arg) = args[index].as_str().filter(|a| a.starts_with('-')) else {
            safe = false;
            index += 1;
            continue;
        };
        let stripped = arg.trim_start_matches('-');
        let (option, separator, value) = match stripped.split_once('=') {
            Some((o, v)) => (o, true, v),
            None => (stripped, false, ""),
        };
        let normalized = option.to_lowercase().replace('_', "-");
        let next = args.get(index + 1).and_then(Value::as_str);
        if normalized == "config" {
            safe = false;
            if separator && !value.is_empty() {
                config_paths.push(value.to_string());
                index += 1;
            } else if !separator && index + 1 < args.len() {
                match next {
                    Some(n) if !n.starts_with('-') => config_paths.push(n.to_string()),
                    _ => invalid = true,
                }
                index += 2;
            } else {
                invalid = true;
                index += 1;
            }
            continue;
        }
        if RCLONE_SAFE_FLAG_ARGS.contains(&normalized.as_str()) && !separator {
            index += 1;
            continue;
        }
        if RCLONE_SAFE_VALUE_ARGS.contains(&normalized.as_str()) {
            if separator && !value.is_empty() {
                index += 1;
                continue;
            }
            if !separator && index + 1 < args.len() && next.is_some_and(|n| !n.starts_with('-')) {
                index += 2;
                continue;
            }
        }
        safe = false;
        index += 1;
    }
    (safe, config_paths, invalid)
}

/// Remove every key of `map` not in `safe`; whether any was removed.
fn strip_to(map: &mut Map<String, Value>, safe: &[&str]) -> bool {
    let extra: Vec<String> = map
        .keys()
        .filter(|k| !safe.contains(&k.as_str()))
        .cloned()
        .collect();
    for key in &extra {
        map.remove(key);
    }
    !extra.is_empty()
}

fn entry_type(entry: &Map<String, Value>) -> Option<&str> {
    entry.get("type").and_then(Value::as_str)
}

fn registered(entry_type: &str) -> Option<(bool, bool)> {
    // `(is_dir, is_mount)`
    if let Some((_, is_dir, _)) = ENTRY_TYPES.iter().find(|(t, _, _)| *t == entry_type) {
        return Some((*is_dir, false));
    }
    MOUNT_TYPES
        .iter()
        .any(|(t, _)| *t == entry_type)
        .then_some((false, true))
}

fn is_extension_mount(entry_type: &str) -> bool {
    EXTENSION_MOUNT_TYPES.iter().any(|(t, _)| *t == entry_type)
}

/// An entry whose `children` are not part of its own serialization: extension mounts and
/// registered non-directory classes.
fn is_leaf(entry_type: Option<&str>) -> bool {
    match entry_type {
        Some(t) => is_extension_mount(t) || registered(t).is_some_and(|(dir, _)| !dir),
        None => false,
    }
}

fn raw_entry_tree_is_valid(entries: &Value) -> bool {
    let Some(entries) = entries.as_object() else {
        return false;
    };
    entries.values().all(|entry| {
        let Some(map) = entry.as_object() else {
            return false;
        };
        let Some(t) = entry_type(map) else {
            return false;
        };
        if is_leaf(Some(t)) {
            return true;
        }
        map.get("children").is_none_or(raw_entry_tree_is_valid)
    })
}

fn join(parent: &str, name: &str) -> String {
    if name.starts_with('/') || parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

/// Port of `_absolute_manifest_path`: resolve against `root` and fold `.` and `..`.
fn absolute_manifest_path(root: &str, value: &str) -> Vec<String> {
    let full = if value.starts_with('/') {
        value.to_string()
    } else {
        format!("{root}/{value}")
    };
    let mut parts: Vec<String> = Vec::new();
    for part in full.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other.to_string()),
        }
    }
    parts
}

/// Sanitize one raw mount entry in place. Returns `(redacted, credential file paths)`.
fn sanitize_raw_mount(entry: &mut Map<String, Value>) -> Result<(bool, Vec<String>)> {
    let raw_type = entry_type(entry).map(str::to_string);
    let mount_type = raw_type.clone().unwrap_or_default();
    let mut redacted = false;
    let mut file_paths = Vec::new();

    let safe_fields: Option<Vec<&str>> =
        if let Some((_, fields)) = MOUNT_TYPES.iter().find(|(t, _)| *t == mount_type) {
            Some(
                MOUNT_BASE_FIELDS
                    .iter()
                    .chain(fields.iter())
                    .copied()
                    .collect(),
            )
        } else {
            lookup(EXTENSION_MOUNT_TYPES, &mount_type).map(|f| f.to_vec())
        };
    if let Some(safe) = safe_fields {
        redacted |= strip_to(entry, &safe);
    }

    let file_fields = lookup(AUTHORITY_FILE_FIELDS, &mount_type).unwrap_or(&[]);
    for name in lookup(AUTHORITY_FIELDS, &mount_type).unwrap_or(&[]) {
        let Some(value) = entry.get(*name).filter(|v| !v.is_null()).cloned() else {
            continue;
        };
        entry.insert((*name).to_string(), Value::Null);
        redacted = true;
        if file_fields.contains(name) {
            match value {
                Value::String(path) => file_paths.push(path),
                _ => return Err(invalid_manifest()),
            }
        }
    }
    for name in lookup(URL_FIELDS, &mount_type).unwrap_or(&[]) {
        if entry.get(*name).is_some_and(url_contains_inline_authority) {
            entry.insert((*name).to_string(), Value::Null);
            redacted = true;
        }
    }
    for name in lookup(RCLONE_CONFIG_VALUE_FIELDS, &mount_type).unwrap_or(&[]) {
        if entry.get(*name).is_some_and(contains_line_break) {
            entry.insert((*name).to_string(), Value::String(String::new()));
            redacted = true;
        }
    }
    if mount_type == "s3_files_mount" && entry.get("extra_options").is_some_and(is_truthy) {
        entry.insert("extra_options".into(), Value::Object(Map::new()));
        redacted = true;
    }

    // Snapshot what the strategy checks need from the entry before borrowing the strategy.
    let invalid_s3fs = lookup(BLAXEL_S3FS_OPTION_FIELDS, &mount_type)
        .unwrap_or(&[])
        .iter()
        .any(|name| entry.get(*name).is_some_and(contains_s3fs_delimiter));

    let Some(strategy) = entry.get_mut("mount_strategy") else {
        return Ok((redacted, file_paths));
    };
    let strategy = match strategy {
        Value::Object(map) => map,
        Value::Null => return Ok((redacted, file_paths)),
        _ => return Err(invalid_manifest()),
    };
    let raw_strategy_type = strategy
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_string);
    let strategy_type = match &raw_strategy_type {
        Some(t) => t.clone(),
        None => {
            strategy.insert("type".into(), Value::Null);
            redacted = true;
            String::new()
        }
    };
    if strategy_type == "blaxel_cloud_bucket" && invalid_s3fs {
        return Err(invalid_manifest());
    }
    let serialized = lookup(STRATEGY_FIELDS, &strategy_type);
    if raw_strategy_type.is_some() && serialized.is_none() {
        return Err(invalid_manifest());
    }
    let strip_unknown = serialized.is_none();
    if let Some(fields) = serialized {
        redacted |= strip_to(strategy, fields);
    }
    for name in lookup(OPAQUE_STRATEGY_FIELDS, &strategy_type).unwrap_or(&[]) {
        if let Some(value) = strategy.get(*name).filter(|v| is_truthy(v)) {
            let replacement = if value.is_object() {
                Value::Object(Map::new())
            } else {
                Value::Null
            };
            strategy.insert((*name).to_string(), replacement);
            redacted = true;
        }
    }

    let finish = |strategy: &mut Map<String, Value>, redacted: &mut bool| {
        if strip_unknown {
            *redacted |= strip_to(strategy, &["type"]);
        }
    };

    match strategy.get("pattern") {
        Some(Value::Object(_)) => {}
        Some(Value::Null) | None => {
            finish(strategy, &mut redacted);
            return Ok((redacted, file_paths));
        }
        Some(_) => return Err(invalid_manifest()),
    }
    let Some(Value::Object(pattern)) = strategy.get_mut("pattern") else {
        return Err(invalid_manifest());
    };
    let pattern_type = pattern
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_string);
    if pattern_type.is_none() {
        pattern.insert("type".into(), Value::Null);
        redacted = true;
    }
    // Runtime-only broad credential acknowledgement is rebound after serialization even when
    // there is no explicit credential field to remove.
    let broad_discovery = matches!(
        (
            mount_type.as_str(),
            strategy_type.as_str(),
            pattern_type.as_deref()
        ),
        ("azure_blob_mount", "in_container", Some("fuse"))
            | ("s3_files_mount", "in_container", Some("s3files"))
    );
    if broad_discovery {
        redacted = true;
    }
    let pattern_fields = match &pattern_type {
        Some(t) => match lookup(PATTERN_FIELDS, t) {
            Some(fields) => fields,
            None => return Err(invalid_manifest()),
        },
        None => &["type"][..],
    };
    if !pattern
        .get("remote_name")
        .is_none_or(rclone_remote_name_is_safe)
    {
        pattern.insert("remote_name".into(), Value::Null);
        redacted = true;
    }
    if let Some(config_path) = pattern
        .get("config_file_path")
        .filter(|v| !v.is_null())
        .cloned()
    {
        pattern.insert("config_file_path".into(), Value::Null);
        redacted = true;
        match config_path {
            Value::String(path) => file_paths.push(path),
            _ => return Err(invalid_manifest()),
        }
    }
    let extra_args = pattern
        .get("extra_args")
        .cloned()
        .unwrap_or(Value::Array(Vec::new()));
    let (mut safe_args, mut extra_config_paths, mut invalid_config) = (false, Vec::new(), false);
    if let Value::Array(args) = &extra_args {
        (safe_args, extra_config_paths, invalid_config) = rclone_extra_args_analysis(args);
    }
    if invalid_config {
        return Err(invalid_manifest());
    }
    if !safe_args {
        if is_truthy(&extra_args) {
            pattern.insert("extra_args".into(), Value::Array(Vec::new()));
            redacted = true;
        }
        file_paths.append(&mut extra_config_paths);
    }
    match pattern.get_mut("options") {
        Some(Value::Object(options)) => {
            if options
                .get("endpoint_url")
                .is_some_and(url_contains_inline_authority)
            {
                options.insert("endpoint_url".into(), Value::Null);
                redacted = true;
            }
            if options.get("extra_options").is_some_and(is_truthy)
                && (mount_type == "s3_files_mount"
                    || pattern_type.as_deref() == Some("s3files")
                    || pattern_type.is_none())
            {
                options.insert("extra_options".into(), Value::Object(Map::new()));
                redacted = true;
            }
            let option_fields = pattern_type
                .as_deref()
                .and_then(|t| lookup(OPTIONS_FIELDS, t))
                .unwrap_or(&[]);
            redacted |= strip_to(options, option_fields);
        }
        Some(Value::Null) | None => {}
        Some(_) => {
            pattern.insert("options".into(), Value::Object(Map::new()));
            redacted = true;
        }
    }
    redacted |= strip_to(pattern, pattern_fields);
    finish(strategy, &mut redacted);
    Ok((redacted, file_paths))
}

/// Python truthiness of a JSON value.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}

/// Sanitize every mount of the raw entry tree, collecting credential-file paths.
fn sanitize_entries(
    entries: &mut Map<String, Value>,
    redacted: &mut bool,
    file_paths: &mut Vec<String>,
) -> Result<()> {
    for entry in entries.values_mut() {
        let Value::Object(map) = entry else {
            continue;
        };
        let kind = entry_type(map).map(str::to_string);
        let is_mount_class = kind
            .as_deref()
            .is_some_and(|t| registered(t).is_some_and(|(_, mount)| mount));
        match kind.as_deref() {
            Some(t) if is_mount_class || is_extension_mount(t) => {
                let (r, paths) = sanitize_raw_mount(map)?;
                *redacted |= r;
                file_paths.extend(paths);
            }
            Some(t) if registered(t).is_some() => {}
            _ => {
                // An unregistered entry that carries a mount strategy is a mount we cannot
                // classify, which is never restored.
                if map.contains_key("mount_strategy") {
                    return Err(invalid_manifest());
                }
            }
        }
        if is_leaf(kind.as_deref()) {
            continue;
        }
        if let Some(Value::Object(children)) = map.get_mut("children") {
            sanitize_entries(children, redacted, file_paths)?;
        }
    }
    Ok(())
}

/// Blank inline credential-file content and refuse non-inline materializers of those paths.
fn sanitize_credential_file_sources(
    entries: &mut Map<String, Value>,
    parent: &str,
    root: &str,
    target: &[String],
    redacted: &mut bool,
) -> Result<()> {
    for (name, entry) in entries.iter_mut() {
        let Value::Object(map) = entry else {
            continue;
        };
        let path = join(parent, name);
        let entry_path = absolute_manifest_path(root, &path);
        let kind = entry_type(map).map(str::to_string);
        if entry_path == target && kind.as_deref() == Some("file") {
            map.insert("content".into(), Value::String(String::new()));
            *redacted = true;
        } else if entry_path == target
            || (entry_path.len() < target.len() && target.starts_with(&entry_path))
        {
            let structural = kind.as_deref() == Some("dir")
                || (kind.as_deref() == Some("local_dir")
                    && map.get("src").is_none_or(Value::is_null));
            if !structural {
                return Err(invalid_manifest());
            }
        }
        if is_leaf(kind.as_deref()) {
            continue;
        }
        if let Some(Value::Object(children)) = map.get_mut("children") {
            sanitize_credential_file_sources(children, &path, root, target, redacted)?;
        }
    }
    Ok(())
}

fn sanitize_raw_manifest(manifest: &mut Value) -> Result<bool> {
    let Value::Object(manifest) = manifest else {
        return Ok(false);
    };
    if manifest
        .get("entries")
        .is_some_and(|e| !raw_entry_tree_is_valid(e))
    {
        return Err(invalid_manifest());
    }
    let root = manifest
        .get("root")
        .and_then(Value::as_str)
        .unwrap_or("/workspace")
        .to_string();
    let mut redacted = false;
    let mut file_paths = Vec::new();
    if let Some(Value::Object(entries)) = manifest.get_mut("entries") {
        sanitize_entries(entries, &mut redacted, &mut file_paths)?;
    }
    let targets: BTreeSet<Vec<String>> = file_paths
        .iter()
        .map(|p| absolute_manifest_path(&root, p))
        .collect();
    if let Some(Value::Object(entries)) = manifest.get_mut("entries") {
        for target in targets {
            sanitize_credential_file_sources(entries, "", &root, &target, &mut redacted)?;
        }
    }
    Ok(redacted)
}

/// Sanitize one raw sandbox session state. Returns the sanitized state and whether anything
/// was removed.
fn sanitize_raw_session_state(state: Value) -> Result<(Value, bool)> {
    let Value::Object(mut state) = state else {
        return Ok((state, false));
    };
    if state.get("manifest").is_some_and(|m| !m.is_object()) {
        return Err(invalid_manifest());
    }
    state.remove(CREDENTIALLESS_MOUNT_AUTHORITY_KEY);
    let mut redacted = match state.get_mut("manifest") {
        Some(manifest) => sanitize_raw_manifest(manifest)?,
        None => false,
    };
    if redacted || state.get(REDACTED_MOUNT_AUTHORITY_KEY) == Some(&Value::Bool(true)) {
        state.insert(REDACTED_MOUNT_AUTHORITY_KEY.into(), Value::Bool(true));
        redacted = true;
    }
    Ok((Value::Object(state), redacted))
}

fn envelope_is_valid(payload: &Value) -> bool {
    let Some(map) = payload.as_object() else {
        return false;
    };
    if map.get("session_state").is_some_and(|s| !s.is_object()) {
        return false;
    }
    match map.get("sessions_by_agent") {
        None | Some(Value::Null) => true,
        Some(Value::Object(sessions)) => sessions.values().all(|entry| {
            entry
                .as_object()
                .is_some_and(|e| e.get("session_state").is_none_or(Value::is_object))
        }),
        Some(_) => false,
    }
}

/// Validate and sanitize the `sandbox` envelope of a run state.
///
/// Any failure reports a fixed message that names no payload value.
pub(super) fn sanitize_sandbox_payload(payload: Value) -> Result<Value> {
    if !envelope_is_valid(&payload) {
        return Err(invalid_envelope());
    }
    let Value::Object(mut sandbox) = payload else {
        return Err(invalid_envelope());
    };
    if let Some(session_state) = sandbox.remove("session_state") {
        sandbox.insert(
            "session_state".into(),
            sanitize_raw_session_state(session_state)?.0,
        );
    }
    if let Some(Value::Object(sessions)) = sandbox.remove("sessions_by_agent") {
        let mut sanitized = Map::new();
        for (key, entry) in sessions {
            let Value::Object(mut entry) = entry else {
                return Err(invalid_envelope());
            };
            if let Some(state) = entry.remove("session_state") {
                entry.insert("session_state".into(), sanitize_raw_session_state(state)?.0);
                sanitized.insert(key, Value::Object(entry));
            } else {
                // The session state is the entry itself.
                sanitized.insert(key, sanitize_raw_session_state(Value::Object(entry))?.0);
            }
        }
        sandbox.insert("sessions_by_agent".into(), Value::Object(sanitized));
    }
    Ok(Value::Object(sandbox))
}
