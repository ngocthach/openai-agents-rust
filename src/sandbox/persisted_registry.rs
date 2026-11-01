//! Typed parse and write entry points for persisted sandbox payloads.
//!
//! Python parses each released payload through a `type`-discriminated registry
//! (`BaseEntry.parse`, `MountPattern` validation, `MountStrategyBase.parse`, `EnvValue.parse`,
//! `SnapshotBase.parse`). Rust keeps those objects as trait objects and concrete structs, so each
//! registry here is a tagged enum or a pair of functions that reads and writes Python's `type`
//! discriminator and converts to the Rust object.
//!
//! A payload whose `type` names no registered kind is rejected, and no error repeats the payload,
//! because a payload can carry credentials. Writing an object is symmetrical: an object of a kind
//! that is not registered is refused instead of being written without its fields. Like Python's
//! `model_dump`, a written mount entry carries its credential fields, so these functions are for
//! an explicit serialization, never for persisted session state or `RunState`.
//!
//! Kinds registered here: manifest entries `file`, `dir`, `local_file`, `local_dir`, `git_repo`
//! and the mounts `s3_mount`, `r2_mount`, `gcs_mount`, `azure_blob_mount`, `box_mount`,
//! `s3_files_mount`; mount patterns `mountpoint`, `rclone`, `fuse`, `s3files`; core mount
//! strategies `bind`, `docker_volume`, `in_container`; env value `str`; snapshots `noop`, `local`,
//! `remote`. Provider mount strategies are registered by the extensions.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::sandbox::entries::{Entry, EntryKind};
use crate::sandbox::environment::{EnvValue, StrEnvValue};
use crate::sandbox::errors::{ErrorCode, SandboxError, SandboxResult};
use crate::sandbox::mount_entries::{
    AzureBlobMount, BoxMount, BuiltinMountPattern, ConfiguredMount, ConfiguredMountBackend,
    ConfiguredMountStrategy, GcsMount, R2Mount, S3FilesMount, S3Mount,
};
use crate::sandbox::mount_patterns::{
    FuseMountPattern, MountpointMountPattern, S3FilesMountPattern,
};
use crate::sandbox::mounts::{
    BindMountStrategy, DockerVolumeMountStrategy, InContainerMountStrategy, Mount, MountPattern,
    MountPatternConfig, MountStrategy, RcloneMountPattern,
};
use crate::sandbox::snapshot::SnapshotSpec;
use crate::sandbox::types::{Group, Permissions};

fn rejected(what: &str) -> SandboxError {
    SandboxError::new(
        ErrorCode::SessionStateInvalid,
        format!("{what} payload is not a known type"),
    )
}

fn unwritable(what: &str) -> SandboxError {
    SandboxError::new(
        ErrorCode::SessionStateInvalid,
        format!("{what} is not a registered kind and cannot be written"),
    )
}

// ---------------------------------------------------------------------------------------------
// Mount patterns
// ---------------------------------------------------------------------------------------------

/// A mount pattern payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MountPatternRecord {
    /// A mountpoint-s3 pattern.
    #[serde(rename = "mountpoint")]
    Mountpoint(MountpointMountPattern),
    /// An rclone pattern.
    #[serde(rename = "rclone")]
    Rclone(RcloneMountPattern),
    /// A blobfuse2 pattern.
    #[serde(rename = "fuse")]
    Fuse(FuseMountPattern),
    /// An S3 Files pattern.
    #[serde(rename = "s3files")]
    S3Files(S3FilesMountPattern),
}

impl MountPatternRecord {
    /// The record for a pattern object, when it is a registered kind.
    pub fn from_pattern(pattern: &dyn MountPattern) -> SandboxResult<Self> {
        let any = pattern
            .as_any()
            .ok_or_else(|| unwritable("mount pattern"))?;
        if let Some(p) = any.downcast_ref::<MountpointMountPattern>() {
            return Ok(MountPatternRecord::Mountpoint(p.clone()));
        }
        if let Some(p) = any.downcast_ref::<RcloneMountPattern>() {
            return Ok(MountPatternRecord::Rclone(p.clone()));
        }
        if let Some(p) = any.downcast_ref::<FuseMountPattern>() {
            return Ok(MountPatternRecord::Fuse(p.clone()));
        }
        if let Some(p) = any.downcast_ref::<S3FilesMountPattern>() {
            return Ok(MountPatternRecord::S3Files(p.clone()));
        }
        Err(unwritable("mount pattern"))
    }

    /// The pattern object this record describes.
    pub fn into_pattern(self) -> Arc<dyn MountPattern> {
        match self {
            MountPatternRecord::Mountpoint(p) => Arc::new(p),
            MountPatternRecord::Rclone(p) => Arc::new(p),
            MountPatternRecord::Fuse(p) => Arc::new(p),
            MountPatternRecord::S3Files(p) => Arc::new(p),
        }
    }

    fn into_builtin(self) -> BuiltinMountPattern {
        match self {
            MountPatternRecord::Mountpoint(p) => BuiltinMountPattern::Mountpoint(p),
            MountPatternRecord::Rclone(p) => BuiltinMountPattern::Rclone(p),
            MountPatternRecord::Fuse(p) => BuiltinMountPattern::Fuse(p),
            MountPatternRecord::S3Files(p) => BuiltinMountPattern::S3Files(p),
        }
    }

    fn from_builtin(pattern: &BuiltinMountPattern) -> Self {
        match pattern {
            BuiltinMountPattern::Mountpoint(p) => MountPatternRecord::Mountpoint(p.clone()),
            BuiltinMountPattern::Rclone(p) => MountPatternRecord::Rclone(p.clone()),
            BuiltinMountPattern::Fuse(p) => MountPatternRecord::Fuse(p.clone()),
            BuiltinMountPattern::S3Files(p) => MountPatternRecord::S3Files(p.clone()),
        }
    }
}

/// Parse a mount pattern payload.
pub fn parse_mount_pattern(payload: &Value) -> SandboxResult<Arc<dyn MountPattern>> {
    let record: MountPatternRecord =
        serde_json::from_value(payload.clone()).map_err(|_| rejected("mount pattern"))?;
    Ok(record.into_pattern())
}

/// Write a mount pattern as its payload.
pub fn mount_pattern_to_value(pattern: &dyn MountPattern) -> SandboxResult<Value> {
    let record = MountPatternRecord::from_pattern(pattern)?;
    serde_json::to_value(record).map_err(|_| unwritable("mount pattern"))
}

// ---------------------------------------------------------------------------------------------
// Mount strategies
// ---------------------------------------------------------------------------------------------

/// A mount strategy payload for the core strategies. The provider strategies are registered by
/// `ProviderMountStrategy` in the extensions.
///
/// An `in_container` payload carries the pattern only: the runtime configuration holds the
/// credentials, and Python does not persist it either.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CoreMountStrategyRecord {
    /// A host directory bound into the workspace.
    #[serde(rename = "bind")]
    Bind(BindMountStrategy),
    /// A Docker named-volume strategy.
    #[serde(rename = "docker_volume")]
    DockerVolume(DockerVolumeMountStrategy),
    /// A mount tool run inside the sandbox.
    #[serde(rename = "in_container")]
    InContainer {
        /// The command pattern that attaches the storage.
        pattern: MountPatternRecord,
    },
}

impl CoreMountStrategyRecord {
    fn into_strategy(self) -> Arc<dyn MountStrategy> {
        match self {
            CoreMountStrategyRecord::Bind(s) => Arc::new(s),
            CoreMountStrategyRecord::DockerVolume(s) => Arc::new(s),
            CoreMountStrategyRecord::InContainer { pattern } => {
                Arc::new(InContainerMountStrategy::new(
                    pattern.into_pattern(),
                    MountPatternConfig::default(),
                ))
            }
        }
    }
}

/// Parse a core mount strategy payload.
pub fn parse_core_mount_strategy(payload: &Value) -> SandboxResult<Arc<dyn MountStrategy>> {
    let record: CoreMountStrategyRecord =
        serde_json::from_value(payload.clone()).map_err(|_| rejected("mount strategy"))?;
    Ok(record.into_strategy())
}

/// Write a core mount strategy as its payload. A configured mount's strategy is written the same
/// way, as the `in_container` or `docker_volume` it attaches with.
pub fn core_mount_strategy_to_value(strategy: &dyn MountStrategy) -> SandboxResult<Value> {
    let any = strategy
        .as_any()
        .ok_or_else(|| unwritable("mount strategy"))?;
    let record = if let Some(s) = any.downcast_ref::<BindMountStrategy>() {
        CoreMountStrategyRecord::Bind(s.clone())
    } else if let Some(s) = any.downcast_ref::<DockerVolumeMountStrategy>() {
        CoreMountStrategyRecord::DockerVolume(s.clone())
    } else if let Some(s) = any.downcast_ref::<InContainerMountStrategy>() {
        CoreMountStrategyRecord::InContainer {
            pattern: MountPatternRecord::from_pattern(s.pattern.as_ref())?,
        }
    } else if let Some(s) = any.downcast_ref::<ConfiguredMountStrategy>() {
        backend_record(&s.backend)
    } else {
        return Err(unwritable("mount strategy"));
    };
    serde_json::to_value(record).map_err(|_| unwritable("mount strategy"))
}

fn backend_record(backend: &ConfiguredMountBackend) -> CoreMountStrategyRecord {
    match backend {
        ConfiguredMountBackend::InContainer(pattern) => CoreMountStrategyRecord::InContainer {
            pattern: MountPatternRecord::from_builtin(pattern),
        },
        ConfiguredMountBackend::DockerVolume {
            driver,
            driver_options,
        } => CoreMountStrategyRecord::DockerVolume(DockerVolumeMountStrategy {
            driver: driver.clone(),
            driver_options: driver_options.clone(),
        }),
    }
}

fn parse_backend(payload: &Value) -> SandboxResult<ConfiguredMountBackend> {
    let record: CoreMountStrategyRecord =
        serde_json::from_value(payload.clone()).map_err(|_| rejected("mount strategy"))?;
    match record {
        CoreMountStrategyRecord::InContainer { pattern } => {
            Ok(ConfiguredMountBackend::InContainer(pattern.into_builtin()))
        }
        CoreMountStrategyRecord::DockerVolume(s) => Ok(ConfiguredMountBackend::DockerVolume {
            driver: s.driver,
            driver_options: s.driver_options,
        }),
        CoreMountStrategyRecord::Bind(_) => Err(rejected("mount strategy")),
    }
}

// ---------------------------------------------------------------------------------------------
// Environment values and snapshots
// ---------------------------------------------------------------------------------------------

/// An environment value payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EnvValueRecord {
    /// A literal value.
    #[serde(rename = "str")]
    Str(StrEnvValue),
}

/// Parse an environment value payload.
pub fn parse_env_value(payload: &Value) -> SandboxResult<Arc<dyn EnvValue>> {
    let record: EnvValueRecord =
        serde_json::from_value(payload.clone()).map_err(|_| rejected("environment value"))?;
    Ok(match record {
        EnvValueRecord::Str(value) => Arc::new(value),
    })
}

/// Write an environment value as its payload.
pub fn env_value_to_value(value: &dyn EnvValue) -> SandboxResult<Value> {
    let any = value
        .as_any()
        .ok_or_else(|| unwritable("environment value"))?;
    let record = any
        .downcast_ref::<StrEnvValue>()
        .map(|v| EnvValueRecord::Str(v.clone()))
        .ok_or_else(|| unwritable("environment value"))?;
    serde_json::to_value(record).map_err(|_| unwritable("environment value"))
}

/// Parse a snapshot payload (`noop`, `local` or `remote`).
pub fn parse_snapshot(payload: &Value) -> SandboxResult<SnapshotSpec> {
    serde_json::from_value(payload.clone()).map_err(|_| rejected("snapshot"))
}

/// Write a snapshot as its payload.
pub fn snapshot_to_value(snapshot: &SnapshotSpec) -> SandboxResult<Value> {
    serde_json::to_value(snapshot).map_err(|_| unwritable("snapshot"))
}

// ---------------------------------------------------------------------------------------------
// Manifest entries
// ---------------------------------------------------------------------------------------------

use crate::sandbox::mount_patterns::{
    MOUNT_TYPE_AZURE_BLOB, MOUNT_TYPE_BOX, MOUNT_TYPE_GCS, MOUNT_TYPE_R2, MOUNT_TYPE_S3,
    MOUNT_TYPE_S3_FILES,
};

fn text_field(map: &Map<String, Value>, key: &str, what: &str) -> SandboxResult<String> {
    map.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| rejected(what))
}

fn optional_text(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key).and_then(Value::as_str).map(str::to_string)
}

fn path_text(path: &std::path::Path) -> SandboxResult<String> {
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| unwritable("entry path"))
}

/// Parse a manifest entry payload, including a directory's children.
pub fn parse_entry(payload: &Value) -> SandboxResult<Entry> {
    let map = payload.as_object().ok_or_else(|| rejected("entry"))?;
    let kind = map
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| rejected("entry"))?;
    let mut entry = match kind {
        "file" => Entry::file(text_field(map, "content", "entry")?.into_bytes()),
        "dir" => {
            let mut entry = Entry::dir();
            if let Some(children) = map.get("children") {
                let children = children.as_object().ok_or_else(|| rejected("entry"))?;
                for (name, child) in children {
                    entry = entry.with_child(name.clone(), parse_entry(child)?);
                }
            }
            entry
        }
        "local_file" => Entry::local_file(text_field(map, "src", "entry")?),
        "local_dir" => {
            let mut entry = Entry::dir();
            entry.kind = EntryKind::LocalDir {
                src: optional_text(map, "src").map(PathBuf::from),
            };
            entry
        }
        "git_repo" => {
            let mut entry = Entry::git_repo(
                text_field(map, "repo", "entry")?,
                text_field(map, "ref", "entry")?,
            );
            if let Some(host) = optional_text(map, "host") {
                entry = entry.with_git_host(host);
            }
            if let Some(subpath) = optional_text(map, "subpath") {
                entry = entry.with_git_subpath(subpath);
            }
            entry
        }
        MOUNT_TYPE_S3 => mount_entry::<S3Mount>(payload, map)?,
        MOUNT_TYPE_R2 => mount_entry::<R2Mount>(payload, map)?,
        MOUNT_TYPE_GCS => mount_entry::<GcsMount>(payload, map)?,
        MOUNT_TYPE_AZURE_BLOB => mount_entry::<AzureBlobMount>(payload, map)?,
        MOUNT_TYPE_BOX => mount_entry::<BoxMount>(payload, map)?,
        MOUNT_TYPE_S3_FILES => mount_entry::<S3FilesMount>(payload, map)?,
        _ => return Err(rejected("entry")),
    };
    if let Some(description) = optional_text(map, "description") {
        entry = entry.with_description(description);
    }
    if let Some(permissions) = map.get("permissions").filter(|v| !v.is_null()) {
        let permissions: Permissions =
            serde_json::from_value(permissions.clone()).map_err(|_| rejected("entry"))?;
        entry = entry.with_permissions(permissions);
    }
    if let Some(group) = map.get("group").filter(|v| !v.is_null()) {
        let group: Group = serde_json::from_value(group.clone()).map_err(|_| rejected("entry"))?;
        entry = entry.with_group(group);
    }
    if let Some(ephemeral) = map.get("ephemeral").and_then(Value::as_bool) {
        entry = entry.with_ephemeral(ephemeral);
    }
    Ok(entry)
}

/// A configured mount type that can be read from a payload.
trait PersistedMount: ConfiguredMount + serde::de::DeserializeOwned + 'static {}
impl<T: ConfiguredMount + serde::de::DeserializeOwned + 'static> PersistedMount for T {}

fn mount_entry<T: PersistedMount>(
    payload: &Value,
    map: &Map<String, Value>,
) -> SandboxResult<Entry> {
    let spec: T = serde_json::from_value(payload.clone()).map_err(|_| rejected("mount entry"))?;
    let backend = parse_backend(
        map.get("mount_strategy")
            .ok_or_else(|| rejected("mount entry"))?,
    )?;
    let ephemeral = map
        .get("ephemeral")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let mount = Mount {
        mount_path: optional_text(map, "mount_path"),
        read_only: map
            .get("read_only")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        ephemeral,
        strategy: Arc::new(ConfiguredMountStrategy::new(Arc::new(spec), backend)),
    };
    Ok(Entry::mount(mount))
}

fn mount_spec_value(spec: &dyn ConfiguredMount) -> SandboxResult<Value> {
    let any = spec.as_any();
    let value = if let Some(m) = any.downcast_ref::<S3Mount>() {
        serde_json::to_value(m)
    } else if let Some(m) = any.downcast_ref::<R2Mount>() {
        serde_json::to_value(m)
    } else if let Some(m) = any.downcast_ref::<GcsMount>() {
        serde_json::to_value(m)
    } else if let Some(m) = any.downcast_ref::<AzureBlobMount>() {
        serde_json::to_value(m)
    } else if let Some(m) = any.downcast_ref::<BoxMount>() {
        serde_json::to_value(m)
    } else if let Some(m) = any.downcast_ref::<S3FilesMount>() {
        serde_json::to_value(m)
    } else {
        return Err(unwritable("mount entry"));
    };
    value.map_err(|_| unwritable("mount entry"))
}

/// Write a manifest entry as its payload, including a directory's children.
pub fn entry_to_value(entry: &Entry) -> SandboxResult<Value> {
    let mut map = Map::new();
    let mut fields = Map::new();
    let type_name: String = match &entry.kind {
        EntryKind::File { content } => {
            let text = String::from_utf8(content.clone()).map_err(|_| unwritable("file entry"))?;
            fields.insert("content".into(), Value::String(text));
            "file".into()
        }
        EntryKind::Dir { children } => {
            let mut written = Map::new();
            for (name, child) in children {
                written.insert(name.clone(), entry_to_value(child)?);
            }
            fields.insert("children".into(), Value::Object(written));
            "dir".into()
        }
        EntryKind::LocalFile { src } => {
            fields.insert("src".into(), Value::String(path_text(src)?));
            "local_file".into()
        }
        EntryKind::LocalDir { src } => {
            let src = match src {
                Some(src) => Value::String(path_text(src)?),
                None => Value::Null,
            };
            fields.insert("src".into(), src);
            "local_dir".into()
        }
        EntryKind::GitRepo {
            host,
            repo,
            git_ref,
            subpath,
        } => {
            fields.insert("host".into(), json!(host));
            fields.insert("repo".into(), json!(repo));
            fields.insert("ref".into(), json!(git_ref));
            fields.insert("subpath".into(), json!(subpath));
            "git_repo".into()
        }
        EntryKind::Mount { mount } => {
            let configured = mount
                .strategy
                .as_any()
                .and_then(|any| any.downcast_ref::<ConfiguredMountStrategy>())
                .ok_or_else(|| unwritable("mount entry"))?;
            let Value::Object(spec) = mount_spec_value(configured.spec.as_ref())? else {
                return Err(unwritable("mount entry"));
            };
            fields.extend(spec);
            fields.insert("mount_path".into(), json!(mount.mount_path));
            fields.insert("read_only".into(), json!(mount.read_only));
            fields.insert(
                "mount_strategy".into(),
                core_mount_strategy_to_value(configured)?,
            );
            configured.spec.mount_type().into()
        }
    };
    map.insert("type".into(), Value::String(type_name));
    map.insert("description".into(), json!(entry.description));
    map.insert("ephemeral".into(), json!(entry.ephemeral));
    map.insert(
        "group".into(),
        serde_json::to_value(&entry.group).map_err(|_| unwritable("entry"))?,
    );
    map.insert("is_dir".into(), json!(entry.is_dir()));
    map.insert(
        "permissions".into(),
        serde_json::to_value(entry.permissions).map_err(|_| unwritable("entry"))?,
    );
    map.extend(fields);
    Ok(Value::Object(map))
}

// ---------------------------------------------------------------------------------------------
// Manifests
// ---------------------------------------------------------------------------------------------

use crate::sandbox::environment::EnvEntry;
use crate::sandbox::manifest::Manifest;
use crate::sandbox::types::User;
use crate::sandbox::workspace_paths::SandboxPathGrant;

/// Manifest keys that would carry the trusted in-container credential acknowledgements. They
/// are application policy, so no payload may set them.
const CREDENTIAL_EXPOSURE_POLICY_KEYS: &[&str] = &[
    "in_container_mount_credential_exposure_allowed_paths",
    "_in_container_mount_credential_exposure_allowed_paths",
    "inContainerMountCredentialExposureAllowedPaths",
    "_inContainerMountCredentialExposureAllowedPaths",
    "in_container_mount_credential_exposure_acknowledged_paths",
    "_in_container_mount_credential_exposure_acknowledged_paths",
    "inContainerMountCredentialExposureAcknowledgedPaths",
    "_inContainerMountCredentialExposureAcknowledgedPaths",
    "in_container_mount_broad_credential_exposure_acknowledged_paths",
    "_in_container_mount_broad_credential_exposure_acknowledged_paths",
    "inContainerMountBroadCredentialExposureAcknowledgedPaths",
    "_inContainerMountBroadCredentialExposureAcknowledgedPaths",
    "mount_credential_exposure_policy",
    "_mount_credential_exposure_policy",
];

fn invalid_manifest(message: impl Into<String>) -> SandboxError {
    SandboxError::new(ErrorCode::SandboxConfigInvalid, message)
}

/// Parse one environment value mapping the way Python's `EnvValue.parse` does: a mapping whose
/// only key is a string `value` is the legacy form of a literal, anything else must name a
/// registered `type`. Only the literal `str` type is registered.
fn parse_env_value_mapping(payload: &Value) -> SandboxResult<Arc<dyn EnvValue>> {
    let Some(map) = payload.as_object() else {
        return Err(invalid_manifest("env value must be an EnvValue or mapping"));
    };
    if map.len() == 1
        && let Some(Value::String(value)) = map.get("value")
    {
        return Ok(Arc::new(StrEnvValue::new(value.clone())));
    }
    let Some(kind) = map.get("type").and_then(Value::as_str) else {
        return Err(invalid_manifest(
            "env value mapping must include a string `type` field",
        ));
    };
    match kind {
        "str" => parse_env_value(payload).map_err(|_| invalid_manifest("env value is invalid")),
        other => Err(invalid_manifest(format!(
            "Unknown env value type `{other}`. Registered types: str"
        ))),
    }
}

fn parse_environment(payload: &Value, manifest: Manifest) -> SandboxResult<Manifest> {
    let Some(map) = payload.as_object() else {
        return Err(invalid_manifest("environment must be a mapping"));
    };
    let Some(members) = map.get("value") else {
        return Ok(manifest);
    };
    let Some(members) = members.as_object() else {
        return Err(invalid_manifest(
            "Environment mapping must be a mapping, got a non-mapping value",
        ));
    };
    let mut manifest = manifest;
    for (key, member) in members {
        match member {
            Value::String(text) => manifest = manifest.with_env(key.clone(), text.clone()),
            Value::Object(fields)
                if fields.contains_key("type")
                    || matches!(fields.get("value"), Some(Value::String(_))) =>
            {
                let value = parse_env_value_mapping(member)?;
                manifest = manifest.with_env_entry(key.clone(), EnvEntry::new(value));
            }
            Value::Object(fields) => {
                let value = fields
                    .get("value")
                    .ok_or_else(|| invalid_manifest("environment entry is missing its `value`"))?;
                let mut entry = EnvEntry::new(parse_env_value_mapping(value)?);
                if let Some(description) = fields.get("description").and_then(Value::as_str) {
                    entry = entry.with_description(description);
                }
                if let Some(ephemeral) = fields.get("ephemeral").and_then(Value::as_bool) {
                    entry = entry.ephemeral(ephemeral);
                }
                manifest = manifest.with_env_entry(key.clone(), entry);
            }
            _ => {
                return Err(invalid_manifest(
                    "environment value must be a str, EnvValue, or EnvEntry",
                ));
            }
        }
    }
    Ok(manifest)
}

fn parse_path_grant(payload: &Value) -> SandboxResult<SandboxPathGrant> {
    let map = payload
        .as_object()
        .ok_or_else(|| invalid_manifest("extra_path_grants entries must be mappings"))?;
    let path = map
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_manifest("sandbox path grant path must be a string"))?;
    let mut grant = SandboxPathGrant::new(path)?;
    if let Some(read_only) = map.get("read_only").and_then(Value::as_bool) {
        grant = grant.read_only(read_only);
    }
    if let Some(description) = map.get("description").and_then(Value::as_str) {
        grant = grant.with_description(description);
    }
    if let Some(host_path) = map.get("host_path").and_then(Value::as_str) {
        grant = grant.with_host_path(host_path)?;
    }
    Ok(grant)
}

/// Parse a manifest payload, like Python's `Manifest.model_validate`.
///
/// This is for a payload the application wrote itself: entries may name host files and the
/// payload may carry `extra_path_grants`. A payload that tries to set the in-container mount
/// credential acknowledgements is refused, because those are trusted application policy that
/// is never serialized. Configuration that arrives from outside the program goes through
/// [`parse_untrusted_manifest`] instead.
pub fn parse_manifest(payload: &Value) -> SandboxResult<Manifest> {
    let map = payload
        .as_object()
        .ok_or_else(|| invalid_manifest("manifest must be a mapping"))?;
    if map
        .keys()
        .any(|key| CREDENTIAL_EXPOSURE_POLICY_KEYS.contains(&key.as_str()))
    {
        return Err(invalid_manifest(
            "In-container mount credential exposure must be configured on a trusted \
             Manifest instance, not in manifest input.",
        ));
    }
    if let Some(version) = map.get("version")
        && version.as_u64() != Some(1)
    {
        return Err(invalid_manifest("manifest version must be 1"));
    }
    let mut manifest = Manifest::new();
    if let Some(root) = map.get("root") {
        manifest = manifest.with_root(
            root.as_str()
                .ok_or_else(|| invalid_manifest("manifest root must be a string"))?,
        );
    }
    match map.get("entries") {
        None | Some(Value::Null) => {}
        Some(Value::Object(entries)) => {
            for (path, entry) in entries {
                manifest = manifest.with_entry(path.clone(), parse_entry(entry)?);
            }
        }
        Some(_) => return Err(invalid_manifest("manifest entries must be a mapping")),
    }
    if let Some(environment) = map.get("environment") {
        manifest = parse_environment(environment, manifest)?;
    }
    if let Some(users) = map.get("users").filter(|v| !v.is_null()) {
        let users: Vec<User> = serde_json::from_value(users.clone())
            .map_err(|_| invalid_manifest("manifest users are invalid"))?;
        for user in users {
            manifest = manifest.with_user(user);
        }
    }
    if let Some(groups) = map.get("groups").filter(|v| !v.is_null()) {
        let groups: Vec<Group> = serde_json::from_value(groups.clone())
            .map_err(|_| invalid_manifest("manifest groups are invalid"))?;
        for group in groups {
            manifest = manifest.with_group(group);
        }
    }
    if let Some(grants) = map.get("extra_path_grants").filter(|v| !v.is_null()) {
        let grants = grants
            .as_array()
            .ok_or_else(|| invalid_manifest("extra_path_grants must be a list"))?;
        for grant in grants {
            manifest = manifest.with_extra_path_grant(parse_path_grant(grant)?);
        }
    }
    if let Some(allowlist) = map
        .get("remote_mount_command_allowlist")
        .filter(|v| !v.is_null())
    {
        let commands: Vec<String> = serde_json::from_value(allowlist.clone()).map_err(|_| {
            invalid_manifest("remote_mount_command_allowlist must be a list of strings")
        })?;
        manifest = manifest.with_remote_mount_command_allowlist(commands);
    }
    Ok(manifest)
}

/// Parse a manifest from configuration that is not trusted application code, like Python's
/// `_coerce_manifest` for a dictionary.
///
/// Everything [`parse_manifest`] accepts is accepted, except what would widen the sandbox's
/// reach on the host: a non-empty `extra_path_grants`, a `local_file` entry, or a `local_dir`
/// entry with a `src`, at any depth. Those must be set on a trusted [`Manifest`] value.
/// `parameter_name` names the setting in the error message.
pub fn parse_untrusted_manifest(payload: &Value, parameter_name: &str) -> SandboxResult<Manifest> {
    if let Some(grants) = payload.get("extra_path_grants")
        && grants.as_array().is_none_or(|grants| !grants.is_empty())
    {
        return Err(invalid_manifest(format!(
            "{parameter_name}.extra_path_grants must be configured on a trusted Manifest \
             instance, not in a dictionary"
        )));
    }
    let manifest = parse_manifest(payload)?;
    fn has_host_source(entry: &Entry) -> bool {
        match &entry.kind {
            EntryKind::LocalFile { .. } => true,
            EntryKind::LocalDir { src } => src.is_some(),
            EntryKind::Dir { children } => children.values().any(has_host_source),
            _ => false,
        }
    }
    if manifest.entries().values().any(has_host_source) {
        return Err(invalid_manifest(format!(
            "{parameter_name} local host sources must be configured on a trusted Manifest \
             instance, not in a dictionary"
        )));
    }
    Ok(manifest)
}

/// Write a manifest as Python's manifest payload.
///
/// Like Python's `model_dump`, this is an explicit serialization: it carries literal environment
/// values and mount credential fields, so it is not for persisted session state. The trusted
/// credential acknowledgements and the materialization concurrency limits are runtime-only and
/// are not written.
///
/// Literal environment variables are written as plain strings and start-up lookups as the long
/// `{description, ephemeral, value}` form, so a lookup written here reads back with the same
/// value type, description and ephemeral flag. A value whose kind is not registered is refused
/// rather than written without its fields, which keeps a secret reference from being flattened.
pub fn manifest_to_value(manifest: &Manifest) -> SandboxResult<Value> {
    let mut entries = Map::new();
    for (path, entry) in manifest.entries_in_order() {
        entries.insert(path.clone(), entry_to_value(entry)?);
    }
    let mut members = Map::new();
    for (key, value) in manifest.environment() {
        members.insert(key.clone(), Value::String(value.clone()));
    }
    for (key, entry) in manifest.resolved_environment().entries() {
        members.insert(
            key.clone(),
            json!({
                "description": entry.description,
                "ephemeral": entry.ephemeral,
                "value": env_value_to_value(entry.value.as_ref())?,
            }),
        );
    }
    let grants: Vec<Value> = manifest
        .extra_path_grants()
        .iter()
        .map(|grant| {
            let mut written = Map::new();
            written.insert("path".into(), json!(grant.path));
            written.insert("read_only".into(), json!(grant.read_only));
            written.insert("description".into(), json!(grant.description));
            if let Some(host_path) = &grant.host_path {
                written.insert("host_path".into(), json!(host_path));
            }
            Value::Object(written)
        })
        .collect();
    Ok(json!({
        "version": manifest.version(),
        "root": manifest.root(),
        "entries": entries,
        "environment": { "value": members },
        "users": manifest.users(),
        "groups": manifest.groups(),
        "extra_path_grants": grants,
        "remote_mount_command_allowlist": manifest.remote_mount_command_allowlist(),
    }))
}
