//! Descriptor-relative filesystem operations for the Unix-local backend.
//!
//! Validation resolves a path and decides whether it may be touched, but the operation that
//! follows names the path again, and a workspace process can swap a directory for a symlink in
//! between. These helpers close that window the way Python's `_FileOps` does: the directory that
//! holds the entry is opened once, component by component, with `O_NOFOLLOW`, and every later
//! step names the entry relative to that descriptor, so replacing any path component afterwards
//! cannot redirect the operation. They are blocking; callers run them on a blocking thread.

use std::ffi::{OsStr, OsString};
use std::io;
use std::os::fd::OwnedFd;
use std::path::{Component, Path};

use rustix::fs::{
    AtFlags, Dir, FileType, Mode, OFlags, Stat, mkdirat, openat, renameat, statat, unlinkat,
};
use rustix::io::Errno;

const DIR_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);

/// Open the directory at the absolute, fully resolved `path`, never following a link.
///
/// `path` must already be canonical, so a link met on the way means a component was swapped after
/// it was resolved and the walk fails instead of following it.
pub(super) fn open_dir_pinned(path: &Path) -> io::Result<OwnedFd> {
    let mut fd = rustix::fs::open("/", DIR_FLAGS, Mode::empty())?;
    for component in path.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(name) => fd = openat(&fd, name, DIR_FLAGS, Mode::empty())?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "a pinned path must be canonical",
                ));
            }
        }
    }
    Ok(fd)
}

/// The entry `name` under `dir`; a leaf link is described as itself unless `follow` is set.
pub(super) fn stat_at(dir: &OwnedFd, name: &OsStr, follow: bool) -> io::Result<Stat> {
    let flags = if follow {
        AtFlags::empty()
    } else {
        AtFlags::SYMLINK_NOFOLLOW
    };
    Ok(statat(dir, name, flags)?)
}

/// What kind of entry `stat` describes.
pub(super) fn kind_of(stat: &Stat) -> FileType {
    FileType::from_raw_mode(stat.st_mode)
}

/// Remove `name` under `dir`: a file or link is unlinked, a directory is removed with `rmdir`
/// (or with its contents when `recursive`). A leaf link is never followed.
pub(super) fn remove_at(dir: &OwnedFd, name: &OsStr, recursive: bool) -> io::Result<()> {
    let stat = stat_at(dir, name, false)?;
    if kind_of(&stat) != FileType::Directory {
        return Ok(unlinkat(dir, name, AtFlags::empty())?);
    }
    if recursive {
        let sub = openat(dir, name, DIR_FLAGS, Mode::empty())?;
        remove_tree(&sub)?;
    }
    Ok(unlinkat(dir, name, AtFlags::REMOVEDIR)?)
}

fn remove_tree(dir: &OwnedFd) -> io::Result<()> {
    for entry in Dir::read_from(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_bytes() == b"." || name.to_bytes() == b".." {
            continue;
        }
        let stat = statat(dir, name, AtFlags::SYMLINK_NOFOLLOW)?;
        if FileType::from_raw_mode(stat.st_mode) == FileType::Directory {
            let sub = openat(dir, name, DIR_FLAGS, Mode::empty())?;
            remove_tree(&sub)?;
            unlinkat(dir, name, AtFlags::REMOVEDIR)?;
        } else {
            unlinkat(dir, name, AtFlags::empty())?;
        }
    }
    Ok(())
}

/// Move `from` under `from_dir` to `to` under `to_dir` with `rename(2)`.
pub(super) fn rename_at(
    from_dir: &OwnedFd,
    from: &OsStr,
    to_dir: &OwnedFd,
    to: &OsStr,
) -> io::Result<()> {
    Ok(renameat(from_dir, from, to_dir, to)?)
}

/// Create the directories `rest` below `base`, one at a time, descending through each with
/// `O_NOFOLLOW` so a component swapped for a link is refused. With `parents` false only the last
/// component may be missing. The final component is accepted when it already is a directory and
/// is an error when it is anything else, a link included.
pub(super) fn mkdir_below(base: &OwnedFd, rest: &[OsString], parents: bool) -> io::Result<()> {
    let Some((last, intermediate)) = rest.split_last() else {
        return Ok(());
    };
    let mut dir = base.try_clone()?;
    for name in intermediate {
        match mkdirat(&dir, name.as_os_str(), Mode::from_raw_mode(0o777)) {
            Ok(()) => {}
            Err(Errno::EXIST) if parents => {}
            Err(error) => return Err(error.into()),
        }
        dir = openat(&dir, name.as_os_str(), DIR_FLAGS, Mode::empty())?;
    }
    match stat_at(&dir, last, false) {
        Ok(stat) if kind_of(&stat) == FileType::Directory => return Ok(()),
        _ => {}
    }
    Ok(mkdirat(&dir, last.as_os_str(), Mode::from_raw_mode(0o777))?)
}

/// The entries of the directory `dir`, each described as itself.
pub(super) fn list_dir(dir: &OwnedFd) -> io::Result<Vec<(OsString, Stat)>> {
    use std::os::unix::ffi::OsStrExt as _;
    let mut entries = Vec::new();
    for entry in Dir::read_from(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_bytes() == b"." || name.to_bytes() == b".." {
            continue;
        }
        // An entry removed between the read and the stat is simply not listed.
        if let Ok(stat) = statat(dir, name, AtFlags::SYMLINK_NOFOLLOW) {
            entries.push((OsStr::from_bytes(name.to_bytes()).to_os_string(), stat));
        }
    }
    Ok(entries)
}
