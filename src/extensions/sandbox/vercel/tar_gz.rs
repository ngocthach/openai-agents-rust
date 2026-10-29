//! A minimal gzipped-`tar` writer for Vercel's `fs/write` endpoint.
//!
//! The endpoint accepts only `application/gzip` bodies holding a `tar` archive whose entry names
//! are relative to the request's `x-cwd` header. The crate's `sandbox` feature pulls in no
//! compression or archive dependency, so this module writes both layers directly:
//!
//! - **tar**: `ustar` headers, 512-byte blocks and the two zero blocks that end an archive.
//! - **gzip**: the 10-byte header, DEFLATE *stored* blocks (`BTYPE=00`, which copy the payload
//!   verbatim), the CRC-32 of the payload and its length. A stored stream is a valid gzip
//!   stream that every `gunzip` and `tar -xz` accepts; it simply does not shrink the payload.
//!
//! The uploads this produces are single small files — a workspace entry or a mount
//! configuration — so skipping compression costs nothing. A whole-workspace archive is produced
//! by `tar` *inside* the sandbox instead, and is already compressed.

/// Builds a gzipped `tar` archive in memory.
#[derive(Debug, Default)]
pub struct TarGzBuilder {
    tar: Vec<u8>,
}

impl TarGzBuilder {
    /// An empty archive.
    pub fn new() -> Self {
        TarGzBuilder::default()
    }

    /// Append a regular file.
    ///
    /// `name` is the entry name as it appears in the archive, so it must be relative to the
    /// directory the archive is extracted into.
    pub fn add_file(&mut self, name: &str, contents: &[u8], mode: u32) {
        self.tar
            .extend_from_slice(&ustar_header(name, contents.len() as u64, mode, b'0'));
        self.tar.extend_from_slice(contents);
        let padding = (512 - contents.len() % 512) % 512;
        self.tar.extend(std::iter::repeat_n(0u8, padding));
    }

    /// Append a directory entry.
    pub fn add_dir(&mut self, name: &str, mode: u32) {
        let name = if name.ends_with('/') {
            name.to_string()
        } else {
            format!("{name}/")
        };
        self.tar
            .extend_from_slice(&ustar_header(&name, 0, mode, b'5'));
    }

    /// Close the archive and return the gzipped bytes.
    pub fn finish(mut self) -> Vec<u8> {
        // Two zero blocks mark the end of a tar archive.
        self.tar.extend(std::iter::repeat_n(0u8, 1024));
        gzip(&self.tar)
    }
}

/// Write one `ustar` header block.
///
/// `type_flag` is `b'0'` for a regular file and `b'5'` for a directory.
fn ustar_header(name: &str, size: u64, mode: u32, type_flag: u8) -> [u8; 512] {
    let mut header = [0u8; 512];
    let (prefix, name) = split_name(name);
    write_bytes(&mut header[0..100], name.as_bytes());
    write_octal(&mut header[100..108], u64::from(mode & 0o7777), 7);
    write_octal(&mut header[108..116], 0, 7);
    write_octal(&mut header[116..124], 0, 7);
    write_octal(&mut header[124..136], size, 11);
    write_octal(&mut header[136..148], 0, 11);
    // The checksum field is treated as spaces while the checksum is computed.
    header[148..156].fill(b' ');
    header[156] = type_flag;
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    write_bytes(&mut header[265..297], b"root");
    write_bytes(&mut header[297..329], b"root");
    write_bytes(&mut header[345..500], prefix.as_bytes());

    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    // `%06o\0 ` is the layout every tar implementation reads back.
    let rendered = format!("{checksum:06o}");
    header[148..154].copy_from_slice(rendered.as_bytes());
    header[154] = 0;
    header[155] = b' ';
    header
}

/// Split an over-long entry name into the `ustar` prefix and name fields.
fn split_name(name: &str) -> (&str, &str) {
    if name.len() <= 100 {
        return ("", name);
    }
    // The prefix holds everything up to a `/` so that `prefix + "/" + name` is the full path.
    for (index, _) in name.match_indices('/') {
        let (prefix, rest) = name.split_at(index);
        let rest = &rest[1..];
        if prefix.len() <= 155 && rest.len() <= 100 {
            return (prefix, rest);
        }
    }
    // Nothing fits: truncate, which is better than emitting a corrupt header.
    ("", &name[name.len() - 100..])
}

/// Copy `value` into `field`, leaving the remaining bytes zeroed.
fn write_bytes(field: &mut [u8], value: &[u8]) {
    let length = value.len().min(field.len());
    field[..length].copy_from_slice(&value[..length]);
}

/// Write `value` as `digits` octal characters followed by a NUL.
fn write_octal(field: &mut [u8], value: u64, digits: usize) {
    let rendered = format!("{value:0digits$o}", digits = digits);
    write_bytes(field, rendered.as_bytes());
    field[digits] = 0;
}

/// Wrap `data` in a gzip stream built from DEFLATE stored blocks.
fn gzip(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 65_535 * 5 + 32);
    // Magic, CM=deflate, no flags, no mtime, no extra flags, unknown OS.
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);
    if data.is_empty() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    } else {
        let mut chunks = data.chunks(65_535).peekable();
        while let Some(chunk) = chunks.next() {
            let final_block = chunks.peek().is_none();
            // Three bits (BFINAL, BTYPE=00) padded to a byte boundary, then LEN and ~LEN.
            out.push(if final_block { 0x01 } else { 0x00 });
            let length = chunk.len() as u16;
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&(!length).to_le_bytes());
            out.extend_from_slice(chunk);
        }
    }
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// The CRC-32 (IEEE polynomial) of `data`, as gzip trailers use it.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_known_crc32_values() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414f_a339
        );
    }

    #[test]
    fn writes_a_valid_ustar_header() {
        let header = ustar_header("task.md", 5, 0o644, b'0');
        assert_eq!(&header[0..7], b"task.md");
        assert_eq!(&header[100..108], b"0000644\0");
        assert_eq!(&header[124..136], b"00000000005\0");
        assert_eq!(header[156], b'0');
        assert_eq!(&header[257..263], b"ustar\0");
        // The recorded checksum must match a recomputation with the field blanked out.
        let recorded = std::str::from_utf8(&header[148..154]).unwrap();
        let mut blanked = header;
        blanked[148..156].fill(b' ');
        let expected: u32 = blanked.iter().map(|byte| u32::from(*byte)).sum();
        assert_eq!(u32::from_str_radix(recorded, 8).unwrap(), expected);
    }

    #[test]
    fn splits_long_entry_names() {
        let long = format!("{}/{}", "a".repeat(120), "b.txt");
        let (prefix, name) = split_name(&long);
        assert_eq!(prefix, "a".repeat(120));
        assert_eq!(name, "b.txt");
        assert_eq!(split_name("short.txt"), ("", "short.txt"));
    }

    #[test]
    fn gzip_frames_large_payloads_in_several_stored_blocks() {
        let payload = vec![7u8; 70_000];
        let framed = gzip(&payload);
        assert_eq!(&framed[..3], &[0x1f, 0x8b, 0x08]);
        // Two stored blocks: only the last one carries BFINAL.
        assert_eq!(framed[10], 0x00);
        let trailer = framed.len() - 8;
        assert_eq!(
            u32::from_le_bytes(framed[trailer..trailer + 4].try_into().unwrap()),
            crc32(&payload)
        );
        assert_eq!(
            u32::from_le_bytes(framed[trailer + 4..].try_into().unwrap()),
            70_000
        );
    }

    /// The archive has to be readable by the `tar` running inside the sandbox, so round-trip it
    /// through the system `tar` rather than trusting the header layout alone.
    #[test]
    fn round_trips_through_system_tar() {
        let mut builder = TarGzBuilder::new();
        builder.add_dir("nested", 0o755);
        builder.add_file("nested/task.md", b"hello sandbox\n", 0o644);
        builder.add_file("top.bin", &vec![0xabu8; 1000], 0o600);
        let archive = builder.finish();

        let dir = tempfile::tempdir().expect("temp dir");
        let archive_path = dir.path().join("archive.tar.gz");
        std::fs::write(&archive_path, &archive).expect("write archive");
        let status = std::process::Command::new("tar")
            .arg("-xzf")
            .arg(&archive_path)
            .arg("-C")
            .arg(dir.path())
            .status()
            .expect("run tar");
        assert!(status.success(), "tar failed to extract the archive");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("nested/task.md")).expect("read entry"),
            "hello sandbox\n"
        );
        assert_eq!(
            std::fs::read(dir.path().join("top.bin")).expect("read entry"),
            vec![0xabu8; 1000]
        );
    }
}
