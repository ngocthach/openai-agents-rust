#!/bin/sh
# WORKSPACE_FINGERPRINT_V2
set -eu

if [ "$#" -lt 4 ]; then
    printf '%s\n'         "usage: $0 <workspace-root> <version> <output-path>"         " <manifest-digest> [exclude-relpath ...]" >&2
    exit 64
fi

workspace_root=$1
version=$2
output_path=$3
manifest_digest=$4
shift 4

if [ ! -d "$workspace_root" ]; then
    printf 'workspace root not found: %s\n' "$workspace_root" >&2
    exit 66
fi

case "$workspace_root" in
    *"'"*)
        printf 'workspace root contains unsupported single quote: %s\n' "$workspace_root" >&2
        exit 65
        ;;
esac

quote_sh() {
    value=$1
    case "$value" in
        *"'"*)
            printf 'unsupported single quote in argument: %s\n' "$value" >&2
            exit 65
            ;;
        *)
            printf "'%s'" "$value"
            ;;
    esac
}

hash_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
        return
    fi
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 | awk '{print $1}'
        return
    fi
    if command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 | awk '{print $NF}'
        return
    fi
    printf 'workspace fingerprint helper requires sha256sum, shasum, or openssl\n' >&2
    exit 127
}

if tar --help 2>&1 | grep -q -- '--no-wildcards'; then
    tar_cmd="tar --no-wildcards"
    escape_tar_patterns=0
else
    tar_cmd="tar"
    escape_tar_patterns=1
fi
for rel in "$@"; do
    case "$rel" in
        ""|"."|"/"|*"/.."|*"/../"*|".."|../*|*/../*|/*)
            printf 'exclude relpath must be a concrete relative path: %s\n' "$rel" >&2
            exit 65
            ;;
    esac
    if [ "$escape_tar_patterns" -eq 1 ]; then
        rel=$(printf '%s\n' "$rel" | sed 's/[][\\*?]/\\&/g')
    fi
    # Reproduce released fingerprints when comparing a preserved live workspace.
    # Switching algorithms before comparison could restore an incomplete old archive.
    if [ "$version" = "workspace_tar_sha256_v1" ]; then
        quoted_rel=$(quote_sh "$rel")
        tar_cmd="$tar_cmd --exclude=$quoted_rel"
    fi
    quoted_dot_rel=$(quote_sh "./$rel")
    tar_cmd="$tar_cmd --exclude=$quoted_dot_rel"
done

tar_cmd="$tar_cmd -C $(quote_sh "$workspace_root") -cf - ."

workspace_fingerprint=$(
    sh -c "$tar_cmd" | hash_stdin
)
fingerprint=$(
    printf '%s\n%s\n' "$workspace_fingerprint" "$manifest_digest" | hash_stdin
)

payload=$(printf '{"fingerprint":"%s","version":"%s"}
' "$fingerprint" "$version")
mkdir -p -- "$(dirname -- "$output_path")"
tmp_output="$output_path.tmp.$$"
printf '%s' "$payload" > "$tmp_output"
mv -f -- "$tmp_output" "$output_path"
printf '%s' "$payload"