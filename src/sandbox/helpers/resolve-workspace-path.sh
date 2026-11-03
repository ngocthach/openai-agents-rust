#!/bin/sh
# RESOLVE_WORKSPACE_REALPATH_V1
set -eu

root="$1"
candidate="$2"
for_write="$3"
shift 3
max_symlink_depth=64

case "$for_write" in
    0|1|2) ;; # read, write, recursive removal
    *)
        printf 'for_write must be 0, 1 or 2: %s\n' "$for_write" >&2
        exit 64
        ;;
esac

if [ $(( $# % 2 )) -ne 0 ]; then
    printf 'extra path grants must be root/read_only pairs\n' >&2
    exit 64
fi

resolve_path() {
    path="$1"
    depth="${2:-0}"
    seen="${3:-}"
    if [ "$path" = "/" ]; then
        printf '/\n'
        return 0
    fi

    if [ "$depth" -ge "$max_symlink_depth" ]; then
        printf 'symlink resolution depth exceeded: %s\n' "$path" >&2
        exit 112
    fi

    if [ -d "$path" ]; then
        (
            cd "$path"
            pwd -P
        )
        return 0
    fi

    parent=${path%/*}
    base=${path##*/}
    if [ -z "$parent" ] || [ "$parent" = "$path" ]; then
        parent="/"
    fi

    resolved_parent=$(resolve_path "$parent" "$depth" "$seen")
    candidate_path="$resolved_parent/$base"
    if [ -L "$candidate_path" ]; then
        case ":$seen:" in
            *":$candidate_path:"*)
                printf 'symlink resolution depth exceeded: %s\n' "$candidate_path" >&2
                exit 112
                ;;
        esac
        target=$(readlink "$candidate_path")
        next_depth=$((depth + 1))
        next_seen="${seen}:$candidate_path"
        case "$target" in
            /*) resolve_path "$target" "$next_depth" "$next_seen" ;;
            *) resolve_path "$resolved_parent/$target" "$next_depth" "$next_seen" ;;
        esac
        return 0
    fi

    printf '%s\n' "$candidate_path"
}

resolved_candidate=$(resolve_path "$candidate" 0)

check_root() {
    allowed_root="$1"
    resolved_root=$(resolve_path "$allowed_root" 0)
    case "$resolved_candidate" in
        "$resolved_root"|"$resolved_root"/*)
            printf '%s\n' "$resolved_candidate"
            exit 0
            ;;
    esac
}

reject_root_grant() {
    allowed_root="$1"
    resolved_root=$(resolve_path "$allowed_root" 0)
    if [ "$resolved_root" = "/" ]; then
        printf 'extra path grant must not resolve to filesystem root: %s\n' "$allowed_root" >&2
        exit 113
    fi
}

consider_extra_grant() {
    allowed_root="$1"
    read_only="$2"
    case "$read_only" in
        0|1) ;;
        *)
            printf 'extra path grant read_only must be 0 or 1: %s\n' "$read_only" >&2
            exit 64
            ;;
    esac

    reject_root_grant "$allowed_root"
    resolved_root=$(resolve_path "$allowed_root" 0)
    case "$resolved_candidate" in
        "$resolved_root"|"$resolved_root"/*)
            root_len=${#resolved_root}
            if [ "$root_len" -gt "$best_grant_len" ]; then
                best_grant_root="$resolved_root"
                best_grant_original="$allowed_root"
                best_grant_read_only="$read_only"
                best_grant_len="$root_len"
            fi
            ;;
    esac
}

authorize_candidate() (
    # A subshell keeps each candidate's grant selection independent.
    resolved_candidate="$1"
    shift
    best_grant_root=""
    best_grant_original=""
    best_grant_read_only="0"
    best_grant_len=0
    while [ "$#" -gt 0 ]; do
        consider_extra_grant "$1" "$2"
        shift 2
    done

    check_root "$root"
    if [ -n "$best_grant_root" ]; then
        if [ "$for_write" != "0" ] && [ "$best_grant_read_only" = "1" ]; then
            printf 'read-only extra path grant: %s\nresolved path: %s\n'                 "$best_grant_original" "$resolved_candidate" >&2
            exit 114
        fi
        printf '%s\n' "$resolved_candidate"
        exit 0
    fi
    printf 'workspace escape: %s\n' "$resolved_candidate" >&2
    exit 111
)

authorize_candidate "$resolved_candidate" "$@" >/dev/null
if [ "$for_write" = "2" ] && [ ! -L "$candidate" ]; then
    # rm unlinks a leaf symlink; only a traversed tree needs descendant checks.
    grant_root=""
    for grant_part in "$@"; do
        if [ -z "$grant_root" ]; then
            grant_root="$grant_part"
            continue
        fi
        if [ "$grant_part" = "1" ]; then
            resolved_grant=$(resolve_path "$grant_root" 0)
            case "$resolved_grant" in
                "$resolved_candidate"|"$resolved_candidate"/*)
                    authorize_candidate "$resolved_grant" "$@" >/dev/null
                    ;;
            esac
        fi
        grant_root=""
    done
fi
printf '%s\n' "$resolved_candidate"