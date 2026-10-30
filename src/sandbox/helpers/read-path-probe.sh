# READ_PATH_PROBE_V3
LC_ALL=C
export LC_ALL
path=$1
resolved_path=
symlink_depth=0

resolve_probe_path() {
    if [ "$symlink_depth" -gt 40 ] || [ "${#1}" -gt 4095 ]; then
        return 2
    fi
    if [ "$1" = "/" ]; then
        resolved_path=/
        return 0
    fi

    parent=${1%/*}
    if [ -z "$parent" ] || [ "$parent" = "$1" ]; then
        parent=/
    fi
    resolve_probe_path "$parent" || return 2
    resolved_parent=$resolved_path
    base=${1##*/}
    if [ "${#base}" -gt 255 ]; then
        return 2
    fi
    if [ "$resolved_parent" = "/" ]; then
        candidate=/$base
    else
        candidate=$resolved_parent/$base
    fi

    if [ -L "$candidate" ]; then
        target_with_marker=$(readlink -n "$candidate" && printf .) || return 2
        target=${target_with_marker%.}
        symlink_depth=$((symlink_depth + 1))
        if [ "$symlink_depth" -gt 40 ]; then
            return 2
        fi
        case "$target" in
            /*)
                resolve_probe_path "$target"
                ;;
            *)
                resolve_probe_path "$resolved_parent/$target"
                ;;
        esac
        return $?
    fi

    resolved_path=$candidate
}

resolve_probe_path "$path" || exit 2
path=$resolved_path
candidate=$path
child=

while :; do
    if [ -e "$candidate" ]; then
        if [ "$candidate" = "$path" ]; then
            exit 0
        fi
        if [ ! -d "$candidate" ] || [ ! -x "$candidate" ]; then
            exit 2
        fi
        lookup_result=$(
            find "$child" -prune -print 2>&1 >/dev/null
            lookup_status=$?
            printf '.%s' "$lookup_status"
        )
        lookup_status=${lookup_result##*.}
        lookup_error=${lookup_result%.*}
        if [ "$lookup_status" -eq 1 ]; then
            lookup_error=$(printf %s "$lookup_error")
            case "$lookup_error" in
                *": No such file or directory")
                    exit 1
                    ;;
            esac
        fi
        exit 2
    fi
    if [ "$candidate" = "/" ]; then
        exit 2
    fi
    child=$candidate
    candidate=${candidate%/*}
    if [ -z "$candidate" ]; then
        candidate=/
    fi
done