target="$1"
parents="$2"
if [ -e "$target" ] || [ -L "$target" ]; then
    [ -d "$target" ] && [ -x "$target" ]
    exit $?
fi
parent=$(dirname "$target")
if [ "$parents" = "1" ]; then
    while [ ! -e "$parent" ]; do
        next=$(dirname "$parent")
        if [ "$next" = "$parent" ]; then
            exit 1
        fi
        parent="$next"
    done
fi
[ -d "$parent" ] && [ -w "$parent" ] && [ -x "$parent" ]
