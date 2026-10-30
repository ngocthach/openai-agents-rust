target="$1"
if [ -e "$target" ]; then
    [ -f "$target" ] && [ -w "$target" ]
    exit $?
fi
parent=$(dirname "$target")
while [ ! -e "$parent" ]; do
    next=$(dirname "$parent")
    if [ "$next" = "$parent" ]; then
        exit 1
    fi
    parent="$next"
done
[ -d "$parent" ] && [ -w "$parent" ] && [ -x "$parent" ]
