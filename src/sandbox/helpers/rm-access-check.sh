target="$1"
recursive="$2"
if [ ! -e "$target" ] && [ ! -L "$target" ]; then
    [ "$recursive" = "1" ]
    exit $?
fi
parent=$(dirname "$target")
[ -d "$parent" ] && [ -w "$parent" ] && [ -x "$parent" ]
