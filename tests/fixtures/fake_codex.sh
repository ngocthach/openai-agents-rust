#!/bin/sh
# Fake `codex` CLI used by tests/codex_tests.rs.
#
# Records its argv, stdin, a few environment variables and any `--output-schema` file into
# $FAKE_CODEX_RECORD_DIR, prints the canned JSONL events of $FAKE_CODEX_EVENTS (or
# $FAKE_CODEX_EVENTS_RESUME when resuming a thread) and exits with $FAKE_CODEX_EXIT.
#
# The SDK replaces the whole environment of the Codex process, so PATH is set here.
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH

record_dir="${FAKE_CODEX_RECORD_DIR:?FAKE_CODEX_RECORD_DIR must be set}"
mkdir -p "$record_dir"
turn=$(ls "$record_dir" | grep -c '^argv-')
turn=$((turn + 1))

: > "$record_dir/argv-$turn.txt"
for arg in "$@"; do
  printf '%s\n' "$arg" >> "$record_dir/argv-$turn.txt"
done

cat > "$record_dir/stdin-$turn.txt"

{
  printf 'CODEX_API_KEY=%s\n' "$CODEX_API_KEY"
  printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY"
  printf 'OPENAI_BASE_URL=%s\n' "$OPENAI_BASE_URL"
  printf 'CODEX_INTERNAL_ORIGINATOR_OVERRIDE=%s\n' "$CODEX_INTERNAL_ORIGINATOR_OVERRIDE"
} > "$record_dir/env-$turn.txt"

schema=""
previous=""
for arg in "$@"; do
  if [ "$previous" = "--output-schema" ]; then
    schema="$arg"
  fi
  previous="$arg"
done
if [ -n "$schema" ] && [ -f "$schema" ]; then
  cp "$schema" "$record_dir/schema-$turn.json"
fi

events="$FAKE_CODEX_EVENTS"
case " $* " in
  *" resume "*)
    if [ -n "$FAKE_CODEX_EVENTS_RESUME" ]; then
      events="$FAKE_CODEX_EVENTS_RESUME"
    fi
    ;;
esac
if [ -n "$events" ] && [ -f "$events" ]; then
  cat "$events"
fi

exit_code="${FAKE_CODEX_EXIT:-0}"
if [ "$exit_code" -ne 0 ]; then
  printf 'fake codex failure\n' >&2
fi
exit "$exit_code"
