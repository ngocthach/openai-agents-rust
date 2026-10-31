#!/bin/sh
# Fake `codex` CLI used by the tests/py_codex_*.rs parity suites.
#
# It is a superset of tests/fixtures/fake_codex.sh: besides recording argv, stdin, a few
# environment variables and any `--output-schema` file into $FAKE_CODEX_RECORD_DIR, it supports
#
#   FAKE_CODEX_EVENTS            default JSONL event file
#   FAKE_CODEX_EVENTS_RESUME     used instead when the argv contains `resume`
#   FAKE_CODEX_EVENTS_<turn>     used instead on that 1-based turn
#   FAKE_CODEX_STDERR_FILE       copied verbatim to stderr (so non-UTF-8 bytes survive)
#   FAKE_CODEX_SLEEP             keep the process alive for N seconds after the events
#   FAKE_CODEX_SLEEP_TURN        restrict FAKE_CODEX_SLEEP to that 1-based turn
#   FAKE_CODEX_EXIT              exit status (default 0)
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
per_turn=$(eval "printf '%s' \"\${FAKE_CODEX_EVENTS_$turn:-}\"")
if [ -n "$per_turn" ]; then
  events="$per_turn"
fi
if [ -n "$events" ] && [ -f "$events" ]; then
  cat "$events"
fi

if [ -n "$FAKE_CODEX_STDERR_FILE" ] && [ -f "$FAKE_CODEX_STDERR_FILE" ]; then
  cat "$FAKE_CODEX_STDERR_FILE" >&2
fi

# Keep the process alive (replacing this shell, so a kill reaches it) to exercise cancellation
# and the idle timeout.
if [ -n "$FAKE_CODEX_SLEEP" ]; then
  if [ -z "$FAKE_CODEX_SLEEP_TURN" ] || [ "$FAKE_CODEX_SLEEP_TURN" = "$turn" ]; then
    exec sleep "$FAKE_CODEX_SLEEP"
  fi
fi

exit_code="${FAKE_CODEX_EXIT:-0}"
if [ "$exit_code" -ne 0 ]; then
  if [ -z "$FAKE_CODEX_STDERR_FILE" ]; then
    printf 'fake codex failure\n' >&2
  fi
fi
exit "$exit_code"
