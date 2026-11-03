"""A scriptable stand-in for the bundled Modal helper, used by the core6 Modal provider tests.

It speaks the JSON protocol of `src/extensions/sandbox/modal/helper.py` (one request per process,
one JSON line back) without calling Modal. Every request is appended to `FAKE_MODAL_LOG`, and
sandbox identity, stopped sandboxes and per-rule use counts live in `FAKE_MODAL_STATE`.

`FAKE_MODAL_RULES` names a JSON file holding a list of rules. A rule is
`{"op": ..., "contains": optional substring of the joined argv, "response": {...},
"once": optional bool, "sleep": optional seconds}`; the first unused matching rule answers the
request instead of the default behaviour.

`FAKE_MODAL_PTY` selects what the `pty` operation does: `cat` echoes stdin until EOF, `burst`
prints 200 lines, `delayed` prints a head, sleeps, then prints a tail and exits 3, `fail` writes
to stderr and exits 7. Anything else prints the argv.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

LOG = os.environ.get("FAKE_MODAL_LOG")
STATE = os.environ.get("FAKE_MODAL_STATE")
RULES = os.environ.get("FAKE_MODAL_RULES")


def _load_state() -> dict:
    if STATE and os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as handle:
            return json.load(handle)
    return {"sandboxes": 0, "files": {}, "terminated": [], "stopped": [], "rules_used": {}}


def _save_state(state: dict) -> None:
    if STATE:
        with open(STATE, "w", encoding="utf-8") as handle:
            json.dump(state, handle)


def _log(request: dict) -> None:
    if LOG:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request, sort_keys=True) + "\n")


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode("ascii")


def _rule_response(request: dict, state: dict):
    if not RULES or not os.path.exists(RULES):
        return None
    with open(RULES, encoding="utf-8") as handle:
        rules = json.load(handle)
    joined = " ".join(request.get("argv") or [])
    for index, rule in enumerate(rules):
        if rule.get("op") != request.get("op"):
            continue
        if "contains" in rule and rule["contains"] not in joined:
            continue
        used = state["rules_used"].get(str(index), 0)
        if rule.get("once") and used:
            continue
        state["rules_used"][str(index)] = used + 1
        if rule.get("sleep"):
            time.sleep(float(rule["sleep"]))
        return rule.get("response", {})
    return None


def _exec(request: dict) -> dict:
    argv = request.get("argv") or []
    joined = " ".join(argv)
    if "tar -czf -" in joined:
        return {"exit_code": 0, "stdout": _b64("fake-tar-bytes"), "stderr": ""}
    if argv[:1] == ["false"]:
        return {"exit_code": 3, "stdout": "", "stderr": _b64("boom\n")}
    if argv[:1] == ["mkdir"] or argv[:2] in (["sh", "-lc"], ["sh", "-c"]):
        return {"exit_code": 0, "stdout": "", "stderr": ""}
    return {"exit_code": 0, "stdout": _b64(f"ran: {joined}\n"), "stderr": ""}


def _pty(request: dict) -> None:
    mode = os.environ.get("FAKE_MODAL_PTY", "")
    out = sys.stdout
    if mode == "cat":
        out.write("ready\n")
        out.flush()
        while True:
            chunk = sys.stdin.readline()
            if not chunk:
                break
            out.write("echo:" + chunk)
            out.flush()
        raise SystemExit(0)
    if mode == "burst":
        for number in range(200):
            out.write(f"line {number}\n")
        out.flush()
        raise SystemExit(0)
    if mode == "delayed":
        out.write("head\n")
        out.flush()
        time.sleep(1.5)
        out.write("tail\n")
        out.flush()
        raise SystemExit(3)
    if mode == "fail":
        sys.stderr.write("pty failed\n")
        sys.stderr.flush()
        raise SystemExit(7)
    out.write(f"pty: {' '.join(request.get('argv') or [])}\n")
    out.flush()
    raise SystemExit(0)


def main() -> None:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.readline()
    request = json.loads(raw)
    _log(request)
    state = _load_state()
    op = request.get("op")

    if op == "pty":
        _save_state(state)
        _pty(request)
        return

    scripted = _rule_response(request, state)
    if scripted is not None:
        _save_state(state)
        json.dump(scripted, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    if op == "create":
        state["sandboxes"] += 1
        response = {
            "sandbox_id": f"sb-{state['sandboxes']}",
            "image_id": request.get("image_id") or "im-from-tag",
        }
    elif op == "exec":
        response = _exec(request)
    elif op == "write_file":
        state["files"][request["path"]] = request.get("data") or ""
        response = {"ok": True}
    elif op == "read_file":
        path = request["path"]
        if path in state["files"]:
            response = {"data": state["files"][path]}
        else:
            response = {"error": f"cat: {path}: No such file", "code": "not_found"}
    elif op == "poll":
        sandbox_id = request.get("sandbox_id")
        running = sandbox_id not in state["terminated"] and sandbox_id not in state["stopped"]
        response = {"running": running}
    elif op == "terminate":
        state["terminated"].append(request.get("sandbox_id"))
        response = {"ok": True}
    elif op == "snapshot_filesystem":
        response = {"image_id": f"im-{request.get('sandbox_id')}"}
    elif op == "snapshot_directory":
        response = {"image_id": f"dir-{request.get('sandbox_id')}"}
    elif op == "mount_image":
        response = {"ok": True}
    elif op == "tunnels":
        response = {"tunnels": {"8080": {"host": "tunnel.modal.host", "port": 443}}}
    else:
        response = {"error": f"unknown op: {op}"}

    _save_state(state)
    json.dump(response, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
