"""A stand-in for the bundled Modal helper, used by the provider tests.

It speaks the same JSON protocol as `src/extensions/sandbox/modal/helper.py` but keeps a
sandbox's files in a local JSON file instead of calling Modal. Every request is appended to
`FAKE_MODAL_LOG` as one JSON line so the tests can assert on the argv, timeout and working
directory the Rust side sent.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

LOG = os.environ.get("FAKE_MODAL_LOG")
STATE = os.environ.get("FAKE_MODAL_STATE")


def _load_state() -> dict:
    if STATE and os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as handle:
            return json.load(handle)
    return {"sandboxes": 0, "files": {}, "terminated": []}


def _save_state(state: dict) -> None:
    if STATE:
        with open(STATE, "w", encoding="utf-8") as handle:
            json.dump(state, handle)


def _log(request: dict) -> None:
    if LOG:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request, sort_keys=True) + "\n")


def _exec(request: dict, state: dict) -> dict:
    argv = request.get("argv") or []
    joined = " ".join(argv)
    if "sleep" in joined:
        # Long enough that the Rust side's own timeout fires first.
        time.sleep(float(os.environ.get("FAKE_MODAL_SLEEP", "5")))
    if argv[:1] == ["mkdir"]:
        return {"exit_code": 0, "stdout": "", "stderr": ""}
    if "tar -czf -" in joined:
        return {
            "exit_code": 0,
            "stdout": base64.b64encode(b"fake-tar-bytes").decode("ascii"),
            "stderr": "",
        }
    if "tar -xzf" in joined:
        state.setdefault("hydrated", []).append(joined)
        return {"exit_code": 0, "stdout": "", "stderr": ""}
    if argv[:1] == ["false"]:
        return {
            "exit_code": 3,
            "stdout": "",
            "stderr": base64.b64encode(b"boom\n").decode("ascii"),
        }
    return {
        "exit_code": 0,
        "stdout": base64.b64encode(f"ran: {joined}\n".encode()).decode("ascii"),
        "stderr": "",
    }


def main() -> None:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.readline()
    request = json.loads(raw)
    _log(request)
    state = _load_state()
    op = request.get("op")

    if op == "create":
        state["sandboxes"] += 1
        response = {
            "sandbox_id": f"sb-{state['sandboxes']}",
            "image_id": request.get("image_id") or "im-from-tag",
        }
    elif op == "exec":
        response = _exec(request, state)
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
        response = {"running": request.get("sandbox_id") not in state["terminated"]}
    elif op == "terminate":
        state["terminated"].append(request.get("sandbox_id"))
        response = {"ok": True}
    elif op == "snapshot_filesystem":
        response = {"image_id": f"im-{request.get('sandbox_id')}"}
    elif op == "snapshot_directory":
        response = {"image_id": f"dir-{request.get('sandbox_id')}"}
    elif op == "mount_image":
        state.setdefault("mounted", []).append(
            {"path": request.get("path"), "image_id": request.get("image_id")}
        )
        response = {"ok": True}
    elif op == "tunnels":
        response = {
            "tunnels": {
                "8080": {"host": "tunnel.modal.host", "port": 443},
                "9000": {"host": "plain.modal.host", "port": 31000},
            }
        }
    elif op == "pty":
        # The PTY bridge writes raw bytes instead of a JSON response.
        sys.stdout.write(f"pty: {' '.join(request.get('argv') or [])}\n")
        sys.stdout.flush()
        _save_state(state)
        raise SystemExit(0)
    else:
        response = {"error": f"unknown op: {op}"}

    _save_state(state)
    json.dump(response, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
