"""Compare the requests the two SDKs sent for one scenario.

Reads ``<out>/<scenario>/py.jsonl`` and ``rs.jsonl``, normalizes both (sorted keys, volatile
ids and timestamps masked, transport-only headers dropped, order-free arrays sorted), removes
the waived JSON pointers, then diffs request by request. Writes ``diff.txt`` next to the
records, prints ``PARITY <scenario> OK`` or ``PARITY <scenario> DIFF <lines>`` and exits 1 on
an unwaived difference. ``<sdk>.events.json`` files, when both exist, are diffed the same way.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

# Headers that only describe the HTTP transport, never the prompt.
TRANSPORT_HEADERS = {"content-length", "host", "accept-encoding", "connection", "accept"}
ID_RE = re.compile(r"^(resp|msg|fc|call|rs|ws|trace|span|chatcmpl|item)_[A-Za-z0-9_-]+$")
VOLATILE_KEYS = {"created_at", "started_at", "ended_at"}
ORDER_FREE_LISTS = {"required", "include"}
# Top-level body lists the Python client sends empty where the Rust client omits them; the
# API treats an empty list and an absent field the same way.
EMPTY_MEANS_ABSENT = {"tools", "include"}


# Keys that start a JSON Schema subtree (tool parameters, output formats); inside one, the
# annotation-only `title` pydantic adds and schemars omits is dropped.
SCHEMA_ROOTS = {"parameters", "schema", "input_schema"}


def normalize(value, key: str | None = None, in_schema: bool = False):
    if isinstance(value, dict):
        return {
            k: normalize(v, k, in_schema or k in SCHEMA_ROOTS)
            for k, v in sorted(value.items())
            if k not in VOLATILE_KEYS and not (in_schema and k == "title")
        }
    if isinstance(value, list):
        items = [normalize(v, key, in_schema) for v in value]
        if key in ORDER_FREE_LISTS and all(isinstance(v, str) for v in items):
            return sorted(items)
        return items
    if isinstance(value, str) and ID_RE.match(value):
        return "<id>"
    if isinstance(value, str) and value[:1] in "{[":
        # JSON carried inside a string (tool arguments and outputs): whitespace and key
        # order are not part of the contract, so compare the canonical form.
        try:
            return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))
        except ValueError:
            return value
    return value


def normalize_record(record: dict) -> dict:
    headers = {
        k: v for k, v in record.get("headers", {}).items() if k not in TRANSPORT_HEADERS
    }
    body = record.get("body")
    if isinstance(body, dict):
        body = {k: v for k, v in body.items() if not (k in EMPTY_MEANS_ABSENT and v == [])}
        if str(record.get("path", "")).endswith("traces/ingest") and isinstance(body.get("data"), list):
            # A trace batch is a set: the backend links spans by id, so the order the two
            # exporters flush them in is not part of the contract.
            body["data"] = sorted(
                body["data"],
                key=lambda item: (
                    str(item.get("object")),
                    str((item.get("span_data") or {}).get("type")),
                    str((item.get("span_data") or {}).get("name")),
                ),
            )
    return {
        "method": record.get("method"),
        "path": record.get("path"),
        "headers": normalize(headers),
        "body": normalize(body),
    }


def remove_pointer(node, parts: list[str]) -> None:
    """Delete a JSON pointer; ``*`` matches every list index or key, a trailing ``*`` on a
    key matches by prefix."""
    if not parts or node is None:
        return
    head, rest = parts[0], parts[1:]
    if isinstance(node, list):
        targets = range(len(node)) if head == "*" else [int(head)] if head.isdigit() else []
        for i in targets:
            if i < len(node):
                if rest:
                    remove_pointer(node[i], rest)
                else:
                    node[i] = None
        return
    if not isinstance(node, dict):
        return
    if head == "*":
        keys = list(node)
    elif head.endswith("*"):
        keys = [k for k in node if k.startswith(head[:-1])]
    else:
        keys = [head] if head in node else []
    for k in keys:
        if rest:
            remove_pointer(node[k], rest)
        else:
            del node[k]


def load_records(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(value) -> list[str]:
    return json.dumps(value, indent=2, sort_keys=True).splitlines(keepends=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--out", required=True)
    parser.add_argument("--waivers", required=True)
    parser.add_argument("--scenarios", help="scenario directory, for per-scenario compare.json")
    args = parser.parse_args()
    folder = Path(args.out) / args.scenario
    waivers_all = json.loads(Path(args.waivers).read_text())
    waivers = {**waivers_all.get("*", {}), **waivers_all.get(args.scenario, {})}

    py = load_records(folder / "py.jsonl")
    rs = load_records(folder / "rs.jsonl")
    options_path = Path(args.scenarios) / args.scenario / "compare.json" if args.scenarios else None
    options = json.loads(options_path.read_text()) if options_path and options_path.exists() else {}
    if options.get("order_free") and py is not None and rs is not None:
        # Concurrent requests (parallel guardrails, parallel tool runs) arrive in any order:
        # pair them by path and instructions instead of arrival.
        def order_key(record: dict):
            body = record.get("body") or {}
            return (record.get("path", ""), str(body.get("instructions", "")), json.dumps(body, sort_keys=True))

        py = sorted(py, key=order_key)
        rs = sorted(rs, key=order_key)
    lines: list[str] = []
    if py is None or rs is None:
        lines.append(f"missing records: py={'ok' if py else 'none'} rs={'ok' if rs else 'none'}\n")
    else:
        if len(py) != len(rs):
            lines.append(f"request count differs: py={len(py)} rs={len(rs)}\n")
        for i, (a, b) in enumerate(zip(py, rs)):
            na, nb = normalize_record(a), normalize_record(b)
            for pointer in waivers:
                parts = [p for p in pointer.split("/") if p]
                remove_pointer(na, parts)
                remove_pointer(nb, parts)
            lines.extend(
                difflib.unified_diff(
                    dump(na), dump(nb), fromfile=f"py request {i}", tofile=f"rs request {i}", n=3
                )
            )
        ev_py, ev_rs = folder / "py.events.json", folder / "rs.events.json"
        if ev_py.exists() and ev_rs.exists():
            lines.extend(
                difflib.unified_diff(
                    dump(json.loads(ev_py.read_text())),
                    dump(json.loads(ev_rs.read_text())),
                    fromfile="py events",
                    tofile="rs events",
                    n=3,
                )
            )
    (folder / "diff.txt").write_text("".join(lines))
    if lines:
        print(f"PARITY {args.scenario} DIFF {len(lines)} (see {folder / 'diff.txt'})")
        return 1
    print(f"PARITY {args.scenario} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
