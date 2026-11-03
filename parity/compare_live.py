"""Check the two SDKs' live shape records against a scenario's ``expect.json``.

``expect.json``::

    {"each": {"<path>": <expectation>, ...}, "across": ["<path>", ...]}

``each`` is evaluated on both records; ``across`` paths must be equal between them. A path is
dotted (``usage.requests``, ``tool_calls.0.arguments.city``) with optional suffixes ``.names``
(list of ``name`` fields), ``.first``, ``.last``, ``.len``. An expectation is an exact JSON
value, ``"nonnull"``, ``"nonempty"``, ``{"gt": n}`` or ``{"contains": v}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MISSING = object()


def lookup(record, path: str):
    node = record
    for part in path.split("."):
        if part == "names" and isinstance(node, list):
            node = [item.get("name") if isinstance(item, dict) else item for item in node]
        elif part == "first" and isinstance(node, list):
            node = node[0] if node else MISSING
        elif part == "last" and isinstance(node, list):
            node = node[-1] if node else MISSING
        elif part == "len" and isinstance(node, (list, str, dict)):
            node = len(node)
        elif isinstance(node, dict):
            node = node.get(part, MISSING)
        elif isinstance(node, list) and part.isdigit():
            node = node[int(part)] if int(part) < len(node) else MISSING
        else:
            return MISSING
        if node is MISSING:
            return MISSING
    return node


def check(value, expectation) -> bool:
    if expectation == "nonnull":
        return value is not MISSING and value is not None
    if expectation == "nonempty":
        return value is not MISSING and bool(value)
    if isinstance(expectation, dict) and "gt" in expectation:
        return isinstance(value, (int, float)) and value > expectation["gt"]
    if isinstance(expectation, dict) and "contains" in expectation:
        return isinstance(value, (list, str)) and expectation["contains"] in value
    return value is not MISSING and value == expectation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenarios", required=True)
    args = parser.parse_args()
    folder = Path(args.out) / args.scenario
    expect = json.loads((Path(args.scenarios) / args.scenario / "expect.json").read_text())
    records = {}
    failures = []
    for sdk in ("py", "rs"):
        path = folder / f"{sdk}.json"
        if not path.exists():
            failures.append(f"{sdk}: no record")
            continue
        records[sdk] = json.loads(path.read_text())
    for sdk, record in records.items():
        for path, expectation in expect.get("each", {}).items():
            value = lookup(record, path)
            if not check(value, expectation):
                shown = "<missing>" if value is MISSING else json.dumps(value)[:120]
                failures.append(f"{sdk}: {path} = {shown}, expected {json.dumps(expectation)}")
    if len(records) == 2:
        for path in expect.get("across", []):
            a, b = lookup(records["py"], path), lookup(records["rs"], path)
            if a != b:
                failures.append(f"across: {path} py={json.dumps(a)[:120]} rs={json.dumps(b)[:120]}")
    (folder / "verdict.txt").write_text("\n".join(failures) + ("\n" if failures else ""))
    if failures:
        print(f"PARITY LIVE {args.scenario} FAIL {len(failures)}")
        for line in failures:
            print("  " + line)
        return 1
    print(f"PARITY LIVE {args.scenario} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
