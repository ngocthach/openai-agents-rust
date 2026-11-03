"""Local stand-in for the OpenAI API used by the parity harness.

Both SDKs point ``OPENAI_BASE_URL`` at ``http://127.0.0.1:PORT/<scenario>/<sdk>/v1``. Every
request is appended to ``<out>/<scenario>/<sdk>.jsonl`` (headers lower-cased, the
authorization value replaced by ``present``), and the reply is the next entry of
``<scenarios>/<scenario>/responses.json`` for that SDK and path family, completed with the
fields the clients require. Stdlib only; run with ``--port 0`` to pick a free port, printed as
``PORT=<n>`` on the first line of stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PATH_RE = re.compile(r"^/(?P<scenario>[\w-]+)/(?P<sdk>py|rs)/v1/(?P<rest>.*)$")
FAMILIES = ("responses", "chat/completions", "traces/ingest")


def family_of(rest: str) -> str | None:
    for family in FAMILIES:
        if rest == family or rest.startswith(family + "?"):
            return family
    return None


def complete_response(entry: dict, request: dict, seq: int) -> dict:
    """Fill the fields a Responses object needs so scripts can stay minimal."""
    out = dict(entry)
    out.setdefault("id", f"resp_parity_{seq}")
    out.setdefault("object", "response")
    out.setdefault("created_at", 1700000000)
    out.setdefault("status", "completed")
    out.setdefault("model", request.get("model", "unknown"))
    out.setdefault("output", [])
    out.setdefault("parallel_tool_calls", True)
    out.setdefault("tool_choice", "auto")
    out.setdefault("tools", [])
    out.setdefault("error", None)
    out.setdefault("incomplete_details", None)
    out.setdefault("instructions", None)
    out.setdefault("metadata", {})
    out.setdefault(
        "usage",
        {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    )
    for i, item in enumerate(out["output"]):
        item.setdefault("id", f"{item.get('type', 'item')}_parity_{seq}_{i}")
        if item.get("type") == "message":
            item.setdefault("role", "assistant")
            item.setdefault("status", "completed")
        if item.get("type") == "function_call":
            item.setdefault("call_id", f"call_parity_{seq}_{i}")
            item.setdefault("status", "completed")
    return out


def complete_chat(entry: dict, request: dict, seq: int) -> dict:
    out = dict(entry)
    out.setdefault("id", f"chatcmpl_parity_{seq}")
    out.setdefault("object", "chat.completion")
    out.setdefault("created", 1700000000)
    out.setdefault("model", request.get("model", "unknown"))
    out.setdefault(
        "usage", {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    )
    for choice in out.get("choices", []):
        choice.setdefault("index", 0)
        choice.setdefault("finish_reason", "stop")
    return out


def response_events(response: dict) -> list[dict]:
    """The scripted Responses object as the SSE events a streamed request receives: the same
    sequence `openai_agents_sdk::testing::response_to_stream_events` produces, with full
    objects so the Python client's typed parsing is satisfied."""
    seq = 0

    def event(payload: dict) -> dict:
        nonlocal seq
        payload["sequence_number"] = seq
        seq += 1
        return payload

    created = {**response, "status": "in_progress", "output": [], "usage": None}
    events = [event({"type": "response.created", "response": created})]
    for index, item in enumerate(response["output"]):
        started = dict(item)
        if item.get("type") == "message":
            started["content"] = []
            started["status"] = "in_progress"
        if item.get("type") == "function_call":
            started["arguments"] = ""
        events.append(
            event({"type": "response.output_item.added", "output_index": index, "item": started})
        )
        if item.get("type") == "message":
            for content_index, part in enumerate(item.get("content", [])):
                if part.get("type") != "output_text":
                    continue
                base = {"item_id": item["id"], "output_index": index, "content_index": content_index}
                events.append(event({"type": "response.content_part.added", **base, "part": {**part, "text": ""}}))
                events.append(event({"type": "response.output_text.delta", **base, "delta": part["text"], "logprobs": []}))
                events.append(event({"type": "response.output_text.done", **base, "text": part["text"], "logprobs": []}))
                events.append(event({"type": "response.content_part.done", **base, "part": part}))
        if item.get("type") == "function_call":
            base = {"item_id": item["id"], "output_index": index}
            events.append(event({"type": "response.function_call_arguments.delta", **base, "delta": item["arguments"]}))
            events.append(event({"type": "response.function_call_arguments.done", **base, "arguments": item["arguments"]}))
        events.append(event({"type": "response.output_item.done", "output_index": index, "item": item}))
    events.append(event({"type": "response.completed", "response": response}))
    return events


def chat_chunks(reply: dict) -> list:
    """The scripted Chat Completions object as streamed chunks followed by `[DONE]`."""
    chunks = []
    for choice in reply.get("choices", []):
        message = choice.get("message", {})
        delta = {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}
        chunks.append(
            {
                "id": reply["id"],
                "object": "chat.completion.chunk",
                "created": reply["created"],
                "model": reply["model"],
                "choices": [{"index": choice.get("index", 0), "delta": delta, "finish_reason": None}],
            }
        )
        chunks.append(
            {
                "id": reply["id"],
                "object": "chat.completion.chunk",
                "created": reply["created"],
                "model": reply["model"],
                "choices": [{"index": choice.get("index", 0), "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
                "usage": reply.get("usage"),
            }
        )
    chunks.append("[DONE]")
    return chunks


class State:
    def __init__(self, out: Path, scenarios: Path) -> None:
        self.out = out
        self.scenarios = scenarios
        self.lock = threading.Lock()
        self.used: dict[tuple[str, str, str], set[int]] = {}
        self.scripts: dict[str, list] = {}

    def script(self, scenario: str) -> list:
        if scenario not in self.scripts:
            path = self.scenarios / scenario / "responses.json"
            self.scripts[scenario] = json.loads(path.read_text()) if path.exists() else []
        return self.scripts[scenario]

    def record(self, scenario: str, sdk: str, line: dict) -> None:
        folder = self.out / scenario
        folder.mkdir(parents=True, exist_ok=True)
        with self.lock, (folder / f"{sdk}.jsonl").open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    def next_entry(self, scenario: str, sdk: str, family: str, body) -> tuple[int, dict | None]:
        """The first unused script entry whose `when` matches the request (entries without
        `when` match anything). Returns its index and the entry, or (-1, None) when the
        script is exhausted."""
        instructions = body.get("instructions") if isinstance(body, dict) else None
        with self.lock:
            key = (scenario, sdk, family)
            used = self.used.setdefault(key, set())
            for index, entry in enumerate(self.script(scenario)):
                if index in used:
                    continue
                when = entry.get("when", {})
                needle = when.get("instructions_contains")
                if needle is not None and needle not in (instructions or ""):
                    continue
                used.add(index)
                return index, entry
            return -1, None


STATE: State


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # keep stdout for the port line
        sys.stderr.write(f"{self.command} {self.path} {fmt % args}\n")

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        match = PATH_RE.match(self.path)
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not match:
            self._send(404, json.dumps({"error": f"unroutable path {self.path}"}).encode())
            return
        scenario, sdk, rest = match["scenario"], match["sdk"], match["rest"]
        family = family_of(rest)
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            body = {"_raw": raw.decode("utf-8", "replace")}
        headers = {k.lower(): v for k, v in self.headers.items()}
        if "authorization" in headers:
            headers["authorization"] = "present"
        STATE.record(
            scenario, sdk, {"method": "POST", "path": "/" + rest, "headers": headers, "body": body}
        )
        if family is None:
            self._send(404, json.dumps({"error": f"unknown path family {rest}"}).encode())
            return
        if family == "traces/ingest":
            self._send(204, b"")
            return
        seq, entry = STATE.next_entry(scenario, sdk, family, body)
        if entry is None:
            msg = f"scenario {scenario}: no unused scripted response matches this {family} request ({sdk})"
            self._send(500, json.dumps({"error": msg}).encode())
            return
        entry = {k: v for k, v in entry.items() if k != "when"}
        if "status" in entry and entry["status"] != 200:
            self._send(entry["status"], json.dumps(entry.get("body", {})).encode())
            return
        streaming = isinstance(body, dict) and bool(body.get("stream"))
        if family == "responses":
            reply = complete_response(entry, body or {}, seq)
            if streaming:
                self._send_sse(response_events(reply))
                return
        else:
            reply = complete_chat(entry, body or {}, seq)
            if streaming:
                self._send_sse(chat_chunks(reply))
                return
        self._send(200, json.dumps(reply).encode())

    def _send_sse(self, events: list) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for event in events:
            if event == "[DONE]":
                payload = b"data: [DONE]\n\n"
            else:
                event_type = event.get("type", "")
                payload = f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        self._send(404, json.dumps({"error": "GET is not scripted"}).encode())


def main() -> None:
    global STATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenarios", required=True)
    args = parser.parse_args()
    STATE = State(Path(args.out), Path(args.scenarios))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PORT={server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        time.sleep(0)


if __name__ == "__main__":
    main()
