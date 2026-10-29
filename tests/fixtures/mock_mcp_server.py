"""Minimal MCP stdio server (stdlib only) used by the Rust MCP integration test."""
import json
import sys

TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "danger_delete",
        "description": "Delete everything",
        "inputSchema": {"type": "object"},
    },
]


def reply(msg_id, result=None, error=None):
    payload = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    msg_id = msg.get("id")
    if msg_id is None:
        continue  # notification
    params = msg.get("params") or {}
    if method == "initialize":
        reply(msg_id, {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}, "prompts": {}},
            "serverInfo": {"name": "mock", "version": "1.0.0"},
        })
    elif method == "tools/list":
        reply(msg_id, {"tools": TOOLS})
    elif method == "tools/call":
        args = params.get("arguments") or {}
        if params.get("name") == "add":
            total = args.get("a", 0) + args.get("b", 0)
            reply(msg_id, {"content": [{"type": "text", "text": str(total)}], "isError": False})
        else:
            reply(msg_id, {"content": [{"type": "text", "text": "deleted"}], "isError": False})
    elif method == "prompts/list":
        reply(msg_id, {"prompts": [{"name": "review", "description": "Review code"}]})
    elif method == "prompts/get":
        reply(msg_id, {"messages": [{"role": "user", "content": {"type": "text", "text": "Review carefully"}}]})
    elif method == "ping":
        reply(msg_id, {})
    else:
        reply(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})
