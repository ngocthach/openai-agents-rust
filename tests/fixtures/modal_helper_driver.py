"""Drives the bundled Modal helper's operations against fake `modal` SDK objects.

Usage: `python3 modal_helper_driver.py <path to helper.py> <case>`; prints one JSON object.
The cases exercise helper code that cannot run without the real SDK: chunked uploads and the
exception classification behind the `retryable` flag.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import types


def _load_helper(path: str):
    spec = importlib.util.spec_from_file_location("modal_helper_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _aio(fn):
    async def run(*args, **kwargs):
        return fn(*args, **kwargs)

    return types.SimpleNamespace(aio=run)


class _Stdin:
    """A stdin that, like Modal's, refuses a write that overflows its buffer until drained."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.buffered = 0
        self.chunks: list[int] = []
        self.eof = 0
        self.drain = types.SimpleNamespace(aio=self._drain)

    def write(self, data) -> None:
        if self.buffered + len(data) > self.limit:
            raise BufferError("Buffer size exceed limit. Call drain to flush the buffer.")
        self.buffered += len(data)

    def write_eof(self) -> None:
        self.eof += 1

    async def _drain(self) -> None:
        self.chunks.append(self.buffered)
        self.buffered = 0


def _write_chunks(helper) -> dict:
    stdin = _Stdin(limit=8 * 1024 * 1024)
    commands: list[list[str]] = []

    class Process:
        def __init__(self) -> None:
            self.stdin = stdin
            self.stderr = types.SimpleNamespace(read=_aio(lambda: b""))
            self.wait = _aio(lambda: 0)

    class Sandbox:
        def __init__(self) -> None:
            self.exec = types.SimpleNamespace(aio=self._exec)

        async def _exec(self, *argv, **kwargs):
            commands.append(list(argv))
            return Process()

    sandbox = Sandbox()

    async def from_id(_sandbox_id):
        return sandbox

    modal = types.SimpleNamespace(
        Sandbox=types.SimpleNamespace(from_id=types.SimpleNamespace(aio=from_id))
    )
    payload = b"a" * (20 * 1024 * 1024 + 3)
    response = asyncio.run(
        helper._op_write_file(
            modal,
            {
                "sandbox_id": "sb-1",
                "path": "/workspace/nested/file.bin",
                "data": base64.b64encode(payload).decode("ascii"),
            },
        )
    )
    return {
        "response": response,
        "chunks": stdin.chunks,
        "eof": stdin.eof,
        "commands": commands,
        "total": sum(stdin.chunks),
    }


def _retryability(helper) -> dict:
    def make(name: str, **attrs):
        return type(name, (Exception,), attrs)("boom")

    out = {}
    for name in (
        "NotFoundError",
        "InvalidError",
        "AuthError",
        "PermissionDeniedError",
        "SandboxFilesystemNotFoundError",
        "ConnectionError",
        "InternalError",
        "ServiceError",
    ):
        out[name] = helper._modal_retryability(make(name))
    out["status_503"] = helper._modal_retryability(make("WeirdError", status_code=503))
    out["status_404"] = helper._modal_retryability(make("WeirdError", status_code=404))
    out["unknown"] = helper._modal_retryability(make("WeirdError"))
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = make("NotFoundError")
    out["wrapped_not_found"] = helper._modal_retryability(wrapper)
    return out


def _read_file(helper) -> dict:
    def run(exit_code: int, stderr: bytes) -> dict:
        class Process:
            def __init__(self) -> None:
                self.stdout = types.SimpleNamespace(read=_aio(lambda: b""))
                self.stderr = types.SimpleNamespace(read=_aio(lambda: stderr))
                self.wait = _aio(lambda: exit_code)

        class Sandbox:
            def __init__(self) -> None:
                self.exec = types.SimpleNamespace(aio=self._exec)

            async def _exec(self, *argv, **kwargs):
                return Process()

        async def from_id(_sandbox_id):
            return Sandbox()

        modal = types.SimpleNamespace(
            Sandbox=types.SimpleNamespace(from_id=types.SimpleNamespace(aio=from_id))
        )
        return asyncio.run(
            helper._op_read_file(modal, {"sandbox_id": "sb-1", "path": "/workspace/target.txt"})
        )

    return {
        "missing": run(1, b"cat: /workspace/target.txt: No such file or directory"),
        "denied": run(1, b"cat: /workspace/target.txt: Permission denied"),
        "silent": run(2, b""),
    }


def _main() -> None:
    helper = _load_helper(sys.argv[1])
    case = sys.argv[2]
    result = {
        "write_chunks": _write_chunks,
        "retryability": _retryability,
        "read_file": _read_file,
    }[case](helper)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    _main()
