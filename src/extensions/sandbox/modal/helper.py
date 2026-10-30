"""JSON bridge between the Rust `ModalSandboxClient` and the official `modal` Python SDK.

Modal's control plane speaks gRPC and publishes no REST surface, so the Rust port drives this
helper instead of re-implementing the protocol. The Rust side writes one JSON request object to
stdin and reads one JSON response object from stdout; every invocation is stateless and addresses
the sandbox by its Modal id, so concurrent calls never share process state.

Requests (`op` selects the operation):

    {"op": "create", "app_name": ..., "image_tag"|"image_id": ..., "workdir": ..., "env": {...},
     "encrypted_ports": [...], "gpu": ..., "cpu": ..., "memory": ..., "timeout": 300,
     (`cpu` and `memory` are a number, or a two-element `[request, limit]` array)
     "idle_timeout": ..., "use_sleep_cmd": true, "image_builder_version": ...,
     "volumes": [{"mount_path": ..., "bucket_name": ..., ...}]}
        -> {"sandbox_id": ..., "image_id": ...}
    {"op": "exec", "sandbox_id": ..., "argv": [...], "timeout": ..., "workdir": ...,
     "env": {...}}
        -> {"exit_code": 0, "stdout": "<base64>", "stderr": "<base64>"}
    {"op": "read_file", "sandbox_id": ..., "path": ...} -> {"data": "<base64>"}
    {"op": "write_file", "sandbox_id": ..., "path": ..., "data": "<base64>"} -> {"ok": true}
    {"op": "poll", "sandbox_id": ...} -> {"running": true}
    {"op": "terminate", "sandbox_id": ...} -> {"ok": true}
    {"op": "snapshot_filesystem", "sandbox_id": ..., "timeout": 60} -> {"image_id": ...}
    {"op": "snapshot_directory", "sandbox_id": ..., "path": ..., "timeout": 60}
        -> {"image_id": ...}
    {"op": "mount_image", "sandbox_id": ..., "path": ..., "image_id": ..., "timeout": 60}
        -> {"ok": true}
    {"op": "tunnels", "sandbox_id": ..., "timeout": 10}
        -> {"tunnels": {"<sandbox port>": {"host": ..., "port": ...}}}
    {"op": "pty", "sandbox_id": ..., "argv": [...], "tty": true, "workdir": ...}

`pty` is the one streaming operation: its request arrives as the first command-line argument
rather than on stdin, and instead of a JSON response it bridges the process's output to stdout and
this helper's stdin to the process, so the Rust side can drive it through its own PTY manager. Every other failure is reported as `{"error": "..."}` with a non-zero exit status.

Credentials reach this helper only through the request body and the usual Modal environment
variables (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_ENVIRONMENT`); nothing is written to
disk and nothing is echoed back in a response.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import sys
import threading
from typing import Any


def _fail(message: str) -> None:
    json.dump({"error": message}, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()
    raise SystemExit(1)


def _import_modal() -> Any:
    try:
        import modal
    except ImportError as error:  # pragma: no cover - depends on the user's environment
        _fail(
            "the `modal` Python package is required by ModalSandboxClient: "
            f"install it in the helper interpreter ({sys.executable}): {error}"
        )
    return modal


def _build_image(modal: Any, request: dict[str, Any]) -> Any:
    image_id = request.get("image_id")
    if image_id:
        image = modal.Image.from_id(image_id)
    else:
        image = modal.Image.from_registry(request.get("image_tag") or "python:3.14-slim")
    if request.get("use_sleep_cmd", True):
        image = image.cmd(["sleep", "infinity"])
    return image


def _build_volumes(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    volumes: dict[str, Any] = {}
    for spec in request.get("volumes") or []:
        secret = None
        credentials = spec.get("credentials")
        if credentials:
            secret = modal.Secret.from_dict(dict(credentials))
        elif spec.get("secret_name"):
            kwargs: dict[str, Any] = {}
            if spec.get("secret_environment_name"):
                kwargs["environment_name"] = spec["secret_environment_name"]
            secret = modal.Secret.from_name(spec["secret_name"], **kwargs)
        volumes[spec["mount_path"]] = modal.CloudBucketMount(
            bucket_name=spec["bucket_name"],
            bucket_endpoint_url=spec.get("bucket_endpoint_url"),
            key_prefix=spec.get("key_prefix"),
            secret=secret,
            read_only=bool(spec.get("read_only", True)),
        )
    return volumes


def _resource_pair(value: Any) -> Any:
    """Turn a JSON `[request, limit]` array into the tuple Modal's resource conversion needs.

    `modal.Sandbox.create` accepts `cpu`/`memory` as a number or as a `(request, limit)` tuple,
    and the conversion behind it tests `isinstance(value, tuple)`. JSON has no tuple, so a pair
    arrives here as a list, which matches neither branch and would silently drop both the request
    and the limit. A pair of any other length is left alone so Modal reports it itself.
    """

    if isinstance(value, list) and len(value) == 2:
        return tuple(value)
    return value


async def _op_create(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    builder_version = request.get("image_builder_version")
    previous_builder_version = os.environ.get("MODAL_IMAGE_BUILDER_VERSION")
    if builder_version:
        os.environ["MODAL_IMAGE_BUILDER_VERSION"] = builder_version
    try:
        app = await modal.App.lookup.aio(request["app_name"], create_if_missing=True)
        image = _build_image(modal, request)
        create = modal.Sandbox.create.aio(
            app=app,
            image=image,
            workdir=request.get("workdir"),
            env=request.get("env") or {},
            encrypted_ports=request.get("encrypted_ports") or [],
            volumes=_build_volumes(modal, request),
            gpu=request.get("gpu"),
            cpu=_resource_pair(request.get("cpu")),
            memory=_resource_pair(request.get("memory")),
            timeout=request.get("timeout"),
            idle_timeout=request.get("idle_timeout"),
        )
        create_timeout = request.get("create_timeout")
        sandbox = (
            await asyncio.wait_for(create, timeout=create_timeout)
            if create_timeout
            else await create
        )
    finally:
        if builder_version:
            if previous_builder_version is None:
                os.environ.pop("MODAL_IMAGE_BUILDER_VERSION", None)
            else:
                os.environ["MODAL_IMAGE_BUILDER_VERSION"] = previous_builder_version
    return {
        "sandbox_id": sandbox.object_id,
        "image_id": getattr(image, "object_id", None),
    }


async def _sandbox(modal: Any, request: dict[str, Any]) -> Any:
    return await modal.Sandbox.from_id.aio(request["sandbox_id"])


async def _op_exec(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    sandbox = await _sandbox(modal, request)
    timeout = request.get("timeout")
    kwargs: dict[str, Any] = {"text": False}
    if timeout is not None:
        # Modal's exec timeout is whole seconds and terminates the command server-side.
        kwargs["timeout"] = max(1, int(timeout))
    if request.get("workdir"):
        kwargs["workdir"] = request["workdir"]
    if request.get("env"):
        kwargs["env"] = dict(request["env"])
    process = await sandbox.exec.aio(*request["argv"], **kwargs)
    stdout = await process.stdout.read.aio()
    stderr = await process.stderr.read.aio()
    exit_code = await process.wait.aio()
    return {
        "exit_code": int(exit_code or 0),
        "stdout": base64.b64encode(stdout or b"").decode("ascii"),
        "stderr": base64.b64encode(stderr or b"").decode("ascii"),
    }


async def _op_read_file(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    sandbox = await _sandbox(modal, request)
    path = request["path"]
    process = await sandbox.exec.aio(
        "sh", "-lc", f"cat -- {shlex.quote(path)}", text=False
    )
    data = await process.stdout.read.aio()
    stderr = await process.stderr.read.aio()
    exit_code = int(await process.wait.aio() or 0)
    if exit_code != 0:
        detail = (stderr or b"").decode("utf-8", "replace").strip()
        code = "not_found" if "No such file" in detail else "read_failed"
        return {"error": detail or f"cat exited with {exit_code}", "code": code}
    return {"data": base64.b64encode(data or b"").decode("ascii")}


async def _op_write_file(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    sandbox = await _sandbox(modal, request)
    path = request["path"]
    payload = base64.b64decode(request.get("data") or "")
    parent = os.path.dirname(path) or "/"
    mkdir = await sandbox.exec.aio("mkdir", "-p", "--", parent, text=False)
    await mkdir.wait.aio()
    # The destination is a positional argument, never interpolated into the script, so a path
    # holding shell metacharacters cannot inject a command.
    process = await sandbox.exec.aio(
        "sh", "-c", 'cat > "$1"', "sh", path, text=False
    )
    view = memoryview(payload)
    chunk_size = 8 * 1024 * 1024
    for start in range(0, len(view), chunk_size):
        process.stdin.write(view[start : start + chunk_size])
        await process.stdin.drain.aio()
    process.stdin.write_eof()
    await process.stdin.drain.aio()
    exit_code = int(await process.wait.aio() or 0)
    if exit_code != 0:
        stderr = await process.stderr.read.aio()
        return {
            "error": (stderr or b"").decode("utf-8", "replace").strip()
            or f"write exited with {exit_code}",
            "code": "write_failed",
        }
    return {"ok": True}


async def _op_poll(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    try:
        sandbox = await _sandbox(modal, request)
        # `poll()` returns an exit code once the sandbox is gone, and None while it runs.
        return {"running": await sandbox.poll.aio() is None}
    except Exception:
        return {"running": False}


async def _op_terminate(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    try:
        sandbox = await _sandbox(modal, request)
        await sandbox.terminate.aio()
    except Exception as error:
        return {"ok": False, "detail": str(error)}
    return {"ok": True}


async def _op_snapshot_filesystem(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    sandbox = await _sandbox(modal, request)
    timeout = request.get("timeout")
    snapshot = await (
        sandbox.snapshot_filesystem.aio(int(timeout))
        if timeout
        else sandbox.snapshot_filesystem.aio()
    )
    image_id = (
        snapshot
        if isinstance(snapshot, str)
        else getattr(snapshot, "object_id", None) or getattr(snapshot, "id", None)
    )
    if not isinstance(image_id, str) or not image_id:
        return {"error": "snapshot_filesystem returned no image id", "code": "snapshot_failed"}
    return {"image_id": image_id}


async def _op_snapshot_directory(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    sandbox = await _sandbox(modal, request)
    timeout = request.get("timeout")
    snap = sandbox.snapshot_directory.aio(request["path"])
    image = await (asyncio.wait_for(snap, timeout=timeout) if timeout else snap)
    image_id = (
        image
        if isinstance(image, str)
        else getattr(image, "object_id", None) or getattr(image, "id", None)
    )
    if not isinstance(image_id, str) or not image_id:
        return {"error": "snapshot_directory returned no image id", "code": "snapshot_failed"}
    return {"image_id": image_id}


async def _op_mount_image(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    sandbox = await _sandbox(modal, request)
    image = modal.Image.from_id(request["image_id"])
    timeout = request.get("timeout")
    mount = sandbox.mount_image.aio(request["path"], image)
    await (asyncio.wait_for(mount, timeout=timeout) if timeout else mount)
    return {"ok": True}


async def _op_tunnels(modal: Any, request: dict[str, Any]) -> dict[str, Any]:
    sandbox = await _sandbox(modal, request)
    timeout = request.get("timeout")
    lookup = sandbox.tunnels.aio()
    tunnels = await (asyncio.wait_for(lookup, timeout=timeout) if timeout else lookup)
    if not isinstance(tunnels, dict):
        return {"error": "tunnels returned no mapping", "code": "tunnels_lookup_failed"}
    # JSON object keys are strings, so the sandbox-side port is stringified here and parsed back
    # on the Rust side.
    resolved: dict[str, Any] = {}
    for port, tunnel in tunnels.items():
        host = getattr(tunnel, "host", None)
        host_port = getattr(tunnel, "port", None)
        if isinstance(host, str) and host and isinstance(host_port, int):
            resolved[str(port)] = {"host": host, "port": host_port}
    return {"tunnels": resolved}


async def _op_pty(modal: Any, request: dict[str, Any]) -> None:
    """Bridge a Modal process to this helper's own stdin and stdout."""
    sandbox = await _sandbox(modal, request)
    kwargs: dict[str, Any] = {"text": False, "pty": bool(request.get("tty", False))}
    if request.get("workdir"):
        kwargs["workdir"] = request["workdir"]
    if request.get("timeout") is not None:
        kwargs["timeout"] = max(1, int(request["timeout"]))
    process = await sandbox.exec.aio(*request["argv"], **kwargs)

    loop = asyncio.get_running_loop()
    stop = threading.Event()

    def pump_stdin() -> None:
        stream = sys.stdin.buffer
        while not stop.is_set():
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                break
            future = asyncio.run_coroutine_threadsafe(_write_stdin(process, chunk), loop)
            try:
                future.result()
            except Exception:
                break

    async def _write_stdin(process: Any, chunk: bytes) -> None:
        process.stdin.write(chunk)
        await process.stdin.drain.aio()

    reader = threading.Thread(target=pump_stdin, daemon=True)
    reader.start()
    try:
        async for chunk in process.stdout:
            sys.stdout.buffer.write(chunk if isinstance(chunk, bytes) else str(chunk).encode())
            sys.stdout.buffer.flush()
    finally:
        stop.set()
    exit_code = int(await process.wait.aio() or 0)
    sys.stdout.buffer.flush()
    raise SystemExit(exit_code)


_OPS = {
    "create": _op_create,
    "exec": _op_exec,
    "read_file": _op_read_file,
    "write_file": _op_write_file,
    "poll": _op_poll,
    "terminate": _op_terminate,
    "snapshot_filesystem": _op_snapshot_filesystem,
    "snapshot_directory": _op_snapshot_directory,
    "mount_image": _op_mount_image,
    "tunnels": _op_tunnels,
}


async def _main() -> None:
    # The `pty` op passes its request on the command line, because the process's stdin belongs to
    # the terminal the caller is driving; every other op reads one request line from stdin.
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.readline()
    if not raw.strip():
        _fail("the modal helper received an empty request")
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as error:
        _fail(f"the modal helper received a malformed request: {error}")
    op = request.get("op")
    modal = _import_modal()
    if op == "pty":
        await _op_pty(modal, request)
        return
    handler = _OPS.get(op)
    if handler is None:
        _fail(f"unknown modal helper op: {op!r}")
    try:
        response = await handler(modal, request)
    except Exception as error:
        _fail(f"{type(error).__name__}: {error}")
    json.dump(response, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(_main())
