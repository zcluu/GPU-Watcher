from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import stat
import struct
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from watchgpu.protocol import PROTOCOL_VERSION, SupervisorProtocol

MAX_MESSAGE_BYTES = 1024 * 1024
DEFAULT_RPC_TIMEOUT_SECONDS = 10.0
RESPONSE_CACHE_SIZE = 1024


class IPCError(RuntimeError):
    pass


class UnixSocketServer:
    def __init__(
        self,
        path: Path,
        protocol: SupervisorProtocol,
        *,
        allowed_uid: int | None = None,
    ) -> None:
        self.path = path
        self._protocol = protocol
        self._allowed_uid = os.getuid() if allowed_uid is None else allowed_uid
        self._server: asyncio.AbstractServer | None = None
        self._lock_fd: int | None = None
        self._responses: OrderedDict[str, tuple[bytes, dict[str, Any]]] = OrderedDict()

    async def start(self) -> None:
        if self._server is not None:
            raise IPCError("server is already running")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._acquire_instance_lock()
        try:
            await self._prepare_socket_path()
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                path=self.path,
                limit=MAX_MESSAGE_BYTES,
            )
            self.path.chmod(0o600)
        except BaseException:
            self._release_instance_lock()
            raise

    async def close(self) -> None:
        owns_instance = self._lock_fd is not None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if owns_instance:
            with suppress(FileNotFoundError):
                self.path.unlink()
        self._release_instance_lock()

    async def serve_forever(self) -> None:
        if self._server is None:
            raise IPCError("server is not running")
        await self._server.serve_forever()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            if _peer_uid(writer) != self._allowed_uid:
                await _write_response(
                    writer,
                    _error_response(None, "PERMISSION_DENIED", "peer UID is not allowed"),
                )
                return
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    await _write_response(
                        writer,
                        _error_response(None, "MESSAGE_TOO_LARGE", "message is too large"),
                    )
                    return
                if not line:
                    return
                if len(line) > MAX_MESSAGE_BYTES:
                    await _write_response(
                        writer,
                        _error_response(None, "MESSAGE_TOO_LARGE", "message is too large"),
                    )
                    return
                response = self._dispatch_line(line)
                await _write_response(writer, response)
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    def _dispatch_line(self, line: bytes) -> dict[str, Any]:
        request_id: str | None = None
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError("message must be a JSON object")
            raw_request_id = decoded.get("request_id")
            request_id = raw_request_id if isinstance(raw_request_id, str) else None
            fingerprint = json.dumps(
                decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            if request_id is not None and request_id in self._responses:
                previous_fingerprint, previous_response = self._responses[request_id]
                if previous_fingerprint != fingerprint:
                    return _error_response(
                        request_id,
                        "REQUEST_ID_CONFLICT",
                        "request_id was already used with different content",
                    )
                self._responses.move_to_end(request_id)
                return previous_response
            response = self._protocol.handle(decoded)
            if request_id is not None:
                self._responses[request_id] = (fingerprint, response)
                self._responses.move_to_end(request_id)
                while len(self._responses) > RESPONSE_CACHE_SIZE:
                    self._responses.popitem(last=False)
            return response
        except (ValueError, TypeError) as exc:
            return _error_response(request_id, type(exc).__name__, str(exc))
        except Exception as exc:
            return _error_response(request_id, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")

    async def _prepare_socket_path(self) -> None:
        try:
            mode = self.path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise IPCError(f"refusing to replace non-socket path: {self.path}")
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.path), timeout=1.0
            )
        except (ConnectionError, OSError, TimeoutError):
            self.path.unlink()
            return
        writer.close()
        with suppress(ConnectionError):
            await writer.wait_closed()
        raise IPCError(f"another WatchGPU daemon is already listening on {self.path}")

    def _acquire_instance_lock(self) -> None:
        lock_path = self.path.with_suffix(".lock")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise IPCError(f"cannot safely open instance lock: {lock_path}") from exc
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != self._allowed_uid:
                raise IPCError("instance lock is not a current-user regular file")
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError, IPCError) as exc:
            os.close(fd)
            if isinstance(exc, IPCError):
                raise
            raise IPCError("another WatchGPU daemon holds the instance lock") from exc
        self._lock_fd = fd

    def _release_instance_lock(self) -> None:
        fd, self._lock_fd = self._lock_fd, None
        if fd is None:
            return
        with suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class AsyncWatchGPUClient:
    def __init__(self, path: Path, *, timeout_seconds: float = DEFAULT_RPC_TIMEOUT_SECONDS) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds

    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        rpc_id = request_id or str(uuid.uuid4())
        message = {
            "version": PROTOCOL_VERSION,
            "request_id": rpc_id,
            "method": method,
            "params": dict(params or {}),
        }
        return await asyncio.wait_for(
            self._call(message, rpc_id), timeout=self.timeout_seconds
        )

    async def _call(self, message: Mapping[str, Any], rpc_id: str) -> dict[str, Any]:
        reader, writer = await asyncio.open_unix_connection(self.path)
        try:
            writer.write(_encode(message))
            await writer.drain()
            line = await reader.readline()
            if not line:
                raise IPCError("server closed the connection without a response")
            response = json.loads(line)
        finally:
            writer.close()
            await writer.wait_closed()

        if not isinstance(response, dict):
            raise IPCError("server returned a non-object response")
        if response.get("request_id") != rpc_id:
            raise IPCError("server response request_id does not match")
        if response.get("ok") is not True:
            error = response.get("error")
            if isinstance(error, dict):
                raise IPCError(str(error.get("message", "request failed")))
            raise IPCError("request failed")
        result = response.get("result")
        if not isinstance(result, dict):
            raise IPCError("server returned an invalid result")
        return result


def _peer_uid(writer: asyncio.StreamWriter) -> int:
    peer_socket = writer.get_extra_info("socket")
    if peer_socket is None:
        raise IPCError("connection has no peer socket")
    credentials = peer_socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    _pid, uid, _gid = struct.unpack("3i", credentials)
    return int(uid)


async def _write_response(
    writer: asyncio.StreamWriter, response: Mapping[str, Any]
) -> None:
    writer.write(_encode(response))
    await writer.drain()


def _encode(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode()


def _error_response(
    request_id: str | None, code: str, message: str
) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": False,
        "error": {"code": code, "message": message},
    }
