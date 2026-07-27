from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from textual.widgets import Static, TabbedContent, TabPane

from watchgpu.console import WatchGPUConsole
from watchgpu.ipc import IPCError


class ReconnectingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self._attempt = 0

    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, dict(params or {})))
        self._attempt += 1
        if self._attempt == 1:
            raise IPCError("socket is restarting")
        return {
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-0",
                    "name": "A40",
                    "total_mib": 46_068,
                    "free_mib": 2048,
                    "reserved_mib": 30_000,
                    "leased_mib": 8192,
                    "utilization_percent": 12,
                    "worker_state": "HOLDING",
                }
            ],
            "leases": [],
        }

    async def aclose(self) -> None:
        self.closed = True


class InteractiveClient:
    def __init__(
        self,
        *,
        policy_status: str = "APPLIED",
        leases: list[dict[str, Any]] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.policy_status = policy_status
        self.leases = list(leases or [])

    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del request_id
        payload = dict(params or {})
        self.calls.append((method, payload))
        if method == "status.get":
            return {
                "gpus": [
                    {
                        "index": 0,
                        "uuid": "GPU-0",
                        "name": "A40",
                        "total_mib": 46_068,
                        "free_mib": 2048,
                        "reserved_mib": 30_000,
                        "leased_mib": 0,
                        "utilization_percent": 5,
                        "worker_state": "HOLDING",
                    },
                    {
                        "index": 1,
                        "uuid": "GPU-1",
                        "name": "A40",
                        "total_mib": 46_068,
                        "free_mib": 3072,
                        "reserved_mib": 29_000,
                        "leased_mib": 0,
                        "utilization_percent": 6,
                        "worker_state": "HOLDING",
                    },
                ],
                "leases": self.leases,
                "policy": {
                    "revision": 4,
                    "status": "APPLIED",
                    "config": {
                        "leave_free_mib": 2048,
                        "maintenance_duty_cycle_percent": 5,
                        "cpu_budget_percent": 100,
                        "gpus": [
                            {"selector": "GPU-0"},
                            {"selector": "GPU-1"},
                        ],
                    },
                },
            }
        if method == "policy.apply":
            return {
                "status": self.policy_status,
                "revision": 5,
                "config": payload["config"],
                "reason": (
                    None
                    if self.policy_status == "APPLIED"
                    else "waiting for worker-safe reconciliation"
                ),
            }
        return {"workers": [{"gpu_uuid": payload.get("gpu_uuid")}]}


def test_console_reconnects_at_narrow_width_and_exit_only_closes_client() -> None:
    async def scenario() -> None:
        client = ReconnectingClient()
        app = WatchGPUConsole(client, poll_interval_seconds=0.01)

        async with app.run_test(size=(40, 20)) as pilot:
            await pilot.pause(0.08)

            assert app.connected
            assert len(app.query(TabPane)) == 6
            assert "GPU 0 [HOLDING]" in str(
                app.query_one("#gpus-view", Static).content
            )
            assert app.query_one("#connection-status", Static).content == (
                "WatchGPU | connected | policy r?"
            )

        assert client.closed
        assert len(client.calls) >= 2
        assert all(call == ("status.get", {}) for call in client.calls)

    asyncio.run(scenario())


def test_active_lease_is_visible_in_the_tasks_tab() -> None:
    async def scenario() -> None:
        client = InteractiveClient(
            leases=[
                {
                    "lease_id": "lease-active",
                    "task_name": "test",
                    "state": "ACTIVE",
                    "client_pid": 3306015,
                    "gpu_uuids": ["GPU-1"],
                    "memory_per_gpu_mib": 2048,
                    "released_by_gpu_mib": {"GPU-1": 2048},
                    "created_at": 2849004.6,
                    "expires_at": 2849664.9,
                }
            ]
        )
        app = WatchGPUConsole(client, poll_interval_seconds=60)

        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("2")
            await pilot.pause()

            assert app.query_one("#main-tabs", TabbedContent).active == "tasks-tab"
            assert app.query_one("#tasks-tab", TabPane).region.height > 0
            assert "test [ACTIVE]" in str(
                app.query_one("#tasks-view", Static).content
            )

    asyncio.run(scenario())


def test_operator_selects_a_gpu_and_pauses_it_with_keyboard() -> None:
    async def scenario() -> None:
        client = InteractiveClient()
        app = WatchGPUConsole(client, poll_interval_seconds=60)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("g", "p")
            await pilot.pause()

        assert ("worker.pause", {"gpu_uuid": "GPU-0"}) in client.calls

    asyncio.run(scenario())


def test_operator_resumes_the_selected_gpu_with_keyboard() -> None:
    async def scenario() -> None:
        client = InteractiveClient()
        app = WatchGPUConsole(client, poll_interval_seconds=60)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("g", "u")
            await pilot.pause()

        assert ("worker.resume", {"gpu_uuid": "GPU-0"}) in client.calls

    asyncio.run(scenario())


def test_operator_releases_memory_from_the_selected_gpu_with_keyboard() -> None:
    async def scenario() -> None:
        client = InteractiveClient()
        app = WatchGPUConsole(client, poll_interval_seconds=60)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("g", "x")
            await pilot.pause()
            await pilot.press(*"2GiB", "enter")
            await pilot.pause()

        assert (
            "worker.release",
            {"gpu_uuid": "GPU-0", "memory_mib": 2048},
        ) in client.calls

    asyncio.run(scenario())


def test_restart_shortcut_requires_keyboard_confirmation() -> None:
    async def scenario() -> None:
        client = InteractiveClient()
        app = WatchGPUConsole(client, poll_interval_seconds=60)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            assert not any(method == "daemon.restart" for method, _params in client.calls)
            await pilot.press("y")
            await pilot.pause()

        assert ("daemon.restart", {}) in client.calls

    asyncio.run(scenario())


def test_stop_shortcut_requires_confirmation_and_always_releases() -> None:
    async def scenario() -> None:
        client = InteractiveClient()
        app = WatchGPUConsole(client, poll_interval_seconds=60)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert not any(method == "daemon.stop" for method, _params in client.calls)
            await pilot.press("y")
            await pilot.pause()

        assert ("daemon.stop", {"release": True}) in client.calls

    asyncio.run(scenario())


def test_policy_editor_applies_with_revision_and_displays_result() -> None:
    async def scenario() -> None:
        client = InteractiveClient()
        app = WatchGPUConsole(client, poll_interval_seconds=60)

        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("e")
            await pilot.pause()
            await pilot.press("backspace", "backspace", "backspace", "backspace")
            await pilot.press(*"3GiB", "f5")
            await pilot.pause()

            action_status = str(app.query_one("#action-status", Static).content)
            assert "APPLIED" in action_status
            assert "r5" in action_status

        policy_calls = [params for method, params in client.calls if method == "policy.apply"]
        assert policy_calls == [
            {
                "config": {
                    "leave_free_mib": 3072,
                    "maintenance_duty_cycle_percent": 5,
                    "cpu_budget_percent": 100,
                    "gpus": [
                        {"selector": "GPU-0"},
                        {"selector": "GPU-1"},
                    ],
                },
                "expected_revision": 4,
                "save": False,
            }
        ]

    asyncio.run(scenario())


def test_policy_editor_displays_pending_and_rejected_results() -> None:
    async def scenario(status: str) -> None:
        client = InteractiveClient(policy_status=status)
        app = WatchGPUConsole(client, poll_interval_seconds=60)

        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            await pilot.press("e", "f5")
            await pilot.pause()

            action_status = str(app.query_one("#action-status", Static).content)
            assert status in action_status
            assert "waiting for worker-safe reconciliation" in action_status

    asyncio.run(scenario("PENDING"))
    asyncio.run(scenario("REJECTED"))
