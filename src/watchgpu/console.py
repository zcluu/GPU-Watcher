from __future__ import annotations

import asyncio
import copy
import inspect
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static, TabbedContent, TabPane

from watchgpu.ipc import AsyncWatchGPUClient
from watchgpu.units import parse_user_capacity_mib

STATUS_METHOD = "status.get"
PAUSE_METHOD = "worker.pause"
RESUME_METHOD = "worker.resume"
RELEASE_METHOD = "worker.release"
POLICY_APPLY_METHOD = "policy.apply"
RESTART_METHOD = "daemon.restart"
STOP_METHOD = "daemon.stop"

_SECTIONS = ("gpus", "tasks", "processes", "profiles", "policy", "events")


class ConsoleClient(Protocol):
    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ConsoleAction:
    method: str
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ReleaseSelection:
    memory_mib: int | None


@dataclass(frozen=True, slots=True)
class PolicySubmission:
    config: Mapping[str, Any]
    expected_revision: int
    save: bool


class ReleaseScreen(ModalScreen[ReleaseSelection | None]):
    """Collect an optional release amount without obscuring which GPU is targeted."""

    CSS = """
    ReleaseScreen {
        align: center middle;
    }
    #release-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: round $accent;
    }
    #release-error {
        height: auto;
        color: $error;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, gpu_label: str) -> None:
        super().__init__()
        self._gpu_label = gpu_label

    def compose(self) -> ComposeResult:
        with Vertical(id="release-dialog"):
            yield Static(f"Release WatchGPU reservation on {self._gpu_label}")
            yield Static("Amount (for example 2GiB); leave blank to release all:")
            yield Input(placeholder="2GiB", id="release-memory")
            yield Static("", id="release-error", markup=False)
            with Horizontal():
                yield Button("Release", id="confirm-release", variant="warning")
                yield Button("Cancel", id="cancel-release")

    def on_mount(self) -> None:
        self.query_one("#release-memory", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "release-memory":
            self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-release":
            self._submit()
        elif event.button.id == "cancel-release":
            self.dismiss(None)

    def _submit(self) -> None:
        raw_value = self.query_one("#release-memory", Input).value.strip()
        try:
            memory_mib = None if not raw_value else parse_user_capacity_mib(raw_value)
            if memory_mib is not None and memory_mib <= 0:
                raise ValueError("release amount must be positive")
        except (TypeError, ValueError) as exc:
            self.query_one("#release-error", Static).update(str(exc))
            return
        self.dismiss(ReleaseSelection(memory_mib))


class ConfirmationScreen(ModalScreen[bool]):
    """Require an explicit yes/no decision for lifecycle operations."""

    CSS = """
    ConfirmationScreen {
        align: center middle;
    }
    #confirmation-dialog {
        width: 64;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: round $warning;
    }
    """

    BINDINGS = [
        Binding("y", "confirm", "Confirm", show=False, priority=True),
        Binding("n", "cancel", "Cancel", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, message: str, *, confirm_label: str) -> None:
        super().__init__()
        self._message = message
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirmation-dialog"):
            yield Static(self._message, markup=False)
            yield Static("Press y to confirm or n/Esc to cancel.")
            with Horizontal():
                yield Button(
                    self._confirm_label,
                    id="confirm-operation",
                    variant="error",
                )
                yield Button("Cancel", id="cancel-operation")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-operation")


class PolicyEditScreen(ModalScreen[PolicySubmission | None]):
    """Edit the common runtime policy fields without requiring TOML knowledge."""

    CSS = """
    PolicyEditScreen {
        align: center middle;
    }
    #policy-dialog {
        width: 72;
        height: auto;
        max-height: 95%;
        padding: 1 2;
        background: $surface;
        border: round $accent;
    }
    .policy-label {
        height: auto;
        margin-top: 1;
    }
    #policy-error {
        height: auto;
        color: $error;
    }
    """

    BINDINGS = [
        Binding("f5", "apply_runtime", "Apply runtime", show=False, priority=True),
        Binding("ctrl+s", "apply_saved", "Apply & save", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        revision: int,
        selected_gpu_uuid: str,
    ) -> None:
        super().__init__()
        self._config = copy.deepcopy(dict(config))
        self._revision = revision
        self._selected_gpu_uuid = selected_gpu_uuid

    def compose(self) -> ComposeResult:
        reserve_limit = self._selected_reserve_limit()
        with Vertical(id="policy-dialog"):
            yield Static(f"Edit policy r{self._revision}")
            yield Static("Global leave-free", classes="policy-label")
            yield Input(
                value=_editable_mib(self._config.get("leave_free_mib")),
                placeholder="2GiB",
                id="policy-leave-free",
            )
            yield Static(
                f"Reserve limit for {self._selected_gpu_uuid}",
                classes="policy-label",
            )
            yield Input(
                value=_editable_mib(reserve_limit),
                placeholder="blank = no per-GPU limit",
                id="policy-reserve-limit",
                disabled=self._selected_gpu_uuid == "all",
            )
            yield Static("Maintenance duty cycle (1-20%)", classes="policy-label")
            yield Input(
                value=str(self._config.get("maintenance_duty_cycle_percent", 5)),
                id="policy-duty-cycle",
                type="integer",
            )
            yield Static("Whole-service CPU budget (1-100%)", classes="policy-label")
            yield Input(
                value=str(self._config.get("cpu_budget_percent", 100)),
                id="policy-cpu-budget",
                type="integer",
            )
            yield Static("CPU maintenance target (0-100% of one core)", classes="policy-label")
            yield Input(
                value=str(self._config.get("maintenance_cpu_target_percent", 0)),
                id="policy-cpu-target",
                type="integer",
            )
            yield Static("F5: runtime only · Ctrl+S: apply and save")
            yield Static("", id="policy-error", markup=False)
            with Horizontal():
                yield Button("Apply runtime", id="apply-runtime", variant="primary")
                yield Button("Apply & save", id="apply-saved", variant="success")
                yield Button("Cancel", id="cancel-policy")

    def on_mount(self) -> None:
        self.query_one("#policy-leave-free", Input).focus()

    def action_apply_runtime(self) -> None:
        self._submit(save=False)

    def action_apply_saved(self) -> None:
        self._submit(save=True)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-runtime":
            self._submit(save=False)
        elif event.button.id == "apply-saved":
            self._submit(save=True)
        elif event.button.id == "cancel-policy":
            self.dismiss(None)

    def _selected_reserve_limit(self) -> object:
        raw_gpus = self._config.get("gpus")
        if not isinstance(raw_gpus, list):
            return None
        for gpu in raw_gpus:
            if isinstance(gpu, Mapping) and gpu.get("selector") == self._selected_gpu_uuid:
                return gpu.get("reserve_limit_mib")
        return None

    def _submit(self, *, save: bool) -> None:
        try:
            leave_free = parse_user_capacity_mib(
                self.query_one("#policy-leave-free", Input).value.strip()
            )
            duty_cycle = int(self.query_one("#policy-duty-cycle", Input).value)
            cpu_budget = int(self.query_one("#policy-cpu-budget", Input).value)
            cpu_target = int(self.query_one("#policy-cpu-target", Input).value)
            if not 1 <= duty_cycle <= 20:
                raise ValueError("duty cycle must be between 1 and 20")
            if not 1 <= cpu_budget <= 100:
                raise ValueError("CPU budget must be between 1 and 100")
            if not 0 <= cpu_target <= 100:
                raise ValueError("CPU maintenance target must be between 0 and 100")
            candidate = copy.deepcopy(self._config)
            candidate["leave_free_mib"] = leave_free
            candidate["maintenance_duty_cycle_percent"] = duty_cycle
            candidate["cpu_budget_percent"] = cpu_budget
            if cpu_target or "maintenance_cpu_target_percent" in candidate:
                candidate["maintenance_cpu_target_percent"] = cpu_target
            if self._selected_gpu_uuid != "all":
                raw_limit = self.query_one("#policy-reserve-limit", Input).value.strip()
                limit = None if not raw_limit else parse_user_capacity_mib(raw_limit)
                raw_gpus = candidate.get("gpus")
                if isinstance(raw_gpus, list):
                    for gpu in raw_gpus:
                        if (
                            isinstance(gpu, dict)
                            and gpu.get("selector") == self._selected_gpu_uuid
                        ):
                            if limit is None:
                                gpu.pop("reserve_limit_mib", None)
                            else:
                                gpu["reserve_limit_mib"] = limit
                            break
        except (TypeError, ValueError) as exc:
            self.query_one("#policy-error", Static).update(str(exc))
            return
        self.dismiss(PolicySubmission(candidate, self._revision, save))


class WatchGPUConsole(App[None]):
    """SSH-friendly status and control client for a local WatchGPU supervisor."""

    CSS = """
    Screen {
        layout: vertical;
        overflow-x: hidden;
    }

    #connection-status {
        width: 100%;
        height: auto;
        min-height: 1;
        max-height: 3;
        padding: 0 1;
        background: $surface;
        color: $text;
        overflow-x: hidden;
    }

    #main-tabs {
        width: 100%;
        height: 1fr;
        overflow-x: hidden;
    }

    #operator-controls {
        width: 100%;
        height: auto;
        padding: 0 1;
    }

    #gpu-select {
        width: 30;
    }

    #action-status {
        width: 1fr;
        height: auto;
        padding: 0 1;
    }

    TabPane {
        width: 100%;
        height: 1fr;
        padding: 0 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    .section-view {
        width: 100%;
        height: auto;
        overflow-x: hidden;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit console", show=False),
        Binding("1", "show_gpus", "GPU", show=False),
        Binding("2", "show_tasks", "Tasks", show=False),
        Binding("3", "show_processes", "Processes", show=False),
        Binding("4", "show_profiles", "Profiles", show=False),
        Binding("5", "show_policy", "Policy", show=False),
        Binding("6", "show_events", "Events", show=False),
        Binding("g", "cycle_gpu", "Select GPU", show=True),
        Binding("p", "pause_selected", "Pause", show=True),
        Binding("u", "resume_selected", "Resume", show=True),
        Binding("x", "release_selected", "Release", show=True),
        Binding("r", "confirm_restart", "Restart", show=True),
        Binding("s", "confirm_stop", "Stop", show=True),
        Binding("e", "edit_policy", "Edit policy", show=True),
    ]

    def __init__(
        self,
        client: ConsoleClient | None = None,
        *,
        socket_path: Path | None = None,
        poll_interval_seconds: float = 1.0,
        request_timeout_seconds: float = 3.0,
    ) -> None:
        super().__init__()
        if client is None:
            if socket_path is None:
                raise ValueError("client or socket_path is required")
            client = AsyncWatchGPUClient(socket_path)
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._client = client
        self._poll_interval_seconds = poll_interval_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._connected = False
        self._snapshot: dict[str, Any] = {}
        self._last_error: str | None = None
        self._last_action_result: dict[str, Any] | None = None
        self._selected_gpu_uuid = "all"
        self._polling = False
        self._closing = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def snapshot(self) -> Mapping[str, Any]:
        return self._snapshot

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_action_result(self) -> Mapping[str, Any] | None:
        return self._last_action_result

    def compose(self) -> ComposeResult:
        yield Static("WatchGPU | connecting | policy r?", id="connection-status", markup=False)
        with Horizontal(id="operator-controls"):
            yield Select[str]((("All GPUs", "all"),), value="all", id="gpu-select")
            yield Button("Pause", id="pause-button", variant="warning")
            yield Button("Resume", id="resume-button", variant="success")
            yield Button("Release", id="release-button", variant="warning")
            yield Button("Restart", id="restart-button", variant="error")
            yield Button("Stop", id="stop-button", variant="error")
            yield Button("Policy", id="policy-button", variant="primary")
            yield Static("Ready", id="action-status", markup=False)
        with TabbedContent(initial="gpus-tab", id="main-tabs"):
            yield TabPane("GPU", self._section_widget("gpus"), id="gpus-tab")
            yield TabPane("Tasks", self._section_widget("tasks"), id="tasks-tab")
            yield TabPane(
                "Processes", self._section_widget("processes"), id="processes-tab"
            )
            yield TabPane("Profiles", self._section_widget("profiles"), id="profiles-tab")
            yield TabPane("Policy", self._section_widget("policy"), id="policy-tab")
            yield TabPane("Events", self._section_widget("events"), id="events-tab")

    async def on_mount(self) -> None:
        self.set_interval(self._poll_interval_seconds, self.refresh_status)
        await self.refresh_status()

    async def on_unmount(self) -> None:
        self._closing = True
        await _close_console_client(self._client)

    def on_resize(self, _event: events.Resize) -> None:
        self._render_snapshot()

    async def refresh_status(self) -> None:
        if self._polling or self._closing:
            return
        self._polling = True
        try:
            result = await asyncio.wait_for(
                self._client.call(STATUS_METHOD, {}),
                timeout=self._request_timeout_seconds,
            )
            self._snapshot = dict(result)
            self._update_gpu_selector()
            self._connected = True
            self._last_error = None
        except Exception as exc:
            self._connected = False
            self._last_error = str(exc) or type(exc).__name__
        finally:
            self._polling = False
            self._render_snapshot()

    async def execute_action(self, action: ConsoleAction) -> Mapping[str, Any] | None:
        """Send one explicit control action, then refresh the visible state."""
        try:
            result = await asyncio.wait_for(
                self._client.call(action.method, action.params),
                timeout=self._request_timeout_seconds,
            )
            self._last_action_result = dict(result)
            self._last_error = None
        except Exception as exc:
            self._last_action_result = None
            self._last_error = str(exc) or type(exc).__name__
            self._connected = False
            self._render_snapshot()
            return None
        await self.refresh_status()
        return self._last_action_result

    async def pause_reservation(self, gpu_uuid: str | None = None) -> Mapping[str, Any] | None:
        return await self.execute_action(pause_action(gpu_uuid))

    def action_cycle_gpu(self) -> None:
        options = self._gpu_options()
        values = [value for _label, value in options]
        try:
            current = values.index(self._selected_gpu_uuid)
        except ValueError:
            current = 0
        self._selected_gpu_uuid = values[(current + 1) % len(values)]
        self.query_one("#gpu-select", Select).value = self._selected_gpu_uuid

    async def action_pause_selected(self) -> None:
        await self.pause_reservation(self._selected_gpu_uuid)

    async def action_resume_selected(self) -> None:
        await self.resume_reservation(self._selected_gpu_uuid)

    def action_release_selected(self) -> None:
        self.push_screen(
            ReleaseScreen(self._selected_gpu_uuid),
            callback=self._finish_release,
        )

    def _finish_release(self, selection: ReleaseSelection | None) -> None:
        if selection is None:
            return
        self.run_worker(
            self.release_reservation(
                self._selected_gpu_uuid,
                memory_mib=selection.memory_mib,
            ),
            name="release-reservation",
            exclusive=True,
        )

    def action_confirm_restart(self) -> None:
        self.push_screen(
            ConfirmationScreen(
                "Restart WatchGPU workers and supervisor? Managed training processes "
                "will not be signalled.",
                confirm_label="Restart",
            ),
            callback=self._finish_restart,
        )

    def _finish_restart(self, confirmed: bool | None) -> None:
        if confirmed:
            self.run_worker(
                self.restart_daemon(),
                name="restart-daemon",
                exclusive=True,
            )

    def action_confirm_stop(self) -> None:
        self.push_screen(
            ConfirmationScreen(
                "Stop WatchGPU and release every WatchGPU-owned reservation? "
                "Managed training and external processes will not be signalled.",
                confirm_label="Stop & release",
            ),
            callback=self._finish_stop,
        )

    def _finish_stop(self, confirmed: bool | None) -> None:
        if confirmed:
            self.run_worker(
                self.stop_and_release(),
                name="stop-daemon",
                exclusive=True,
            )

    def action_edit_policy(self) -> None:
        raw_policy = self._snapshot.get("policy")
        if not isinstance(raw_policy, Mapping):
            self._last_error = "Supervisor did not report an editable policy"
            self._render_snapshot()
            return
        revision = raw_policy.get("revision")
        config = raw_policy.get("config")
        if not isinstance(revision, int) or isinstance(revision, bool) or not isinstance(
            config, Mapping
        ):
            self._last_error = "Supervisor returned an invalid policy snapshot"
            self._render_snapshot()
            return
        self.push_screen(
            PolicyEditScreen(
                config,
                revision=revision,
                selected_gpu_uuid=self._selected_gpu_uuid,
            ),
            callback=self._finish_policy_edit,
        )

    def _finish_policy_edit(self, submission: PolicySubmission | None) -> None:
        if submission is None:
            return
        self.run_worker(
            self.apply_policy(
                submission.config,
                expected_revision=submission.expected_revision,
                save=submission.save,
            ),
            name="apply-policy",
            exclusive=True,
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pause-button":
            await self.action_pause_selected()
        elif event.button.id == "resume-button":
            await self.action_resume_selected()
        elif event.button.id == "release-button":
            self.action_release_selected()
        elif event.button.id == "restart-button":
            self.action_confirm_restart()
        elif event.button.id == "stop-button":
            self.action_confirm_stop()
        elif event.button.id == "policy-button":
            self.action_edit_policy()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "gpu-select" and isinstance(event.value, str):
            self._selected_gpu_uuid = event.value

    async def resume_reservation(
        self, gpu_uuid: str | None = None
    ) -> Mapping[str, Any] | None:
        return await self.execute_action(resume_action(gpu_uuid))

    async def release_reservation(
        self,
        gpu_uuid: str | None = None,
        *,
        memory_mib: int | None = None,
    ) -> Mapping[str, Any] | None:
        return await self.execute_action(release_action(gpu_uuid, memory_mib=memory_mib))

    async def apply_policy(
        self,
        config: Mapping[str, Any],
        *,
        expected_revision: int,
        save: bool,
    ) -> Mapping[str, Any] | None:
        return await self.execute_action(
            policy_apply_action(
                config,
                expected_revision=expected_revision,
                save=save,
            )
        )

    async def restart_daemon(self) -> Mapping[str, Any] | None:
        return await self.execute_action(restart_action())

    async def stop_and_release(self) -> Mapping[str, Any] | None:
        return await self.execute_action(stop_action())

    def action_show_gpus(self) -> None:
        self._show_tab("gpus")

    def action_show_tasks(self) -> None:
        self._show_tab("tasks")

    def action_show_processes(self) -> None:
        self._show_tab("processes")

    def action_show_profiles(self) -> None:
        self._show_tab("profiles")

    def action_show_policy(self) -> None:
        self._show_tab("policy")

    def action_show_events(self) -> None:
        self._show_tab("events")

    @staticmethod
    def _section_widget(section: str) -> Static:
        return Static(
            "Waiting for supervisor status...",
            id=f"{section}-view",
            classes="section-view",
            markup=False,
        )

    def _show_tab(self, section: str) -> None:
        self.query_one("#main-tabs", TabbedContent).active = f"{section}-tab"

    def _render_snapshot(self) -> None:
        width = max(12, self.size.width - 4)
        widgets = {widget.id: widget for widget in self.query(Static) if widget.id}
        for section in _SECTIONS:
            widget = widgets.get(f"{section}-view")
            if widget is not None:
                widget.update(render_section(section, self._snapshot, width=width))
        connection_status = widgets.get("connection-status")
        if connection_status is not None:
            connection_status.update(self._connection_line())
        action_status = widgets.get("action-status")
        if action_status is not None:
            action_status.update(self._action_line())

    def _gpu_options(self) -> list[tuple[str, str]]:
        options = [("All GPUs", "all")]
        for gpu in _mapping_items(self._snapshot.get("gpus")):
            gpu_uuid = gpu.get("uuid")
            if isinstance(gpu_uuid, str) and gpu_uuid:
                options.append((f"GPU {gpu.get('index', '?')} · {gpu_uuid}", gpu_uuid))
        return options

    def _update_gpu_selector(self) -> None:
        selector = self.query_one("#gpu-select", Select)
        options = self._gpu_options()
        values = {value for _label, value in options}
        if self._selected_gpu_uuid not in values:
            self._selected_gpu_uuid = "all"
        selector.set_options(options)
        selector.value = self._selected_gpu_uuid

    def _connection_line(self) -> str:
        if self._connected:
            policy = self._snapshot.get("policy")
            if isinstance(policy, Mapping):
                revision = policy.get("revision", self._snapshot.get("policy_revision", "?"))
            else:
                revision = self._snapshot.get("policy_revision", "?")
            return f"WatchGPU | connected | policy r{revision}"
        detail = self._last_error or "waiting for supervisor"
        return f"WatchGPU | disconnected | retrying: {detail}"

    def _action_line(self) -> str:
        if self._last_error is not None:
            return f"Action error: {self._last_error}"
        result = self._last_action_result
        if result is None:
            return "Ready"
        status = result.get("status")
        revision = result.get("revision")
        if isinstance(status, str):
            revision_text = f" r{revision}" if isinstance(revision, int) else ""
            reason = result.get("reason")
            reason_text = f" · {reason}" if reason else ""
            return f"Policy {status}{revision_text}{reason_text}"
        return "Action completed"


def pause_action(gpu_uuid: str | None = None) -> ConsoleAction:
    return ConsoleAction(PAUSE_METHOD, _gpu_params(gpu_uuid))


def resume_action(gpu_uuid: str | None = None) -> ConsoleAction:
    return ConsoleAction(RESUME_METHOD, _gpu_params(gpu_uuid))


def release_action(
    gpu_uuid: str | None = None,
    *,
    memory_mib: int | None = None,
) -> ConsoleAction:
    params = _gpu_params(gpu_uuid)
    if memory_mib is not None:
        if isinstance(memory_mib, bool) or memory_mib <= 0:
            raise ValueError("memory_mib must be positive")
        params["memory_mib"] = memory_mib
    return ConsoleAction(RELEASE_METHOD, params)


def policy_apply_action(
    config: Mapping[str, Any],
    *,
    expected_revision: int,
    save: bool,
) -> ConsoleAction:
    if isinstance(expected_revision, bool) or expected_revision < 0:
        raise ValueError("expected_revision must be non-negative")
    return ConsoleAction(
        POLICY_APPLY_METHOD,
        {
            "config": dict(config),
            "expected_revision": expected_revision,
            "save": save,
        },
    )


def restart_action() -> ConsoleAction:
    return ConsoleAction(RESTART_METHOD, {})


def stop_action() -> ConsoleAction:
    return ConsoleAction(STOP_METHOD, {"release": True})


def render_section(
    section: str,
    snapshot: Mapping[str, Any],
    *,
    width: int = 80,
) -> str:
    """Render a status section without requiring a wide terminal."""
    renderers = {
        "gpus": _render_gpus,
        "tasks": _render_tasks,
        "processes": _render_processes,
        "profiles": _render_profiles,
        "policy": _render_policy,
        "events": _render_events,
    }
    renderer = renderers.get(section)
    if renderer is None:
        raise ValueError(f"unknown console section: {section}")
    content = renderer(snapshot)
    return _wrap_lines(content, width=max(width, 12))


def _render_gpus(snapshot: Mapping[str, Any]) -> str:
    gpus = _mapping_items(snapshot.get("gpus"))
    if not gpus:
        return "No managed GPUs reported."

    blocks: list[str] = []
    for gpu in gpus:
        index = gpu.get("index", "?")
        state = gpu.get("worker_state", "UNKNOWN")
        total_mib = _integer(gpu.get("total_mib"))
        free_mib = _integer(gpu.get("free_mib"))
        reserved_mib = _integer(gpu.get("reserved_mib"))
        leased_mib = _integer(gpu.get("leased_mib"))
        utilization = _integer(gpu.get("utilization_percent"))
        temperature = _integer(gpu.get("temperature_c"))
        external_mib = (
            None
            if total_mib is None or free_mib is None or reserved_mib is None
            else max(0, total_mib - free_mib - reserved_mib)
        )
        lines = [
            f"GPU {index} [{state}]",
            f"Name: {gpu.get('name', 'Unknown GPU')}",
            f"UUID: {gpu.get('uuid', 'unknown')}",
            f"Total: {_format_mib(total_mib)}",
            f"Free: {_format_mib(free_mib)}",
            f"Reserved: {_format_mib(reserved_mib)}",
            f"Leased: {_format_mib(leased_mib)}",
            f"External used: {_format_mib(external_mib)}",
            f"Utilization: {utilization}%" if utilization is not None else "Utilization: -",
            f"Temperature: {temperature}°C" if temperature is not None else "Temperature: -",
            f"MIG: {gpu.get('mig_mode') or '-'}",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_tasks(snapshot: Mapping[str, Any]) -> str:
    tasks = _mapping_items(snapshot.get("tasks"))
    if not tasks:
        tasks = _mapping_items(snapshot.get("leases"))
    if not tasks:
        return "No managed tasks reported."

    # The supervisor intentionally retains terminal leases for operator history.
    # Put live work first so repeated launches cannot push the current training
    # below the visible part of a short terminal.  Within each group, show the
    # newest entry first.
    tasks.sort(key=_task_sort_key)

    blocks: list[str] = []
    for task in tasks:
        name = task.get("task_name", task.get("lease_id", "Unnamed task"))
        state = task.get("state", "UNKNOWN")
        released = task.get("released_by_gpu_mib")
        lines = [
            f"{name} [{state}]",
            f"Lease: {task.get('lease_id', '-')}",
            f"PID: {task.get('client_pid', task.get('pid', '-'))}",
            f"GPUs: {_join_values(task.get('gpu_uuids'))}",
            f"Requested/GPU: {_format_mib(_integer(task.get('memory_per_gpu_mib')))}",
            f"Released: {_format_memory_mapping(released)}",
            f"Created: {task.get('created_at', '-')}",
            f"Expires: {task.get('expires_at', '-')}",
        ]
        queue_position = task.get("queue_position")
        if queue_position is not None:
            lines.append(f"Queue position: {queue_position}")
        heartbeat = task.get("heartbeat_age_seconds")
        if heartbeat is not None:
            lines.append(f"Heartbeat age: {heartbeat}s")
        error = task.get("error")
        if error:
            lines.append(f"Error: {error}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _task_sort_key(task: Mapping[str, Any]) -> tuple[int, float, str]:
    state = str(task.get("state", "UNKNOWN")).upper()
    state_priority = {
        "ACTIVE": 0,
        "ORPHANED": 1,
        "QUEUED": 2,
    }.get(state, 3)
    created_at = task.get("created_at")
    created_order = (
        -float(created_at)
        if isinstance(created_at, (int, float)) and not isinstance(created_at, bool)
        else 0.0
    )
    return state_priority, created_order, str(task.get("lease_id", ""))


def _render_processes(snapshot: Mapping[str, Any]) -> str:
    processes = _mapping_items(snapshot.get("processes"))
    if not processes:
        return "No GPU process data reported."

    blocks: list[str] = []
    for process in processes:
        classification = process.get(
            "classification",
            process.get("kind", process.get("type", "UNCLASSIFIED")),
        )
        lines = [
            f"Process {process.get('pid', '?')} [{classification}]",
            f"Name: {process.get('name', '-')}",
            f"GPU: {process.get('gpu_uuid', process.get('gpu_index', '-'))}",
            f"Memory: {_format_mib(_integer(process.get('used_memory_mib')))}",
        ]
        task_name = process.get("task_name")
        if task_name:
            lines.append(f"Task: {task_name}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_profiles(snapshot: Mapping[str, Any]) -> str:
    profiles = _mapping_items(snapshot.get("profiles"))
    if not profiles:
        return "No memory profiles reported."

    blocks: list[str] = []
    for profile in profiles:
        key = profile.get("task_key", profile.get("key", "Unnamed profile"))
        status = profile.get("status", "UNKNOWN")
        lines = [
            f"{key} [{status}]",
            f"World size: {profile.get('world_size', '-')}",
            f"Peak/GPU: {_format_mib(_integer(profile.get('peak_per_gpu_mib')))}",
            f"Safety margin: {_format_mib(_integer(profile.get('margin_mib')))}",
            f"Recommended: {_format_mib(_integer(profile.get('recommended_mib')))}",
            f"Fingerprint valid: {profile.get('fingerprint_valid', '-')}",
        ]
        rounding = profile.get("rounding")
        if rounding is not None:
            lines.append(f"Rounding: {rounding}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_policy(snapshot: Mapping[str, Any]) -> str:
    raw_policy = snapshot.get("policy")
    policy = raw_policy if isinstance(raw_policy, Mapping) else {}
    revision = policy.get("revision", snapshot.get("policy_revision", "?"))
    status = policy.get("status", policy.get("apply_status", "CURRENT"))
    lines = [f"Policy r{revision} [{status}]"]
    raw_config = policy.get("config", policy.get("desired_config"))
    config = raw_config if isinstance(raw_config, Mapping) else {}
    if not config:
        lines.append("No policy configuration reported.")
    else:
        for key, value in config.items():
            lines.append(f"{key}: {_display_value(value)}")
    reason = policy.get("reason")
    if reason:
        lines.append(f"Reason: {reason}")
    cpu = snapshot.get("cpu")
    if isinstance(cpu, Mapping):
        lines.extend(
            (
                "",
                "Process-tree CPU",
                f"Usage: {cpu.get('process_tree_percent', '-')}%",
                f"Budget: {cpu.get('budget_percent', '-')}%",
                f"Maintenance target: {cpu.get('maintenance_target_percent', '-')}%",
                f"Maintenance state: {cpu.get('maintenance_state', '-')}",
                f"Processes: {cpu.get('process_count', '-')}",
                f"Affinity cores: {_join_values(cpu.get('affinity_cores'))}",
                f"Worker threads: {cpu.get('worker_cpu_threads', '-')}",
                f"Maintenance throttled: {cpu.get('over_budget', False)}",
            )
        )
    restart = snapshot.get("maintenance_restart")
    if isinstance(restart, Mapping):
        lines.extend(
            (
                "",
                f"Maintenance restart: {restart.get('state', '-')}",
                f"Scheduled for: {restart.get('scheduled_for', '-')}",
                f"Last executed date: {restart.get('last_executed_local_date', '-')}",
            )
        )
    return "\n".join(lines)


def _render_events(snapshot: Mapping[str, Any]) -> str:
    events = _mapping_items(snapshot.get("events"))
    runtime = snapshot.get("runtime")
    runtime_error = (
        runtime.get("last_error") if isinstance(runtime, Mapping) else None
    )
    if not events and not runtime_error:
        return "No events reported."

    blocks: list[str] = []
    if runtime_error:
        blocks.append(f"DAEMON_POLL_ERROR\nSeverity: ERROR\nMessage: {runtime_error}")
    for event in events:
        event_type = event.get("type", event.get("event", "EVENT"))
        lines = [
            f"{event_type}",
            f"Time: {event.get('timestamp', event.get('created_at', '-'))}",
            f"Message: {event.get('message', '-')}",
        ]
        severity = event.get("severity")
        if severity is not None:
            lines.append(f"Severity: {severity}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _mapping_items(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _gpu_params(gpu_uuid: str | None) -> dict[str, Any]:
    if gpu_uuid is None:
        return {}
    if not gpu_uuid.strip():
        raise ValueError("gpu_uuid must be non-empty")
    return {"gpu_uuid": gpu_uuid}


def _join_values(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return "-"
    return ", ".join(str(item) for item in value) or "-"


def _format_memory_mapping(value: object) -> str:
    if not isinstance(value, Mapping) or not value:
        return "-"
    return ", ".join(
        f"{key}={_format_mib(_integer(memory))}" for key, memory in value.items()
    )


def _display_value(value: object) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _editable_mib(value: object) -> str:
    normalized = _integer(value)
    return "" if normalized is None else f"{normalized}MiB"


def _format_mib(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1024:.1f} GiB"


def _wrap_lines(content: str, width: int) -> str:
    wrapped: list[str] = []
    for line in content.splitlines():
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(
            textwrap.wrap(
                line,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
        )
    return "\n".join(wrapped)


async def _close_console_client(client: ConsoleClient) -> None:
    closer = getattr(client, "aclose", None)
    if closer is None:
        closer = getattr(client, "close", None)
    if not callable(closer):
        return
    result = closer()
    if inspect.isawaitable(result):
        await result
