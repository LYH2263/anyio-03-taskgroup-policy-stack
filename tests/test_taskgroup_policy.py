from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

import pytest

from anyio import (
    TASK_STATUS_IGNORED,
    CancelScope,
    Event,
    GroupExitSurface,
    SpawnScopeBind,
    StartReparentGate,
    TaskGroupPolicyFrame,
    create_task_group,
    current_task_status,
    get_cancelled_exc_class,
    get_current_task_group_policy,
    pop_task_group_policy,
    push_task_group_policy,
    sleep,
    sleep_forever,
    task_group_policy,
    wait_all_tasks_blocked,
)
from anyio._core._taskgroup_policy import DEFAULT_TASK_GROUP_POLICY_FRAME

if TYPE_CHECKING:
    from anyio.abc import TaskGroup, TaskStatus

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


def _flatten(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        flattened: list[BaseException] = []
        for child in exc.exceptions:
            flattened.extend(_flatten(child))

        return flattened

    return [exc]


def _eager_task_factory_active() -> bool:
    """Return ``True`` when the current asyncio loop uses the eager factory."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    return loop.get_task_factory() is asyncio.eager_task_factory


#
# The stack itself: push / pop / context manager / defaults
#


async def test_empty_stack_observes_default_frame() -> None:
    assert get_current_task_group_policy() == DEFAULT_TASK_GROUP_POLICY_FRAME
    frame = get_current_task_group_policy()
    assert frame.spawn_scope_bind is SpawnScopeBind.PIN_DEFAULT
    assert frame.start_reparent_gate is StartReparentGate.DEFER_UNTIL_STARTED
    assert (
        frame.group_exit_surface is GroupExitSurface.AGGREGATE_EXCEPTION_GROUP
    )


async def test_push_pop_restores_outer_user_frame() -> None:
    base = get_current_task_group_policy()
    token1 = push_task_group_policy(
        spawn_scope_bind=SpawnScopeBind.FORCE_GROUP_IMMEDIATE
    )
    assert token1.frame.spawn_scope_bind is SpawnScopeBind.FORCE_GROUP_IMMEDIATE
    try:
        frame1 = get_current_task_group_policy()
        assert frame1.spawn_scope_bind is SpawnScopeBind.FORCE_GROUP_IMMEDIATE
        # Fields not given fall back to their defaults.
        assert (
            frame1.start_reparent_gate is StartReparentGate.DEFER_UNTIL_STARTED
        )

        token2 = push_task_group_policy(
            group_exit_surface=GroupExitSurface.CANCEL_PEERS_THEN_WAIT
        )
        try:
            frame2 = get_current_task_group_policy()
            assert (
                frame2.group_exit_surface
                is GroupExitSurface.CANCEL_PEERS_THEN_WAIT
            )
            # The outer frame is not a mutable singleton: the new frame has its
            # own spawn binding back at the default.
            assert frame2.spawn_scope_bind is SpawnScopeBind.PIN_DEFAULT
        finally:
            pop_task_group_policy(token2)

        # Popping restores the outer *user* frame, not a process global.
        assert (
            get_current_task_group_policy().spawn_scope_bind
            is SpawnScopeBind.FORCE_GROUP_IMMEDIATE
        )
    finally:
        pop_task_group_policy(token1)

    assert get_current_task_group_policy() is base


async def test_pop_in_wrong_order_rejected() -> None:
    token1 = push_task_group_policy(
        spawn_scope_bind=SpawnScopeBind.FORCE_GROUP_IMMEDIATE
    )
    token2 = push_task_group_policy(
        group_exit_surface=GroupExitSurface.CANCEL_PEERS_THEN_WAIT
    )
    with pytest.raises(RuntimeError, match="topmost"):
        pop_task_group_policy(token1)

    pop_task_group_policy(token2)
    pop_task_group_policy(token1)
    assert (
        get_current_task_group_policy() == DEFAULT_TASK_GROUP_POLICY_FRAME
    )


async def test_context_manager_pops_on_exception() -> None:
    with pytest.raises(ValueError, match="boom"):
        with task_group_policy(
            spawn_scope_bind=SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED
        ):
            assert (
                get_current_task_group_policy().spawn_scope_bind
                is SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED
            )
            raise ValueError("boom")

    assert get_current_task_group_policy() == DEFAULT_TASK_GROUP_POLICY_FRAME


async def test_push_accepts_ready_made_frame() -> None:
    frame = TaskGroupPolicyFrame(
        spawn_scope_bind=SpawnScopeBind.FORCE_GROUP_IMMEDIATE,
        start_reparent_gate=StartReparentGate.EAGER_SAFE_REPARENT,
        group_exit_surface=GroupExitSurface.SHIELDED_WAIT_ON_OUTER_CANCEL,
    )
    with task_group_policy(frame) as active:
        assert active is frame
        assert get_current_task_group_policy() is frame


#
# Merged view of frame + group state
#


async def test_policy_view_fields() -> None:
    reached = Event()

    async def waiter() -> None:
        reached.set()
        await sleep_forever()

    async with create_task_group() as tg:
        view = tg.current_policy_view()
        assert view.frame == DEFAULT_TASK_GROUP_POLICY_FRAME
        assert view.cancel_scope is tg.cancel_scope
        assert view.pending_tasks == 0
        assert view.entered is True
        assert view.exiting is False

        tg.start_soon(waiter)
        await reached.wait()
        await wait_all_tasks_blocked()
        view = tg.current_policy_view()
        assert view.pending_tasks == 1
        assert view.exiting is False
        tg.cancel_scope.cancel()


#
# spawn_scope_bind: start() + task_status.started() reparent path
#


async def test_start_defers_group_cancel_until_started() -> None:
    observations: dict[str, bool] = {}
    release = Event()

    async def just_wait(*, task_status: TaskStatus[str]) -> None:
        await release.wait()
        await sleep(0)  # must not be cancelled here
        observations["not_cancelled_before_started"] = True
        task_status.started("started")
        try:
            await sleep_forever()  # must be cancelled here
        except get_cancelled_exc_class():
            observations["cancelled_after_started"] = True
            raise

    async def group_task(
        *, task_status: TaskStatus[TaskGroup] = TASK_STATUS_IGNORED
    ) -> None:
        with task_group_policy():  # explicit default frame
            async with create_task_group() as inner:
                task_status.started(inner)
                await sleep_forever()

    async def outside_task(inner: TaskGroup) -> str:
        return await inner.start(just_wait)

    async with create_task_group() as tg:
        inner = await tg.start(group_task)
        handle = tg.start_soon(outside_task, inner)
        await wait_all_tasks_blocked()
        inner.cancel()
        release.set()
        await handle.wait()
        assert handle.return_value == "started"

    assert observations["not_cancelled_before_started"] is True
    assert observations["cancelled_after_started"] is True


async def test_force_group_immediate_start_is_cancelled_before_started() -> None:
    release = Event()
    result: dict[str, Any] = {}

    async def just_wait(*, task_status: TaskStatus[None]) -> None:
        try:
            await release.wait()
            task_status.started()
            await sleep_forever()
        except get_cancelled_exc_class() as exc:
            result["child_seen"] = type(exc)
            raise

    async def group_task(
        *, task_status: TaskStatus[TaskGroup] = TASK_STATUS_IGNORED
    ) -> None:
        async with create_task_group() as inner:
            task_status.started(inner)
            await sleep_forever()

    async def outside_task(inner: TaskGroup) -> None:
        try:
            with task_group_policy(
                spawn_scope_bind=SpawnScopeBind.FORCE_GROUP_IMMEDIATE
            ):
                await inner.start(just_wait)
        except BaseException as exc:
            result["start_seen"] = exc
        else:
            result["start_seen"] = None

    async with create_task_group() as tg:
        inner = await tg.start(group_task)
        handle = tg.start_soon(outside_task, inner)
        await wait_all_tasks_blocked()
        inner.cancel()
        release.set()
        await handle.wait()

    assert isinstance(
        result["start_seen"], get_cancelled_exc_class()
    ), result
    assert result.get("child_seen") is get_cancelled_exc_class()


#
# spawn_scope_bind: start_soon deferred-membership path
#


async def test_keep_caller_start_soon_defers_group_cancel() -> None:
    observations: dict[str, bool] = {}
    entered = Event()
    release = Event()

    async def deferred_child() -> None:
        status = current_task_status()
        assert status is not None
        entered.set()
        await release.wait()
        try:
            await sleep(0)  # group is already cancelled here
        except get_cancelled_exc_class():
            observations["cancelled_before_started"] = True
            raise

        status.started()
        try:
            await sleep_forever()  # joined the group scope now
        except get_cancelled_exc_class():
            observations["cancelled_after_started"] = True
            raise

    async def driver(inner: TaskGroup) -> None:
        await entered.wait()
        await wait_all_tasks_blocked()
        inner.cancel()
        await wait_all_tasks_blocked()
        release.set()

    async def group_task(
        *, task_status: TaskStatus[TaskGroup] = TASK_STATUS_IGNORED
    ) -> None:
        with task_group_policy(
            spawn_scope_bind=SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED
        ):
            async with create_task_group() as inner:
                task_status.started(inner)
                inner.start_soon(deferred_child)
                await entered.wait()
                await sleep_forever()

    async with create_task_group() as tg:
        inner = await tg.start(group_task)
        tg.start_soon(driver, inner)

    assert observations.get("cancelled_before_started") is not True
    assert observations.get("cancelled_after_started") is True


async def test_keep_caller_start_soon_outer_cancel_reaches_child() -> None:
    entered = Event()
    cancelled = Event()

    async def child() -> None:
        assert current_task_status() is not None
        entered.set()
        try:
            await sleep_forever()
        except get_cancelled_exc_class():
            cancelled.set()
            raise

    with CancelScope() as outer:
        with task_group_policy(
            spawn_scope_bind=SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED
        ):
            async with create_task_group() as tg:
                tg.start_soon(child)
                await entered.wait()
                await wait_all_tasks_blocked()
                outer.cancel()

    assert cancelled.is_set()


async def test_keep_caller_start_soon_child_need_not_signal() -> None:
    # A deferred start_soon child that never calls started() must still be
    # awaited on exit and must complete normally.
    finished = Event()

    async def child() -> None:
        assert current_task_status() is not None
        finished.set()

    with task_group_policy(
        spawn_scope_bind=SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED
    ):
        async with create_task_group() as tg:
            tg.start_soon(child)
            await finished.wait()


async def test_keep_caller_start_exposes_spawn_status() -> None:
    async def service(*, task_status: TaskStatus[int]) -> None:
        assert current_task_status() is task_status
        task_status.started(7)

    with task_group_policy(
        spawn_scope_bind=SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED
    ):
        async with create_task_group() as tg:
            assert await tg.start(service) == 7


async def test_plain_start_soon_has_no_spawn_status() -> None:
    seen: dict[bool] = {}

    async def child() -> None:
        seen["status"] = current_task_status() is None

    async with create_task_group() as tg:
        tg.start_soon(child)
        await wait_all_tasks_blocked()

    assert seen["status"] is True


#
# start_reparent_gate: REJECT_STARTED_AFTER_GROUP_CANCEL
#


async def test_reject_started_after_group_cancel() -> None:
    release = Event()
    result: dict[str, BaseException | None] = {}

    async def just_wait(*, task_status: TaskStatus[None]) -> None:
        await release.wait()
        task_status.started()
        pytest.fail("started() should have raised")

    async def group_task(
        *, task_status: TaskStatus[TaskGroup] = TASK_STATUS_IGNORED
    ) -> None:
        async with create_task_group() as inner:
            task_status.started(inner)
            await sleep_forever()

    async def outside_task(inner: TaskGroup) -> None:
        try:
            with task_group_policy(
                start_reparent_gate=(
                    StartReparentGate.REJECT_STARTED_AFTER_GROUP_CANCEL
                )
            ):
                await inner.start(just_wait)
        except BaseException as exc:
            result["error"] = exc
        else:
            result["error"] = None

    async with create_task_group() as tg:
        inner = await tg.start(group_task)
        handle = tg.start_soon(outside_task, inner)
        await wait_all_tasks_blocked()
        inner.cancel()
        release.set()
        await handle.wait()

    error = result["error"]
    assert isinstance(error, RuntimeError)
    assert "after the target task group had been cancelled" in str(error)


#
# group_exit_surface: aggregation across multiple child failures
#


@pytest.mark.parametrize(
    "surface",
    [
        GroupExitSurface.AGGREGATE_EXCEPTION_GROUP,
        GroupExitSurface.CANCEL_PEERS_THEN_WAIT,
    ],
)
async def test_multiple_child_failures_are_aggregated(
    surface: GroupExitSurface,
) -> None:
    async def boom(text: str) -> None:
        # A self-installed shield briefly delays peer cancellation so that both
        # children genuinely fail; under CANCEL_PEERS_THEN_WAIT cancellation is
        # delivered through the child's own handle scope anyway.
        with CancelScope(shield=True):
            await sleep(0.03)

        raise ValueError(text)

    with task_group_policy(group_exit_surface=surface):
        with pytest.raises(BaseException) as exc_info:
            async with create_task_group() as tg:
                tg.start_soon(boom, "one")
                tg.start_soon(boom, "two")

    messages = {str(exc) for exc in _flatten(exc_info.value)}
    assert messages == {"one", "two"}


#
# group_exit_surface: outer cancellation during the exit wait (#695 path)
#


@pytest.mark.parametrize(
    "surface",
    [
        GroupExitSurface.AGGREGATE_EXCEPTION_GROUP,
        GroupExitSurface.SHIELDED_WAIT_ON_OUTER_CANCEL,
    ],
)
async def test_outer_cancel_during_exit_wait_shields_and_cancels_group(
    surface: GroupExitSurface,
) -> None:
    started = Event()
    peer_cancelled = Event()

    async def waiter() -> None:
        started.set()
        try:
            await sleep_forever()
        except get_cancelled_exc_class():
            peer_cancelled.set()
            raise

    async def host(
        *, task_status: TaskStatus[TaskGroup] = TASK_STATUS_IGNORED
    ) -> None:
        with task_group_policy(group_exit_surface=surface):
            async with create_task_group() as inner:
                inner.start_soon(waiter)
                await started.wait()
                task_status.started(inner)
                await sleep_forever()

    async with create_task_group() as tg:
        handle = await tg.start(host, return_handle=True)
        inner: TaskGroup = handle.start_value
        await wait_all_tasks_blocked()
        assert not peer_cancelled.is_set()
        # The group's own shield flag is the pre-existing value before the
        # outer cancellation arrives.
        assert inner.cancel_scope.shield is False
        handle.cancel()
        await handle.wait()

    # The exit wait completed (no hang), the peer was cancelled and the
    # group's own shield was never toggled (no permanent-shield fakery).
    assert peer_cancelled.is_set()
    assert inner.cancel_scope.shield is False


#
# Per-task-tree isolation of the stack
#


async def test_concurrent_host_tasks_do_not_share_stack() -> None:
    keep_seen: dict[str, Any] = {}
    force_seen: dict[str, Any] = {}
    keep_child_ready = Event()
    force_child_ready = Event()
    release = Event()

    async def host(
        bind: SpawnScopeBind,
        surface: GroupExitSurface,
        seen: dict[str, Any],
        ready: Event,
    ) -> None:
        with task_group_policy(
            spawn_scope_bind=bind, group_exit_surface=surface
        ):
            frame = get_current_task_group_policy()
            seen["frame"] = frame

            async def child() -> None:
                # Observable spawn difference: only the KEEP_CALLER host's
                # start_soon child gets an explicit spawn task status.
                seen["spawn_status_present"] = current_task_status() is not None
                ready.set()
                await release.wait()

            async with create_task_group() as tg:
                tg.start_soon(child)
                await release.wait()

    async with create_task_group() as tg:
        tg.start_soon(
            host,
            SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED,
            GroupExitSurface.CANCEL_PEERS_THEN_WAIT,
            keep_seen,
            keep_child_ready,
        )
        tg.start_soon(
            host,
            SpawnScopeBind.FORCE_GROUP_IMMEDIATE,
            GroupExitSurface.AGGREGATE_EXCEPTION_GROUP,
            force_seen,
            force_child_ready,
        )
        await keep_child_ready.wait()
        await force_child_ready.wait()

        assert (
            keep_seen["frame"].spawn_scope_bind
            is SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED
        )
        assert (
            keep_seen["frame"].group_exit_surface
            is GroupExitSurface.CANCEL_PEERS_THEN_WAIT
        )
        assert keep_seen["spawn_status_present"] is True
        assert (
            force_seen["frame"].spawn_scope_bind
            is SpawnScopeBind.FORCE_GROUP_IMMEDIATE
        )
        assert (
            force_seen["frame"].group_exit_surface
            is GroupExitSurface.AGGREGATE_EXCEPTION_GROUP
        )
        assert force_seen["spawn_status_present"] is False

        # The outer host's own stack is untouched by either child host.
        assert (
            get_current_task_group_policy() == DEFAULT_TASK_GROUP_POLICY_FRAME
        )
        release.set()


#
# Late pushes against an in-flight __aexit__ wait loop
#


async def test_push_during_group_exit_wait_is_rejected() -> None:
    entered = Event()

    async def child() -> None:
        entered.set()
        # Once this checkpoint returns, the host task is blocked in the
        # group's __aexit__ wait loop.
        await wait_all_tasks_blocked()
        with pytest.raises(RuntimeError, match="__aexit__ wait loop"):
            push_task_group_policy(
                spawn_scope_bind=SpawnScopeBind.FORCE_GROUP_IMMEDIATE
            )

    async with create_task_group() as tg:
        tg.start_soon(child)
        await entered.wait()


#
# EAGER_SAFE_REPARENT (asyncio eager task factory)
#


async def test_eager_safe_reparent_runs_first_segment_synchronously() -> None:
    eager = _eager_task_factory_active()
    release = Event()
    observations: dict[str, bool] = {}

    async def eager_child() -> None:
        observations["ran_before_scheduling"] = True
        await release.wait()

    async def scheduled_child() -> None:
        observations["baseline_ran_before_scheduling"] = True
        await release.wait()

    with task_group_policy(
        start_reparent_gate=StartReparentGate.EAGER_SAFE_REPARENT
    ):
        async with create_task_group() as tg:
            tg.start_soon(eager_child)
            assert observations.get("ran_before_scheduling", False) is eager

    # The baseline frame keeps defeating the eager factory even on an eager
    # loop, so spawns never run a first segment synchronously.
    async with create_task_group() as tg:
        observations.clear()
        tg.start_soon(scheduled_child)
        assert "baseline_ran_before_scheduling" not in observations
        release.set()


async def test_eager_safe_reparent_exception_after_started_not_lost() -> None:
    async def service(*, task_status: TaskStatus[None]) -> None:
        task_status.started()
        raise ValueError("eager boom")

    with task_group_policy(
        start_reparent_gate=StartReparentGate.EAGER_SAFE_REPARENT
    ):
        with pytest.raises(BaseException) as exc_info:
            async with create_task_group() as tg:
                await tg.start(service)

    assert any(
        isinstance(exc, ValueError) and str(exc) == "eager boom"
        for exc in _flatten(exc_info.value)
    )


async def test_eager_safe_reparent_start_value_without_tip_switch() -> None:
    async def service(*, task_status: TaskStatus[int]) -> None:
        task_status.started(123)

    with task_group_policy(
        start_reparent_gate=StartReparentGate.EAGER_SAFE_REPARENT
    ):
        async with create_task_group() as tg:
            assert await tg.start(service) == 123
