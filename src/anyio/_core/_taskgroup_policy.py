"""
Task tree level task group policy stack.

This module implements a nested, resettable policy stack that controls how
:class:`~anyio.abc.TaskGroup` binds freshly spawned child tasks to cancel scopes,
how ``task_status.started()`` reparents them, and how the group exit aggregates
errors. The stack is deliberately independent of cancellation deadline
policies, memory object stream bookkeeping and HTTP/WSGI request stacks.

Mount point and isolation
-------------------------

The stack is *not* stored in a :class:`~anyio.lowlevel.RunVar`.  On asyncio an
AnyIO run variable is bucketed by the native event loop token (see
:data:`anyio.lowlevel._run_vars`), and on Trio it is bucketed per ``trio.run()``
call; in both cases every task running in that bucket – including concurrently
running unrelated host tasks – shares the value.  Pushing different policies
from two concurrent host tasks would therefore cross-contaminate each other.

Instead the stack lives in a :class:`~contextvars.ContextVar` holding an
immutable tuple of frames.  Each spawned child task inherits a copy of the
current context at spawn time, so a frame pushed in a host task is visible to
that task's own spawn decisions and to descendants spawned while the frame is
active, while two sibling host tasks (and separate task tree roots) get
independent stacks.  Popping restores the previous user frame; when the stack
is empty, readers observe :data:`DEFAULT_TASK_GROUP_POLICY_FRAME`, an immutable
sentinel – never a process wide mutable singleton.

Decision timing
---------------

* ``spawn_scope_bind`` is resolved at each individual ``create_task()`` /
  ``start_soon()`` / ``start()`` call.  Frames pushed later never rebind tasks
  that are already running.
* ``start_reparent_gate`` is resolved live when ``task_status.started()`` runs.
* ``group_exit_surface`` is resolved when ``TaskGroup.__aexit__`` starts.
  Pushing a frame while a task group of the current task tree is in its exit
  wait loop is rejected with :class:`RuntimeError` instead of silently
  replacing or being ignored by the in-flight exit.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from ._eventloop import get_async_backend


class SpawnScopeBind(Enum):
    """
    Selects which cancel scope a child task spawned in the current frame is
    bound to before it reports readiness.
    """

    PIN_DEFAULT = auto()
    """
    ``create_task()`` / ``start_soon()`` children bind to the group's cancel
    scope immediately, while ``start()`` children stay pinned to the caller's
    cancel scope until ``task_status.started()`` (the historical backend
    specific behavior).
    """

    FORCE_GROUP_IMMEDIATE = auto()
    """
    Every child spawned in this frame, including a ``start()`` child that has
    not called ``task_status.started()`` yet, binds immediately and solely to
    the group's cancel scope.  This is achieved by real scope parenting, never
    by toggling ``shield``.
    """

    KEEP_CALLER_UNTIL_STARTED = auto()
    """
    Even ``start_soon()`` children stay outside the group cancel scope until an
    explicit readiness signal (``task_status.started()`` on the spawn status
    returned by :func:`current_task_status`).  This is observably different
    from :attr:`PIN_DEFAULT`; the group cancel scope cannot cancel such a task
    before the signal.
    """


class StartReparentGate(Enum):
    """Selects how ``task_status.started()`` moves a task into the group scope."""

    DEFER_UNTIL_STARTED = auto()
    """
    Group cancellation must not reach a task that has not reported readiness;
    the task is reparented into the group scope when ``started()`` is called.
    """

    REJECT_STARTED_AFTER_GROUP_CANCEL = auto()
    """
    ``started()`` fails with :class:`RuntimeError` and does not reparent the
    task into a group whose cancel scope is already cancelled.
    """

    EAGER_SAFE_REPARENT = auto()
    """
    On asyncio, allow the loop's configured eager task factory to run the
    child's first segment synchronously.  The ordering between reparenting and
    the first scheduler tip is deterministic, and exceptions raised after
    ``started()`` without a tip switch are still delivered.  On Trio (which has
    no eager task factories) this behaves like :attr:`DEFER_UNTIL_STARTED`.
    """


class GroupExitSurface(Enum):
    """Selects how ``TaskGroup.__aexit__()`` waits for children and surfaces errors."""

    AGGREGATE_EXCEPTION_GROUP = auto()
    """
    Wait for all children and aggregate unhandled errors into a
    :class:`BaseExceptionGroup` (the historical behavior, including the
    shielded wait when the wait is cancelled from outside, #695).
    """

    CANCEL_PEERS_THEN_WAIT = auto()
    """
    When any child fails while the frame is active, cancel the other
    unfinished children directly (their own task scopes, bypassing self
    installed shields) before waiting, and still aggregate every unhandled
    error observably.
    """

    SHIELDED_WAIT_ON_OUTER_CANCEL = auto()
    """
    Outer cancellation during the exit wait must enter the transient
    "shield the wait scope + cancel the group" path.  Faking this by setting
    ``cancel_scope.shield`` to a permanent value before entering the group is
    forbidden; the group's own shield flag remains untouched.
    """


@dataclass(frozen=True, slots=True)
class TaskGroupPolicyFrame:
    """An immutable set of task group policy choices."""

    spawn_scope_bind: SpawnScopeBind = SpawnScopeBind.PIN_DEFAULT
    start_reparent_gate: StartReparentGate = StartReparentGate.DEFER_UNTIL_STARTED
    group_exit_surface: GroupExitSurface = GroupExitSurface.AGGREGATE_EXCEPTION_GROUP


DEFAULT_TASK_GROUP_POLICY_FRAME = TaskGroupPolicyFrame()
"""
The frame observed when no policy has been pushed.  Its behavior is identical
to the historical task group behavior on every backend.
"""


@dataclass(frozen=True, slots=True)
class TaskGroupPolicyView:
    """
    A snapshot that merges the active policy frame with the runtime state of a
    specific task group.

    :ivar frame: the resolved policy frame
    :ivar cancel_scope: the group's cancel scope
    :vartype cancel_scope: ~anyio.CancelScope | None
    :ivar pending_tasks: number of child tasks that have not finished
    :ivar entered: ``True`` when the group context has been entered
    :ivar exiting: ``True`` while the group is running its ``__aexit__`` wait
    """

    frame: TaskGroupPolicyFrame
    cancel_scope: Any
    pending_tasks: int
    entered: bool
    exiting: bool


class TaskGroupPolicyToken:
    """Handle returned by :func:`push_task_group_policy` for manual popping."""

    __slots__ = ("_frame", "_token")

    def __init__(
        self, token: Token[tuple[TaskGroupPolicyFrame, ...]], frame: TaskGroupPolicyFrame
    ):
        self._token = token
        self._frame = frame

    @property
    def frame(self) -> TaskGroupPolicyFrame:
        """The frame that was pushed."""
        return self._frame


_policy_stack_var: ContextVar[tuple[TaskGroupPolicyFrame, ...]] = ContextVar(
    "_anyio_taskgroup_policy_stack", default=()
)

#: The spawn status of the task currently being spawned under a deferred
#: membership frame, or ``None``.  Inherited by child tasks via context copy.
_spawn_status_var: ContextVar[Any] = ContextVar(
    "_anyio_spawn_task_status", default=None
)


class _TreeRuntime:
    """Mutable, per spawn tree bookkeeping propagated by context inheritance."""

    __slots__ = ("exiting_host_ids",)

    def __init__(self) -> None:
        #: ids of host tasks that are currently inside a group __aexit__ wait
        self.exiting_host_ids: set[int] = set()


_tree_runtime_var: ContextVar[_TreeRuntime | None] = ContextVar(
    "_anyio_taskgroup_policy_tree", default=None
)


def _get_tree_runtime() -> _TreeRuntime:
    runtime = _tree_runtime_var.get()
    if runtime is None:
        runtime = _TreeRuntime()
        _tree_runtime_var.set(runtime)

    return runtime


def _bind_tree_runtime() -> None:
    """
    Ensure the spawn tree runtime exists in the current context.

    Called by ``TaskGroup.__aenter__`` before any child is spawned, so that
    child tasks inherit the same runtime object as the host task.
    """
    _get_tree_runtime()


def _begin_group_exit() -> int:
    """Mark the current host task's task group as being in its exit wait."""
    host_id = get_async_backend().get_current_task().id
    _get_tree_runtime().exiting_host_ids.add(host_id)
    return host_id


def _end_group_exit(host_id: int) -> None:
    if runtime := _tree_runtime_var.get():
        runtime.exiting_host_ids.discard(host_id)


def _ancestor_or_self_is_exiting() -> bool:
    runtime = _tree_runtime_var.get()
    if runtime is None or not runtime.exiting_host_ids:
        return False

    backend = get_async_backend()
    current = backend.get_current_task()
    parent_by_id = {
        task_info.id: task_info.parent_id for task_info in backend.get_running_tasks()
    }
    ancestor_id: int | None = current.id
    while ancestor_id is not None:
        if ancestor_id in runtime.exiting_host_ids:
            return True

        ancestor_id = parent_by_id.get(ancestor_id)

    return False


def get_current_task_group_policy() -> TaskGroupPolicyFrame:
    """
    Return the policy frame on top of the current task tree's stack, or
    :data:`DEFAULT_TASK_GROUP_POLICY_FRAME` when the stack is empty.
    """
    stack = _policy_stack_var.get()
    return stack[-1] if stack else DEFAULT_TASK_GROUP_POLICY_FRAME


def push_task_group_policy(
    frame: TaskGroupPolicyFrame | None = None,
    *,
    spawn_scope_bind: SpawnScopeBind = SpawnScopeBind.PIN_DEFAULT,
    start_reparent_gate: StartReparentGate = StartReparentGate.DEFER_UNTIL_STARTED,
    group_exit_surface: GroupExitSurface = GroupExitSurface.AGGREGATE_EXCEPTION_GROUP,
) -> TaskGroupPolicyToken:
    """
    Push a task group policy frame onto the current task tree's stack.

    Either pass a ready made :class:`TaskGroupPolicyFrame` or the individual
    fields.  Pair every push with :func:`pop_task_group_policy` (or use the
    :func:`task_group_policy` context manager, which also pops on exception
    paths).

    :raises RuntimeError: if a task group in the current task tree is currently
        in its ``__aexit__`` wait loop – frames cannot be applied to an
        in-flight group exit.
    """
    if _ancestor_or_self_is_exiting():
        raise RuntimeError(
            "cannot push a task group policy frame while a task group in the "
            "current task tree is in its __aexit__ wait loop"
        )

    if frame is None:
        frame = TaskGroupPolicyFrame(
            spawn_scope_bind=spawn_scope_bind,
            start_reparent_gate=start_reparent_gate,
            group_exit_surface=group_exit_surface,
        )

    stack = _policy_stack_var.get()
    token = _policy_stack_var.set((*stack, frame))
    return TaskGroupPolicyToken(token, frame)


def pop_task_group_policy(token: TaskGroupPolicyToken) -> None:
    """
    Pop a frame previously pushed with :func:`push_task_group_policy`.

    :raises RuntimeError: if the given frame is not the current stack top
        (frames must be popped in reverse push order)
    """
    stack = _policy_stack_var.get()
    if not stack or stack[-1] is not token.frame:
        raise RuntimeError(
            "the task group policy frame being popped is not the topmost frame"
        )

    _policy_stack_var.reset(token._token)


@contextmanager
def task_group_policy(
    frame: TaskGroupPolicyFrame | None = None,
    *,
    spawn_scope_bind: SpawnScopeBind = SpawnScopeBind.PIN_DEFAULT,
    start_reparent_gate: StartReparentGate = StartReparentGate.DEFER_UNTIL_STARTED,
    group_exit_surface: GroupExitSurface = GroupExitSurface.AGGREGATE_EXCEPTION_GROUP,
) -> Generator[TaskGroupPolicyFrame]:
    """
    Context manager that pushes a frame for the duration of the ``with`` block
    and pops it again on exit, including when the block raises.
    """
    token = push_task_group_policy(
        frame,
        spawn_scope_bind=spawn_scope_bind,
        start_reparent_gate=start_reparent_gate,
        group_exit_surface=group_exit_surface,
    )
    try:
        yield token.frame
    finally:
        stack = _policy_stack_var.get()
        if stack and stack[-1] is token.frame:
            _policy_stack_var.reset(token._token)


def current_task_status() -> Any:
    """
    Return the :class:`~anyio.abc.TaskStatus` associated with the current
    spawn.

    This is set for child tasks spawned while a frame with
    :attr:`SpawnScopeBind.KEEP_CALLER_UNTIL_STARTED` is active (on every
    spawn path).  Such a task must call ``started()`` on the returned status
    to explicitly move into the group cancel scope.  Returns ``None`` outside
    such a spawn.
    """
    return _spawn_status_var.get()


@contextmanager
def _install_spawn_status(status: Any) -> Generator[None]:
    token = _spawn_status_var.set(status)
    try:
        yield
    finally:
        _spawn_status_var.reset(token)
