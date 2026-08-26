"""Per-user (multi-tenant) authentication for streamable-http deployments.

By default this server authenticates to Odoo ONCE at startup using the
credentials in the environment (``ODOO_API_KEY`` / ``ODOO_USER`` /
``ODOO_PASSWORD``) and reuses that single connection for every MCP client
that connects — fine for a local ``stdio`` process (one process per user
already), but wrong for a shared ``streamable-http`` deployment: every
action in Odoo ends up attributed to the same user, breaking audit trails
and per-user permissions.

This module adds an *opt-in* per-session mode (``ODOO_MCP_PER_SESSION_AUTH=true``).
When enabled, each MCP request may carry the caller's own Odoo identity via
HTTP headers (default ``X-Odoo-User`` / ``X-Odoo-Api-Key``). A small pool
keeps one authenticated ``OdooConnection`` per distinct set of credentials,
reused across calls, and evicted after an idle timeout.

Design notes
------------
- Credentials are extracted from ``ctx.request_context.request`` — the raw
  Starlette ``Request`` the MCP Python SDK attaches per JSON-RPC call (not
  per transport session), so this works even though streamable-http
  sessions are long-lived and multiplex many calls.
- ``ConnectionProxy`` / ``AccessControllerProxy`` are transparent
  passthrough objects. ``tools.py`` and ``resources.py`` are written against
  a single ``self.connection`` / ``self.access_controller`` attribute set
  once at registration time — swapping the *real* objects for these proxies
  means zero changes are needed in either file. Each proxy resolves the
  actual object from a contextvar that is set for the duration of a single
  tool/resource call by ``SessionBindingApp``.
- ``SessionBindingApp`` wraps the FastMCP ``app`` object passed into
  ``register_tools`` / ``register_resources`` so every ``@app.tool()`` /
  ``@app.resource()`` registered function is transparently pre-wrapped to
  resolve and bind the caller's connection before running, and to clear it
  afterward. This is the single choke point — no per-tool edits required.
- The wrapper relies on ``functools.wraps`` making ``inspect.signature()``
  (and, since neither tools.py nor resources.py use
  ``from __future__ import annotations``, ``typing.get_type_hints()`` too)
  transparently see through to the original function. FastMCP uses both to
  find the ``ctx: Context`` parameter and to build the tool's argument
  schema. If either module ever switches to string annotations, that
  resolution would need the wrapper's own module globals to also expose
  ``Context``, ``Optional``, etc.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from .access_control import AccessController
from .config import OdooConfig
from .odoo_connection import OdooConnection, OdooConnectionError
from .performance import PerformanceManager

logger = logging.getLogger(__name__)


class MissingSessionCredentialsError(Exception):
    """Raised when per-session auth is required but the request has none."""


@dataclass(frozen=True)
class SessionCredentials:
    """The Odoo identity a single caller presented via HTTP headers."""

    username: str
    api_key: str


# Holds the (connection, access_controller) pair resolved for the call
# currently in flight. Set by SessionBindingApp before invoking the wrapped
# tool/resource function, cleared afterward. Never touched by tools.py or
# resources.py directly — they only ever see the proxies below.
_active_pair_var: ContextVar[Optional[Tuple[OdooConnection, AccessController]]] = ContextVar(
    "odoo_mcp_active_pair", default=None
)


def _active_pair() -> Tuple[OdooConnection, AccessController]:
    pair = _active_pair_var.get()
    if pair is None:  # pragma: no cover - defensive, SessionBindingApp always sets this
        raise RuntimeError(
            "No Odoo connection bound for this call. This is a bug in the "
            "session-binding wrapper — the tool/resource ran outside of it."
        )
    return pair


class ConnectionProxy:
    """Transparent stand-in for OdooConnection, resolved per-call."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_active_pair()[0], name)


class AccessControllerProxy:
    """Transparent stand-in for AccessController, resolved per-call."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_active_pair()[1], name)


def extract_session_credentials(
    ctx: Any,
    user_header: str,
    api_key_header: str,
) -> Optional[SessionCredentials]:
    """Pull per-user Odoo credentials from the current call's HTTP headers.

    Returns None for stdio (no HTTP request at all) or when the headers are
    simply absent — callers decide whether that's a fallback-to-global
    situation or an error.
    """
    if ctx is None:
        return None
    request_context = getattr(ctx, "request_context", None)
    request = getattr(request_context, "request", None) if request_context else None
    headers = getattr(request, "headers", None)
    if headers is None:
        return None

    username = headers.get(user_header)
    api_key = headers.get(api_key_header)
    if not username or not api_key:
        return None
    return SessionCredentials(username=username.strip(), api_key=api_key.strip())


@dataclass
class _PoolEntry:
    connection: OdooConnection
    access_controller: AccessController
    last_used: float


class SessionConnectionPool:
    """Caches one authenticated OdooConnection per distinct set of credentials.

    Re-authenticating with Odoo on every single tool call would add a
    network round trip to every request, so entries are kept alive and
    reused until they've been idle for ``idle_timeout`` seconds.
    """

    def __init__(
        self,
        base_config: OdooConfig,
        performance_manager: Optional[PerformanceManager] = None,
        idle_timeout: float = 1800.0,
    ):
        self._base_config = base_config
        self._performance_manager = performance_manager
        self._idle_timeout = idle_timeout
        self._entries: Dict[SessionCredentials, _PoolEntry] = {}
        self._locks: Dict[SessionCredentials, asyncio.Lock] = {}
        self._directory_lock = asyncio.Lock()

    async def get(self, creds: SessionCredentials) -> Tuple[OdooConnection, AccessController]:
        """Return a live, authenticated (connection, access_controller) pair for these credentials."""
        await self._evict_idle()

        async with self._directory_lock:
            lock = self._locks.setdefault(creds, asyncio.Lock())

        async with lock:
            entry = self._entries.get(creds)
            if entry is not None and entry.connection.is_authenticated:
                entry.last_used = time.monotonic()
                return entry.connection, entry.access_controller

            connection, access_controller = await asyncio.to_thread(self._authenticate, creds)
            self._entries[creds] = _PoolEntry(
                connection=connection,
                access_controller=access_controller,
                last_used=time.monotonic(),
            )
            return connection, access_controller

    def _authenticate(self, creds: SessionCredentials) -> Tuple[OdooConnection, AccessController]:
        """Blocking: connect + authenticate a fresh OdooConnection for one user. Runs off the event loop."""
        per_user_config = dataclasses.replace(
            self._base_config,
            username=creds.username,
            api_key=creds.api_key,
            password=None,
        )
        connection = OdooConnection(per_user_config, performance_manager=self._performance_manager)
        try:
            connection.connect()
            connection.authenticate()
        except OdooConnectionError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise OdooConnectionError(f"Failed to authenticate session user: {e}") from e

        access_controller = AccessController(
            per_user_config,
            database=connection.database,
            auth_method=connection.auth_method,
        )
        logger.info("Authenticated per-session Odoo connection for user '%s'", creds.username)
        return connection, access_controller

    async def _evict_idle(self) -> None:
        if self._idle_timeout <= 0:
            return
        now = time.monotonic()
        async with self._directory_lock:
            stale = [
                creds
                for creds, entry in self._entries.items()
                if now - entry.last_used > self._idle_timeout
            ]
            for creds in stale:
                entry = self._entries.pop(creds, None)
                self._locks.pop(creds, None)
                if entry is not None:
                    logger.info(
                        "Evicting idle per-session Odoo connection for user '%s'", creds.username
                    )
                    await asyncio.to_thread(entry.connection.disconnect, True)

    async def close_all(self) -> None:
        """Disconnect every pooled connection. Call on server shutdown."""
        async with self._directory_lock:
            entries, self._entries = self._entries, {}
            self._locks.clear()
        for entry in entries.values():
            await asyncio.to_thread(entry.connection.disconnect, True)

    def __len__(self) -> int:
        return len(self._entries)


class SessionBindingApp:
    """Wraps a FastMCP app so every registered tool/resource resolves its
    own per-caller Odoo connection before running.

    Passed into ``register_tools`` / ``register_resources`` in place of the
    raw FastMCP instance — those modules are unaware anything changed.
    """

    def __init__(
        self,
        app: Any,
        resolve: Callable[[Any], Awaitable[Tuple[OdooConnection, AccessController]]],
    ):
        self._app = app
        self._resolve = resolve

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._app, name)
        if name not in ("tool", "resource"):
            return attr

        def decorator_factory(*d_args: Any, **d_kwargs: Any):
            real_decorator = attr(*d_args, **d_kwargs)

            def wrap_fn(fn: Callable[..., Awaitable[Any]]):
                @functools.wraps(fn)
                async def bound(*args: Any, **kwargs: Any) -> Any:
                    ctx = kwargs.get("ctx")
                    if ctx is None and args:
                        ctx = args[-1]
                    pair = await self._resolve(ctx)
                    token = _active_pair_var.set(pair)
                    try:
                        return await fn(*args, **kwargs)
                    finally:
                        _active_pair_var.reset(token)

                return real_decorator(bound)

            return wrap_fn

        return decorator_factory
