"""Tests for per-session (multi-user) Odoo authentication.

Covers the pieces in ``mcp_server_odoo/multiuser.py`` (credential
extraction, the connection pool, the proxies, and the app-wrapping
decorator interceptor) plus the ``OdooMCPServer`` integration points that
wire it up: ``_resolve_connection``, config validation, and the
per-session-auth branches of ``_ensure_connection`` / ``_register_*``.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_server_odoo.config import OdooConfig
from mcp_server_odoo.multiuser import (
    AccessControllerProxy,
    ConnectionProxy,
    MissingSessionCredentialsError,
    SessionBindingApp,
    SessionConnectionPool,
    SessionCredentials,
    _active_pair_var,
    extract_session_credentials,
)
from mcp_server_odoo.odoo_connection import OdooConnectionError
from mcp_server_odoo.server import OdooMCPServer


def _make_ctx(headers=None):
    """Build a fake FastMCP Context exposing ctx.request_context.request.headers."""
    ctx = Mock()
    if headers is None:
        ctx.request_context = None
        return ctx
    ctx.request_context.request.headers = headers
    return ctx


class TestExtractSessionCredentials:
    def test_no_ctx_returns_none(self):
        assert extract_session_credentials(None, "X-Odoo-User", "X-Odoo-Api-Key") is None

    def test_no_request_context_returns_none(self):
        ctx = _make_ctx(headers=None)
        assert extract_session_credentials(ctx, "X-Odoo-User", "X-Odoo-Api-Key") is None

    def test_request_context_without_request_returns_none(self):
        ctx = Mock()
        ctx.request_context.request = None
        assert extract_session_credentials(ctx, "X-Odoo-User", "X-Odoo-Api-Key") is None

    def test_both_headers_present_returns_credentials(self):
        ctx = _make_ctx({"X-Odoo-User": " alice@example.com ", "X-Odoo-Api-Key": " secret "})
        creds = extract_session_credentials(ctx, "X-Odoo-User", "X-Odoo-Api-Key")
        assert creds == SessionCredentials(username="alice@example.com", api_key="secret")

    def test_missing_api_key_header_returns_none(self):
        ctx = _make_ctx({"X-Odoo-User": "alice@example.com"})
        assert extract_session_credentials(ctx, "X-Odoo-User", "X-Odoo-Api-Key") is None

    def test_missing_user_header_returns_none(self):
        ctx = _make_ctx({"X-Odoo-Api-Key": "secret"})
        assert extract_session_credentials(ctx, "X-Odoo-User", "X-Odoo-Api-Key") is None

    def test_empty_header_values_return_none(self):
        ctx = _make_ctx({"X-Odoo-User": "", "X-Odoo-Api-Key": ""})
        assert extract_session_credentials(ctx, "X-Odoo-User", "X-Odoo-Api-Key") is None

    def test_custom_header_names_are_honored(self):
        ctx = _make_ctx({"X-Custom-User": "bob", "X-Custom-Key": "k"})
        creds = extract_session_credentials(ctx, "X-Custom-User", "X-Custom-Key")
        assert creds == SessionCredentials(username="bob", api_key="k")


class TestConnectionProxies:
    def test_raises_when_no_pair_bound(self):
        proxy = ConnectionProxy()
        with pytest.raises(RuntimeError, match="No Odoo connection bound"):
            _ = proxy.some_attr

    def test_forwards_to_bound_connection(self):
        mock_connection = Mock()
        mock_connection.search_read.return_value = ["record"]
        mock_access = Mock()

        token = _active_pair_var.set((mock_connection, mock_access))
        try:
            conn_proxy = ConnectionProxy()
            ac_proxy = AccessControllerProxy()
            assert conn_proxy.search_read() == ["record"]
            assert ac_proxy.check_permission is mock_access.check_permission
        finally:
            _active_pair_var.reset(token)

        # Once unbound, the proxy raises again
        with pytest.raises(RuntimeError):
            _ = ConnectionProxy().anything


class TestSessionConnectionPool:
    def _make_pool(self, idle_timeout=1800.0):
        config = OdooConfig(
            url="http://localhost:8069", per_session_auth=True, transport="streamable-http"
        )
        return SessionConnectionPool(config, idle_timeout=idle_timeout)

    @pytest.mark.asyncio
    async def test_get_authenticates_new_credentials(self):
        pool = self._make_pool()
        creds = SessionCredentials(username="alice", api_key="key1")

        with (
            patch("mcp_server_odoo.multiuser.OdooConnection") as mock_conn_cls,
            patch("mcp_server_odoo.multiuser.AccessController") as mock_ac_cls,
        ):
            mock_connection = Mock()
            mock_connection.is_authenticated = True
            mock_connection.database = "db"
            mock_connection.auth_method = "api_key"
            mock_conn_cls.return_value = mock_connection

            connection, access_controller = await pool.get(creds)

            mock_connection.connect.assert_called_once()
            mock_connection.authenticate.assert_called_once()
            assert connection is mock_connection
            assert access_controller is mock_ac_cls.return_value
            assert len(pool) == 1

    @pytest.mark.asyncio
    async def test_get_reuses_cached_connection(self):
        pool = self._make_pool()
        creds = SessionCredentials(username="alice", api_key="key1")

        with (
            patch("mcp_server_odoo.multiuser.OdooConnection") as mock_conn_cls,
            patch("mcp_server_odoo.multiuser.AccessController"),
        ):
            mock_connection = Mock()
            mock_connection.is_authenticated = True
            mock_conn_cls.return_value = mock_connection

            first = await pool.get(creds)
            second = await pool.get(creds)

            assert first == second
            mock_conn_cls.assert_called_once()
            mock_connection.authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_reauthenticates_when_stale(self):
        pool = self._make_pool()
        creds = SessionCredentials(username="alice", api_key="key1")

        with (
            patch("mcp_server_odoo.multiuser.OdooConnection") as mock_conn_cls,
            patch("mcp_server_odoo.multiuser.AccessController"),
        ):
            mock_connection = Mock()
            mock_connection.is_authenticated = False
            mock_conn_cls.return_value = mock_connection

            await pool.get(creds)
            await pool.get(creds)

            # A second, fresh OdooConnection is built each time the cached
            # entry is no longer authenticated.
            assert mock_conn_cls.call_count == 2

    @pytest.mark.asyncio
    async def test_get_separates_different_users(self):
        pool = self._make_pool()
        alice = SessionCredentials(username="alice", api_key="key1")
        bob = SessionCredentials(username="bob", api_key="key2")

        with (
            patch("mcp_server_odoo.multiuser.OdooConnection") as mock_conn_cls,
            patch("mcp_server_odoo.multiuser.AccessController"),
        ):
            mock_conn_cls.side_effect = lambda *a, **k: Mock(is_authenticated=True)

            conn_a, _ = await pool.get(alice)
            conn_b, _ = await pool.get(bob)

            assert conn_a is not conn_b
            assert len(pool) == 2

    @pytest.mark.asyncio
    async def test_authenticate_wraps_unexpected_errors(self):
        pool = self._make_pool()
        creds = SessionCredentials(username="alice", api_key="bad")

        with patch("mcp_server_odoo.multiuser.OdooConnection") as mock_conn_cls:
            mock_connection = Mock()
            mock_connection.connect.side_effect = RuntimeError("boom")
            mock_conn_cls.return_value = mock_connection

            with pytest.raises(OdooConnectionError, match="Failed to authenticate session user"):
                await pool.get(creds)

    @pytest.mark.asyncio
    async def test_authenticate_propagates_odoo_connection_error(self):
        pool = self._make_pool()
        creds = SessionCredentials(username="alice", api_key="bad")

        with patch("mcp_server_odoo.multiuser.OdooConnection") as mock_conn_cls:
            mock_connection = Mock()
            mock_connection.connect.side_effect = OdooConnectionError("auth rejected")
            mock_conn_cls.return_value = mock_connection

            with pytest.raises(OdooConnectionError, match="auth rejected"):
                await pool.get(creds)

    @pytest.mark.asyncio
    async def test_idle_entries_are_evicted_and_disconnected(self):
        pool = self._make_pool(idle_timeout=0.01)
        alice = SessionCredentials(username="alice", api_key="key1")
        bob = SessionCredentials(username="bob", api_key="key2")

        with (
            patch("mcp_server_odoo.multiuser.OdooConnection") as mock_conn_cls,
            patch("mcp_server_odoo.multiuser.AccessController"),
        ):
            mock_alice_conn = Mock(is_authenticated=True)
            mock_bob_conn = Mock(is_authenticated=True)
            mock_conn_cls.side_effect = [mock_alice_conn, mock_bob_conn]

            await pool.get(alice)
            assert len(pool) == 1

            import asyncio

            await asyncio.sleep(0.05)
            await pool.get(bob)

            # Alice's idle entry was evicted (and disconnected) before Bob's
            # was inserted.
            mock_alice_conn.disconnect.assert_called_once_with(True)
            assert len(pool) == 1

    @pytest.mark.asyncio
    async def test_close_all_disconnects_every_entry(self):
        pool = self._make_pool()
        alice = SessionCredentials(username="alice", api_key="key1")
        bob = SessionCredentials(username="bob", api_key="key2")

        with (
            patch("mcp_server_odoo.multiuser.OdooConnection") as mock_conn_cls,
            patch("mcp_server_odoo.multiuser.AccessController"),
        ):
            mock_alice_conn = Mock(is_authenticated=True)
            mock_bob_conn = Mock(is_authenticated=True)
            mock_conn_cls.side_effect = [mock_alice_conn, mock_bob_conn]

            await pool.get(alice)
            await pool.get(bob)

            await pool.close_all()

            mock_alice_conn.disconnect.assert_called_once_with(True)
            mock_bob_conn.disconnect.assert_called_once_with(True)
            assert len(pool) == 0


class _FakeFastMCP:
    """Minimal stand-in for FastMCP's tool()/resource() decorator API."""

    def __init__(self):
        self.registered = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.registered[fn.__name__] = fn
            return fn

        return decorator

    def resource(self, *args, **kwargs):
        return self.tool(*args, **kwargs)

    def custom_route(self, *args, **kwargs):
        return "not-a-tool-decorator"


class TestSessionBindingApp:
    @pytest.mark.asyncio
    async def test_binds_resolved_pair_for_call_duration_then_clears(self):
        fake_app = _FakeFastMCP()
        mock_connection = Mock()
        mock_access = Mock()
        resolve = AsyncMock(return_value=(mock_connection, mock_access))

        wrapped = SessionBindingApp(fake_app, resolve)

        @wrapped.tool()
        async def my_tool(x: int, ctx=None) -> str:
            assert ConnectionProxy().dummy is mock_connection.dummy
            assert AccessControllerProxy().dummy is mock_access.dummy
            return str(x)

        # The real function object registered on the underlying app is the
        # wrapper — tools.py/resources.py never change, so FastMCP always
        # calls through this wrapper.
        registered = fake_app.registered["my_tool"]
        result = await registered(x=5, ctx="fake-ctx")

        assert result == "5"
        resolve.assert_awaited_once_with("fake-ctx")
        # Bound only for the duration of the call
        assert _active_pair_var.get() is None

    @pytest.mark.asyncio
    async def test_resolve_receives_ctx_from_positional_args(self):
        fake_app = _FakeFastMCP()
        resolve = AsyncMock(return_value=(Mock(), Mock()))
        wrapped = SessionBindingApp(fake_app, resolve)

        @wrapped.tool()
        async def positional_tool(x, ctx) -> int:
            return x

        registered = fake_app.registered["positional_tool"]
        await registered(1, "ctx-positional")

        resolve.assert_awaited_once_with("ctx-positional")

    def test_non_tool_attributes_pass_through_unwrapped(self):
        fake_app = _FakeFastMCP()
        wrapped = SessionBindingApp(fake_app, AsyncMock())

        assert wrapped.custom_route() == "not-a-tool-decorator"

    @pytest.mark.asyncio
    async def test_resolve_error_prevents_call_and_binding(self):
        fake_app = _FakeFastMCP()
        resolve = AsyncMock(side_effect=MissingSessionCredentialsError("no creds"))
        wrapped = SessionBindingApp(fake_app, resolve)

        called = False

        @wrapped.tool()
        async def my_tool(ctx=None) -> str:
            nonlocal called
            called = True
            return "ok"

        registered = fake_app.registered["my_tool"]
        with pytest.raises(MissingSessionCredentialsError):
            await registered(ctx="fake-ctx")

        assert called is False
        assert _active_pair_var.get() is None


class TestConfigPerSessionAuth:
    def test_per_session_auth_requires_streamable_http(self):
        with pytest.raises(ValueError, match="requires ODOO_MCP_TRANSPORT=streamable-http"):
            OdooConfig(url="http://localhost:8069", per_session_auth=True, transport="stdio")

    def test_per_session_auth_allows_missing_global_credentials(self):
        config = OdooConfig(
            url="http://localhost:8069", per_session_auth=True, transport="streamable-http"
        )
        assert config.api_key is None
        assert config.username is None

    def test_per_session_auth_keeps_global_credentials_as_fallback(self):
        config = OdooConfig(
            url="http://localhost:8069",
            per_session_auth=True,
            transport="streamable-http",
            api_key="fallback-key",
            username="fallback-user",
        )
        assert config.api_key == "fallback-key"

    def test_non_positive_idle_timeout_raises(self):
        with pytest.raises(ValueError, match="SESSION_CONNECTION_IDLE_TIMEOUT"):
            OdooConfig(url="http://localhost:8069", api_key="k", session_connection_idle_timeout=0)

    def test_default_per_session_auth_is_off(self):
        config = OdooConfig(url="http://localhost:8069", api_key="k")
        assert config.per_session_auth is False
        assert config.session_user_header == "X-Odoo-User"
        assert config.session_api_key_header == "X-Odoo-Api-Key"
        assert config.session_connection_idle_timeout == 1800.0

    def test_load_config_reads_per_session_env_vars(self, monkeypatch):
        monkeypatch.setenv("ODOO_URL", "http://localhost:8069")
        monkeypatch.setenv("ODOO_API_KEY", "k")
        monkeypatch.setenv("ODOO_MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("ODOO_MCP_PER_SESSION_AUTH", "true")
        monkeypatch.setenv("ODOO_MCP_SESSION_USER_HEADER", "X-Custom-User")
        monkeypatch.setenv("ODOO_MCP_SESSION_API_KEY_HEADER", "X-Custom-Key")
        monkeypatch.setenv("ODOO_MCP_SESSION_CONNECTION_IDLE_TIMEOUT", "60")

        from mcp_server_odoo.config import load_config

        config = load_config()

        assert config.per_session_auth is True
        assert config.session_user_header == "X-Custom-User"
        assert config.session_api_key_header == "X-Custom-Key"
        assert config.session_connection_idle_timeout == 60.0


class TestServerResolveConnection:
    def _config(self, **overrides):
        kwargs = dict(url="http://localhost:8069", api_key="global-key", username="global-user")
        kwargs.update(overrides)
        return OdooConfig(**kwargs)

    @pytest.mark.asyncio
    async def test_returns_shared_connection_when_pool_disabled(self):
        server = OdooMCPServer(self._config())
        server.connection = Mock()
        server.access_controller = Mock()

        result = await server._resolve_connection(ctx=None)

        assert result == (server.connection, server.access_controller)

    @pytest.mark.asyncio
    async def test_uses_pool_when_headers_present(self):
        server = OdooMCPServer(self._config(per_session_auth=True, transport="streamable-http"))
        pooled_pair = (Mock(), Mock())
        server._session_pool = Mock()
        server._session_pool.get = AsyncMock(return_value=pooled_pair)

        ctx = _make_ctx({"X-Odoo-User": "alice", "X-Odoo-Api-Key": "key"})
        result = await server._resolve_connection(ctx)

        assert result == pooled_pair
        server._session_pool.get.assert_awaited_once_with(
            SessionCredentials(username="alice", api_key="key")
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_global_connection_without_headers(self):
        server = OdooMCPServer(self._config(per_session_auth=True, transport="streamable-http"))
        server._session_pool = Mock()
        server.connection = Mock()
        server.access_controller = Mock()

        result = await server._resolve_connection(ctx=_make_ctx(headers=None))

        assert result == (server.connection, server.access_controller)
        server._session_pool.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_no_headers_and_no_fallback(self):
        server = OdooMCPServer(
            OdooConfig(
                url="http://localhost:8069",
                per_session_auth=True,
                transport="streamable-http",
            )
        )
        server._session_pool = Mock()

        with pytest.raises(MissingSessionCredentialsError, match="X-Odoo-User"):
            await server._resolve_connection(ctx=_make_ctx(headers=None))


class TestServerEnsureConnectionPerSessionAuth:
    def test_skips_eager_connect_without_fallback_credentials(self):
        config = OdooConfig(
            url="http://localhost:8069", per_session_auth=True, transport="streamable-http"
        )
        server = OdooMCPServer(config)

        with patch("mcp_server_odoo.server.OdooConnection") as mock_conn_cls:
            server._ensure_connection()
            mock_conn_cls.assert_not_called()

        assert server.connection is None

    def test_still_connects_eagerly_with_fallback_credentials(self):
        config = OdooConfig(
            url="http://localhost:8069",
            per_session_auth=True,
            transport="streamable-http",
            api_key="fallback-key",
            username="fallback-user",
        )
        server = OdooMCPServer(config)

        with (
            patch("mcp_server_odoo.server.OdooConnection") as mock_conn_cls,
            patch("mcp_server_odoo.server.AccessController"),
        ):
            mock_connection = Mock()
            mock_connection.is_authenticated = True
            mock_conn_cls.return_value = mock_connection

            server._ensure_connection()

            mock_connection.connect.assert_called_once()
        assert server.connection is mock_connection


class TestServerRegistrationPerSessionAuth:
    def test_register_tools_and_resources_use_proxies_and_wrapped_app(self):
        config = OdooConfig(
            url="http://localhost:8069", per_session_auth=True, transport="streamable-http"
        )
        server = OdooMCPServer(config)

        with (
            patch("mcp_server_odoo.server.register_resources") as mock_reg_res,
            patch("mcp_server_odoo.server.register_tools") as mock_reg_tools,
        ):
            mock_reg_res.return_value = Mock()
            mock_reg_tools.return_value = Mock()

            server._register_resources()
            server._register_tools()

            res_args = mock_reg_res.call_args[0]
            assert isinstance(res_args[0], SessionBindingApp)
            assert isinstance(res_args[1], ConnectionProxy)
            assert isinstance(res_args[2], AccessControllerProxy)

            tools_args = mock_reg_tools.call_args[0]
            assert isinstance(tools_args[0], SessionBindingApp)
            assert isinstance(tools_args[1], ConnectionProxy)
            assert isinstance(tools_args[2], AccessControllerProxy)

    @pytest.mark.asyncio
    async def test_run_http_closes_pool_on_shutdown(self):
        config = OdooConfig(
            url="http://localhost:8069", per_session_auth=True, transport="streamable-http"
        )
        server = OdooMCPServer(config)
        server._session_pool.close_all = AsyncMock()
        server.app.run_streamable_http_async = AsyncMock()
        server._preseed_session_manager = Mock()
        server._warn_if_exposed = Mock()

        await server.run_http(host="localhost", port=8000)

        server._session_pool.close_all.assert_awaited_once()
