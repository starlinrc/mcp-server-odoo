"""MCP Server implementation for Odoo.

This module provides the FastMCP server that exposes Odoo data
and functionality through the Model Context Protocol.
"""

import asyncio
import contextlib
from typing import Any, Dict, Optional, Tuple

from mcp.server import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from . import __version__
from .access_control import AccessController
from .config import OdooConfig, get_config
from .error_handling import (
    ConfigurationError,
    ErrorContext,
    error_handler,
)
from .logging_config import get_logger, logging_config, perf_logger
from .multiuser import (
    AccessControllerProxy,
    ConnectionProxy,
    MissingSessionCredentialsError,
    SessionBindingApp,
    SessionConnectionPool,
    extract_session_credentials,
)
from .odoo_connection import OdooConnection, OdooConnectionError
from .performance import PerformanceManager
from .resources import register_resources
from .tools import register_tools

# Set up logging
logger = get_logger(__name__)

# Server version — single-sourced from the package
SERVER_VERSION = __version__


class OdooMCPServer:
    """Main MCP server class for Odoo integration.

    This class manages the FastMCP server instance and maintains
    the connection to Odoo. The server lifecycle is managed by
    establishing connection before starting and cleaning up on exit.
    """

    def __init__(self, config: Optional[OdooConfig] = None):
        """Initialize the Odoo MCP server.

        Args:
            config: Optional OdooConfig instance. If not provided,
                   will load from environment variables.
        """
        # Load configuration
        self.config = config or get_config()

        # Set up structured logging with the validated config level
        logging_config.setup(log_level=self.config.log_level)

        # Initialize connection and access controller (will be created on startup)
        self.connection: Optional[OdooConnection] = None
        self.access_controller: Optional[AccessController] = None
        self.performance_manager: Optional[PerformanceManager] = None
        self.resource_handler = None
        self.tool_handler = None

        # Serializes connection setup/reauth across concurrent lifespan
        # entries (streamable-http enters the lifespan per session)
        self._connect_lock = asyncio.Lock()

        # Multi-user mode: each caller authenticates with their own Odoo
        # identity (see multiuser.py) instead of everyone sharing
        # self.connection. None when the feature is off — the fast path
        # used by every other deployment stays untouched.
        self._session_pool: Optional[SessionConnectionPool] = (
            SessionConnectionPool(
                self.config, idle_timeout=self.config.session_connection_idle_timeout
            )
            if self.config.per_session_auth
            else None
        )

        # Configure transport security for DNS rebinding protection. Left as
        # None (no allowed_hosts configured) the SDK middleware defaults to
        # protection DISABLED — preserving prior behavior for stdio and for
        # HTTP deployments that don't set ODOO_MCP_ALLOWED_HOSTS.
        transport_security = self._build_transport_security()

        # Create FastMCP instance with server metadata
        self.app = FastMCP(
            name="odoo-mcp-server",
            instructions="MCP server for accessing and managing Odoo ERP data through the Model Context Protocol",
            lifespan=self._odoo_lifespan,
            host=self.config.host,
            transport_security=transport_security,
        )

        @self.app.custom_route("/health", methods=["GET"])
        async def health_check(request):
            from starlette.responses import JSONResponse

            return JSONResponse(self.get_health_status())

        @self.app.completion()
        async def handle_completion(ref, argument, context):
            from mcp.types import Completion

            if argument.name == "model":
                model_names = await asyncio.to_thread(self._get_model_names)
                partial = argument.value or ""
                if partial:
                    matches = [m for m in model_names if partial.lower() in m.lower()]
                else:
                    matches = model_names
                return Completion(values=matches[:20])
            return None

        logger.info(f"Initialized Odoo MCP Server v{SERVER_VERSION}")

    @contextlib.asynccontextmanager
    async def _odoo_lifespan(self, app: FastMCP):
        """Manage Odoo connection lifecycle for FastMCP.

        Sets up connection, registers resources/tools before serving.

        The low-level MCP server enters this context PER SESSION. Under
        stdio there is exactly one session per process, so cleaning up on
        exit is correct. Under streamable-http every client session (and
        every ``DELETE /mcp``) exits and re-enters it — tearing down the
        authenticated Odoo connection there broke every call after the
        first (#70). The connection must persist across HTTP sessions;
        the OS reclaims it at process exit.
        """
        try:
            with perf_logger.track_operation("server_startup"):
                # Connection setup is sync XML-RPC/urllib I/O (up to the
                # socket timeout) — keep it off the event loop. The lock
                # preserves the serialization that running on the loop's
                # single thread used to provide.
                async with self._connect_lock:
                    await asyncio.to_thread(self._ensure_connection)
                self._register_resources()
                self._register_tools()
            yield {}
        finally:
            if self.config.transport != "streamable-http":
                self._cleanup_connection()

    def _ensure_connection(self):
        """Ensure connection to Odoo is established.

        Reuses an existing authenticated connection (streamable-http
        re-enters the lifespan per session — see ``_odoo_lifespan``).

        Raises:
            ConnectionError: If connection fails
            ConfigurationError: If configuration is invalid
        """
        if (
            self._session_pool is not None
            and not self.connection
            and not (self.config.api_key or (self.config.username and self.config.password))
        ):
            logger.info(
                "Per-session auth enabled with no global fallback credentials — "
                "skipping eager connection; each call authenticates as its own caller"
            )
            return

        if self.connection and self.connection.is_authenticated:
            logger.info("Reusing existing authenticated Odoo connection")
            return
        if self.connection:
            # Reconnect the existing object IN PLACE: registered tool and
            # resource handlers hold references to this connection, so it
            # must never be replaced with a new instance.
            logger.warning("Existing connection is not authenticated; reconnecting")
            try:
                with perf_logger.track_operation("connection_reauth"):
                    if not self.connection.is_connected:
                        self.connection.connect()
                    self.connection.authenticate()
                # Reauth re-runs the api-key→password fallback chain, so the
                # effective auth method may differ from the initial connect.
                # The controller may not exist at all if the first startup
                # failed after self.connection was assigned but before auth
                # succeeded — without it, handler registration silently skips.
                if self.access_controller is None:
                    self.access_controller = AccessController(
                        self.config,
                        database=self.connection.database,
                        auth_method=self.connection.auth_method,
                    )
                else:
                    self.access_controller.auth_method = self.connection.auth_method
                return
            except Exception as e:
                context = ErrorContext(operation="connection_reauth")
                if isinstance(e, (OdooConnectionError, ConfigurationError)):
                    raise
                # handle_error reraises (reraise defaults to True) — reauth
                # failures always propagate to the session
                error_handler.handle_error(e, context=context)
        if not self.connection:
            try:
                logger.info("Establishing connection to Odoo...")
                with perf_logger.track_operation("connection_setup"):
                    # Create performance manager (shared across components)
                    self.performance_manager = PerformanceManager(self.config)

                    # Create connection with performance manager
                    self.connection = OdooConnection(
                        self.config, performance_manager=self.performance_manager
                    )

                    # Connect and authenticate
                    self.connection.connect()
                    self.connection.authenticate()

                logger.info(f"Successfully connected to Odoo at {self.config.url}")

                # Initialize access controller (pass resolved DB for session
                # auth and the EFFECTIVE auth method — after a password
                # fallback, permission checks must not send the rejected key)
                self.access_controller = AccessController(
                    self.config,
                    database=self.connection.database,
                    auth_method=self.connection.auth_method,
                )
            except Exception as e:
                context = ErrorContext(operation="connection_setup")
                # Let specific errors propagate as-is
                if isinstance(e, (OdooConnectionError, ConfigurationError)):
                    raise
                # Handle other unexpected errors
                error_handler.handle_error(e, context=context)

    def _cleanup_connection(self):
        """Clean up Odoo connection."""
        if self.connection:
            try:
                logger.info("Closing Odoo connection...")
                self.connection.disconnect()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                # Always clear connection reference
                self.connection = None
                self.access_controller = None
                self.resource_handler = None
                self.tool_handler = None

    async def _resolve_connection(self, ctx: Any) -> Tuple[OdooConnection, AccessController]:
        """Resolve which Odoo identity a single tool/resource call should use.

        Called by SessionBindingApp before every tool/resource invocation
        when per-session auth is enabled. A caller that sends the configured
        headers gets its own pooled connection; a caller that doesn't falls
        back to the shared connection when one is configured, or gets a
        clear error when it isn't.
        """
        creds = extract_session_credentials(
            ctx, self.config.session_user_header, self.config.session_api_key_header
        )
        if creds is not None:
            logger.info(
                "Per-session auth: routing call to pooled connection for '%s'", creds.username
            )
            return await self._session_pool.get(creds)

        if self.connection and self.access_controller:
            logger.info(
                "Per-session auth: no per-call credentials — routing to shared fallback connection"
            )
            return self.connection, self.access_controller

        raise MissingSessionCredentialsError(
            "This server requires per-user Odoo credentials. Configure the "
            f"{self.config.session_user_header} and {self.config.session_api_key_header} "
            "headers in your MCP client."
        )

    def _register_resources(self):
        """Register resource handlers after connection is established.

        Idempotent: streamable-http re-enters the lifespan per session and
        handlers must not be registered twice on the shared FastMCP app.
        """
        if self.resource_handler is not None:
            logger.debug("Resources already registered, skipping")
            return
        if self._session_pool is not None:
            self.resource_handler = register_resources(
                SessionBindingApp(self.app, self._resolve_connection),
                ConnectionProxy(),
                AccessControllerProxy(),
                self.config,
            )
            logger.info("Registered MCP resources (per-session auth)")
        elif self.connection and self.access_controller:
            self.resource_handler = register_resources(
                self.app, self.connection, self.access_controller, self.config
            )
            logger.info("Registered MCP resources")

    def _register_tools(self):
        """Register tool handlers after connection is established.

        Idempotent — see ``_register_resources``.
        """
        if self.tool_handler is not None:
            logger.debug("Tools already registered, skipping")
            return
        if self._session_pool is not None:
            self.tool_handler = register_tools(
                SessionBindingApp(self.app, self._resolve_connection),
                ConnectionProxy(),
                AccessControllerProxy(),
                self.config,
            )
            logger.info("Registered MCP tools (per-session auth)")
        elif self.connection and self.access_controller:
            self.tool_handler = register_tools(
                self.app, self.connection, self.access_controller, self.config
            )
            logger.info("Registered MCP tools")

    async def run_stdio(self):
        """Run the server using stdio transport."""
        try:
            logger.info("Starting MCP server with stdio transport...")
            await self.app.run_stdio_async()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except (OdooConnectionError, ConfigurationError):
            raise
        except Exception as e:
            context = ErrorContext(operation="server_run")
            error_handler.handle_error(e, context=context)

    def run_stdio_sync(self):
        """Synchronous wrapper for run_stdio.

        This is provided for compatibility with synchronous code.
        """
        import asyncio

        asyncio.run(self.run_stdio())

    # SSE transport has been deprecated in MCP protocol version 2025-03-26
    # Use streamable-http transport instead

    async def run_http(self, host: str = "localhost", port: int = 8000):
        """Run the server using streamable HTTP transport.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            logger.info(f"Starting MCP server with HTTP transport on {host}:{port}...")
            self._warn_if_exposed(host)
            self.app.settings.host = host
            self.app.settings.port = port
            self._preseed_session_manager()
            try:
                await self.app.run_streamable_http_async()
            finally:
                if self._session_pool is not None:
                    await self._session_pool.close_all()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except (OdooConnectionError, ConfigurationError):
            raise
        except Exception as e:
            context = ErrorContext(operation="server_run_http")
            error_handler.handle_error(e, context=context)

    def _preseed_session_manager(self) -> None:
        """Apply ODOO_MCP_SESSION_IDLE_TIMEOUT to the streamable-http transport.

        The SDK's StreamableHTTPSessionManager supports evicting idle sessions
        (freeing their transport state, which otherwise accumulates until
        process restart), but FastMCP does not yet expose the parameter. Its session manager is created lazily in
        streamable_http_app(), so constructing it here first — mirroring the
        arguments FastMCP would pass, plus the timeout — makes FastMCP reuse
        this instance. Remove once FastMCP plumbs session_idle_timeout through.
        """
        if self.config.session_idle_timeout is None:
            return

        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        self.app._session_manager = StreamableHTTPSessionManager(
            app=self.app._mcp_server,
            event_store=self.app._event_store,
            retry_interval=self.app._retry_interval,
            json_response=self.app.settings.json_response,
            stateless=self.app.settings.stateless_http,
            security_settings=self.app.settings.transport_security,
            session_idle_timeout=self.config.session_idle_timeout,
        )
        logger.info(
            "Streamable-http session idle timeout enabled: %.0fs",
            self.config.session_idle_timeout,
        )

    def _build_transport_security(self) -> Optional[TransportSecuritySettings]:
        """Build DNS-rebinding-protection settings from ODOO_MCP_ALLOWED_HOSTS.

        Returns None when no hosts are configured, which leaves the SDK
        middleware at its default (protection disabled) — unchanged behavior
        for stdio and for HTTP deployments behind a proxy that don't set the
        variable. When hosts are configured, each is allowed on any port and
        matching http/https origins are derived.
        """
        if not self.config.allowed_hosts:
            return None

        allowed_hosts: list[str] = []
        allowed_origins: list[str] = []
        for host in self.config.allowed_hosts:
            base = host.split(":")[0] if ":" in host else host
            allowed_hosts.append(host if ":" in host else f"{host}:*")
            allowed_origins.extend([f"http://{base}:*", f"https://{base}:*"])

        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )

    def _warn_if_exposed(self, host: str) -> None:
        """Warn loudly when the HTTP transport binds a non-loopback host.

        The streamable-http transport has NO built-in client authentication:
        anyone who can reach the port gets Odoo access through the server's
        stored credentials. Remote deployments must front it with a reverse
        proxy that enforces authentication.
        """
        if host in ("localhost", "127.0.0.1", "::1"):
            return

        message = (
            f"HTTP transport binding to '{host}' — this transport has NO built-in "
            "authentication. Anyone who can reach this port gets Odoo access with "
            "the server's stored credentials. Bind to localhost or front this "
            "server with an authenticating reverse proxy."
        )
        if self.config.yolo_mode == "true":
            message += (
                " YOLO FULL-ACCESS MODE IS ENABLED: unauthenticated clients could "
                "read, write and delete ANY record"
            )
            if self.config.enable_method_calls:
                message += " and call arbitrary model methods"
            message += "."
        logger.warning(message)

    def get_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """Get server capabilities.

        Returns:
            Dict with server capabilities
        """
        return {
            "capabilities": {
                "resources": True,  # Exposes Odoo data as resources
                "tools": True,  # Provides tools for Odoo operations
                "prompts": False,  # Prompts will be added in later phases
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get server health status.

        Returns:
            Dict with health status
        """
        is_connected = bool(self.connection is not None and self.connection.is_authenticated)
        # Per-session auth authenticates lazily, per caller — no global
        # connection is expected until someone without a fallback calls in.
        healthy = is_connected or self._session_pool is not None

        return {
            "status": "healthy" if healthy else "unhealthy",
            "version": SERVER_VERSION,
            "connection": {
                "connected": is_connected,
                "per_session_auth": self._session_pool is not None,
            },
        }

    def _get_model_names(self) -> list[str]:
        """Get available model names for autocomplete."""
        if not self.access_controller:
            return []
        try:
            models = self.access_controller.get_enabled_models()
            if models:
                return [m["model"] for m in models]
            # YOLO mode returns [] meaning "all allowed" — query ir.model directly
            if self.connection and self.connection.is_authenticated:
                records = self.connection.search_read("ir.model", [], ["model"], limit=200)
                return [r["model"] for r in records]
            return []
        except Exception as e:
            logger.debug(f"Failed to get model names for autocomplete: {e}")
            return []
