"""GitHub Copilot provider (OpenAI-compatible chat completions).

``GITHUB_COPILOT_TOKEN`` is a GitHub OAuth token (see ``scripts/github_copilot_login.py``).
It is exchanged for a short-lived Copilot API token that is cached and refreshed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.base import ProviderConfig
from providers.defaults import GITHUB_COPILOT_DEFAULT_BASE
from providers.exceptions import AuthenticationError, InvalidRequestError
from providers.openai_compat import OpenAIChatTransport

COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
_TOKEN_REFRESH_MARGIN_S = 60.0
_COPILOT_HEADERS = {
    "Editor-Version": "vscode/1.99.0",
    "Editor-Plugin-Version": "copilot-chat/0.26.0",
    "Copilot-Integration-Id": "vscode-chat",
    "User-Agent": "GitHubCopilotChat/0.26.0",
}


class GitHubCopilotProvider(OpenAIChatTransport):
    """GitHub Copilot API using ``https://api.githubcopilot.com/chat/completions``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="GITHUB_COPILOT",
            base_url=config.base_url or GITHUB_COPILOT_DEFAULT_BASE,
            api_key="",
            default_headers=_COPILOT_HEADERS,
        )
        self._oauth_token = config.api_key
        self._copilot_token = ""
        self._copilot_token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    def _token_is_fresh(self) -> bool:
        return bool(self._copilot_token) and (
            time.time() < self._copilot_token_expires_at - _TOKEN_REFRESH_MARGIN_S
        )

    async def _ensure_copilot_token(self) -> None:
        """Exchange the OAuth token for a Copilot API token when missing or expiring."""
        if self._token_is_fresh():
            return
        async with self._token_lock:
            if self._token_is_fresh():
                return
            config = self._config
            async with httpx.AsyncClient(
                proxy=config.proxy or None,
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    read=config.http_read_timeout,
                    write=config.http_write_timeout,
                ),
            ) as http:
                response = await http.get(
                    COPILOT_TOKEN_URL,
                    headers={
                        **_COPILOT_HEADERS,
                        "Authorization": f"token {self._oauth_token}",
                        "Accept": "application/json",
                    },
                )
            if response.status_code in (401, 403, 404):
                raise AuthenticationError(
                    "GitHub rejected GITHUB_COPILOT_TOKEN (HTTP "
                    f"{response.status_code}). Check that the account has an active "
                    "Copilot subscription and re-run scripts/github_copilot_login.py."
                )
            response.raise_for_status()
            payload = response.json()
            token = payload.get("token")
            expires_at = payload.get("expires_at")
            if not isinstance(token, str) or not token:
                raise AuthenticationError(
                    "GitHub Copilot token exchange returned no token."
                )
            self._copilot_token = token
            self._copilot_token_expires_at = (
                float(expires_at)
                if isinstance(expires_at, int | float)
                else time.time() + 25 * 60
            )
            self._client.api_key = token

    async def list_model_ids(self) -> frozenset[str]:
        await self._ensure_copilot_token()
        return await super().list_model_ids()

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        await self._ensure_copilot_token()
        return await super()._create_stream(body)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        thinking = self._is_thinking_enabled(request, thinking_enabled)
        try:
            body = build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
                if thinking
                else ReasoningReplayMode.DISABLED,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc
        extra = getattr(request, "extra_body", None)
        if isinstance(extra, dict) and extra:
            body["extra_body"] = dict(extra)
        return body
