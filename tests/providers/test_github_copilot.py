"""Tests for the GitHub Copilot provider (token exchange + OpenAI-compat chat)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from providers.base import ProviderConfig
from providers.exceptions import AuthenticationError
from providers.github_copilot import GITHUB_COPILOT_DEFAULT_BASE, GitHubCopilotProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self):
        self.model = "gpt-4.1"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = False


@pytest.fixture
def provider():
    config = ProviderConfig(
        api_key="oauth-token",
        base_url=GITHUB_COPILOT_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=False,
    )
    with patch("providers.openai_compat.AsyncOpenAI"):
        return GitHubCopilotProvider(config)


def _patch_http(response: httpx.Response):
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return patch("httpx.AsyncClient", return_value=cm), client


def _json_response(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status, json=payload, request=httpx.Request("GET", "https://x")
    )


def test_default_base_url_constant():
    assert GITHUB_COPILOT_DEFAULT_BASE == "https://api.githubcopilot.com"


def test_init_passes_copilot_headers():
    config = ProviderConfig(api_key="oauth-token", base_url=None)
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        GitHubCopilotProvider(config)
    headers = mock_openai.call_args.kwargs["default_headers"]
    assert headers["Copilot-Integration-Id"] == "vscode-chat"
    assert mock_openai.call_args.kwargs["base_url"] == GITHUB_COPILOT_DEFAULT_BASE


@pytest.mark.asyncio
async def test_token_exchange_sets_client_api_key_and_caches(provider):
    expires = time.time() + 1800
    patcher, client = _patch_http(
        _json_response(200, {"token": "copilot-tok", "expires_at": expires})
    )
    with patcher:
        await provider._ensure_copilot_token()
        await provider._ensure_copilot_token()

    assert provider._client.api_key == "copilot-tok"
    assert client.get.await_count == 1
    assert client.get.await_args.kwargs["headers"]["Authorization"] == (
        "token oauth-token"
    )


@pytest.mark.asyncio
async def test_token_refreshes_when_expiring(provider):
    soon = time.time() + 10
    patcher, client = _patch_http(
        _json_response(200, {"token": "t1", "expires_at": soon})
    )
    with patcher:
        await provider._ensure_copilot_token()
        await provider._ensure_copilot_token()

    assert client.get.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_token_exchange_rejected_raises_authentication_error(provider, status):
    patcher, _ = _patch_http(_json_response(status, {"message": "nope"}))
    with patcher, pytest.raises(AuthenticationError):
        await provider._ensure_copilot_token()


@pytest.mark.asyncio
async def test_token_exchange_missing_token_raises(provider):
    patcher, _ = _patch_http(_json_response(200, {"expires_at": 1}))
    with patcher, pytest.raises(AuthenticationError):
        await provider._ensure_copilot_token()


@pytest.mark.asyncio
async def test_create_stream_ensures_token_first(provider):
    provider._ensure_copilot_token = AsyncMock()
    with patch(
        "providers.openai_compat.OpenAIChatTransport._create_stream",
        new=AsyncMock(return_value=("stream", {})),
    ) as parent:
        result = await provider._create_stream({"model": "gpt-4.1"})

    provider._ensure_copilot_token.assert_awaited_once()
    parent.assert_awaited_once()
    assert result == ("stream", {})


def test_build_request_body_basic(provider):
    body = provider._build_request_body(MockRequest())

    assert body["model"] == "gpt-4.1"
    assert body["messages"][0]["role"] == "system"
