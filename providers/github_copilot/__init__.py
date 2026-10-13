"""GitHub Copilot (OpenAI-compat) adapter."""

from providers.defaults import GITHUB_COPILOT_DEFAULT_BASE

from .client import GitHubCopilotProvider

__all__ = ["GITHUB_COPILOT_DEFAULT_BASE", "GitHubCopilotProvider"]
