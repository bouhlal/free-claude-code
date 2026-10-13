"""Obtain a GitHub OAuth token for ``GITHUB_COPILOT_TOKEN`` via GitHub's device flow.

Usage: ``uv run python scripts/github_copilot_login.py``
"""

from __future__ import annotations

import sys
import time

import httpx

# OAuth app id used by GitHub's official Copilot editor plugins.
CLIENT_ID = "Iv1.b507a08c87ecfe98"
DEVICE_CODE_URL = "https://github.com/login/device/code"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
_HEADERS = {"Accept": "application/json"}


def main() -> int:
    with httpx.Client(timeout=30.0, headers=_HEADERS) as http:
        device = http.post(
            DEVICE_CODE_URL, data={"client_id": CLIENT_ID, "scope": "read:user"}
        )
        device.raise_for_status()
        info = device.json()
        print(f"Open {info['verification_uri']} and enter code: {info['user_code']}")
        interval = int(info.get("interval", 5))
        deadline = time.time() + int(info.get("expires_in", 900))
        while time.time() < deadline:
            time.sleep(interval)
            resp = http.post(
                ACCESS_TOKEN_URL,
                data={
                    "client_id": CLIENT_ID,
                    "device_code": info["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            token = payload.get("access_token")
            if token:
                print("\nAdd this to your .env (or paste it in the Admin UI):")
                print(f'GITHUB_COPILOT_TOKEN="{token}"')
                return 0
            error = payload.get("error")
            if error == "slow_down":
                interval += 5
            elif error != "authorization_pending":
                print(f"Login failed: {error}", file=sys.stderr)
                return 1
    print("Login timed out.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
