import time
from typing import Any, cast

import compute_api_client
import requests
from oauthlib.oauth2 import Client

from quantuminspire.util.configuration import AuthSettings, TokenInfo


class AuthorisationError(Exception):
    """Indicates that the authorisation permanently went wrong."""

    pass


class AuthorisationPending(Exception):
    """Indicates that we should continue polling."""


class OauthDeviceSession:
    def __init__(self, settings: AuthSettings):
        self._settings = settings
        self._client_id = settings.client_id
        self._token_info = settings.tokens
        self._token_endpoint, self._device_endpoint = self._get_endpoints()
        self._oauth_client = Client(settings.client_id)
        self._headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self.expires_in = 600
        self.interval = 5
        self.expires_at = time.time()
        self._device_code = ""

    def _get_endpoints(self) -> tuple[str, str]:
        response = requests.get(self._settings.well_known_endpoint)
        response.raise_for_status()
        config = response.json()
        return config["token_endpoint"], config["device_authorization_endpoint"]

    def initialize_authorization(self, scope: str = "api-access openid") -> dict[str, Any]:
        code_verifier = self._oauth_client.create_code_verifier(self._settings.code_verifyer_length)
        self._oauth_client.create_code_challenge(code_verifier, self._settings.code_challenge_method)
        data = {
            "client_id": self._client_id,
            "code_challenge_method": self._settings.code_challenge_method,
            "code_challenge": self._oauth_client.code_challenge,
            "scope": scope,
        }

        response = requests.post(self._device_endpoint, data=data, headers=self._headers).json()
        self.expires_in = int(response["expires_in"])
        self.interval = response["interval"]
        self.expires_at = time.monotonic() + self.expires_in
        self._device_code = str(response["device_code"])

        return cast(dict[str, Any], response)

    def request_token(self) -> TokenInfo:
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": self._client_id,
            "code_verifier": self._oauth_client.code_verifier,
            "device_code": self._device_code,
        }

        response = requests.post(self._token_endpoint, data=data, headers=self._headers)
        content = response.json()
        if response.status_code == 400:
            if content["error"] == "authorization_pending":
                raise AuthorisationPending(content["error"])
            if content["error"] == "slow_down":
                self.interval += 5
                raise AuthorisationPending(content["error"])

        if response.status_code == 200:
            self._token_info = TokenInfo(**content)
            return self._token_info

        raise AuthorisationError(f"Received status code: {response.status_code}\n {response.text}")

    def poll_for_tokens(self) -> TokenInfo:

        while time.monotonic() < self.expires_at:
            try:
                return self.request_token()
            except AuthorisationPending:
                time.sleep(self.interval)

        raise AuthorisationError("Login session timed out, please login again.")

    def refresh(self) -> TokenInfo:
        if self._token_info is None:
            raise AuthorisationError("You should authenticate first before you can refresh")

        if self._token_info.expires_at > time.time() + 5:
            return self._token_info

        data = {
            "grant_type": "refresh_token",
            "client_id": self._client_id,
            "refresh_token": self._token_info.refresh_token,
        }

        response = requests.post(self._token_endpoint, data=data, headers=self._headers)

        if response.status_code == 200:
            self._token_info = TokenInfo(**response.json())
            return self._token_info

        raise AuthorisationError(f"Received status code: {response.status_code}\n {response.text}")


class Configuration(compute_api_client.Configuration):  # type: ignore[misc]

    def __init__(self, host: str | None = None, oauth_session: OauthDeviceSession | None = None, **kwargs: Any):
        self._oauth_session = oauth_session
        super().__init__(host=host, **kwargs)

    def auth_settings(self) -> Any:
        if self._oauth_session is not None:
            token_info = self._oauth_session.refresh()
            self.access_token = token_info.access_token
        return super().auth_settings()
