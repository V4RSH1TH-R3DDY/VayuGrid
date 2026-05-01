from __future__ import annotations

import base64
import secrets


class VaultUnavailableError(Exception):
    """Raised when HashiCorp Vault is unreachable or not configured."""


class VaultClient:
    """Thin wrapper around the HashiCorp Vault HTTP API (KV secrets engine v2).

    Uses the ``hvac`` library. Falls back gracefully when Vault is offline.
    """

    def __init__(self, vault_url: str, token: str) -> None:
        import hvac

        self._client = hvac.Client(url=vault_url, token=token)

    def is_available(self) -> bool:
        try:
            return self._client.is_authenticated()
        except Exception:
            return False

    def get_secret(self, mount_path: str, secret_path: str) -> dict:
        """Read a KV-v2 secret. Returns the ``data`` dict."""
        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=secret_path, mount_point=mount_path
            )
            return response["data"]["data"]
        except Exception as exc:
            raise VaultUnavailableError(str(exc)) from exc

    def put_secret(self, mount_path: str, secret_path: str, data: dict) -> None:
        """Write / update a KV-v2 secret."""
        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=secret_path, secret=data, mount_point=mount_path
            )
        except Exception as exc:
            raise VaultUnavailableError(str(exc)) from exc

    def rotate_encryption_key(self) -> str:
        """Generate a new 32-byte AES key, store it in Vault, and return the base64url value."""
        new_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
        self.put_secret("vayugrid", "encryption-key", {"value": new_key})
        return new_key
