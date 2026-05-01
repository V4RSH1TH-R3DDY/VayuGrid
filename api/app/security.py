from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from .config import settings

bearer_scheme = HTTPBearer(auto_error=False)


class Encryption:
    @staticmethod
    def encrypt(data: str) -> str:
        aesgcm = AESGCM(get_encryption_key_bytes())
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, data.encode(), None)
        return (nonce + ct).hex()

    @staticmethod
    def decrypt(hex_data: str) -> str:
        raw = bytes.fromhex(hex_data)
        nonce = raw[:12]
        ct = raw[12:]
        aesgcm = AESGCM(get_encryption_key_bytes())
        return aesgcm.decrypt(nonce, ct, None).decode()


def get_encryption_key_bytes() -> bytes:
    """Return the AES-GCM key bytes.

    Tries Vault first (if VAULT_ADDR and VAULT_TOKEN are configured).
    Falls back to the ENCRYPTION_KEY env var.
    """
    from .config import settings

    if settings.vault_url and settings.vault_token:
        try:
            from .vault_client import VaultClient

            client = VaultClient(settings.vault_url, settings.vault_token)
            raw = client.get_secret("vayugrid", "encryption-key").get("value", "")
            if raw:
                return base64.b64decode(raw)
        except Exception:
            pass  # fall through to env var
    return base64.b64decode(settings.encryption_key)


class UserClaims(BaseModel):
    username: str
    role: str
    node_id: int | None = None


def verify_password(plain_password: str, expected_password: str) -> bool:
    return hmac.compare_digest(plain_password, expected_password)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def create_access_token(data: Dict[str, Any]) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> UserClaims:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        role = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return UserClaims(username=username, role=role, node_id=payload.get("node_id"))
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        ) from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> UserClaims:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    return decode_access_token(credentials.credentials)


def require_roles(allowed_roles: Iterable[str]) -> Callable[[UserClaims], UserClaims]:
    def _dependency(user: UserClaims = Depends(get_current_user)) -> UserClaims:
        if user.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user

    return _dependency
