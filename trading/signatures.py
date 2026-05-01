from __future__ import annotations

import base64
import json
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def generate_keypair() -> tuple[str, str]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_bytes = private_key.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    public_bytes = public_key.public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
    return (
        base64.b64encode(private_bytes).decode("ascii"),
        base64.b64encode(public_bytes).decode("ascii"),
    )


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


def sign_payload(private_key_b64: str, payload: dict[str, Any]) -> str:
    private_key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    signature = private_key.sign(_canonical_payload(payload))
    return base64.b64encode(signature).decode("ascii")


def verify_payload(public_key_b64: str, payload: dict[str, Any], signature_b64: str) -> bool:
    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
    try:
        public_key.verify(base64.b64decode(signature_b64), _canonical_payload(payload))
    except InvalidSignature:
        return False
    return True
