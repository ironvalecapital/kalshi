from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class KalshiAuthError(Exception):
    pass


def load_private_key(path: str) -> rsa.RSAPrivateKey:
    key_path = Path(path)
    if not key_path.exists():
        raise KalshiAuthError(f"Private key file not found: {path}")
    key_bytes = key_path.read_bytes()
    return serialization.load_pem_private_key(key_bytes, password=None)


def _sign(private_key: rsa.RSAPrivateKey, message: bytes) -> str:
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("ascii")


def build_auth_headers(
    api_key_id: str,
    private_key: rsa.RSAPrivateKey,
    method: str,
    path: str,
) -> Dict[str, str]:
    if not api_key_id:
        raise KalshiAuthError("API key id missing")
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}{method.upper()}{path}".encode("utf-8")
    signature = _sign(private_key, message)
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }
