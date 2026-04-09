from __future__ import annotations

import hashlib


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()
