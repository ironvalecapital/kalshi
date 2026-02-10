from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class TokenBucket:
    rate_per_sec: int
    capacity: int
    tokens: float
    last_refill: float

    def refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    def __init__(self, read_per_sec: int, write_per_sec: int, burst: int) -> None:
        now = time.monotonic()
        self.buckets: Dict[str, TokenBucket] = {
            "read": TokenBucket(read_per_sec, burst, burst, now),
            "write": TokenBucket(write_per_sec, burst, burst, now),
        }

    def wait_for_token(self, bucket: str, tokens: float = 1.0) -> None:
        while not self.buckets[bucket].consume(tokens):
            time.sleep(0.05)


def tier_to_limits(tier: str) -> Dict[str, int]:
    tier = tier.lower()
    if tier == "basic":
        return {"read": 20, "write": 10, "burst": 40}
    if tier == "advanced":
        return {"read": 30, "write": 30, "burst": 60}
    if tier == "premier":
        return {"read": 100, "write": 100, "burst": 200}
    if tier == "prime":
        return {"read": 400, "write": 400, "burst": 800}
    return {"read": 20, "write": 10, "burst": 40}
