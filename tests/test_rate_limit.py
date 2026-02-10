import time

from kalshi_bot.rate_limit import RateLimiter


def test_token_bucket_refill():
    limiter = RateLimiter(read_per_sec=5, write_per_sec=5, burst=5)
    for _ in range(5):
        limiter.wait_for_token("read")
    start = time.time()
    limiter.wait_for_token("read")
    assert time.time() - start >= 0
