"""Token-bucket rate limiter, one bucket per (namespace, client)."""
import threading
import time


class RateLimiter:
    def __init__(self, per_minute: int, clock=time.monotonic, max_keys: int = 10_000):
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0          # tokens added per second
        self.clock = clock
        self.max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}   # key -> (tokens, last_seen)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Take one token. Returns (allowed, retry_after_seconds)."""
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                self._evict(now)
                return True, 0
            self._buckets[key] = (tokens, now)
            wait = (1.0 - tokens) / self.rate if self.rate else 60.0
            return False, max(1, int(wait + 0.999))

    def _evict(self, now: float) -> None:
        """Bound memory: drop buckets that have fully refilled (they behave like a fresh one)."""
        if len(self._buckets) <= self.max_keys:
            return
        full_after = self.capacity / self.rate if self.rate else 60.0
        for key in [k for k, (_, seen) in self._buckets.items() if now - seen >= full_after]:
            del self._buckets[key]
