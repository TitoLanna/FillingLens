"""Per-request stage timing."""
import time
import uuid
from contextlib import contextmanager


class Trace:
    def __init__(self):
        self.trace_id = "tr_" + uuid.uuid4().hex[:4]
        self.stages: list[dict] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages.append({"name": name, "ms": round((time.perf_counter() - start) * 1000, 2)})

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_ms": round((time.perf_counter() - self._t0) * 1000),
            "stages": self.stages,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
        }
