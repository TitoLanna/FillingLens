"""Gateway settings from environment variables (and backend/.env)."""
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from filinglens.config import ROOT, load_dotenv


@dataclass(frozen=True)
class GatewaySettings:
    host: str
    port: int
    upstream_host: str
    upstream_port: int
    allowed_origins: tuple[str, ...]   # ("*",) allows any origin
    admin_token: str                   # empty = admin routes disabled
    rate_query_per_min: int
    rate_read_per_min: int
    max_inflight_queries: int
    max_body_bytes: int
    upstream_timeout: float
    trust_proxy: bool                  # read the client IP from the last X-Forwarded-For entry


def load() -> GatewaySettings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    upstream = urlparse(env("UPSTREAM_URL", "http://127.0.0.1:8000"))
    if upstream.scheme != "http" or not upstream.hostname:
        raise ValueError("UPSTREAM_URL must be http://host:port (the backend is internal; TLS ends at the edge)")
    return GatewaySettings(
        host=env("GATEWAY_HOST", "127.0.0.1"),
        port=int(env("GATEWAY_PORT", "8080")),
        upstream_host=upstream.hostname,
        upstream_port=upstream.port or 80,
        allowed_origins=tuple(o.strip() for o in env("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()),
        admin_token=env("ADMIN_TOKEN", ""),
        rate_query_per_min=int(env("RATE_QUERY_PER_MIN", "10")),
        rate_read_per_min=int(env("RATE_READ_PER_MIN", "120")),
        max_inflight_queries=int(env("MAX_INFLIGHT_QUERIES", "8")),
        max_body_bytes=int(env("MAX_BODY_BYTES", "16384")),
        upstream_timeout=float(env("UPSTREAM_TIMEOUT", "130")),
        trust_proxy=env("TRUST_PROXY", "false").lower() == "true",
    )
