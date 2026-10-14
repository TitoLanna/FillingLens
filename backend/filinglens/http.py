"""Tiny JSON-over-HTTPS helpers on urllib."""
import json
import os
import ssl
import urllib.error
import urllib.request


class UpstreamError(Exception):
    pass


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS. If Python ships with no CA roots (python.org macOS builds), use the OS bundle."""
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get("x509_ca"):
        for cafile in (os.environ.get("SSL_CERT_FILE"), "/etc/ssl/cert.pem",
                       "/etc/ssl/certs/ca-certificates.crt"):
            if cafile and os.path.exists(cafile):
                ctx.load_verify_locations(cafile)
                break
    return ctx


_CTX = _ssl_context()


def _open(url: str, headers: dict, body: dict | None, timeout: float):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=_CTX)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise UpstreamError(f"{url} -> HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise UpstreamError(f"{url} -> {exc.reason}") from None


def get_json(url: str, headers: dict, timeout: float = 30):
    with _open(url, headers, None, timeout) as resp:
        return json.loads(resp.read())


def get_text(url: str, headers: dict, timeout: float = 60) -> str:
    with _open(url, headers, None, timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def post_json(url: str, headers: dict, body: dict, timeout: float = 60):
    with _open(url, {"Content-Type": "application/json", **headers}, body, timeout) as resp:
        return json.loads(resp.read())


def post_sse(url: str, headers: dict, body: dict, timeout: float = 120):
    """Yield (event, parsed_json_data) for each server-sent event."""
    with _open(url, {"Content-Type": "application/json", **headers}, body, timeout) as resp:
        event = None
        for raw in resp:
            line = raw.decode().rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                yield event, json.loads(line[5:].strip())
