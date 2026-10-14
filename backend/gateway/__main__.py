"""Run: python3 -m gateway   (from the backend/ folder)"""
from . import config
from .server import make_server

if __name__ == "__main__":
    cfg = config.load()
    server = make_server(cfg)
    admin = "enabled" if cfg.admin_token else "disabled (set ADMIN_TOKEN)"
    print(f"Gateway on http://{cfg.host}:{cfg.port} -> backend http://{cfg.upstream_host}:{cfg.upstream_port} "
          f"| admin routes {admin} | origins {', '.join(cfg.allowed_origins)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
