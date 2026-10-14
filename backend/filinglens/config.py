"""Settings from environment variables, with a minimal .env reader."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    db_path: Path
    cors_origin: str
    pipeline: str
    refusal_min_bm25: float
    refusal_min_dense: float
    sec_user_agent: str
    raw_dir: Path
    embedding_provider: str   # hash | voyage
    voyage_api_key: str
    voyage_model: str
    llm_provider: str         # anthropic | stub
    anthropic_api_key: str
    llm_model: str
    judge_model: str
    llm_price_in: float       # USD per million input tokens
    llm_price_out: float      # USD per million output tokens


def load() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    return Settings(
        host=env("HOST", "127.0.0.1"),
        port=int(env("PORT", "8000")),
        db_path=ROOT / env("DB_PATH", "data/filinglens.db"),
        cors_origin=env("CORS_ORIGIN", "*"),
        pipeline=env("PIPELINE", "baseline"),
        refusal_min_bm25=float(env("REFUSAL_MIN_BM25", "0.3")),
        refusal_min_dense=float(env("REFUSAL_MIN_DENSE", "0.2")),
        sec_user_agent=env("SEC_USER_AGENT", "FilingLens research contact@example.com"),
        raw_dir=ROOT / env("RAW_DIR", "data/raw"),
        embedding_provider=env("EMBEDDING_PROVIDER", "hash"),
        voyage_api_key=env("VOYAGE_API_KEY", ""),
        voyage_model=env("VOYAGE_MODEL", "voyage-3.5"),
        llm_provider=env("LLM_PROVIDER", "anthropic" if env("ANTHROPIC_API_KEY") else "stub"),
        anthropic_api_key=env("ANTHROPIC_API_KEY", ""),
        llm_model=env("LLM_MODEL", "claude-sonnet-5-5"),
        judge_model=env("JUDGE_MODEL", env("LLM_MODEL", "claude-sonnet-5-5")),
        llm_price_in=float(env("LLM_PRICE_IN", "0")),
        llm_price_out=float(env("LLM_PRICE_OUT", "0")),
    )
