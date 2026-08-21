"""Configuration: AppConfig dataclass + precedence load/save.

Precedence: env/.env > config.json > defaults.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

# JSON-persistable fields (the `config` tool get/set surface).
JSON_FIELDS = ("score_threshold", "top_k", "chunk_size", "chunk_overlap", "model")

# Env vars that override config.json / defaults.
ENV_FIELDS = {
    "OLLAMA_API_KEY": "api_key",
    "OLLAMA_URL": "ollama_url",
    "QDRANT_URL": "qdrant_url",
    "ADMIN_PORT": "admin_port",
    "VAULT_ROOT": "vault_root",
    "COLLECTION": "collection",
    "DEFAULT_SOURCE": "default_source",
}


@dataclass
class AppConfig:
    score_threshold: float = 0.5
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 100
    model: str = "deepseek-v4-flash"
    embed_model: str = "bge-m3"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_base_url: str = "https://ollama.com/v1"
    qdrant_url: str = "http://127.0.0.1:6333"
    admin_port: int = 8310
    vault_root: str = ""
    default_source: str = "vault"
    collection: str = "supervisor"
    ext: tuple[str, ...] = (".md", ".txt")
    api_key: str = ""


def _config_path() -> Path:
    override = os.environ.get("MCP_RAG_CONFIG")
    if override:
        return Path(override)
    return Path("config.json")


def _load_json() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in data.items() if k in JSON_FIELDS}


def load_config(env: dict | None = None) -> AppConfig:
    """Build AppConfig with precedence defaults -> config.json -> env.

    `env` (optional) is a dict that overrides os.environ for testability.
    """
    load_dotenv()

    # 1. defaults
    cfg = AppConfig()

    # 2. config.json overrides defaults
    for key, value in _load_json().items():
        setattr(cfg, key, value)

    # 3. env overrides JSON + defaults
    source = env if env is not None else os.environ
    for env_key, attr in ENV_FIELDS.items():
        if env_key in source and source[env_key] != "":
            value = source[env_key]
            if attr == "admin_port":
                value = int(value)
            setattr(cfg, attr, value)

    return cfg


def save_config(cfg: AppConfig) -> None:
    """Write config.json with only the JSON-persistable fields."""
    data = {k: getattr(cfg, k) for k in JSON_FIELDS}
    with open(_config_path(), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
