import os

import pytest

from mcp_rag.config import AppConfig, load_config, save_config


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point config.json at a temp path and clear relevant env vars."""
    monkeypatch.setenv("MCP_RAG_CONFIG", str(tmp_path / "config.json"))
    for key in (
        "OLLAMA_API_KEY",
        "OLLAMA_URL",
        "QDRANT_URL",
        "ADMIN_PORT",
        "VAULT_ROOT",
        "COLLECTION",
        "DEFAULT_SOURCE",
    ):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_defaults_with_no_env_and_no_config(isolated_config):
    cfg = load_config(env={})
    assert cfg.score_threshold == 0.5
    assert cfg.top_k == 5
    assert cfg.chunk_size == 800
    assert cfg.chunk_overlap == 100
    assert cfg.model == "deepseek-v4-flash"
    assert cfg.embed_model == "bge-m3"
    assert cfg.ollama_url == "http://127.0.0.1:11434"
    assert cfg.llm_base_url == "https://ollama.com/v1"
    assert cfg.qdrant_url == "http://127.0.0.1:6333"
    assert cfg.admin_port == 8310
    assert cfg.vault_root == ""
    assert cfg.default_source == "vault"
    assert cfg.collection == "supervisor"
    assert cfg.ext == (".md", ".txt")
    assert cfg.api_key == ""


def test_env_overrides_defaults(isolated_config):
    cfg = load_config(
        env={
            "OLLAMA_API_KEY": "secret",
            "OLLAMA_URL": "http://ollama:11434",
            "QDRANT_URL": "http://qdrant:6333",
            "ADMIN_PORT": "9000",
            "VAULT_ROOT": "/vault",
            "COLLECTION": "notes",
            "DEFAULT_SOURCE": "obsidian",
        }
    )
    assert cfg.api_key == "secret"
    assert cfg.ollama_url == "http://ollama:11434"
    assert cfg.qdrant_url == "http://qdrant:6333"
    assert cfg.admin_port == 9000
    assert cfg.vault_root == "/vault"
    assert cfg.collection == "notes"
    assert cfg.default_source == "obsidian"


def test_json_overrides_defaults_but_env_wins(isolated_config):
    # config.json sets top_k=7, model="json-model"
    save_config(AppConfig(top_k=7, model="json-model"))
    # env overrides top_k but not model
    cfg = load_config(env={"ADMIN_PORT": "9999"})
    assert cfg.top_k == 7  # from JSON
    assert cfg.model == "json-model"  # from JSON
    assert cfg.admin_port == 9999  # from env
    assert cfg.score_threshold == 0.5  # default untouched


def test_save_reload_round_trip(isolated_config):
    cfg = AppConfig(score_threshold=0.6, top_k=7, chunk_size=1000, chunk_overlap=150, model="other")
    save_config(cfg)
    reloaded = load_config(env={})
    assert reloaded.score_threshold == 0.6
    assert reloaded.top_k == 7
    assert reloaded.chunk_size == 1000
    assert reloaded.chunk_overlap == 150
    assert reloaded.model == "other"
    # non-persisted fields keep defaults
    assert reloaded.collection == "supervisor"
