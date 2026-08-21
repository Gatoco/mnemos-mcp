# Proposal: mcp-rag-opencode

## Summary

Build a complete MCP server implementing RAG for opencode: index files (Obsidian vault + docs) → bge-m3 embeddings via **local** Ollama → store/retrieve in Qdrant → answer questions from the user's own notes with `deepseek-v4-flash`. Server exposes full MCP tools (`index`, `query`, `search`, `delete`, `list`, `stats`, `config`) over stdio, registered as a local command in `opencode.json`. Plus a **local admin web page** to monitor and control the RAG state (indexed documents, collection stats, index triggers, query testing, raw Qdrant exploration via its native dashboard).

## Motivation

University final project ("proyecto integrado") + a real personal tool. The user wants semantic search over a 19GB Obsidian vault (Spanish) and the ability to get LLM answers sourced from his own notes. Embeddings must be multilingual (Spanish), hence bge-m3 over nomic.

## Scope

### In Scope
- MCP server (official `mcp` SDK, stdio, pin `mcp==1.12.x`) with tools: `index`, `query` (RAG+LLM), `search` (raw vector), `delete`, `list`, `stats`, `config`.
- Embedding provider (bge-m3 via local Ollama `/api/embed`, 1024 dims) and LLM provider (`deepseek-v4-flash` via OpenAI-compatible `https://ollama.com/v1/chat/completions`, `OLLAMA_API_KEY`).
- Qdrant integration via Docker (`qdrant/qdrant`, ports 6333/6334, volume); one collection, payload `{source, path, mtime, md5, heading_path}`, filter by source/path.
- Stdlib heading-aware chunker (~600-900 tok, ~100 overlap), markdown-aware, not langchain.
- Incremental indexer: mtime+md5 skip, batch upsert (100-256) with retry/backoff, progress via `ctx.report_progress()`.
- **Local admin page** (bind 127.0.0.1): read-only RAG status (collection stats, per-source counts, last index time, recent indexed files), index trigger (source/path + force-rescan), query playground (test search/query without LLM cost), delete documents by path/source, link to Qdrant dashboard. FastAPI (or stdlib http.server) serving an HTML/JS single page; same core API as MCP tools, no auth (local only).
- Tests (pytest, `:memory:` Qdrant, mocked embed/LLM), docker-compose, `.env.example`, Makefile/scripts.
- Docs: README (ES), memoria; opencode.json registration snippet; AGENTS.md note.

### Out of Scope
- Auth, multi-user (admin page binds 127.0.0.1 only, single user).
- Runtime embedding-model swapping (config-only later).
- GPU/ROCm support (CPU-only path for v1).
- Watch/inotify mode (defer; rescan-on-demand).
- Custom vector-database viewer (Qdrant's native dashboard at `http://localhost:6333/dashboard` covers raw DB exploration; admin page links to it).

## Key Decisions (with rationale)

- **Official `mcp` SDK pinned** — canonical; API churns (user already pins `mcp==1.8.0` for arch-mcp), so pin exact.
- **bge-m3 embeddings, local Ollama** — native multilingual (Spanish vault); Ollama Cloud has NO embeddings endpoint (verified 404).
- **Cloud LLM for generation** — `deepseek-v4-flash` via Ollama Cloud; embeddings stay local only.
- **One Qdrant collection + payload filters** — filters suffice; per-source collections are over-engineering for v1.
- **Stdlib chunker** — tiny, correct for Obsidian MD; no heavy langchain dep.
- **Admin page reuses core API** — same functions behind MCP tools and HTTP endpoints; no duplicate logic.

## Options Considered

- `fastmcp` fork vs official `mcp` → official (canonical, documented).
- nomic-embed-text vs bge-m3 → nomic weak Spanish, rejected.
- Multiple collections → rejected (over-engineering).
- langchain chunkers → heavy, rejected.

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Slow cold index (19GB, CPU-only) | High | Batches + incremental rescan; acceptable to user; optionally pre-index on Garuda GPU |
| No Ollama running today | High | Document install/start; `.env.example` points to `127.0.0.1:11434` or Garuda |
| MCP SDK churn | Med | Pin exact version |
| score_threshold ~0.5 tuning | Med | Config knob via `config` tool; test against vault |
| RAM ceiling 16GiB | Med | bge-m3 ~1GB OK; skip >16MB files / chunk big files |
| Admin page scope creep | Low | Read-only status + triggers only; Qdrant dashboard covers raw DB |

## User Impact

- Register in `opencode.json` (local command).
- Run `docker compose up qdrant`.
- `ollama pull bge-m3` locally.
- Create `.env` with `OLLAMA_API_KEY` + endpoints.
- First `index` command over vault (slow cold start).
- Ask via `query`; open `http://127.0.0.1:8310` for the admin page; `http://localhost:6333/dashboard` for raw Qdrant.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `~/proyectos_github/mcp-rag-opencode/` | New | Server package, tests, docker, docs, Makefile |
| `~/.config/opencode/opencode.json` | Modified | Register `mcp-rag-opencode` local command |
| `~/Documentos/obsidian/vault/` | Read-only source | Indexing target |

## Rollback Plan

Remove the `mcp-rag-opencode` entry from `opencode.json`; delete the Qdrant container/volume; repo stays isolated in its own project folder. Non-invasive to existing tools.

## Dependencies

- Docker (Qdrant image), Ollama with `bge-m3` model, `OLLAMA_API_KEY`.
- Python 3.14, `mcp==1.12.x`, `qdrant-client`, `httpx`.

## Success Criteria

- [ ] `index` over a sample vault subset succeeds; `query` returns an LLM answer grounded in retrieved chunks.
- [ ] Incremental re-index skips unchanged files (mtime+md5).
- [ ] `pytest` green with in-memory Qdrant + mocked providers (no network).
- [ ] Server registered and functional inside opencode (manual via MCP inspector).
- [ ] Admin page shows collection stats, indexed docs, and can trigger an index + test a search.
