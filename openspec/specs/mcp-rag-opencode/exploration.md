# Exploration: mcp-rag-opencode

A full MCP server implementing RAG for opencode. University "proyecto integrado" + real personal use.

## Current State

- Empty git repo at `~/proyectos_github/mcp-rag-opencode` (`git init` done, `openspec/specs/mcp-rag-opencode/` empty, no `config.yaml`).
- Vault is **19GB** of Markdown today (target >2GB is already exceeded — actual is 19GB, expect 30-50GB with attachments).
- **No Ollama service running** on this machine (`127.0.0.1:11434` fails). Radeon 780M iGPU has no working ROCm → all embeddings run on **CPU** (16 cores, ~16GiB free RAM). Second machine "Garuda" (192.168.18.172, RTX 3050) can host Ollama.
- **Generation LLM**: `ollama-cloud/deepseek-v4-flash`. opencode provider `ollama-cloud` uses baseURL `https://ollama.com/v1` (OpenAI-compatible), requires API key.
- **CRITICAL**: Ollama Cloud **does NOT expose an embeddings endpoint** — `POST https://ollama.com/v1/embeddings` returns `{"error":"path \"/v1/embeddings\" not found"}`. Embeddings must be generated **locally** (own machine or Garuda). Cloud is chat-generation only.
- opencode registers MCP servers as local commands (`command` + `args`, e.g. `python3 /home/iwakura/bin/vision_mcp.py`, `uvx duckduckgo-mcp-server`). New server must follow this pattern (stdlib transport over stdio).
- Existing config pins `mcp==1.8.0` for arch-mcp (API removed note) — confirms the MCP Python SDK is actively breaking its API surface; pin versions.

## Affected Areas

- `~/proyectos_github/mcp-rag-opencode/` — new project root (server package, tests, Dockerfile, README).
- `~/.config/opencode/opencode.json` — register `mcp-rag-opencode` under `mcp` as a `local` command.
- Qdrant via Docker (new) — collection(s) for vault + project docs.
- Ollama (own machine or Garuda) — local embedding model, `/api/embed`.
- Vault `~/Documentos/obsidian/vault/` — indexing source (read-only).

## Approaches

### 1. MCP server framework
- **`mcp` official SDK** (FastMCP-style API is the standard). The official SDK *incorporated FastMCP 1.0's design*; a separate `fastmcp` PyPI package exists (FastMCP 2.0, actively maintained) but is a third-party fork with a tangled history. For a single-project local server the official SDK is the safe, documented path.
  - Pros: official, `@mcp.tool()`, `ctx.report_progress()` for long index ops, `ctx.info/debug`, stdio default transport via `mcp.run()`.
  - Cons: API actively churns (pin exact version, e.g. `mcp==1.12.x`).
  - Effort: Low
- **`fastmcp` PyPI** — 2.0, ergonomic.
  - Pros: nice ergonomics, active.
  - Cons: third-party fork, less canonical.
  - Effort: Low

### 2. Vector DB — Qdrant via Docker (recommended) vs local embedded mode
- **Docker Qdrant** (recommended): production-grade, snapshots, REST/gRPC, WebUI.
  - Pros: snapshots/backup via `create_snapshot`/`recover_snapshot`, payload indexing, filters, scales to GB.
  - Cons: one more service to run.
  - Effort: Low
- **QdrantClient `path="..."` local embedded mode** — great for tests (`:memory:`), but single-process, no snapshot API parity, fine for unit tests only.

### 3. Embedding model (LOCAL, CPU)
Vault is Spanish → **multilingual matters**. Ranking for Spanish/multilingual on CPU:
- **`bge-m3`** (567M, 1024 dims + sparse + colbert, ~1.2GB): native multilingual (100+ langs incl. Spanish), best quality for Spanish RAG. Heavier on CPU.
- **`nomic-embed-text` v1.5** (137M, 768 dims, ~274MB): lightweight, fastest on CPU, but **English-focused** (MTEB English) — weak for Spanish vault.
- **`snowflake-arctic-embed`** (568M, 1024 dims, ~1.2GB): strong multilingual alternative to bge-m3.
- **`mxbai-embed-large`** (335M, 1024 dims, ~670MB): English MRL focus, weaker Spanish.
- **`qwen3-embedding:0.6b`** (0.6B, 1024 dims via MRL, ~639MB, 32K ctx): multilingual, MTEB 64.33, strong sub-1GB option — good compromise.

**Recommendation**: `bge-m3` for best Spanish quality (it's the canonical multilingual choice and emits sparse+colbert for future hybrid retrieval). If CPU speed is painful at index time (19GB!), fall back to `snowflake-arctic-embed` (same 1024 dims, faster) or `qwen3-embedding:0.6b`. Run on Garuda (GPU) if local CPU is too slow for initial 19GB index.

### 4. Chunking — markdown/heading-aware
Do NOT pull langchain. Write a small heading-aware splitter (split on `#` headings, target ~600-900 tokens, ~100 token overlap). Store heading path in payload for context. Pure stdlib `re` + `pathlib`.
  - Pros: tiny, no heavy dep, correct for Obsidian MD.
  - Cons: naive (no semantic split) — acceptable, headings give structure.
  - Effort: Low

### 5. RAG query flow
Retrieve top-k=5-8 with `query_points(query_filter=<path prefix>, score_threshold=<~0.5>)`, then LLM call to `deepseek-v4-flash` via `https://ollama.com/v1/chat/completions` (OpenAI-compatible) with retrieved chunks in system prompt. Use stdlib `urllib`/`requests` or `httpx`.

### 6. Update/index strategy (19GB now, growing)
- Rescan with **mtime+size+hash** tracking; store `path`, `mtime`, `md5`, `source`, `heading_path` in Qdrant payload.
- Skip unchanged files. Batch upsert (e.g. 100-256 vectors/batch) with retry/backoff.
- **One collection** with `source` + `path` payload filters (collection-per-source multiplies collections; filters are sufficient and simpler). Actually: use ONE collection, filter by `source` and `path` prefix via payload index.
- Long index = progress reporting via `ctx.report_progress()`. Watch-mode optional (inotify) — defer; rescan-on-demand is enough.

### 7. Testing
- `pytest`: unit-test chunker + Qdrant with `:memory:` local client; mock embeddings (deterministic vector) and mock LLM — no network.
- `mcp dev` / `mcp inspector` from `mcp[cli]` for manual protocol testing.

## Recommendation

Python official **`mcp` SDK** (pin `mcp==1.12.x`) + **Qdrant via Docker** + local **`bge-m3`** embeddings via Ollama `/api/embed` (run on Garuda GPU for initial index, own machine for incremental) + **`deepseek-v4-flash`** (cloud) for generation. One Qdrant collection, payload-filtered by `source`/`path`. Small stdlib heading-aware chunker (~600-900 tok, 100 overlap). Tools: `index`, `query`, `search`, `delete`, `list`, `stats`, `config`. Progress via `ctx.report_progress()`. Register as local command in `opencode.json`.

## Risks

- **Embeddings only local**: Ollama Cloud has no embeddings endpoint (verified). Embedding generation is a CPU bottleneck on 19GB+ vault → must decide where embedding runs (local CPU vs Garuda GPU). Mitigate with batching + incremental rescan + possibly `qwen3-embedding:0.6b` (fast, multilingual).
- **MCP SDK API churn**: official SDK breaks its API; pin exact version (config already pins `mcp==1.8.0` for arch-mcp).
- **bge-m3 CPU speed on first 19GB index**: cold-start cost; consider indexing on Garuda GPU, then incremental local.
- **Spanish retrieval quality**: nomic-embed-text is English-focused and will hurt Spanish recall; bge-m3/snowflake/qwen3 multilingual required.
- **No Ollama running locally today**: must install/start Ollama (systemd per existing setup doc) or point at Garuda before index/embed works.
- **OpenCode host context limits**: RAG responses should cap retrieved context (top-k=5-8) to avoid bloating the LLM context.

## Ready for Proposal

Yes. User must decide (tell them in proposal):
1. Embedding model: `bge-m3` (best Spanish) vs `qwen3-embedding:0.6b` (faster) vs `snowflake-arctic-embed`.
2. Where embeddings run: local CPU vs Garuda (192.168.18.172, RTX 3050) — matters for 19GB cold index.
3. Confirm one-collection + payload-filter model.
