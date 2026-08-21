# Tasks: mcp-rag-opencode

Sequential, small phases for sdd-apply. Each task is independently verifiable offline.

## Phase summary table

| ID | Title | Depends on | Key REQ | Verify |
|----|-------|-----------|---------|--------|
| T0 | Bootstrap | — | CONF-001, DOC-001 | `python -m build` / `make install` |
| T1 | Config + dataclasses | T0 | CONF-001 | `pytest tests/test_config.py -q` |
| T2 | Chunker | T1 | CHUNK-001, CHUNK-002 | `pytest tests/test_chunker.py -q` |
| T3 | QdrantStore | T1 | QDR-001, QDR-002 | `pytest tests/test_qdrant.py -q` |
| T4 | Embedder | T1 | EMB-001, EMB-002 | `pytest tests/test_embed.py -q` |
| T5 | LLM provider | T1 | LLM-001 | `pytest tests/test_llm.py -q` |
| T6 | Indexer | T2, T3, T4 | IDX-001, CHUNK-002 | `pytest tests/test_indexer.py -q` |
| T7 | Core | T2, T3, T4, T5, T6 | ADMIN-002, MCP-002..008 | `pytest tests/test_core.py -q` |
| T8 | MCP server | T7 | MCP-001..008 | `pytest tests/test_tools.py -q` |
| T9 | Admin page | T7 | ADMIN-001, ADMIN-003 | `pytest tests/test_admin.py -q` |
| T10 | Integration + docs | T8, T9 | TEST-001, DOC-001 | `pytest -q` (offline) |
| T11 | (stretch) .pdf v2 | — | — | not in scope |

---

## T0 — Bootstrap

**Goal:** Repo skeleton, packaging, deps, env template, docker, make, gitignore, README stub.

**Files:** `pyproject.toml`, `mcp_rag/__init__.py`, `.env.example`, `docker-compose.yml`, `Makefile`, `.gitignore`, `README.md` (stub), `tests/__init__.py` (or conftest-only).

**Details:**
- `pyproject.toml`: `[project]` name `mcp-rag-opencode`, requires-python `>=3.11`, deps `mcp==1.12.*`, `qdrant-client`, `httpx`, `python-dotenv`. `[project.scripts]` entry point `mcp-rag-opencode = mcp_rag.server:main`. `[tool.pytest.ini_options]` `testpaths=["tests"]`.
- `mcp_rag/__init__.py`: `__version__`, re-export `AppConfig`.
- `.env.example`: `OLLAMA_API_KEY=`, `OLLAMA_URL=http://127.0.0.1:11434`, `QDRANT_URL=http://127.0.0.1:6333`, `ADMIN_PORT=8310`, `VAULT_ROOT=`, `COLLECTION=supervisor`, `DEFAULT_SOURCE=vault`.
- `docker-compose.yml`: `qdrant/qdrant`, ports `6333:6333`, `6334:6334`, volume `qdrant_storage:/qdrant/storage`.
- `Makefile`: `install` (pip install -e .), `test` (pytest -q), `run` (mcp-rag-opencode), `admin` (python -m mcp_rag.admin).
- `.gitignore`: `.env`, `__pycache__/`, `*.egg-info/`, `.venv/`, `config.json`, `.pytest_cache/`.
- `README.md`: stub with title + placeholder sections (filled in T10).
- Git: already inited, branch `main` (verify `git branch`; create if missing).

**Verification:** `make install` succeeds; `python -c "import mcp_rag"` works; `docker compose config` valid.

**Depends on:** none.

**REQ:** CONF-001 (env template), DOC-001 (README stub).

---

## T1 — Config + dataclasses

**Goal:** `AppConfig` dataclass + precedence load/save.

**Files:** `mcp_rag/config.py`, `tests/test_config.py`.

**Details:**
- `@dataclass AppConfig` with all fields from design (score_threshold 0.5, top_k 5, chunk_size 800, chunk_overlap 100, model, embed_model, ollama_url, llm_base_url, qdrant_url, admin_port 8310, vault_root, default_source "vault", collection "supervisor", ext (".md",".txt")).
- `load_config(env=None) -> AppConfig`: precedence env/.env > config.json > defaults. Read `.env` via `python-dotenv` (load_dotenv). Env vars: `OLLAMA_API_KEY`, `OLLAMA_URL`, `QDRANT_URL`, `ADMIN_PORT`, `VAULT_ROOT`, `COLLECTION`, `DEFAULT_SOURCE`. JSON file `config.json` (score_threshold, top_k, chunk_size, chunk_overlap, model).
- `save_config(cfg) -> None`: write `config.json` (only the JSON-persistable fields).
- `config.json` path: repo root (or `MCP_RAG_CONFIG` env override — optional, keep simple: repo root).

**Verification:** `pytest tests/test_config.py -q`. Tests: defaults with no env; env overrides; JSON overrides env; save→reload round-trip.

**Depends on:** T0.

**REQ:** CONF-001.

---

## T2 — Chunker

**Goal:** stdlib heading-aware chunker.

**Files:** `mcp_rag/chunker.py`, `tests/test_chunker.py`.

**Details:**
- `@dataclass Chunk: path, heading_path, text, mtime, md5`.
- `Chunker.chunk(path, text, mtime, md5) -> list[Chunk]`:
  - Skip Obsidian YAML frontmatter (leading `---\n...\n---`).
  - Track heading stack from `#`/`##`/`###`; build `heading_path` like `"Proyecto > Setup > Hardware"`.
  - Target ~800 tokens (~450-700 words ES), ~100 token overlap between chunks.
  - Skip empty/heading-only sections (no body → no chunk).
  - `>16MB` guard: raise/flag so indexer records in `errors[]` (return sentinel or raise `RagError`; indexer catches).
- Token estimate: simple heuristic (words/0.75 or chars/4) — stdlib only, no tokenizer dep.

**Verification:** `pytest tests/test_chunker.py -q`. Cases: nested headings → correct heading_path; frontmatter skipped; empty/heading-only → no chunk; overlap present; >16MB → error path.

**Depends on:** T1.

**REQ:** CHUNK-001, CHUNK-002.

---

## T3 — QdrantStore

**Goal:** Qdrant collection mgmt + CRUD + search + stats + health, `:memory:` support.

**Files:** `mcp_rag/qdrant_store.py`, `tests/test_qdrant.py`.

**Details:**
- `@dataclass DocHit: path, heading_path, score, snippet, mtime`.
- `QdrantStore(cfg, client=None)` — accept injected `QdrantClient` (tests pass `QdrantClient(":memory:")`).
- `ensure_collection()`: create `docs` if absent — cosine, 1024 dims; payload indexes: `path` (keyword), `source` (keyword), `mtime` (int).
- `upsert(points: list[dict])`: batch 128, retry/backoff (3x, 1s/2s/4s). Point payload `{source, path, mtime, md5, heading_path}`.
- `search(vector, filter, limit, threshold) -> list[DocHit]`: filter by source/path_prefix, score threshold.
- `delete_by_path(path) -> int`, `delete_by_source(source) -> int`.
- `list_documents(source=None, path_prefix=None, limit=100, offset=0) -> list[dict]` (paginated, `{path, chunks, last_indexed}`).
- `existing_paths(source) -> set[str]`: scroll by filter for stale check.
- `get_collection_stats() -> dict`: vectors_count, by_source.
- `health() -> tuple[bool, str]`: never raises.

**Verification:** `pytest tests/test_qdrant.py -q` (in-memory). Tests: collection created with cosine/1024; upsert+search; filter by source/path_prefix; delete by path/source; paginated list; stats; health.

**Depends on:** T1.

**REQ:** QDR-001, QDR-002.

---

## T4 — Embedder

**Goal:** `OllamaEmbedder` via `/api/embed`.

**Files:** `mcp_rag/embed.py`, `tests/test_embed.py`.

**Details:**
- `OllamaEmbedder(cfg, client=None)` — injectable `httpx.Client` for tests.
- `dimensions = 1024`.
- `embed(texts) -> list[list[float]]`: POST `{ollama_url}/api/embed`, model `bge-m3`, batch 32, timeout 120s, retry 3x backoff 1s/2s/4s.
- HTTP 404 → raise error with exact message `"model not found, run: ollama pull bge-m3"`.
- `health() -> tuple[bool, str]`: GET `/api/tags`; 404 → model-not-found message; never raises.

**Verification:** `pytest tests/test_embed.py -q` (mock httpx). Tests: batch of 100 → 4 requests of 32; retry then fail; 404 message exact; health.

**Depends on:** T1.

**REQ:** EMB-001, EMB-002.

---

## T5 — LLM provider

**Goal:** `LLMProvider` OpenAI-compatible chat.

**Files:** `mcp_rag/llm.py`, `tests/test_llm.py`.

**Details:**
- `@dataclass Answer: text, sources, model`.
- `LLMProvider(cfg, client=None)` — injectable `httpx.Client`.
- `answer(question, context: list[DocHit]) -> Answer`: POST `{llm_base_url}/chat/completions`, model `deepseek-v4-flash`, temperature 0.3, `Authorization: Bearer {OLLAMA_API_KEY}`.
- System prompt: "Answer ONLY from context; cite source paths." Context = `heading_path` + 400-char snippet per chunk, max 8 chunks.
- Missing `OLLAMA_API_KEY` → clear error identifying the missing key.
- `health() -> bool`: key presence only, no network.

**Verification:** `pytest tests/test_llm.py -q` (mock httpx). Tests: request shape (endpoint/model/temp/key); missing key error; prompt includes citations.

**Depends on:** T1.

**REQ:** LLM-001.

---

## T6 — Indexer

**Goal:** incremental scan with mtime+md5 skip, batch upsert, stale delete, progress.

**Files:** `mcp_rag/indexer.py`, `tests/test_indexer.py`.

**Details:**
- `@dataclass IndexReport: files_scanned, files_indexed, files_skipped, chunks_upserted, stale_removed, duration_s, errors`.
- `Indexer(store, embedder, chunker, cfg)`.
- `scan(root, source, force=False, max_files=None, progress_cb=None) -> IndexReport`:
  - Walk `root` for `cfg.ext` files; skip `>16MB` → append to `errors[]`.
  - Skip unchanged: compare stored payload `mtime`+`md5` (via `store.existing_paths`/scroll) → `files_skipped`.
  - Changed/new: chunk → embed (batch 32) → upsert (batch 128).
  - Stale delete: Qdrant paths not on disk → `delete_by_path`, count `stale_removed`.
  - `progress_cb(fraction)` called periodically.
- `max_files` for tests.

**Verification:** `pytest tests/test_indexer.py -q` (tmp_path + os.utime, FakeEmbedder). Tests: unchanged skipped; changed re-indexed; deleted file stale-cleanup; >16MB error; batch.

**Depends on:** T2, T3, T4.

**REQ:** IDX-001, CHUNK-002.

---

## T7 — Core

**Goal:** shared core layer used by BOTH MCP and admin (ADMIN-002).

**Files:** `mcp_rag/core.py`, `tests/test_core.py`.

**Details:**
- Functions: `index_files(...)`, `query_rag(...)`, `search_vec(...)`, `delete_docs(...)`, `list_docs(...)`, `get_stats(...)`, `config_get()`, `config_set(...)`.
- Orchestrates chunker→embed→store (index) and store→llm (query).
- `query_rag`: embed question → `store.search(top_k, filter, threshold)` → no hit ≥ threshold → return "no relevant docs" (no LLM) → else build prompt → `LLMProvider.answer` → `Answer`.
- `get_stats`: `{collection, vectors_count, sources[], last_index_at, qdrant_health, ollama_health, llm_configured}`.
- `RagError` exception hierarchy; health never raises.
- No transport logic here — pure core.

**Verification:** `pytest tests/test_core.py -q` (in-memory qdrant + FakeEmbedder + FakeLLM). Tests: index→search→query end-to-end; no-hit path skips LLM; stats; delete; list; config get/set.

**Depends on:** T2, T3, T4, T5, T6.

**REQ:** ADMIN-002, MCP-002..008 (core behavior).

---

## T8 — MCP server

**Goal:** FastMCP app with 7 tools + progress + error mapping.

**Files:** `mcp_rag/server.py`, `tests/test_tools.py`.

**Details:**
- `FastMCP` app; tools `index`, `query`, `search`, `delete`, `list`, `stats`, `config` — each with JSON schema args, delegating to `core.py`.
- `index`: args `source?`, `path?`, `force_rescan?`, `max_files?`; report progress via `ctx.report_progress()`; return `{files_scanned, files_indexed, files_skipped, chunks_upserted, duration_s, errors[]}`.
- `query`: `question` (required), `top_k` (default 5, clamp max 8), `source?`, `path_prefix?`, `score_threshold?` → `{answer, sources[], model}`.
- `search`: raw retrieval, no LLM → `[{path, heading_path, score, snippet, mtime}]`.
- `delete`: `path` (exact) or `source` → `{deleted_points}`.
- `list`: `source?`, `path_prefix?`, `limit` (default 100, max 1000), `offset` → `[{path, chunks, last_indexed}]`.
- `stats`, `config` (get/set).
- Map `RagError` → MCP `ClearError` with message.
- `main()` entry point for stdio.

**Note (mcp SDK version):** Spec pins `mcp==1.12.x`. In 1.12.x the FastMCP API is `@mcp.tool()` decorator and progress via `ctx.report_progress(progress, total)` (the `Context` is injected as a param named `ctx`). **Fallback:** the user's opencode config pins `mcp==1.8.0` elsewhere as known-good for tool registration — if 1.12.x tool-decorator/progress shape differs at apply time, pin `mcp==1.8.0` in pyproject and use its API. Verify the actual installed shape first (`python -c "import mcp.server.fastmcp as f; print(f.FastMCP)"`).

**Verification:** `pytest tests/test_tools.py -q` (call tools in-process, in-memory qdrant). Tests: each tool returns correct shape; invalid args rejected; top_k clamped; progress reported.

**Depends on:** T7.

**REQ:** MCP-001..008.

---

## T9 — Admin page

**Goal:** stdlib HTTP admin server + vanilla-JS page + async jobs.

**Files:** `mcp_rag/admin.py`, `mcp_rag/jobs.py`, `mcp_rag/static/index.html`, `tests/test_admin.py`.

**Details:**
- `jobs.py`: `@dataclass Job: id, status, progress, result, error`; `JobRegistry.submit(fn, *a, **k) -> id` (ThreadPoolExecutor), `get(id)`.
- `admin.py`: `ThreadingHTTPServer` bound `127.0.0.1:8310` (from cfg.admin_port). Routes:
  - `GET /` → static/index.html
  - `GET /api/stats`, `GET /api/documents` (paginated, filter `source`), `GET /api/search`, `GET /api/health`
  - `POST /api/index` → 202 + job id; `GET /api/jobs/{id}` → progress/status
  - `DELETE /api/documents` (path or source_prefix)
  - All endpoints delegate to `core.py` (ADMIN-002). JSON `{error}` on failure.
- `static/index.html`: vanilla JS, tabs Dashboard / Documents / Index / Search playground. No build step.

**Verification:** `pytest tests/test_admin.py -q` (in-process server, no LLM). Tests: GET / serves page; POST /api/index → 202 + job; poll job → done; DELETE; search; stats.

**Depends on:** T7.

**REQ:** ADMIN-001, ADMIN-003.

---

## T10 — Integration + docs

**Goal:** end-to-end smoke + Spanish README + full offline suite green.

**Files:** `README.md` (full), `scripts/smoke.py` (or `tests/test_smoke.py`), final pass over all tests.

**Details:**
- End-to-end smoke: index tiny sample → search → query with FakeLLM (offline).
- `README.md` (Spanish, DOC-001): setup (`docker compose up qdrant`, `ollama pull bge-m3`, `.env`, register in `opencode.json`, `make install`), usage (index/query via opencode, admin page), ASCII architecture diagram, troubleshooting.
- Final check: `pytest -q` all green offline (TEST-001). Note manual `mcp dev` for protocol testing.

**Verification:** `pytest -q` (offline, all green).

**Depends on:** T8, T9.

**REQ:** TEST-001, DOC-001.

---

## T11 — (stretch, NOT in scope) .pdf support v2

**Goal:** Future work only — listed, not built.

**Details:**
- Add `.pdf` to `cfg.ext`; needs `pypdf` (or text extraction) dependency. Deferred to v2 (vault is markdown). No code in this change.

**Depends on:** — (future).

**REQ:** none (out of scope).

---

## Execution order + rough effort

| Order | Task | Effort |
|-------|------|--------|
| 1 | T0 Bootstrap | S |
| 2 | T1 Config | S |
| 3 | T2 Chunker | M |
| 4 | T3 QdrantStore | M |
| 5 | T4 Embedder | S |
| 6 | T5 LLM | S |
| 7 | T6 Indexer | M |
| 8 | T7 Core | M |
| 9 | T8 MCP server | M |
| 10 | T9 Admin | M |
| 11 | T10 Integration+docs | M |
| — | T11 (.pdf v2) | future |

Effort: S = small (<0.5d), M = medium (0.5–1d). T2–T6 are independent of each other (all depend only on T1) and can be built in any order after T1; T7 depends on all of them.
