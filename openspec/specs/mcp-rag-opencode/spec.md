# mcp-rag-opencode Specification

## Purpose

A local RAG MCP server for opencode: index a Markdown vault (Obsidian) into Qdrant via local bge-m3 embeddings, retrieve semantically, and answer questions from the user's own notes using a cloud LLM. Exposes MCP tools (`index`, `query`, `search`, `delete`, `list`, `stats`, `config`) over stdio plus a local read-only admin web page on 127.0.0.1:8310. A single core layer is shared by both transports (no duplicate logic). All requirements are new (no prior behavior).

## Requirements

### Requirement: MCP-001 — Server transport and registration

The system MUST run as an MCP server over stdio using the official `mcp` SDK, pinned to `mcp==1.12.x`, and MUST register as a local command in `opencode.json`. Every tool MUST accept and return JSON-schema-validated inputs/outputs.

#### Scenario: Registered and invoked
- GIVEN `opencode.json` contains a `mcp-rag-opencode` local command entry pointing at the package entry point
- WHEN opencode connects via stdio
- THEN the server handshakes and advertises tools `index`, `query`, `search`, `delete`, `list`, `stats`, `config`

#### Scenario: Invalid args rejected
- GIVEN a tool call with args violating its JSON schema (e.g. `query` without `question`)
- WHEN the tool is invoked
- THEN the server returns a validation error and performs no side effect

### Requirement: MCP-002 — `index` tool

The `index` tool MUST accept args `source` (optional, default configured vault root), `path` (optional sub-path), `force_rescan` (bool, default false), `max_files` (optional int, for tests). It MUST report progress via `ctx.report_progress()` and return `{files_scanned, files_indexed, files_skipped, chunks_upserted, duration_s, errors[]}`.

#### Scenario: Happy path index
- GIVEN a source dir with 3 markdown files and a reachable Qdrant + Ollama
- WHEN `index` is called with default `source`
- THEN it returns `files_scanned=3`, `files_indexed=3`, `chunks_upserted>0`, `errors=[]`, and reports progress during the run

#### Scenario: Index single sub-path
- GIVEN a source with subfolders and `path="Setup"`
- WHEN `index` is called with that `path`
- THEN only files under that sub-path are scanned and reported

### Requirement: MCP-003 — `query` tool

The `query` tool MUST accept `question` (required string), `top_k` (int, default 5, max 8), `source` (optional), `path_prefix` (optional), `score_threshold` (optional float override). It MUST retrieve top-k chunks then call the LLM, and MUST return `{answer, sources[{path, score, snippet}], model}`.

#### Scenario: Grounded answer
- GIVEN indexed vault containing the answer and a configured LLM key
- WHEN `query` is called with a question
- THEN `answer` cites retrieved content and `sources` lists the retrieved paths with scores

#### Scenario: top_k cap
- GIVEN `top_k=20`
- THEN the value is clamped to 8

### Requirement: MCP-004 — `search` tool

The `search` tool MUST perform raw vector retrieval WITHOUT an LLM. Args: `text`, `top_k`, `source?`, `path_prefix?`, `score_threshold?`. Returns `[{path, heading_path, score, snippet, mtime}]`.

#### Scenario: Raw search
- GIVEN indexed docs and `search("cuando se configuro X")`
- WHEN called
- THEN returns ranked hits with `score` and no LLM call is made

### Requirement: MCP-005 — `delete` tool

The `delete` tool MUST accept `path` (exact, required when `source` absent) or `source` (all docs of source). Returns `{deleted_points}`.

#### Scenario: Delete by path
- GIVEN a previously indexed `path`
- WHEN `delete` is called with that exact `path`
- THEN `deleted_points >= 1` and subsequent `list` no longer shows it

#### Scenario: Delete by source
- GIVEN multiple docs under a source
- WHEN `delete` called with `source`
- THEN all points for that source are removed and `deleted_points` reflects the count

### Requirement: MCP-006 — `list` tool

The `list` tool MUST accept `source?`, `path_prefix?`, `limit` (default 100, max 1000), `offset`. Returns `[{path, chunks, last_indexed}]` (paginated).

#### Scenario: Paged list
- GIVEN 250 indexed docs
- WHEN `list` called with `limit=100, offset=0` then `offset=100`
- THEN each call returns at most 100 entries and the second starts after the first

#### Scenario: Path prefix filter
- GIVEN `path_prefix="Proyecto"`
- THEN only docs under that prefix are returned

### Requirement: MCP-007 — `stats` tool

The `stats` tool MUST return `{collection, vectors_count, sources[{source, files, chunks}], last_index_at, qdrant_health, ollama_health, llm_configured}`.

#### Scenario: Health snapshot
- GIVEN running Qdrant and Ollama, configured key
- WHEN `stats` called
- THEN `qdrant_health` and `ollama_health` are healthy and `llm_configured=true`

#### Scenario: Service down
- GIVEN Ollama not reachable
- THEN `ollama_health` reports the failure without raising

### Requirement: MCP-008 — `config` tool

The `config` tool MUST support `get` (returns full config) and `set` (accepts `{score_threshold?, top_k?, chunk_size?, chunk_overlap?, model?}`), persisting changes to the config file.

#### Scenario: Get current config
- GIVEN a config file with defaults
- WHEN `config` `get` called
- THEN returns `score_threshold=0.5`, `top_k=5`, and effective chunk/model

#### Scenario: Set persisted
- GIVEN `set` with `{top_k: 7}`
- WHEN invoked
- THEN the new value is returned by subsequent `get` and survives a server restart

### Requirement: EMB-001 — Embedding provider

The embedding provider MUST call Ollama `/api/embed` locally, model `bge-m3`, 1024 dims, batch size 32, retry 3x with backoff 1s/2s/4s, timeout 120s.

#### Scenario: Batch embed
- GIVEN 100 chunks
- WHEN embedding requested
- THEN vectors are 1024-dim and requests are sent in batches of 32

#### Scenario: Retry then fail
- GIVEN transient Ollama failure
- WHEN embedding attempted
- THEN it retries 3 times with increasing backoff, then surfaces the error

### Requirement: EMB-002 — Embedding model-not-found

The provider MUST detect HTTP 404 from `/api/embed` and return the error message "model not found, run: ollama pull bge-m3".

#### Scenario: Model absent
- GIVEN Ollama without `bge-m3`
- WHEN embedding attempted
- THEN the error message exactly matches the required text

### Requirement: LLM-001 — LLM provider

The LLM provider MUST call the OpenAI-compatible `https://ollama.com/v1/chat/completions` endpoint, model `deepseek-v4-flash`, using `OLLAMA_API_KEY`, temperature 0.3. The system prompt MUST ground answers in retrieved context and cite sources.

#### Scenario: Grounded generation
- GIVEN a valid key and retrieved chunks
- WHEN `query` runs
- THEN the request uses the configured endpoint/model and the answer cites source paths

#### Scenario: Missing key
- GIVEN `OLLAMA_API_KEY` unset
- WHEN LLM call attempted
- THEN a clear error identifies the missing key

### Requirement: CHUNK-001 — Heading-aware chunker

The chunker MUST be stdlib-only, markdown-heading-aware, target 600-900 tokens (~450-700 words ES), ~100 token overlap, and store `heading_path` (e.g. "Proyecto > Setup > Hardware").

#### Scenario: Heading nesting
- GIVEN a doc with nested `#`/`##` headings
- WHEN chunked
- THEN chunks carry the correct `heading_path` hierarchy and stay within the token budget

#### Scenario: Frontmatter skipped
- GIVEN a doc with Obsidian YAML frontmatter
- WHEN chunked
- THEN frontmatter is excluded from chunks

#### Scenario: Empty/heading-only chunk skipped
- GIVEN a section with no body text
- WHEN chunked
- THEN no empty chunk is produced

### Requirement: CHUNK-002 — Oversized file

The chunker MUST skip files >16MB, count them in the indexer's `errors[]`.

#### Scenario: Oversized skip
- GIVEN a >16MB markdown file in the scan
- WHEN indexing
- THEN the file is skipped and an entry is appended to `errors[]`

### Requirement: IDX-001 — Incremental indexer

The indexer MUST skip files whose `mtime` and `md5` are unchanged since the stored payload, upsert in batches of 128 with retry/backoff, and MUST delete stale points for files that disappeared from disk (v1 policy: delete stale points). It MUST report progress via MCP progress tokens.

#### Scenario: Unchanged file skipped
- GIVEN a previously indexed file with unchanged `mtime`+`md5`
- WHEN re-indexing
- THEN `files_skipped` increments and `chunks_upserted` does not

#### Scenario: Changed file re-indexed
- GIVEN a file whose content changed
- WHEN re-indexing
- THEN the new chunks are upserted and its count reflects the change

#### Scenario: Deleted file stale-cleanup
- GIVEN a file removed from disk since last index
- WHEN re-indexing
- THEN its Qdrant points are deleted and reported as stale-removed

### Requirement: QDR-001 — Qdrant integration

The system MUST use one Qdrant collection `docs` (cosine, 1024 dims), run via Docker `qdrant/qdrant` on ports 6333/6334 with a volume, payload `{source, path, mtime, md5, heading_path}`, and payload indexes on `path` (keyword), `source` (keyword), `mtime` (int). A `:memory:` client MUST support the tests.

#### Scenario: Collection ready
- GIVEN the container is up
- WHEN the server connects
- THEN collection `docs` exists with cosine distance and 1024-dims

#### Scenario: Filtered retrieval
- GIVEN points from two sources
- WHEN `search` filters by `source` and `path_prefix`
- THEN only matching points return

### Requirement: QDR-002 — Health check

The server MUST check Qdrant health and surface it in `stats`.

#### Scenario: Qdrant down
- GIVEN Qdrant not running
- WHEN `stats` called
- THEN `qdrant_health` reports failure without crashing

### Requirement: ADMIN-001 — Admin server binding and endpoints

The admin page MUST be an HTTP server bound to `127.0.0.1:8310` (configurable via `ADMIN_PORT`), a single vanilla-JS HTML page (no build step), and expose `GET /`, `GET /api/stats`, `GET /api/documents` (paginated, filterable by `source`), `POST /api/index`, `GET /api/search`, `DELETE /api/documents`, `GET /api/health`. No auth (local only).

#### Scenario: Page served
- GIVEN the server running
- WHEN `GET /` on `127.0.0.1:8310`
- THEN returns the admin HTML page

#### Scenario: Trigger index via HTTP
- GIVEN `POST /api/index` with `{source?, path?, force_rescan?}`
- THEN it returns HTTP 202 with a job id, and `GET /api/jobs/{id}` reports progress/status

#### Scenario: Delete via HTTP
- GIVEN `DELETE /api/documents` with `path` or `source_prefix`
- THEN matching points are deleted

### Requirement: ADMIN-002 — Shared core layer

The admin HTTP endpoints MUST call the SAME core functions as the MCP tools (no duplicate logic); MCP tools and HTTP endpoints share one core layer.

#### Scenario: Single implementation
- GIVEN a core function for search
- WHEN both `search` MCP tool and `GET /api/search` are invoked
- THEN both delegate to the same core function (verified by inspection/call-path)

### Requirement: ADMIN-003 — Async index job

`POST /api/index` MUST return HTTP 202 with a job id and `GET /api/jobs/{id}` MUST poll status/progress from an in-memory job registry.

#### Scenario: Long index async
- GIVEN a large index triggered via HTTP
- THEN the POST returns 202 immediately, and polling the job id yields progress then `done`

### Requirement: CONF-001 — Configuration sources

The system MUST load `.env` (no secrets committed; `.env` gitignored) with `OLLAMA_API_KEY`, `OLLAMA_URL` (default `http://127.0.0.1:11434`), `QDRANT_URL` (default `http://127.0.0.1:6333`), `ADMIN_PORT` (default 8310), `VAULT_ROOT`, `COLLECTION` (default `supervisor`), `DEFAULT_SOURCE` (default `vault`), plus a JSON config file (`score_threshold` 0.5, `top_k` 5, `chunk_size`, `overlap`) overridable via the `config` tool. Sensible defaults must apply when values are absent; no secrets in repo.

#### Scenario: Defaults without env
- GIVEN no `.env` and no config overrides
- THEN server uses defaults for URL, ports, collection, `top_k`, `score_threshold`

#### Scenario: Override via config tool
- GIVEN `config set {score_threshold: 0.6}`
- THEN subsequent `query` uses 0.6 as the default threshold

### Requirement: TEST-001 — Offline pytest suite

The test suite MUST run offline (pytest) and cover: chunker (headings, frontmatter skip, overlap, >16MB skip), embedding provider (mocked), LLM provider (mocked), indexer incremental (mtime/md5 skip, stale deletion, batch), tools via `:memory:` Qdrant, and admin HTTP endpoints (no LLM). A manual `mcp dev` note documents protocol testing.

#### Scenario: Full suite offline
- GIVEN no network and mocked providers
- WHEN `pytest` runs
- THEN all tests pass without contacting Qdrant/Ollama/LLM

### Requirement: DOC-001 — Deliverables

The system MUST ship a Spanish `README.md` covering: setup (`docker compose up qdrant`, `ollama pull bge-m3`, `.env`, register in `opencode.json`, `make install`), usage (index/query via opencode, admin page), an ASCII architecture diagram, and troubleshooting. The `memoria` outline is a user-side deliverable (not written by the implementation).

#### Scenario: README completeness
- GIVEN the README
- THEN it contains setup, usage, ASCII diagram, and troubleshooting sections in Spanish

## Scenarios: error paths

### Scenario: No Ollama running
- GIVEN Ollama not reachable
- WHEN `index` or `search` attempted
- THEN a clear connection error surfaces and `ollama_health` reflects it

### Scenario: No Qdrant running
- GIVEN Qdrant not reachable
- WHEN `index` attempted
- THEN a clear connection error surfaces and `qdrant_health` reflects it

### Scenario: No API key
- GIVEN `OLLAMA_API_KEY` unset
- WHEN `query` attempted
- THEN a clear error identifies the missing key; `stats.llm_configured=false`
