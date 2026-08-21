# Design: mcp-rag-opencode

## Technical Approach

Python 3.14 MCP server (`mcp==1.12.x`, stdio) implementing RAG: heading-aware chunker → local `bge-m3` embeddings via Ollama `/api/embed` → upsert into one Qdrant collection → top-k retrieval filtered by source/path → LLM answer (`deepseek-v4-flash` via `https://ollama.com/v1/chat/completions`). A thin transport layer (MCP tools + stdlib admin HTTP) delegates to a single shared `core.py` (spec ADMIN-002). Admin page is a dependency-free vanilla-JS single page served by `http.server` on `127.0.0.1:8310`.

## Architecture Decisions

| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| Admin server | FastAPI+uvicorn vs stdlib `http.server` | **stdlib `ThreadingHTTPServer`** | No-auth, single-user, local only; needs GET/POST/DELETE + JSON only — no streaming/SSE/websockets/CORS. Adding FastAPI+uvicorn is a build/runtime dep for zero benefit. Async index via `ThreadPoolExecutor`. |
| Admin page | SPA framework vs vanilla JS | **vanilla JS, ~250 lines** | No build step, no npm. Layout = tabs (Dashboard, Documents, Index, Search, Health). |
| jobs.py | async job registry vs synchronous | **tiny async registry** | `POST /api/index` MUST return 202 + pollable job (ADMIN-003). In-memory `dict[id]→Job` + `ThreadPoolExecutor`. No persistence (jobs die on restart — acceptable, local only). |
| File types v1 | `.md` + `.txt` vs +`.pdf` | **`.md` + `.txt`** | PDF parsing needs a dep (pypdf). Out of v1; flag v2. |
| Config | JSON file + env + defaults | **env/.env > JSON file > defaults** | `config` tool persists to JSON (CONF-001). |
| Docker | server in container vs host | **Qdrant only in Docker** | Server needs MCP stdio + local Ollama; containerizing adds nothing. |

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `mcp_rag/__init__.py` | Create | Package marker, exports `AppConfig`, version. |
| `mcp_rag/config.py` | Create | `AppConfig` dataclass + env/.env/JSON load + save + defaults. |
| `mcp_rag/embed.py` | Create | `OllamaEmbedProvider` (`bge-m3`, `/api/embed`, batch 32, retry 3x/1s/2s/4s, timeout 120s). |
| `mcp_rag/llm.py` | Create | `LLMProvider.answer()` → `Answer`, OpenAI-compatible chat, temp 0.3, `health()`. |
| `mcp_rag/chunker.py` | Create | Heading-aware `Chunker.chunk()` → `list[Chunk]`. |
| `mcp_rag/qdrant_store.py` | Create | `QdrantStore` — collection mgmt, upsert, search, delete, list, stats, payload indexes. |
| `mcp_rag/indexer.py` | Create | `Indexer.scan()` incremental (mtime+md5), batch 128, progress cb, stale delete. |
| `mcp_rag/core.py` | Create | Shared `index_files/query_rag/search_vec/delete_docs/list_docs/get_stats`; called by BOTH MCP + admin. |
| `mcp_rag/server.py` | Create | Thin `FastMCP` app: tool registrations → core.py, `ctx.report_progress()`. |
| `mcp_rag/jobs.py` | Create | In-memory async job registry + `ThreadPoolExecutor`. |
| `mcp_rag/admin.py` | Create | `ThreadingHTTPServer` routes → core.py + jobs. |
| `mcp_rag/static/index.html` | Create | Vanilla JS admin page. |
| `tests/{conftest,test_chunker,test_embed,test_llm,test_indexer,test_tools,test_admin}.py` | Create | Offline pytest suite, `:memory:` Qdrant, fake providers. |
| `docker-compose.yml` | Create | Qdrant only (6333/6334, `qdrant_storage` volume). |
| `.env.example`, `Makefile`, `pyproject.toml` | Create | Setup/deps: `mcp==1.12.x`, `qdrant-client`, `httpx`, `python-dotenv`. |
| `README.md` | Create | Spanish setup/usage/diagram/troubleshooting (DOC-001). |

## Interfaces / Contracts

```python
# config.py
@dataclass
class AppConfig:
    score_threshold: float = 0.5
    top_k: int = 5
    chunk_size: int = 800       # ~tokens target
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
def load_config(env=None) -> AppConfig      # .env > config.json > defaults
def save_config(cfg: AppConfig) -> None        # write config.json

# embed.py
class OllamaEmbedder:
    def __init__(self, cfg, client): ...
    @property def dimensions: int = 1024
    def embed(self, texts: list[str]) -> list[list[float]]   # batch 32, retry 3x/1s/2s/4s, timeout 120
    def health(self) -> tuple[bool, str]                     # GET /api/tags; 404 → "model not found, run: ollama pull bge-m3"

# llm.py
@dataclass
class Answer: text: str; sources: list[DocHit]; model: str
class LLMProvider:
    def answer(self, question: str, context: list[DocHit]) -> Answer
    def health(self) -> bool   # key presence only, no network

# qdrant_store.py
@dataclass
class DocHit: path: str; heading_path: str; score: float; snippet: str; mtime: int
class QdrantStore:
    def ensure_collection(self) -> None          # cosine, 1024, payload indexes path/source/mtime
    def upsert(self, points: list[dict]) -> None # batch 128, retry/backoff
    def search(self, vector, filter, limit, threshold) -> list[DocHit]
    def delete_by_path(self, path: str) -> int
    def delete_by_source(self, source: str) -> int
    def list_documents(self, source=None, path_prefix=None, limit=100, offset=0) -> list[dict]
    def get_collection_stats(self) -> dict       # vectors_count, by_source
    def existing_paths(self, source) -> set[str] # for stale check (scroll by filter)
    def health(self) -> tuple[bool, str]
    # `:memory:` mode via QdrantClient(":memory:") in tests

# chunker.py
@dataclass
class Chunk: path: str; heading_path: str; text: str; mtime: int; md5: str
class Chunker:
    def chunk(self, path: str, text: str, mtime: float, md5: str) -> list[Chunk]

# indexer.py
@dataclass
class IndexReport:
    files_scanned: int; files_indexed: int; files_skipped: int
    chunks_upserted: int; stale_removed: int; duration_s: float; errors: list[str]
class Indexer:
    def scan(self, root, source, force=False, max_files=None, progress_cb=None) -> IndexReport

# jobs.py
@dataclass
class Job: id: str; status: str; progress: float; result: dict | None; error: str | None
class JobRegistry:
    def submit(self, fn, *a, **k) -> str   # returns job id
    def get(self, job_id: str) -> Job | None
```

## Data Flow

```
INDEX:  fs walk → stat → mtime?md5 vs payload → embed batch 32 → upsert batch 128
        → progress_cb → stale delete (qdrant paths not on disk)
QUERY:  embed(question) → store.search(top_k, filter, threshold)
        → no hit≥thr → "no relevant docs" (no LLM) → else prompt → LLM → Answer
ADMIN:  GET/POST/DELETE → core.py → same as tools; POST /api/index → job registry → poll
```

Query prompt: system = "Answer ONLY from context; cite source paths."; context = `heading_path` + 400-char snippet per chunk; max 8 chunks.

## Testing Strategy

| Layer | What | Approach |
|-------|------|----------|
| Unit | chunker | headings, frontmatter skip, overlap, >16MB skip |
| Unit | embed | batch 32, retry, 404 message |
| Unit | llm | key missing, prompt build (mock httpx) |
| Integration | indexer | `tmp_path` fake files, `os.utime` mtime, skip/change/stale |
| Integration | tools | `:memory:` qdrant + FakeEmbedder(4-dim)/FakeLLM |
| Integration | admin | HTTP endpoints, 202 job, no LLM |

`conftest.py`: `FakeEmbedder` (deterministic 4-dim), `FakeLLM` (echo), `QdrantClient(":memory:")`. All offline (TEST-001).

## Error Handling & Observability

Core exceptions (`RagError` subclasses) → MCP `ClearError` with message / HTTP 4xx|5xx with JSON `{error}`. Health never raises — returns `(bool, msg)`. `logging` to stderr with timestamps (index, embed, query, http). Health endpoints: `/api/health`, stats includes qdrant/ollama/llm health.

## Config Precedence

`env/.env` (`OLLAMA_API_KEY`, `OLLAMA_URL`, `QDRANT_URL`, `ADMIN_PORT`, `VAULT_ROOT`, `COLLECTION`, `DEFAULT_SOURCE`) → `config.json` (`score_threshold`, `top_k`, `chunk_size`, `chunk_overlap`, `model`) → defaults. `config` tool get/set persists to JSON. No secrets committed (`.env` gitignored).

## Docker

`docker-compose.yml`: `qdrant/qdrant`, ports 6333/6334, volume `qdrant_storage:/qdrant/storage`. Server runs on host (needs MCP stdio + local Ollama — not containerized).

## Migration / Rollout

No migration (greenfield). Rollout: `make install` → `docker compose up qdrant` → `ollama pull bge-m3` → `.env` → register in `opencode.json` → first `index`.

## Open Questions

- [ ] `.pdf` support deferred to v2 (needs pypdf/text extraction; not needed for vault which is markdown).
- [ ] None blocking — all v1 decisions resolved.
