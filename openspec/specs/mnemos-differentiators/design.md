# Design: Mnemos Differentiators

## Technical Approach

Evolve `mcp_rag` into **Mnē-MCP** (display-only rename; import package unchanged) by adding five retrieval/measurement differentiators on top of the existing dense-only pipeline: cross-encoder rerank, dense+sparse hybrid (Qdrant BM25 + RRF), MCP Resources/Prompts, semantic cache, and an integrated eval harness. All new features are config-gated (default-on) and degrade gracefully to the prior dense path. The query flow becomes: **cache → hybrid search → rerank → LLM**. Specs: RRK-001..003, HYB-001..005, RES-001/002, CAH-001..003, EVL-001..004, RNM-001/002, SCC-001.

## Architecture Decisions

| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| Rerank transport | LLM-rerank vs cross-encoder | **`POST {ollama_url}/api/rerank`** (RRK-001) | Spec-mandated; mirrors `embed.py` httpx/retry pattern. ⚠️ Official Ollama docs list no `/api/rerank` — design keeps vector-order fallback so absence is non-fatal. |
| Hybrid sparse | Qdrant BM25 vs client weights | **Qdrant `SparseVectorParams(IDF)` + client `_build_sparse`** | Ollama `/api/embed` returns dense only; BM25 scores server-side. |
| Fusion | RRF vs weighted sum | **RRF k=60** | Rank-based, scale-free; robust to dense/sparse score mismatch. |
| Dedupe | by point id vs (path,heading) | **extend `DocHit` with `point_id`** | Dense+sparse return same points; id is the canonical key (md5(path#index)). |
| Cache | SQLite vs in-memory | **in-memory LRU `OrderedDict` + RLock** | Lazy, no persistence; invalidate on index/delete. |
| Eval | separate vs integrated | **`eval` tool + `golden_set.json`** | Integrated tool + offline synthetic tests. |
| Sparse unsupported | hard-fail vs degrade | **`sparse_enabled=False` → dense-only** | Old `:memory:`/server without sparse must not break. |

## Data Flow

```
QUERY:  embed(question) → dense + _build_sparse(text)
        → cache.lookup(emb, filters)? hit → return (skip Qdrant+LLM)
        → store.search(dense + sparse, RRF k=60, candidates=rerank_candidates)
        → rerank? healthy → rerank(query, texts) → sort → top_k
        → else vector order → top_k
        → threshold only if user passed score_threshold
        → cache.store(emb, filters, result)
        → query_rag: LLM.answer(question, hits) → Answer
EVAL:   golden_set.json → for each q: search_vec (hybrid+rerank, no LLM)
        → recall@k / MRR / nDCG@k → eval_results.json
RES:    mnemos://doc/{path} → store.get_chunks_by_path(path) → join text
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `mcp_rag/reranker.py` | Create | `CrossEncoderReranker` — `/api/rerank`, retry 3x/1s/2s/4s, `health()`. |
| `mcp_rag/cache.py` | Create | `SemanticCache(maxsize=256, threshold=0.85)` LRU + RLock. |
| `mcp_rag/eval.py` | Create | `load_golden`, `run_eval`, `add_golden`, `EvalReport`. |
| `mcp_rag/qdrant_store.py` | Modify | Named vectors dense+sparse, `_build_sparse`, hybrid `search`, `get_chunks_by_path`, `DocHit.point_id`. |
| `mcp_rag/config.py` | Modify | New fields + `JSON_FIELDS` + `RERANK_MODEL` env. |
| `mcp_rag/core.py` | Modify | `RagService` gains reranker/cache; revised `query_rag`/`search_vec`; `get_stats` new blocks; invalidate on delete/index. |
| `mcp_rag/server.py` | Modify | `eval` tool, `@mcp.resource`, 2 `@mcp.prompt`. |
| `mcp_rag/admin.py` | Modify | Search playground rerank checkbox + scores; dashboard chips. |
| `mcp_rag/static/index.html` | Modify | Title Mnē-MCP; rerank UI; new stat chips. |
| `mcp_rag/__init__.py` | Modify | Docstring → Mnē-MCP (RNM-001). |
| `README.md` | Modify | Rename + comparison section + `ollama pull bge-reranker-v2-m3`. |
| `golden_set.json` | Create | 5-query sample (synthetic paths, documented as demo). |
| `tests/{test_reranker,test_hybrid,test_cache,test_eval,test_resources}.py` | Create | New offline suites. |
| `tests/{test_core,test_config,test_qdrant,test_tools}.py` | Modify | New fields/flow coverage. |

## Interfaces / Contracts

```python
# reranker.py
class CrossEncoderReranker:
    def __init__(self, cfg, client=None, retry_sleep=None): ...
    def rerank(self, query: str, documents: list[str]) -> list[float]  # aligned to input order
    def health(self) -> tuple[bool, str]  # GET /api/tags contains cfg.rerank_model

# cache.py
class SemanticCache:
    def __init__(self, maxsize=256, threshold=0.85): ...
    def lookup(self, question_embedding, filters_tuple) -> result | None
    def store(self, question_embedding, filters_tuple, result) -> None
    def invalidate(self) -> None
    @property def size(self) -> int

# eval.py
@dataclass class GoldenQuery: query: str; expected_paths: list[str]
@dataclass class EvalReport: metrics: dict; per_query: list[dict]
def load_golden(path) -> list[GoldenQuery]
def run_eval(service, golden, k_values=(1,3,5)) -> EvalReport
def add_golden(query, expected_paths, path) -> None

# qdrant_store.py
@dataclass class DocHit: path; heading_path; score; snippet; mtime; point_id: str
class QdrantStore:
    def ensure_collection(self) -> None  # dense(1024,COSINE)+sparse(IDF); guard→sparse_enabled=False
    def _build_sparse(self, text) -> SparseVector  # lowercase, split [^a-z0-9áéíóúüñ]+, tf counts
    def upsert(self, points) -> None  # vectors={"dense":..,"sparse":..} if sparse_enabled
    def search(self, query_text, query_dense, source=None, path_prefix=None,
               limit=5, threshold=0.0, hybrid=True, sparse_vector=None) -> list[DocHit]
    def get_chunks_by_path(self, path) -> list[str]  # scroll filter exact path, with_payload=["text"]

# config.py — new fields
rerank: bool = True; rerank_candidates: int = 24; rerank_model: str = "bge-reranker-v2-m3"
hybrid: bool = True; cache: bool = True; cache_threshold: float = 0.85
JSON_FIELDS += ("rerank","rerank_candidates","hybrid","cache")
ENV_FIELDS["OLLAMA_RERANK_MODEL"] = "rerank_model"

# core.py — RagService
def __init__(self, store, embedder, llm, chunker, indexer, cfg,
             reranker=None, cache=None): ...
def query_rag(self, question, top_k=5, source=None, path_prefix=None, score_threshold=None) -> dict
def search_vec(self, query, top_k=5, source=None, path_prefix=None, score_threshold=None) -> list[DocHit]
def get_stats(self) -> dict  # + rerank/hybrid/cache/eval blocks
```

## Testing Strategy

| Layer | What | Approach |
|-------|------|----------|
| Unit | reranker | MockTransport: request shape `{model,query,documents}`, retries, scores order, health False → fallback |
| Unit | hybrid | `_build_sparse` token counts (es accents); store upsert/search fused deduped; `:memory:` sparse guard skip |
| Unit | cache | hit/miss/invalidate/LRU eviction |
| Unit | eval | synthetic corpus known ground truth → recall@k/MRR/nDCG; missing golden → empty metrics |
| Unit | resources | `app.list_resources`/`read_resource`; unknown path → error; `list_prompts` |
| Integration | core | query flow with rerank/cache; cache hit skips Qdrant+LLM; invalidate on delete |
| Integration | config | new fields persist + env override |

## Threat Matrix

N/A — no routing, shell, subprocess, VCS/PR automation, executable-file classification, or process-integration boundary. (Rerank is an HTTP POST to a local Ollama service, not a shell/subprocess boundary.)

## Migration / Rollout

Existing collections lack the `sparse` vector. `ensure_collection` upgrades the schema in place (adds sparse params); existing points have no sparse → hybrid degrades to dense-only for them (HYB-003). To enable hybrid on old data, reindex with `force_rescan=true` (rebuilds points with sparse). No data loss; dense vectors untouched. Rollout: `ollama pull bge-reranker-v2-m3` (user runs; graceful fallback if missing) → restart server → `index force_rescan=true`.

## Open Questions

- [ ] **`/api/rerank` availability** — official Ollama docs list no rerank endpoint. Design follows RRK-001 contract; if absent at apply, rerank stays disabled (health False) and vector order is used. Verify at apply; no design change needed.
