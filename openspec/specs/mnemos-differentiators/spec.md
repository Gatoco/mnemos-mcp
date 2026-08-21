# Mnemos Differentiators Specification

## Purpose

Evolve `mcp-rag-opencode` (display **Mnē-MCP**; package import stays `mcp_rag`) with five differentiating capabilities: cross-encoder reranking, dense+sparse hybrid search, MCP Resources/Prompts, semantic caching, and an integrated eval harness. Also rename the display surface and add a README comparison section. All requirements are new; existing tools `index/query/search/delete/list/stats/config` and collection schema remain backward-compatible.

## Requirements

| ID | Requirement | Verifiable via |
|----|-------------|----------------|
| RNM-001 | Display name **Mnē-MCP** used in README title, admin page `<title>`, and `mcp_rag/__init__.py` docstring. Import package `mcp_rag` MUST NOT change; README notes the display-only rename. | grep/tests on README, admin index.html, `__init__.py`; `import mcp_rag` succeeds |
| RNM-002 | README gains a **Comparison with similar projects** section citing ximot/knowledge-mcp, qdrant/mcp-server-qdrant, ancoleman/qdrant-rag-mcp, doitmagic/rag-code-mcp, weverkley/qdrant-mcp-server, w3-mcp-server-qdrant, rageval-mcp — a table and our differentiators (rerank, hybrid, eval, resources, grounded refusal). | README content test / manual review |
| RRK-001 | New `mcp_rag/reranker.py` exposing `CrossEncoderReranker(cfg, client=None)` with `rerank(query, documents: list[str]) -> list[float]`, `POST {cfg.ollama_url}/api/rerank` `{"model": cfg.rerank_model, "query": q, "documents": docs}`; returns scores aligned to input order; injectable httpx client; 3 retries + backoff mirroring `embed.py`. | test_reranker (MockTransport: request shape, retries, scores order) |
| RRK-002 | `AppConfig` gains `rerank: bool=True`, `rerank_candidates: int=24`, `rerank_model: str="bge-reranker-v2-m3"`. `JSON_FIELDS` adds `rerank`, `rerank_candidates` (exposed via `config` tool get/set). | test_config.py |
| RRK-003 | `query_rag`/`search_vec`: retrieve `min(rerank_candidates, 100)` hits via dense search → if `rerank` enabled AND reranker healthy → rerank → keep top_k. After rerank, threshold applies only if user explicitly passed `score_threshold`; otherwise take top_k. If reranker health is False → warn and use vector order. | test_core.py / test_reranker.py |
| HYB-001 | `QdrantStore.ensure_collection` creates named vectors `dense` (1024, cosine) + `sparse` (`SparseVectorParams(modifier=Modifier.IDF)`); old single-vector collections upgraded in place. Sparse unsupported by local Qdrant → guard and keep dense-only collection. | test_qdrant.py :memory: create |
| HYB-002 | Client-side `_build_sparse(text) -> SparseVector(indices, values)`: lowercase, split on non-alphanumeric, count term frequency, no stopwords, values raw counts. | unit test on token counts |
| HYB-003 | `upsert` writes both `dense` (normalized) and `sparse` vectors; points written before hybrid lack sparse → hybrid query degrades to dense-only for those points. | test_qdrant.py upsert/search |
| HYB-004 | `search` runs BOTH dense and sparse queries and fuses via RRF `score=Σ 1/(k+rank_i)`, k=60; returns fused `DocHit.score`; default threshold 0.0 when hybrid on (rely on top_k); user-set threshold applies to fused score. | test_qdrant.py fusion + test_core.py |
| HYB-005 | `AppConfig` gains `hybrid: bool=True` in `JSON_FIELDS`; sparse builder unit tests + store upsert/search with sparse in :memory: (fallback dense-only path tested if :memory: lacks sparse). | test_config.py, test_qdrant.py |
| RES-001 | Server exposes MCP Resource `mnemos://doc/{path}` = chunk text (listable, via new `QdrantStore.get_chunks_by_path(path)` scroll filter); unknown resource → MCP error. | test_tools.py (app.list_resources/read_resource) |
| RES-002 | Server exposes MCP Prompts `rag-query` (arg `question` → system grounding + user) and `index-vault` (arg `path`/`source` → instruct index). | test_tools.py (app.list_prompts) |
| CAH-001 | `mcp_rag/cache.py` `SemanticCache(maxsize=256)` LRU dict, thread-safe lock, `get(question_embedding, filters) -> hits|None` (cosine≥0.85), `put(embedding, filters, result)`, `invalidate()`. | test_cache.py |
| CAH-002 | `query_rag`/`search_vec` check cache before retrieval (when `cache` enabled AND embedding available); hit → return fused hits (+answer for query_rag) skipping Qdrant+LLM; embed once reused for search; `invalidate()` called after index/delete. | test_cache.py + test_core.py |
| CAH-003 | `AppConfig` gains `cache: bool=True` in `JSON_FIELDS`. | test_config.py |
| EVL-001 | `golden_set.json` `[{query, expected_paths:[...]}]` in repo; `eval` tool subcommands `run` (args `k_values="1,3,5"`) computing per-query recall@k, MRR, nDCG@k over retrieval (no LLM) → `{metrics, per_query[]}`; `eval add` (query, expected_paths) appends. | test_eval.py synthetic corpus |
| EVL-002 | `get_stats` gains `eval: {golden_count, last_run, recall@5, mrr}` from last run persisted in `eval_results.json`. | test_core.py stats |
| EVL-003 | `golden_set.json` committed (not gitignored) with a 5-query sample. | file presence |
| EVL-004 | `eval run` with missing/empty golden → empty metrics + message, no crash. | test_eval.py |
| SCC-001 | Success criteria mapping: rerank improves recall@k → RRK-001/003/005 + EVL-001/004; `eval run` returns metrics → EVL-001/004; resources list/read → RES-001; cache hit instant → CAH-002; rename → RNM-001; hybrid/rerank/cache configurable → RRK-002, HYB-005, CAH-003. | coverage check |

## Scenario: Query with rerank + hybrid + cache hit

- GIVEN `rerank=true`, `hybrid=true`, `cache=true`, index has points with dense+sparse, and a prior identical question cached (cosine ≥0.85)
- WHEN `query_rag("…")` is called
- THEN cache hit returns the stored answer+sources instantly — Qdrant search and LLM are NOT called

## Scenario: Query with rerank + hybrid, cache miss

- GIVEN `rerank=true`, `hybrid=true`, cache miss, reranker healthy
- WHEN `query_rag` runs
- THEN dense search returns `min(rerank_candidates,100)` hits, sparse search runs, RRF fuses to top_k, rerank reorders, LLM answers with reranked sources

## Scenario: Rerank model missing

- GIVEN `rerank=true` but reranker.health() is False
- WHEN `query_rag` runs
- THEN a warning is logged and vector (RRF) order is used; no exception; results returned

## Scenario: Sparse absent on old points

- GIVEN existing collection created pre-hybrid (no `sparse` vector on points)
- WHEN `search` runs with `hybrid=true`
- THEN sparse query degrades to dense-only for those points and still returns hits

## Scenario: Cache invalidation on delete

- GIVEN a cached result for question Q
- WHEN `delete_docs(path)` deletes a source chunk in Q's hits
- THEN `SemanticCache.invalidate()` clears the cache; a subsequent `query_rag(Q)` re-retrieves

## Scenario: Eval run on synthetic corpus

- GIVEN `golden_set.json` with known ground truth and an indexed corpus
- WHEN `eval run k_values="1,3,5"` executes
- THEN returns `{metrics:{recall@1,recall@3,recall@5,mrr,ndcg@1,ndcg@3,ndcg@5}, per_query:[...]}` computed over retrieval only

## Scenario: Eval with missing golden file

- GIVEN no `golden_set.json`
- WHEN `eval run` executes
- THEN returns empty `metrics` and a message stating no golden set, no crash

## Scenario: Unknown resource

- GIVEN an indexed doc path `a.md`
- WHEN reading `mnemos://doc/nonexistent.md`
- THEN the server returns an MCP resource error; reading `mnemos://doc/a.md` returns the chunk text

## MODIFIED Requirements (existing spec `mcp-rag-opencode`)

### Requirement: MCP-004 — `search` tool

The `search` tool MUST perform raw vector retrieval WITHOUT an LLM. Args: `text`, `top_k`, `source?`, `path_prefix?`, `score_threshold?`. Returns `[{path, heading_path, score, snippet, mtime}]`.
(Previously: single dense vector search. Now: dense+sparse hybrid fused via RRF when `hybrid=true`, reranked when `rerank=true`; `score` is the fused/reranked score, threshold semantics documented.)

#### Scenario: Raw search
- GIVEN indexed docs and `search("cuando se configuro X")`
- WHEN called
- THEN returns ranked hits with fused `score` and no LLM call is made

#### Scenario: Hybrid + rerank path
- GIVEN `hybrid=true`, `rerank=true`, reranker healthy
- WHEN `search` is called
- THEN hits come from RRF fusion reordered by rerank scores; `score` reflects the post-fusion order

## REMOVED / RENAMED Requirements

_(none — all prior behavior preserved; additions are backward-compatible.)_
