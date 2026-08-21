# Tasks: Mnemos Differentiators (Mnē-MCP)

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 900–1300 |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR1 T0–T1 → PR2 T2–T4 → PR3 T5–T6 → PR4 T7–T9 |
| Delivery strategy | **size:exception — RESOLVED 2026-08-21 (user chose "Un solo commit")** |
| Chain strategy | **single commit (no remote; repo local)** |

```
Decision needed before apply: RESOLVED — user picked single-commit (size:exception)
Chained PRs: declined — one commit for the whole change
400-line budget risk: accepted (size:exception)
```

```
Decision needed before apply: RESOLVED — user picked single-commit (size:exception)
Chained PRs: declined — one commit for the whole change
400-line budget risk: accepted (size:exception)
```

### Suggested Work Units

| Unit | Goal | Likely PR | Focused test command | Runtime harness | Rollback boundary |
|------|------|-----------|----------------------|-----------------|-------------------|
| 1 | Rename + config deltas (T0–T1) | PR 1 | `pytest tests/test_config.py` | `grep -r "Mnē-MCP" README.md mcp_rag/` | Revert README/init/config edits; no schema change |
| 2 | Hybrid + cache + reranker (T2–T4) | PR 2 | `pytest tests/test_qdrant.py tests/test_cache.py tests/test_reranker.py` | In-memory Qdrant; no network | Config knobs `hybrid/cache/rerank` off → old dense path |
| 3 | Core flow + resources/prompts (T5–T6) | PR 3 | `pytest tests/test_core.py tests/test_tools.py` | In-memory store + fake embed/llm/rerank | Cache invalidate + rerank guard isolated in core.py |
| 4 | Eval + admin + finalize (T7–T9) | PR 4 | `pytest tests/test_eval.py tests/test_admin.py; pytest` | `smoke.py` offline | Remove eval tool/decorators/extra deps, revert pyproject |

## Phase 1: Display + Config Foundation (T0–T1)

- [ ] T0 Rename display: README title → **Mnē-MCP** (keep pkg `mcp_rag`); add README `## Comparativa con proyectos similares` (table vs ximot/knowledge-mcp, qdrant/mcp-server-qdrant, ancoleman/qdrant-rag-mcp, doitmagic/rag-code-mcp, weverkley/qdrant-mcp-server, w3-mcp-server-qdrant, rageval-mcp; our differentiators: grounded+refusal, heading_path citations, cross-encoder rerank, hybrid, cache, integrated eval, admin page). Files: README.md, mcp_rag/static/index.html (`<title>`), mcp_rag/__init__.py docstring. Verify: `grep -rn "Mnē-MCP" README.md mcp_rag/static/index.html mcp_rag/__init__.py`; `python -c "import mcp_rag"`. REQ: RNM-001/002.
- [ ] T1 Config deltas: mcp_rag/config.py add `rerank=True, rerank_candidates=24, rerank_model="bge-reranker-v2-m3", hybrid=True, cache=True, cache_threshold=0.85`; `JSON_FIELDS += ("rerank","rerank_candidates","hybrid","cache")`; `ENV_FIELDS["OLLAMA_RERANK_MODEL"]="rerank_model"`. Update tests/test_config.py for new fields + env override. Verify: `pytest tests/test_config.py -q`. REQ: RRK-002, HYB-005, CAH-003.

## Phase 2: Retrieval Features (T2–T4)

- [ ] T2 Hybrid sparse: mcp_rag/qdrant_store.py — `ensure_collection` dense(1024,COSINE)+`sparse(SparseVectorParams(IDF))` with try/except → `self.sparse_enabled` flag; `_build_sparse(text)` (lowercase, split `[^a-z0-9áéíóúüñ]+`, tf counts) → SparseVector; `upsert` writes both vectors when sparse_enabled; `search(query_text, query_dense, source, path_prefix, limit, threshold, hybrid, sparse_vector)` dual query + RRF k=60 + dedupe by `DocHit.point_id`; dense-only fast path when not hybrid/unsupported; `DocHit` gains `point_id`. Add tests/test_hybrid.py: `_build_sparse` es-accent counts; in-memory upsert/search fused deduped, `:memory:` sparse-guard skip. Verify: `pytest tests/test_qdrant.py tests/test_hybrid.py -q`. REQ: HYB-001..004.

### T3 Semantic cache

- [ ] T3 mcp_rag/cache.py `SemanticCache(maxsize=256, threshold=0.85)`: LRU OrderedDict + RLock; `lookup(embedding, filters_tuple)` cosine≥threshold → result; `store(embedding, filters, result)`; `invalidate()`; `size`. tests/test_cache.py: hit/miss/evict/invalidate. Verify: `pytest tests/test_cache.py -q`. REQ: CAH-001.

### T4 Reranker

- [ ] T4 mcp_rag/reranker.py `CrossEncoderReranker(cfg, client=None, retry_sleep=None)`: try `POST {ollama_url}/api/rerank` → 404 raises unsupported; fallback local sentence-transformers (guarded import, optional extra, model bge-reranker-v2-m3); `health()` False if neither available → vector-order fallback; `rerank(query, documents) -> scores aligned`; 3 retries/backoff mirroring embed.py. tests/test_reranker.py MockTransport: request shape `{model,query,documents}`, retries, score order, health-False path. Verify: `pytest tests/test_reranker.py -q`. REQ: RRK-001/003. Note: rerank is a NO-OP (graceful) if sentence-transformers absent — acceptable; eval+hybrid+cache+resources are the reliable differentiators.

## Phase 3: Core + Server (T5–T6)

### T5 Core flow

- [ ] T5 mcp_rag/core.py `RagService` gains `reranker=None, cache=None`; revised `query_rag`/`search_vec`: embed once → cache lookup (cosine≥0.85+filters) hit→skip Qdrant+LLM → hybrid search (candidates=rerank_candidates if rerank else top_k) → rerank if enabled+healthy else warn+vector order → threshold only if user-explicit → top_k → cache store → LLM; `delete_docs`/`index_files` call `cache.invalidate()`; `get_stats` gains rerank/hybrid/cache/eval blocks; `from_config` wires reranker+cache. Update tests/test_core.py (fakes for reranker/cache). Verify: `pytest tests/test_core.py -q`. REQ: RRK-003, CAH-002, EVL-002.

### T6 Resources + Prompts

- [ ] T6 server.py: `@mcp.resource("mnemos://doc/{path}")` → `store.get_chunks_by_path(path)` (scroll filter exact path, `with_payload=["text"]`, join) — unknown→MCP error; `@mcp.prompt("rag-query")` (arg question→grounding+user), `@mcp.prompt("index-vault")` (path/source). Add `QdrantStore.get_chunks_by_path`. tests/test_resources.py: store.get_chunks_by_path; app.list_resources/read_resource + list_prompts. Verify: `pytest tests/test_qdrant.py tests/test_resources.py tests/test_tools.py -q`. REQ: RES-001/002, HYB-003(resource read).

## Phase 4: Eval + Admin + Finalize (T7–T9)

### T7 Eval

- [ ] T7 mcp_rag/eval.py: `GoldenQuery`, `EvalReport`, `load_golden(path)`, `run_eval(service, golden, k_values=(1,3,5))` → recall@1/3/5, MRR, nDCG@k (retrieval only, no LLM) → persist eval_results.json; `add_golden(query, expected_paths, path)`; missing file→empty metrics+message. server.py tool `eval(action run|add)`. `golden_set.json` committed (5-query sample). tests/test_eval.py synthetic corpus exact metrics + missing-file path. Verify: `pytest tests/test_eval.py -q`. REQ: EVL-001/003/004.

### T8 Admin

- [ ] T8 mcp_rag/static/index.html: `<title>` Mnē-MCP, search playground rerank toggle, dashboard chips (rerank/hybrid/cache/eval); admin.py passes through new stats fields. tests/test_admin.py `/api/stats` contains rerank/hybrid/cache/eval. Verify: `pytest tests/test_admin.py -q`. REQ: RNM-001, EVL-002.

### T9 Finalize

- [ ] T9 pyproject.toml optional extra `[project.optional-dependencies] rerank=["sentence-transformers"]`; README install note (rerank optional pip extra; `ollama pull bge-reranker-v2-m3` NOT needed now); full `pytest -q` offline green; update smoke.py with hybrid/rerank/cache/eval flags. Verify: `pytest -q` (all suites offline). REQ: SCC-001.

## Acceptance

All tasks check off only after: named-vector collection upgrades in place (HYB-001), hybrid degrades to dense-only on old points (HYB-003), cache invalidates on index/delete (CAH-002), rerank graceful NO-OP when sentence-transformers absent (RRK-003), eval run returns recall@k/MRR/nDCG offline (EVL-001/004), resources listable/readable (RES-001), README+admin titled Mnē-MCP (RNM-001).
