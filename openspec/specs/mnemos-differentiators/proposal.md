# Proposal: Mnemos Differentiators

## 1. Summary

Evolve `mcp-rag-opencode` (display name **Mnē-MCP**) into a differentiated RAG server. The retrieval-plumbing field is crowded; the moat is **answer quality + measurement**. Add 5 features nobody in the competitor set has: cross-encoder reranking, dense+sparse hybrid search, MCP Resources/Prompts, semantic caching, and an integrated eval harness. Plus rename to **Mnē-MCP** (display only; package stays `mcp_rag`).

## 2. Motivation

University differentiator: competitors (ximot/knowledge-mcp, qdrant/mcp-server-qdrant, ancoleman/qdrant-rag-mcp, doitmagic/rag-code-mcp, weverkley/qdrant-mcp-server, w3-mcp-server-qdrant, rageval-mcp) all stop at retrieval plumbing. None offer: grounded server-side generation with refusal, heading_path chunk citations, reranking, hybrid search, eval, or a RAG admin page.

## 3. Scope IN / OUT

**IN**: rerank (bge-reranker-v2-m3), hybrid dense+sparse (Qdrant BM25 + RRF), MCP Resources + 2 Prompts, semantic cache, `eval` tool + golden set + `stats` eval fields, rename display to Mnē-MCP. Tests + README comparison section.

**OUT** (deferred): fsnotify watching, one-command installer, all-in-one docker-compose with Ollama, point-ID determinism refactor (already md5(path#index)), multi-user.

## 4. Approach per feature

- **Reranking** — after vector search fetch `rerank_candidates` (24) hits, rerank with **local cross-encoder via Ollama `/api/rerank`** (model `bge-reranker-v2-m3`). **⚠️ VERIFIED 2026-08-21: Ollama 0.32.15 has NO `/api/rerank` endpoint (404)** — so rerank is implemented with a **local cross-encoder fallback via `sentence-transformers` if installed, else pure vector-order fallback**. Config: `rerank: bool=true`, `rerank_candidates=24`, `rerank_model=bge-reranker-v2-m3`. Failure: model missing → warn + vector-order fallback (graceful).
- **Hybrid** — build sparse vectors client-side (lowercase, split, count-tf); store `dense` (1024) + `sparse` named vectors per point; query both, fuse with RRF (k=60); score_threshold applies to fused score. Config: `hybrid: bool=true`. Failure: sparse absent on old points → dense-only query.
- **Resources/Prompts** — `@mcp.resource("mnemos://doc/{path}")` (text = chunk text, listable) + `@mcp.prompt("rag-query")` (question arg) + `@mcp.prompt("index-vault")`. Failure: unknown resource → MCP error.
- **Semantic cache** — key = question embedding cosine ≥0.85 + same filter params; store fused hits + answer; in-memory LRU dict `maxsize=256`, invalidate on delete/index. Config: `cache: bool=true`. Failure: cache miss = normal path.
- **Eval** — `eval add-golden` / `eval run` over `./golden_set.json` ({query, expected_paths[]}); compute recall@k (1,3,5), MRR, nDCG@k on retrieval (no LLM); `stats` gains eval fields. Failure: missing golden file → empty metrics + message.

## 5. Rename decision

Display name **Mnē-MCP** everywhere: README title, admin title, package display name. Repo folder → `mnemos-mcp` (optional). **Python import package stays `mcp_rag`** — renaming breaks imports/tests/docs; note in README, defer package rename unless user insists.

## 6. Options considered

- **Cross-encoder vs LLM-rerank** → cross-encoder (fast, domain-proper, local Ollama).
- **Qdrant BM25 vs client sparse** → Qdrant BM25 (server-side scoring, no client sparse weights from Ollama — Ollama `/api/embed` returns dense only).
- **SQLite vs in-memory cache** → in-memory LRU (lazy, no persistence needed).
- **eval integrated vs separate** → integrated tool + unit tests (synthetic corpus).

## 7. Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Ollama `/api/rerank` unsupported (VERIFIED: 404) | Certain | Cross-encoder via sentence-transformers local; else vector-order fallback |
| Sparse storage cost (19GB) | Med | Sparse ~ tokens only; acceptable |
| RRF fused score ≠ cosine 0.5 | Med | Document new threshold semantics; tune on golden set |
| Golden set quality | Med | Start 20-30 queries from vault topics |
| Cache invalidation correctness | Low | Invalidate on delete/index; TTL 1h |
| sentence-transformers dep weight | Med | Optional extra; graceful degrade if absent |

## 8. User impact

- `opencode.json` registration unchanged (same command).
- New env `OLLAMA_RERANK_MODEL` default `bge-reranker-v2-m3`; `ollama pull bge-reranker-v2-m3` locally.
- README new comparison section; admin page title → Mnē-MCP.

## 9. Success criteria

- [ ] Rerank improves recall@k vs no-rerank on golden set.
- [ ] `eval run` returns recall@k/MRR/nDCG + per-query breakdown.
- [ ] Resources list + read works in MCP inspector (mnemos://doc/...).
- [ ] Cache hit returns instantly (2nd identical query).
- [ ] README/admin titled Mnē-MCP; hybrid + rerank + cache configurable.

## Affected Areas

| Area | Impact |
|------|--------|
| `mcp_rag/core.py`, `qdrant_store.py`, `config.py`, `server.py`, `admin.py` | Modified — rerank/hybrid/cache/eval/resources |
| `mcp_rag/reranker.py`, `mcp_rag/cache.py`, `mcp_rag/eval.py` | New |
| `README.md`, `tests/` | Modified — rename + comparison; eval/hybrid tests |
| `golden_set.json` | New — eval fixture |

## Rollback

Config knobs default-on; set `rerank:false, hybrid:false, cache:false` restores prior retrieval path. Remove Resources/Prompts/eval decorators if MCP client chokes. No collection schema change (named vectors are additive).

## Dependencies

`bge-reranker-v2-m3` via local Ollama; Qdrant supports multiple named vectors + sparse (existing image); `mcp` SDK (already pinned).

## Key Learnings

1. Cross-encoder reranking, Qdrant BM25 sparse fusion, and semantic caching are the differentiators over retrieval-only competitors.
2. Ollama `/api/embed` returns dense only; sparse weights must be built client-side for Qdrant BM25.
3. The rename is display-only (Mnē-MCP); the `mcp_rag` import package stays to avoid churn.
