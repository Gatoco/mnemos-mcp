# Mnē-MCP — Servidor MCP de RAG para opencode

**Mnē-MCP** (del griego *Mnēmosynē*, titánide de la memoria; el macron en `ē`, U+0113, marca la vocal larga en la transliteración) es un servidor [MCP](https://modelcontextprotocol.io) que implementa un pipeline **RAG** (Retrieval-Augmented Generation) completo para [opencode](https://opencode.ai): indexa un vault de Obsidian o carpetas de documentos y responde preguntas del asistente usando las notas propias como fuente, citando la ruta exacta de cada fragmento utilizado.

Stack: `#RAG` `#MCP` `#mcp-server` `#opencode` `#Obsidian` `#Qdrant` `#Ollama` `#DeepSeek` `#bge-m3` `#Python`

## Funcionalidad

1. **Indexado incremental** — escanea directorios (`.md`, `.txt`) desde disco con detección de cambios por `mtime` + `md5`, eliminación de puntos obsoletos (stale cleanup) y soporte de volúmenes grandes (~19 GB).
2. **Embeddings locales** — modelo `bge-m3` vía Ollama local (`/api/embed`), 1024 dimensiones, multilingüe (español incluido), normalizados L2 para escalar correctamente con similitud coseno.
3. **Recuperación semántica** — Qdrant (Docker), colección única con filtros por payload (`source`, `path`, `mtime`, `md5`, `heading_path`) y umbral de relevancia configurable.
4. **Generación fundamentada** — respuestas con `deepseek-v4-flash` (Ollama Cloud) únicamente cuando la evidencia supera el umbral; en caso contrario devuelve rechazo explícito ("no relevant documents found"). Sin evidencia, no genera.

## Arquitectura

```
MCP tools (stdio) ──┐
                    ├── RagService (core) ────▶ Qdrant (vectorial, Docker)
Admin page (:8310) ─┘       │
                            ├── OllamaEmbedder (bge-m3 local)
                            └── LLMProvider (deepseek-v4-flash, Ollama Cloud)
```

- Python 3.14 · SDK MCP oficial (`mcp==1.12.4`) · `qdrant-client` · `httpx`
- 7 herramientas: `index` · `query` · `search` · `delete` · `list` · `stats` · `config`
- Página de administración local (`http://127.0.0.1:8310`): dashboard, documentos indexados, indexado asíncrono con progreso y playground de búsqueda. Comparte el mismo core que las herramientas MCP (cero lógica duplicada).
- Suite de 70 tests offline (Qdrant `:memory:` + mocks) + smoke test end-to-end.

## Uso para desarrollador

```bash
docker compose up -d qdrant   # Qdrant
ollama pull bge-m3            # embeddings locales
cp .env.example .env          # OLLAMA_API_KEY + VAULT_ROOT
make install                  # instala el paquete (venv)
make admin                    # panel: http://127.0.0.1:8310
```

Registro en `opencode.json` (sección `mcp`, tipo `local`, comando `python3 mcp_rag/server.py` o el entry point `mcp-rag-opencode`).

| Solicitud | Comportamiento |
|---|---|
| `index` (source, path, force_rescan, max_files) | Escaneo incremental, omite archivos sin cambios, sube vectores por lotes |
| `query` (question, top_k ≤ 8, source, path_prefix, score_threshold) | Recuperación → reranker/LLM → respuesta con citas (`path` + `heading_path`) |
| `search` | Hits crudos con `score`, sin generación |
| `stats` | Salud de Qdrant/Ollama, estado del LLM, recuentos por fuente |

## Roadmap

Búsqueda híbrida densa+dispersa (BM25 + RRF) · cross-encoder reranking · caché semántica · harness de evaluación (recall@k, MRR, nDCG) — planificado en `openspec/specs/mnemos-differentiators/`.

## Académico

**Proyecto Integrado** · Prof. Christian Pérez · Alumno: Gat · 2026
