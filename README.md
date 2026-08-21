# ⚡ Mnē-MCP — Memoria semántica para opencode

**Mnē-MCP** (de *Mnemosyne*, la titánide griega de la memoria) es un **servidor MCP de RAG completo** para [opencode](https://opencode.ai): indexa tu vault de **Obsidian** y carpetas de documentos, y le permite al asistente responder preguntas **usando tus propias notas como fuente**, citando la ruta exacta de cada afirmación.

`#RAG` `#MCP` `#opencode` `#Obsidian` `#Qdrant` `#Ollama` `#DeepSeek` `#bge-m3` `#Python` `#ProyectoIntegrado`

---

## ▸ Qué hace

1. **Indexa** carpetas reales (`.md`, `.txt`) desde disco — incremental (mtime+md5), limpieza de archivos eliminados, maneja volúmenes grandes (19 GB).
2. **Embede** con `bge-m3` vía Ollama **local**: 1024 dimensiones, multilingüe (clave para notas en español).
3. **Recupera** por similitud semántica desde **Qdrant** (Docker), con filtros por fuente/ruta y umbral de relevancia.
4. **Responde** con `deepseek-v4-flash` (Ollama Cloud) **solo si hay evidencia**: si no encuentra contexto suficiente, lo dice — no inventa.

## 🧱 Cómo está hecho

```
MCP tools (stdio) ──┐
                    ├── RagService (core) ──────▶ Qdrant (vectorial, Docker)
Admin page (:8310) ─┘      │
                           ├── OllamaEmbedder (bge-m3 local)
                           └── LLMProvider (deepseek-v4-flash)
```

- **Python 3.14** · SDK MCP oficial (`mcp==1.12.4`) · `qdrant-client` · `httpx`
- **7 tools**: `index` · `query` · `search` · `delete` · `list` · `stats` · `config`
- **Admin page** local (`http://127.0.0.1:8310`): Dashboard, Documentos, Indexado asíncrono con progreso y playground de búsqueda — misma lógica que las tools MCP, cero duplicación
- **70 tests** offline (Qdrant `:memory:` + mocks) + smoke end-to-end

## 🚀 Para el desarrollador

```bash
docker compose up -d qdrant     # requisito 1: Qdrant
ollama pull bge-m3              # requisito 2: embeddings locales
cp .env.example .env            # OLLAMA_API_KEY + VAULT_ROOT
make install                    # instala el server (venv)
make admin                      # panel de control → http://127.0.0.1:8310
```

**Registralo en `opencode.json`** (sección `mcp`) como comando local, y desde el asistente:

| Pedís | Qué pasa |
|---|---|
| `indexá mi vault` | Escanea, salta lo no modificado, sube vectores |
| `¿cuándo configuré X? respondé con mis notas` | Recupera fragmentos → responde con **citas** (`path` + sección) |
| `buscá en mis notas "hardware del poco x3"` | Hits crudos con `score` |
| Una pregunta sin datos en el índice | Rechazo honesto: *"No se encontraron documentos relevantes"* |

## 🧠 Roadmap

Búsqueda **híbrida** densa+dispersa (BM25 + RRF) · **cross-encoder rerank** · **caché semántica** · **eval harness** (recall@k / MRR / nDCG) — planificados en `openspec/specs/mnemos-differentiators/`.

## 🏛️ Académico

**Proyecto Integrado** · Prof. **Christian Pérez** · Alumno: Gat · 2026 · [Repo](https://github.com/Gatoco/mnemos-mcp)
