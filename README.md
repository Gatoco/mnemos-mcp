# mcp-rag-opencode

Servidor MCP de RAG para **opencode**: indexa tu vault de Obsidian (o cualquier carpeta de Markdown) en **Qdrant** usando embeddings locales **bge-m3**, recupera por similitud semántica y responde preguntas con **deepseek-v4-flash** usando tus propias notas como fuente. Expone las herramientas `index`, `query`, `search`, `delete`, `list`, `stats` y `config` sobre stdio, más una página de administración local en `http://127.0.0.1:8310`.

## Requisitos

- **Python ≥ 3.11**
- **Docker** con Qdrant (`docker compose up -d qdrant`)
- **Ollama local** con el modelo de embeddings `bge-m3` (`ollama pull bge-m3`)
- **`OLLAMA_API_KEY`** — clave de [Ollama Cloud](https://ollama.com) para el LLM en la nube (`deepseek-v4-flash`)

## Instalación

1. Instala el paquete (crea un venv con Python ≥3.11 primero):

   ```bash
   make install
   ```

2. Copia el template de entorno y rellénalo:

   ```bash
   cp .env.example .env
   # edita .env: OLLAMA_API_KEY=tu-clave  y  VAULT_ROOT=/ruta/a/tu/vault
   ```

   Variables principales:

   | Variable | Descripción | Defecto |
   |----------|-------------|---------|
   | `OLLAMA_API_KEY` | Clave de Ollama Cloud (LLM) | *(vacía)* |
   | `OLLAMA_URL` | URL de Ollama local (embeddings) | `http://127.0.0.1:11434` |
   | `QDRANT_URL` | URL de Qdrant | `http://127.0.0.1:6333` |
   | `ADMIN_PORT` | Puerto de la página de admin | `8310` |
   | `VAULT_ROOT` | Carpeta raíz a indexar | *(vacía — se pasa por tool)* |
   | `COLLECTION` | Nombre de la colección Qdrant | `supervisor` |
   | `DEFAULT_SOURCE` | Fuente por defecto | `vault` |

4. Levanta Qdrant y descarga el modelo de embeddings:

   ```bash
   docker compose up -d qdrant
   ollama pull bge-m3
   ```

5. Registra el servidor en `opencode.json` (sección `mcp`):

   ```json
   {
     "mcp": {
       "mcp-rag-opencode": {
         "type": "local",
         "command": ["python3", "/ruta/a/mcp-rag-opencode/mcp_rag/server.py"],
         "enabled": true
       }
     }
   }
   ```

   Si instalaste con `make install`, también puedes usar la entrada de consola:

   ```json
   {
     "mcp": {
       "mcp-rag-opencode": {
         "type": "local",
         "command": ["mcp-rag-opencode"],
         "enabled": true
       }
     }
   }
   ```

## Uso

Desde opencode, el agente dispone de estas tools:

- **`index`** — escanea e indexa el vault (embeddings y subida a Qdrant). El primer índice en frío es lento en CPU.
- **`query`** — pregunta en lenguaje natural; responde con `deepseek-v4-flash` usando tus notas como contexto.
- **`search`** — recuperación semántica cruda (sin LLM): lista de hits con `score`.
- **`delete`** — borra por `path` (exacto) o por `source`.
- **`list`** — lista documentos indexados (paginado).
- **`stats`** — estado de Qdrant/Ollama, configuración del LLM y recuentos.
- **`config`** — lee/ajusta la configuración en `config.json`.

Ejemplos de prompts tal como se ven desde opencode/LLM:

```
indexa mi vault ahora
  → index tool: {files_scanned, files_indexed, files_skipped, chunks_upserted, duration_s, errors[]}

¿cuándo configuré el servidor? responde usando mis notas
  → query tool: {answer, sources[], model}

busca en mis notas "hardware del poco x3" sin resumir
  → search tool: [{path, heading_path, score, snippet, mtime}]

muestra el estado del sistema y qué hay indexado
  → stats tool: {collection, vectors_count, sources[], qdrant_health, ollama_health, llm_configured}
```

## Admin page

```bash
make admin
```

Abre `http://127.0.0.1:8310`. Pestañas:

- **Dashboard** — stats y estado de Qdrant/Ollama/LLM.
- **Documents** — lista paginada de documentos indexados, filtrable por `source`.
- **Index** — dispara un índice (asíncrono) y muestra su progreso.
- **Search** — playground de búsqueda semántica (raw retrieval, sin LLM).

El panel de Qdrant en vivo está en `http://localhost:6333/dashboard`.

## Arquitectura

```
                 ┌────────────────────────────────────────────────┐
                 │                 RagService (core)                │
  MCP tools ─────┤  index / query / search / delete / list / stats │──── admin HTTP :8310
  (stdio)  ──────┤  (compartido con la página de admin)            │──── page vanilla JS
                 └───────────────┬────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
        chunker → embedder (bge-m3)       store → query vectorial
        (headings, overlap)               (Qdrant, 1024 dims)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                            Qdrant (Docker)
                                 ▲
                    index nuevo/chunk cambiado
                                 │
                 LLM en la nube (deepseek-v4-flash, ollama.com/v1)
                 responde citando las rutas de tus notas
```

## Troubleshooting

- **Ollama no arranca / embeddings fallan**: comprueba que Ollama responde con `curl http://127.0.0.1:11434/api/tags`. Si devuelve 404, falta el modelo: `ollama pull bge-m3`. El error exacto es `model not found, run: ollama pull bge-m3`.
- **Falta `OLLAMA_API_KEY`**: la tool `query` falla con un mensaje claro que identifica la clave ausente. Revisa que `.env` la tenga y que `stats.llm_configured` sea `true`.
- **Qdrant caído**: comprueba con `docker compose ps` que el contenedor está arriba. Si no, `docker compose up -d qdrant`. `stats.qdrant_health` lo refleja.
- **Índice lento**: el primer índice en frío embebe todo en CPU. Acota el escaneo con `max_files` (parámetro de `index`) para probar con pocos archivos.
- **Pocos o malos resultados en `query`**: baja `score_threshold` (p. ej. `config set {score_threshold: 0.4}`) o sube `top_k` (máx. 8) para recuperar más contexto.

## Tests

```bash
make test
```

Suite de pytest 100% offline (Qdrant en memoria y providers simulados), sin red, ni Ollama ni Qdrant reales. Verificación rápida end-to-end con `make smoke`.

## Descargo de responsabilidad sobre modelos

Los embeddings (`bge-m3`) corren en tu Ollama local; el LLM (`deepseek-v4-flash`) es en la nube. Tus notas nunca salen de tu máquina para indexarse, pero el texto recuperado como contexto se envía al LLM al hacer una `query`.
