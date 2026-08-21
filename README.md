# NotaRAG — Servidor MCP de RAG para opencode

> **Proyecto Integrado** — Nombre del ramo: **_[completar nombre del ramo]_** · Profesor: **Christian Pérez**
> **Alumno**: [tu nombre]
> **Fecha**: 2026-08-21

`#RAG` `#MCP` `#opencode` `#Obsidian` `#Qdrant` `#Ollama` `#DeepSeek` `#ProyectoIntegrado` `#InteligenciaArtificial` `#Python`

---

## 1. ¿Qué es este proyecto?

**NotaRAG** es un servidor que implementa un sistema de **RAG** (Retrieval-Augmented
Generation, generación aumentada por recuperación) y lo expone como un
**servidor MCP** para que el asistente de código **opencode** lo use como una
herramienta más.

En términos simples: NotaRAG toma las notas que tengo en mi vault de Obsidian
(y cualquier carpeta de documentos), las indexa en una base de datos vectorial
llamada Qdrant, y cuando el asistente necesita responder una pregunta sobre mi
propio conocimiento, primero **busca** en mis notas los fragmentos más
relevantes y **luego** genera la respuesta basándose en ellos, citando la
fuente exacta de cada afirmación.

La diferencia con un asistente normal es clave: en vez de responder con
conocimiento general (a veces desactualizado o inventado), el asistente
responde **usando mis propios apuntes**, con la ruta del archivo que respalda
cada parte de la respuesta. Esto elimina el problema de "alucinación" y
convierte el vault de notas en una base de conocimiento aprovechable.

## 2. ¿Qué es un MCP? (contexto)

MCP (Model Context Protocol) es un estándar abierto que permite conectar
asistentes de inteligencia artificial con herramientas y datos externos. Funciona
como un "USB-C para la IA": en lugar de que cada asistente invente su propia
forma de conectarse con bases de datos, archivos o servicios, MCP define un
protocolo común sobre el cual cualquier asistente compatible puede usar
cualquier herramienta compatible.

En este proyecto, opencode (el asistente) actúa como "cliente" y NotaRAG como
"servidor" de herramientas. El servidor se comunica con el asistente por
stdin/stdout (protocolo stdio), y expone herramientas con nombre, parámetros
y resultados tipados. Esto hace que el asistente pueda, de forma autónoma
y en medio de una conversación, invocar acciones como "indexa mi vault" o
"busca en mis notas X", igual que un humano usaría una herramienta.

La razón por la que esto es valioso para el proyecto integrado es que
demuestra no solo el uso de IA generativa, sino **la construcción de un
sistema de IA completo y productivo** — con persistencia, recuperación,
orquestación y una interfaz de uso real — en lugar de una demo de
"copiar-pegar".

## 3. Funcionalidades y justificación

NotaRAG expone **7 herramientas** (cada una está justificada por un problema
real que resuelve):

### `index` — Construir la base de conocimiento
Escanea una carpeta (por defecto el vault de Obsidian), divide los documentos
Markdown en fragmentos con contexto jerárquico (heading_path, tipo
"Proyecto > Hardware > Setup"), genera los embeddings y los guarda en Qdrant.
Es incremental: no vuelve a indexar lo que no cambió (compara fecha y hash),
y limpia los documentos que ya no existen. Sin esta herramienta el resto del
sistema no tiene de dónde recuperar.

### `query` — La funcionalidad estrella (RAG completo)
Recibe una pregunta en lenguaje natural, busca los fragmentos más similares
en Qdrant, los pasa como contexto a un LLM (DeepSeek V4 Flash, en la nube
de Ollama) y devuelve una respuesta **fundamentada**: cada afirmación
acompañada de la ruta de la nota y la puntuación de relevancia. Si no hay
fragmentos suficientemente relevantes, el sistema lo dice y **no** usa
la respuesta: preferimos decir "no encontré" antes que inventar.

### `search` — Recuperación pura sin LLM
Igual que query pero sin generación de texto: devuelve la lista cruda de
fragmentos con su score. Sirve para auditorías, debugging y para
herramientas que solo necesitan recuperación (no generar texto).

### `delete` — Control y privacidad
Borra documentos del índice por ruta exacta o por fuente completa. Permite
quitar información del índice sin tocar los archivos originales, necesario
para corregir errores o respetar privacidad.

### `list` — Transparencia
Lista qué documentos hay indexados, cuántos fragmentos tiene cada uno y la
fecha de último indexado. Sin esto, el sistema es una caja negra: no se sabe
qué conoce.

### `stats` — Salud del sistema
Estado de Qdrant (cantidad de vectores, fuentes), estado de Ollama
(embeddings), si la clave del LLM está configurada y la salud de la base.
Es el "semáforo" del sistema: permite saber de un vistazo si todo
funciona.

### `config` — Configuración dinámica
Lee y ajusta en caliente parámetros como el umbral de relevancia
(score_threshold), el top_k (cuántos fragmentos se recuperan), el tamaño
de los fragmentos y el overlap. Permite calibrar la calidad de las
respuestas sin tocar código ni reiniciar el servidor.

**Además, una página de administración local (http://127.0.0.1:8310)** que
permite ver el estado del RAG, revisar los documentos indexados con
búsqueda y paginación, disparar un indexado asíncrono con barra de progreso,
borrar documentos y probar búsquedas semánticas desde un panel visual. Esta
página responde el mismo núcleo lógico que las herramientas MCP (no hay
lógica duplicada), y enlaza al dashboard nativo de Qdrant para explorar la
base de datos vectorial en crudo.

## 4. ¿Por qué sirve? (el problema que resuelve)

1. **El conocimiento personal está disperso**: una vault de Obsidian con
   cientos de notas no se puede buscar de forma semántica con herramientas
   clásicas (grep). NotaRAG permite preguntarle al asistente cosas como
   "¿cómo configuré el servidor?", incluso si las notas no comparten palabras
   exactas con la pregunta — eso es justo lo que hacen los embeddings
   semánticos.
2. **Las respuestas están ancladas a la realidad**: al recuperar fragmentos
   reales y citar la fuente, se reduce la "alucinación" típica de los LLM
   (inventar respuestas). El sistema se negoca a responder cuando no
   encuentra evidencia.
3. **Privacidad y control de datos**: los embeddings (la conversión de texto
   a vectores) se generan en local con Ollama — nunca salen de la máquina.
   Solo el texto recuperado (y solo cuando el usuario pregunta) se envía al
   LLM de nube.
4. **Escalabilidad**: indexado incremental con md5/mtime, batches y retries
   con backoff hacen que indexar 19 GB de notas sea un proceso reanudable y
   acotado, no un monstruo de una sola vez.
5. **Es un sistema real, no una demo**: tiene tests (70, 100% offline), un
   script de humo, configuración por entorno y documento, y un contrato de
   interfaz con el asistente (MCP) que es el estándar de la industria 2026.

## 5. ¿Cómo funciona? (flujo de trabajo)

El ciclo de vida tiene dos momentos:

**1. Fase de indexado (construcción del conocimiento):**
NotaRAG camina la carpeta raíz, filtra los archivos válidos (.md, .txt) y
descarta los que superan un límite de tamaño. Por cada archivo, calcula la
fecha de modificación: si no cambió desde el último indexado, lo saltea
directamente (optimización incremental). Si cambió, divide el documento en
fragmentos de ~800 tokens, respetando la jerarquía de encabezados Markdown
y con un solapamiento de ~100 tokens para no cortar contexto. Cada fragmento
se convierte en un vector de 1024 dimensiones con el modelo multilingüe
bge-m3 vía Ollama local, se normaliza y se sube a Qdrant en lotes de 128
con reintentos. Al finalizar, los fragmentos de documentos que ya no existen
en disco se eliminan del índice (el "stale cleanup").

**2. Fase de consulta (recuperación + generación):**
La pregunta del usuario se convierte en vector con el mismo modelo. Se busca
en Qdrant los k fragmentos más similares (por similitud coseno), filtrados
opcionalmente por fuente (por ejemplo, solo notas de cierto proyecto) y con
un umbral de relevancia configurable. Si los mejores resultados están por
debajo del umbral, el sistema responde "No se encontraron documentos
relevantes" y no gasta una llamada al LLM. Si hay resultados válidos, se
construye un prompt que le dice al LLM (DeepSeek V4 Flash): "Responde
SOLO usando este contexto y cita las rutas", con cada fragmento precedido
de su ruta y jerarquía. El LLM responde, y el sistema empaqueta la
respuesta con la lista de fuentes y el score de cada una.

Tanto el indexado como la consulta son accesibles por dos vías idénticas:
las herramientas MCP (para que el asistente las use autónomamente) y la
página de administración (para que una persona lo controle visualmente).

## 6. Stack tecnológico y justificación

| Tecnología | Rol | ¿Por qué? |
|---|---|---|
| **Python 3.14** | Lenguaje del servidor | Ecosistema maduro para IA y librerías MCP oficiales |
| **MCP SDK (`mcp`)** | Protocolo de comunicación | Estándar de la industria; opencode lo soporta nativamente |
| **Qdrant (Docker)** | Base de datos vectorial | Especialista en búsqueda de similitud; rápido, con filtros por payload; corre local |
| **Ollama local** | Embeddings bge-m3 | Multilingüe (clave para notas en español); 100% local, sin enviar datos |
| **DeepSeek V4 Flash** | Generación de respuestas | LLM en la nube (Ollama Cloud); usa el contexto recuperado |
| **Qdrant en memoria** | Testing | Permite suite de tests 100% offline y determinista |

Decisión importante: los **embeddings son locales** porque Ollama Cloud (el
servicio en la nube del mismo proveedor) **no expone un endpoint de
embeddings** — se verificó experimentalmente. Y aunque lo hiciera, tener el
indexado local es mejor para la privacidad. El LLM de generación sí es en la
nube porque es donde está la calidad del modelo.

## 7. Estado actual del proyecto

- **Planificación**: completa, con metodología SDD (specs, diseño y tareas).
- **Implementación**: 11/11 tareas — el servidor MCP, la página de
  administración, el indexado incremental y los 7 herramientas están
  implementadas y con **70 tests pasando, todos offline** (sin necesidad de
  red ni de servicios reales).
- **Validación real**: probado contra Qdrant real en Docker y Ollama real
  (modelo bge-m3) con un subconjunto de notas del vault — indexado y
  búsqueda funcionando; queda calibrar los scores en frío con el volumen
  completo y registrar el servidor en el asistente.
- **Pendiente**: indexado completo del vault (19 GB, proceso largo en CPU),
  ajuste fino del umbral de relevancia, y memoria/entrega final del ramo.

## 8. Uso en una frase

"Le preguntás al asistente cualquier cosa sobre tu vida de proyectos,
y te responde citando exactamente la nota de dónde lo sacó — o te dice que
no lo tiene."

---

*Proyecto académico — ramo con Prof. Christian Pérez. Repositorio:
`~/proyectos_github/mcp-rag-opencode`.*
