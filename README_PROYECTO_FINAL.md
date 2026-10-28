# Proyecto final — Agente de devoluciones EcoMarket

Aplicación web empaquetable que extiende el sistema RAG de `taller_2.ipynb` con un
agente capaz de consultar pedidos, explicar políticas, verificar la elegibilidad de una
devolución, registrar la solicitud y generar una etiqueta de envío. La solución utiliza
LangGraph, un ejecutor ReAct, replanificación, aprobación humana, Milvus, LangSmith y
Streamlit.

Este documento responde explícitamente las cuatro fases solicitadas en
`Proyecto Final.pdf`: diseño, implementación, análisis crítico y despliegue.

## Caso de estudio y alcance

EcoMarket necesita transformar su asistente de atención al cliente en una herramienta
proactiva. El flujo automatizado es la devolución de productos, desde la consulta inicial
hasta la generación de la Guía Verde.

El sistema realiza una **preevaluación**. Registrar una devolución y generar una guía no
equivale a aprobar el reembolso: la decisión definitiva sigue sujeta a las evidencias y a
la inspección del producto. La política base es `POL-DEV-2026-V4` y la adenda
`POL-DEV-2026-V4-AD1` autoriza expresamente al agente a iniciar y registrar el proceso.

## Fase 1 — Diseño de la arquitectura del agente

### Extensión de la arquitectura RAG

El RAG del Taller 2 se conserva como la herramienta `buscar_politica_devolucion`. De
esta manera, el agente decide cuándo necesita recuperar políticas, catálogo o preguntas
frecuentes y puede combinar el resultado con información transaccional de los pedidos.

También se aplican rutas deterministas para intenciones conocidas y sensibles. Por
ejemplo, una consulta general sobre cómo devolver un producto solo explica el proceso, y
una solicitud para listar todos los pedidos usa `list_all=true`. Estas rutas evitan que el
modelo solicite un código o una aprobación cuando no corresponde. Las consultas abiertas
continúan siendo resueltas por el ciclo Plan-and-Execute.

### Marco de agentes seleccionado

Se eligieron **LangChain y LangGraph** en lugar de LlamaIndex:

- LangChain ofrece contratos de herramientas, mensajes y modelos con salida estructurada.
- LangGraph permite representar explícitamente planificación, ejecución, replanificación,
  interrupciones humanas y terminación.
- Los checkpoints permiten suspender y reanudar una devolución después de solicitar una
  confirmación.
- El estado tipado hace visibles el plan, los resultados, errores, documentos recuperados,
  aprobación y artefactos generados.
- Su integración con LangSmith permite observar el recorrido completo del grafo.

LlamaIndex es especialmente fuerte en indexación y consulta documental, pero el problema
principal de este proyecto es orquestar decisiones y acciones con estado. Por ello,
LangGraph resulta más apropiado; Milvus y `BAAI/bge-m3` continúan resolviendo la capa de
recuperación.

### Patrón de agentes

La solución adapta el patrón **Plan-and-Execute**:

```text
START → prepare_input → planner → executor → replanner
                                      ↑          │
                                      └──────────┤ continue
                                                 ├─ ask_user → interrupt
                                                 ├─ request_approval → interrupt
                                                 ├─ respond → END
                                                 └─ escalate → END
```

- **Planner:** crea el plan mínimo como una estructura Pydantic. Conoce la descripción de
  las cinco herramientas, pero no puede ejecutarlas.
- **Executor ReAct:** ejecuta un solo paso y recibe como máximo una herramienta autorizada
  para ese paso.
- **Replanner:** revisa resultados y errores, elimina pasos terminados y decide si debe
  continuar, preguntar, solicitar aprobación, responder o escalar.
- **LangGraph:** administra estado, checkpoints, límites e interrupciones.
- **Rutas deterministas:** atienden intenciones inequívocas sin delegar decisiones críticas
  al modelo.
- **LangSmith:** registra trazas de nodos, herramientas, latencia, errores y
  replanificaciones.

### Flujo de una devolución

```text
Solicitud del usuario
        │
        ├─ falta código ───────────────→ solicitar código y suspender
        │
        └─ código disponible
                ↓
         consultar pedido
                ↓
         recuperar política RAG
                ↓
         verificar elegibilidad
                │
                ├─ no elegible ────────→ explicar el motivo
                │
                └─ preelegible
                        ↓
                 pedir confirmación
                        │
                        ├─ cancelar ────→ finalizar sin cambios
                        │
                        └─ confirmar
                                ↓
                       registrar devolución
                                ↓
                       generar Guía Verde
                                ↓
                       responder y permitir descarga
```

### Herramientas y contratos

El ejecutor dispone de cinco herramientas; el requisito mínimo del enunciado es de dos,
sin contar RAG. El planner recibe la descripción de las cinco, pero **cero herramientas
ejecutables**.

Todas responden con uno de estos sobres JSON:

```json
{"ok": true, "data": {}}
```

```json
{"ok": false, "error": {"code": "...", "message": "..."}}
```

| Herramienta | Entrada | Salida principal | Posibles fallos |
|---|---|---|---|
| `consultar_pedido` | `order_code: str`, `list_all: bool` | Pedido completo o resumen de todos los pedidos | código ausente, pedido inexistente o error del repositorio |
| `buscar_politica_devolucion` | `query: str`, `product_type: str` | Hasta cuatro fragmentos con fuente, tipo documental, página/fila y relevancia | error de recuperación |
| `verificar_elegibilidad` | `order_code: str`, `list_all: bool` | Decisión `preeligible`, `not_eligible` o `needs_information`, motivo, plazo y condiciones | pedido inexistente o error del repositorio |
| `registrar_devolucion` | `order_code`, `reason`, `idempotency_key`, `confirmed` | Identificador de devolución, estado y fecha | falta de aprobación, no elegible, pedido inexistente o error de escritura |
| `generar_etiqueta` | `return_id`, `idempotency_key` | Identificador y ruta de la Guía Verde | devolución inexistente o error al generar el archivo |

`registrar_devolucion` y `generar_etiqueta` usan claves de idempotencia: repetir la
misma acción no crea devoluciones ni etiquetas duplicadas.

## Fase 2 — Implementación y conexión de componentes

### Extensión del código base

El prototipo del Taller 2 fue convertido de cuaderno a paquete Python instalable. Se
conservaron la base documental, los pedidos simulados, Milvus y los embeddings, y se
separaron presentación, orquestación, reglas de negocio, persistencia y recuperación.

```text
app/streamlit_app.py                  Interfaz web y reanudación de interrupciones
src/ecomarket_agent/config.py         Configuración central y límites
src/ecomarket_agent/models.py         Contratos Pydantic
src/ecomarket_agent/agents.py         Planner, ejecutor ReAct, rutas y replanner
src/ecomarket_agent/application.py    Composición e inyección de dependencias
src/ecomarket_agent/graph/state.py    Estado compartido del grafo
src/ecomarket_agent/graph/workflow.py Nodos, transiciones e interrupciones
src/ecomarket_agent/tools/            Herramientas de negocio
src/ecomarket_agent/retrieval/        Ingestión y recuperación RAG
src/ecomarket_agent/repositories.py   Persistencia transaccional simulada
src/ecomarket_agent/eligibility.py    Reglas deterministas de elegibilidad
scripts/index_knowledge.py            Indexación explícita de Milvus
tests/                                Pruebas unitarias y del flujo
dataset/                              Pedidos y base de conocimiento demostrativa
```

### Estado y conexión de componentes

El estado de LangGraph conserva la entrada, objetivo, código detectado, plan pendiente,
pasos ejecutados, pedido, documentos recuperados, elegibilidad, devolución, etiqueta,
errores, aprobación y contadores. El checkpointer SQLite permite reanudar el mismo hilo
después de un `interrupt()`.

La composición se realiza en `application.py`:

```text
Settings
   ├─ modelo Gemini
   ├─ repositorio de pedidos
   ├─ repositorio de devoluciones
   ├─ repositorio de etiquetas
   ├─ servicio de elegibilidad
   └─ servicio RAG
          ↓
   registro de herramientas
          ↓
   PlanExecuteAgents
          ↓
   StateGraph + checkpointer
          ↓
   Streamlit
```

### Recuperación de conocimiento

El modo predeterminado es `hybrid`: intenta una búsqueda vectorial en Milvus con
`BAAI/bge-m3` y, si la colección no está disponible, aplica recuperación léxica local
conservando las fuentes. La base incluye PDF, CSV y la adenda Markdown.

Para reconstruir la colección:

```bash
python scripts/index_knowledge.py
```

Para ejecutar sin cargar embeddings:

```env
ECOMARKET_RETRIEVAL_MODE=lexical
```

### Manejo de respuestas y errores

- Las respuestas de las herramientas son estructuradas; el agente no infiere éxito a
  partir de texto libre.
- Los pedidos se buscan por coincidencia exacta del código normalizado.
- Si falta información, el grafo se suspende y solicita solamente el dato requerido.
- Una acción con efecto se suspende hasta recibir aprobación explícita.
- Los productos no elegibles terminan el flujo antes del registro.
- Los errores recuperables permiten replanificar; los límites evitan ciclos infinitos.
- Al superar los límites o encontrar una situación insegura, el caso se escala.
- La interfaz presenta mensajes claros y conserva separada la preevaluación de la
  aprobación final del reembolso.

### Evaluación del comportamiento

| Prompt de prueba | Comportamiento esperado |
|---|---|
| `Hola, buenos días.` | Responder directamente, sin herramientas |
| `¿Cómo puedo realizar una devolución?` | Explicar el proceso sin pedir código ni aprobación |
| `¿Qué pedidos he realizado?` | Listar todos los pedidos con `consultar_pedido(list_all=true)` |
| `¿Dónde está PED-2026-0010?` | Consultar un pedido exacto |
| `De mis pedidos, ¿cuáles son elegibles para devolución?` | Comparar todos con `verificar_elegibilidad(list_all=true)` |
| `¿De qué material es PED-2026-0003?` | Combinar pedido y recuperación RAG |
| `Quiero devolver PED-2026-0003.` | Verificar, pedir aprobación, registrar y generar la guía |
| `Quiero devolver PED-2026-0009.` | Rechazar según elegibilidad y explicar el motivo |
| `Quiero devolver mi pedido.` | Solicitar el código mediante una interrupción |
| Código inexistente | Informar que el pedido no existe, sin inventar datos |
| Usuario cancela la confirmación | Finalizar sin registrar la devolución |

Las pruebas automatizadas cubren contratos, detección de intenciones, consulta exacta y
global, reglas de elegibilidad, idempotencia, carga documental y suspensión/reanudación
del grafo.

## Fase 3 — Análisis crítico y propuestas de mejora

### Riesgos éticos y de seguridad

| Riesgo | Escenario | Mitigación implementada o propuesta |
|---|---|---|
| Acción no autorizada | El modelo registra una devolución que el cliente solo estaba consultando | `interrupt()`, confirmación explícita y rutas informativas deterministas |
| Alucinación de políticas | El modelo inventa plazos o promete un reembolso | RAG con fuente, reglas deterministas y distinción entre preevaluación e inspección final |
| Duplicación | Un reintento crea dos devoluciones o guías | Claves de idempotencia y repositorios que reutilizan el resultado existente |
| Prompt injection documental | Un texto recuperado intenta ordenar acciones al agente | Los documentos se tratan como datos; el executor solo recibe la herramienta autorizada |
| Exposición de datos | Códigos de pedido o datos personales aparecen en trazas | No registrar secretos, anonimizar identificadores y aplicar retención y control de acceso en producción |
| Automatización excesiva | El agente decide el reembolso sin revisión | La aprobación económica permanece fuera del agente y requiere inspección humana |
| Política desactualizada | Una regla antigua contradice una adenda vigente | Metadatos de fuente, precedencia explícita de la adenda y reindexación controlada |
| Ciclos o consumo descontrolado | Planner y replanner repiten acciones | Máximos de ejecuciones y replanificaciones, además de escalamiento seguro |

Los datos del proyecto son simulados. En producción sería necesario autenticar al cliente,
autorizar el acceso a cada pedido, cifrar datos en tránsito y reposo, aplicar mínimos
privilegios, registrar auditoría inmutable y definir políticas de eliminación.

### Monitoreo y observabilidad

LangSmith recibe el nombre del proyecto mediante `LANGSMITH_PROJECT` y activa trazas
cuando existe `LANGSMITH_API_KEY`. Las trazas permiten revisar:

- entrada y salida de cada nodo;
- plan inicial y replanificaciones;
- herramienta elegida, duración y resultado;
- errores, escalaciones e interrupciones;
- cantidad de ejecuciones y latencia total.

Para producción se proponen alertas por tasa de errores, latencia elevada, incremento de
replanificaciones, intentos sin aprobación, códigos inexistentes, duplicados, caídas de
Milvus y diferencias entre preelegibilidad y resultado de la inspección. Las trazas deben
anonimizarse y tener acceso y retención limitados.

### Propuestas de mejora

1. Crear automáticamente una orden de reemplazo después de la inspección.
2. Actualizar correo, dirección o preferencias del cliente en un CRM con aprobación.
3. Integrar transportadoras reales para rastreo, tarifas y etiquetas PDF.
4. Sustituir JSON y SQLite por servicios autenticados y PostgreSQL.
5. Añadir autenticación, autorización por pedido y gestión de roles.
6. Incorporar carga segura de fotografías y análisis asistido de evidencias.
7. Versionar políticas, evaluar calidad de recuperación y mostrar citas en la interfaz.
8. Construir un conjunto de evaluación con precisión de herramientas, éxito de tareas,
   groundedness, latencia y costo.
9. Exponer el grafo mediante FastAPI para permitir un frontend React/Next.js o una
   aplicación móvil.
10. Incorporar una cola humana para casos ambiguos, reclamos y excepciones.

## Fase 4 — Despliegue de la aplicación

### Elección de Streamlit

El enunciado permite Streamlit o Gradio. Se eligió **Streamlit** porque la demostración
requiere más que una entrada y una salida del modelo: chat, estado de sesión, botones de
aprobación, descarga de la guía y un panel lateral de monitoreo.

Gradio sería una alternativa adecuada para una demostración centrada principalmente en
el modelo y componentes multimedia. Streamlit encaja mejor con esta pequeña aplicación
operativa y permite integrarla directamente con Python y LangGraph sin introducir una API
y un frontend separados. La interfaz es reemplazable: el dominio y el grafo no dependen
de Streamlit.

### Requisitos

- Python 3.12
- Una clave de Google para el modelo configurado
- Una clave de LangSmith si se desea observabilidad remota

### Instalación

Con `venv` o un entorno Python existente:

```bash
cp .env.example .env
python -m pip install '.[dev]'
```

Con Conda:

```bash
conda env create -f environment.yml
conda activate icesi_ia_generativa
cp .env.example .env
python -m pip install . --no-deps
```

Variables mínimas:

```env
GOOGLE_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ecomarket-final-dev
```

Nunca se deben versionar secretos. La aplicación usa SQLite para checkpoints,
`dataset/returns.json` para la simulación transaccional y `generated_labels/` para las
guías. Estos artefactos están excluidos de Git.

### Ejecución

```bash
python scripts/index_knowledge.py
streamlit run app/streamlit_app.py
```

La aplicación ofrece un campo de chat, visualización de respuestas, aprobación o
cancelación de la devolución, descarga de la Guía Verde y creación de una nueva
conversación.

### Pruebas y calidad

```bash
pytest
ruff check src app scripts tests
```

Las pruebas automáticas no consumen el API del modelo. Las validaciones manuales de
extremo a extremo sí requieren las variables del archivo `.env`.
