# Proyecto final — Agente de devoluciones EcoMarket

Aplicación empaquetable que extiende el RAG de `taller_2.ipynb` con un agente
**Plan-and-Execute**, un ejecutor **ReAct**, replanificación, aprobación humana y
monitoreo en LangSmith.

## Arquitectura

El grafo conserva el ciclo de la guía de Plan-and-Execute:

```text
START → prepare_input → planner → executor → replanner
                                      ↑          │
                                      └──────────┤ continue
                                                 ├─ ask_user → interrupt
                                                 ├─ request_approval → interrupt
                                                 ├─ respond → END
                                                 └─ escalate → END
```

- **Planner:** crea un plan estructurado; conoce el catálogo, pero no ejecuta herramientas.
- **Executor:** agente ReAct que procesa un paso y usa solo las herramientas autorizadas.
- **Replanner:** conserva los pasos terminados y decide continuar, preguntar, aprobar,
  responder o escalar.
- **LangGraph:** mantiene el estado, los límites, los checkpoints y las interrupciones.
- **LangSmith:** registra trazas, latencia, errores, herramientas y replanificaciones.
- **Streamlit:** presenta la conversación, confirmaciones y descarga de etiquetas.

## Herramientas

El ejecutor dispondrá de cinco herramientas:

1. `consultar_pedido`
2. `buscar_politica_devolucion`
3. `verificar_elegibilidad`
4. `registrar_devolucion`
5. `generar_etiqueta`

El planner recibe la descripción de las cinco, pero recibe **cero herramientas
ejecutables**. Las acciones de registro requieren confirmación y claves de idempotencia.

## Estructura

```text
app/streamlit_app.py             Interfaz
src/ecomarket_agent/config.py    Configuración y límites
src/ecomarket_agent/models.py    Contratos Pydantic
src/ecomarket_agent/graph/       Estado y construcción del flujo
src/ecomarket_agent/tools/       Herramientas de negocio
src/ecomarket_agent/retrieval/   Ingestión y recuperación RAG
tests/                           Pruebas unitarias e integración
dataset/                         Datos demostrativos fuera del paquete
```

## Configuración

```bash
cp .env.example .env
python -m pip install -e '.[dev]'
```

Variables mínimas:

```env
GOOGLE_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ecomarket-final-dev
```

Nunca se deben versionar secretos. `LangSmith` proporciona observabilidad; el estado
reanudable pertenece al checkpointer de LangGraph.

## Ejecución

```bash
streamlit run app/streamlit_app.py
```

## Pruebas

```bash
pytest
```

La implementación se desarrollará a partir de los contratos tipados incluidos en el
paquete, manteniendo separados dominio, herramientas, recuperación, orquestación e interfaz.
