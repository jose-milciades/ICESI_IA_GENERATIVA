"""Interfaz conversacional y resolución de interrupciones de LangGraph."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from ecomarket_agent.application import create_application
from ecomarket_agent.config import Settings


@st.cache_resource
def get_application():
    return create_application(Settings.from_env())


def active_config() -> dict:
    return {
        "configurable": {"thread_id": st.session_state.active_thread},
        "recursion_limit": 50,
        "tags": ["ecomarket", "plan-and-execute", "react"],
        "metadata": {
            "environment": application.settings.environment,
            "graph_version": "0.1.0",
        },
    }


def refresh_state() -> None:
    snapshot = application.graph.get_state(active_config())
    st.session_state.interrupt = snapshot.interrupts[0].value if snapshot.interrupts else None
    if not snapshot.interrupts and snapshot.values.get("response"):
        response = snapshot.values["response"]
        if not st.session_state.messages or st.session_state.messages[-1] != {
            "role": "assistant",
            "content": response,
        }:
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.last_label = snapshot.values.get("label")
        st.session_state.active_thread = None


def run_graph(graph_input) -> None:
    try:
        with st.spinner("EcoMarket está procesando la solicitud..."):
            application.graph.invoke(graph_input, config=active_config())
        refresh_state()
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "No pude completar la solicitud. Revisa la configuración del "
                    f"modelo o inténtalo nuevamente. ({type(exc).__name__})"
                ),
            }
        )
        st.session_state.active_thread = None
        st.session_state.interrupt = None


st.set_page_config(page_title="EcoMarket", page_icon="♻️", layout="centered")
st.title("♻️ Asistente de EcoMarket")
st.caption("Plan-and-Execute · ReAct · RAG · LangGraph · LangSmith")

try:
    application = get_application()
except Exception as exc:
    st.error(
        "No fue posible iniciar la aplicación. Verifica `.env` e instala las "
        f"dependencias del proyecto. Detalle: {type(exc).__name__}."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_thread" not in st.session_state:
    st.session_state.active_thread = None
if "interrupt" not in st.session_state:
    st.session_state.interrupt = None
if "last_label" not in st.session_state:
    st.session_state.last_label = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pending = st.session_state.interrupt
if pending and pending.get("kind") == "input":
    st.info(pending.get("question", "Necesito información adicional para continuar."))
if pending and pending.get("kind") == "approval":
    request = pending.get("request") or {}
    st.warning(request.get("summary", "Esta acción requiere confirmación."))
    approve_column, reject_column = st.columns(2)
    if approve_column.button("Confirmar devolución", type="primary"):
        run_graph(Command(resume={"approved": True}))
        st.rerun()
    if reject_column.button("Cancelar"):
        run_graph(Command(resume={"approved": False}))
        st.rerun()

prompt_label = (
    pending.get("question", "Proporciona la información solicitada")
    if pending and pending.get("kind") == "input"
    else "¿En qué puedo ayudarte con tu pedido?"
)

if prompt := st.chat_input(
    prompt_label,
    disabled=bool(pending and pending.get("kind") == "approval"),
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if pending and pending.get("kind") == "input":
        run_graph(Command(resume={"text": prompt}))
    else:
        st.session_state.active_thread = str(uuid4())
        run_graph(
            {
                "input": prompt,
                "thread_id": st.session_state.active_thread,
                "past_steps": [],
            }
        )
    st.rerun()

label = st.session_state.last_label
if label:
    label_path = Path(label["file_path"])
    if label_path.exists():
        st.download_button(
            "Descargar Guía Verde",
            data=label_path.read_bytes(),
            file_name=label_path.name,
            mime="text/plain",
        )

with st.sidebar:
    st.subheader("Monitoreo")
    st.text(f"Ambiente: {application.settings.environment}")
    st.text(f"LangSmith: {application.settings.langsmith_project}")
    st.text(f"Recuperación: {application.settings.retrieval_mode}")
    if st.button("Nueva conversación"):
        st.session_state.messages = []
        st.session_state.active_thread = None
        st.session_state.interrupt = None
        st.session_state.last_label = None
        st.rerun()
