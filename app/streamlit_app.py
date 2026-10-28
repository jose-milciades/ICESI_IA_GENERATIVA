"""Punto de entrada de la interfaz del proyecto final."""

import streamlit as st

from ecomarket_agent.config import Settings


settings = Settings.from_env()

st.set_page_config(page_title="EcoMarket", page_icon="♻️")
st.title("Asistente de EcoMarket")
st.caption(
    "Agente Plan-and-Execute con ReAct, replanificación y monitoreo en LangSmith."
)

if prompt := st.chat_input("¿En qué puedo ayudarte con tu pedido?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.info(
            "El esqueleto de la aplicación está listo. El grafo se conectará en la "
            "siguiente fase de implementación."
        )

with st.sidebar:
    st.subheader("Ejecución")
    st.text(f"Ambiente: {settings.environment}")
    st.text(f"Proyecto: {settings.langsmith_project}")
