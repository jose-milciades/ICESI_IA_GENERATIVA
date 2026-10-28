"""Reconstruye la colección vectorial de EcoMarket de manera explícita."""

from ecomarket_agent.config import Settings
from ecomarket_agent.retrieval import index_knowledge

if __name__ == "__main__":
    total = index_knowledge(Settings.from_env())
    print(f"Colección indexada: {total} documentos.")
