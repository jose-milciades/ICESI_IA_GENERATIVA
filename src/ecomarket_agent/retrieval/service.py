"""Ingestión documental y recuperación con Milvus y respaldo léxico local."""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from ecomarket_agent.config import Settings
from ecomarket_agent.models import RetrievedDocument

logger = logging.getLogger(__name__)


def _split_page(document: Document, size: int = 1000, overlap: int = 200) -> list[Document]:
    """Segmenta por caracteres sin cargar dependencias de embeddings al importar."""
    text = document.page_content
    chunks: list[Document] = []
    start = 0
    while start < len(text):
        tentative_end = min(start + size, len(text))
        end = tentative_end
        if tentative_end < len(text):
            candidates = [
                text.rfind(separator, start, tentative_end) for separator in ("\n\n", "\n", " ")
            ]
            boundary = max(candidates)
            if boundary > start + size // 2:
                end = boundary
        content = text[start:end].strip()
        if content:
            chunks.append(Document(page_content=content, metadata=dict(document.metadata)))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_knowledge_documents(knowledge_dir: Path) -> list[Document]:
    """Carga la política por fragmentos y conserva cada fila CSV como unidad."""
    documents: list[Document] = []
    policy_path = next(knowledge_dir.glob("*.pdf"), None)
    if policy_path:
        pages: list[Document] = []
        for page_number, page in enumerate(PdfReader(str(policy_path)).pages, start=1):
            content = (page.extract_text() or "").strip()
            if content:
                pages.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(policy_path),
                            "document_type": "return_policy",
                            "page": page_number,
                            "row": 0,
                        },
                    )
                )
        for page in pages:
            documents.extend(_split_page(page))

    for csv_path in sorted(knowledge_dir.glob("*.csv")):
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            for row_number, row in enumerate(reader, start=2):
                content = "\n".join(f"{header}: {row.get(header, '')}" for header in headers)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(csv_path),
                            "document_type": csv_path.stem,
                            "page": 0,
                            "row": row_number,
                        },
                    )
                )
    for markdown_path in sorted(knowledge_dir.glob("*.md")):
        content = markdown_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        document = Document(
            page_content=content,
            metadata={
                "source": str(markdown_path),
                "document_type": "return_policy_addendum",
                "page": 0,
                "row": 0,
            },
        )
        documents.extend(_split_page(document))
    if not documents:
        raise FileNotFoundError(f"No hay documentos en {knowledge_dir}.")
    return documents


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return {token for token in re.findall(r"[a-z0-9]{3,}", ascii_text)}


class KnowledgeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.knowledge_dir = settings.data_dir / "knowledge_base"
        self._documents: list[Document] | None = None
        self._vector_store = None
        self._vector_attempted = False

    @property
    def documents(self) -> list[Document]:
        if self._documents is None:
            self._documents = load_knowledge_documents(self.knowledge_dir)
        return self._documents

    def _get_vector_store(self):
        if self._vector_attempted:
            return self._vector_store
        self._vector_attempted = True
        if self.settings.retrieval_mode not in {"hybrid", "vector"}:
            return None
        collection_path = self.settings.milvus_uri / "collections" / self.settings.collection_name
        collection_missing = not self.settings.milvus_uri.exists() or (
            self.settings.milvus_uri.is_dir() and not collection_path.exists()
        )
        if collection_missing:
            logger.warning(
                "La colección Milvus no está indexada en %s; usando respaldo léxico.",
                self.settings.milvus_uri,
            )
            return None
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_milvus import Milvus

            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=self.settings.collection_name,
                connection_args={"uri": str(self.settings.milvus_uri.resolve())},
            )
        except Exception as exc:  # pragma: no cover - depende del servicio local
            logger.warning("Milvus no disponible; se usará recuperación léxica: %s", exc)
        return self._vector_store

    def search(self, query: str, k: int = 4) -> list[RetrievedDocument]:
        vector_store = self._get_vector_store()
        if vector_store is not None:
            try:
                matches = vector_store.similarity_search_with_score(query, k=k)
                return [self._to_result(document, float(score)) for document, score in matches]
            except Exception as exc:  # pragma: no cover - depende de Milvus
                logger.warning("Falló la consulta vectorial; usando respaldo: %s", exc)
        return self._lexical_search(query, k)

    def _lexical_search(self, query: str, k: int) -> list[RetrievedDocument]:
        query_tokens = _tokens(query)
        ranked: list[tuple[float, Document]] = []
        for document in self.documents:
            document_tokens = _tokens(document.page_content)
            overlap = len(query_tokens & document_tokens)
            score = overlap / max(len(query_tokens), 1)
            if overlap:
                ranked.append((score, document))
        if not ranked:
            ranked = [(0.0, document) for document in self.documents[:k]]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [self._to_result(document, score) for score, document in ranked[:k]]

    @staticmethod
    def _to_result(document: Document, score: float) -> RetrievedDocument:
        metadata = document.metadata
        return RetrievedDocument(
            content=document.page_content,
            source=str(metadata.get("source", "desconocida")),
            document_type=str(metadata.get("document_type", "unknown")),
            page=int(metadata.get("page", 0)),
            row=int(metadata.get("row", 0)),
            score=score,
        )


def index_knowledge(settings: Settings) -> int:
    """Reconstruye explícitamente la colección; nunca se ejecuta al importar la app."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus

    documents = load_knowledge_documents(settings.data_dir / "knowledge_base")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    settings.milvus_uri.parent.mkdir(parents=True, exist_ok=True)
    Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.collection_name,
        connection_args={"uri": str(settings.milvus_uri.resolve())},
        drop_old=True,
    )
    return len(documents)
