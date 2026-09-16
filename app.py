# # import logging
# # import os
# # import re
# # import tempfile
# # import uuid
# # from contextlib import asynccontextmanager
# # from functools import lru_cache
# # from typing import Any

# # import chromadb
# # from dotenv import load_dotenv
# # from fastapi import (
# #     APIRouter,
# #     FastAPI,
# #     File,
# #     Form,
# #     HTTPException,
# #     UploadFile,
# # )
# # from openai import OpenAI, OpenAIError
# # from pydantic import BaseModel, Field
# # from pypdf import PdfReader
# # from pypdf.errors import PdfReadError
# # from sentence_transformers import SentenceTransformer


# # # --------------------------------------------------------------------------
# # # Configuration
# # # --------------------------------------------------------------------------

# # load_dotenv()

# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
# # )

# logger = logging.getLogger("technical_rag")


# class Settings:
#     """Centralized application configuration."""

#     def __init__(self) -> None:
#         self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

#         self.openrouter_model = os.getenv(
#             "OPENROUTER_MODEL",
#             "nvidia/nemotron-3.5-lightning:free",
#         )

#         self.openrouter_embed_model = os.getenv(
#             "OPENROUTER_EMBED_MODEL",
#             "nvidia/nemotron-3-embed-1b:free",
#         )

        
#         self.chroma_db_dir = os.getenv(
#             "CHROMA_DB_DIR",
#             "./chroma_db",
#         )

#         self.app_host = os.getenv(
#             "APP_HOST",
#             "0.0.0.0",
#         )

#         self.app_port = int(
#             os.getenv("APP_PORT", "8000")
#         )

#     def require_llm_config(self) -> None:
#         if not self.openrouter_api_key:
#             raise RuntimeError(
#                 "Missing required environment variable: "
#                 "OPENROUTER_API_KEY. "
#                 "Set it in your environment or a .env file."
#             )


# settings = Settings()


# # --------------------------------------------------------------------------
# # PDF handling
# # --------------------------------------------------------------------------

# class PDFLoader:
#     """Extracts plain text from PDF documents."""

#     @staticmethod
#     def load_pdf(path: str) -> str:
#         try:
#             reader = PdfReader(path)
#         except PdfReadError as exc:
#             raise ValueError(
#                 f"Could not read PDF file: {exc}"
#             ) from exc

#         text = " ".join(
#             page.extract_text() or ""
#             for page in reader.pages
#         )

#         text = re.sub(r"\s+", " ", text).strip()

#         if not text:
#             raise ValueError(
#                 "No extractable text found in PDF. "
#                 "The document may be a scanned image."
#             )

#         return text


# # --------------------------------------------------------------------------
# # Text chunking
# # --------------------------------------------------------------------------

# class TextChunker:
#     """
#     Splits long documents into overlapping text chunks.

#     Overlap helps preserve context when important information appears
#     near the boundary between two chunks.
#     """

#     def __init__(
#         self,
#         chunk_size: int = 1000,
#         overlap: int = 200,
#     ) -> None:
#         if overlap >= chunk_size:
#             raise ValueError(
#                 "Chunk overlap must be smaller than chunk size."
#             )

#         self.chunk_size = chunk_size
#         self.overlap = overlap

#     def split_text(self, text: str) -> list[str]:
#         text = text.strip()

#         if not text:
#             return []

#         chunks: list[str] = []
#         start = 0

#         while start < len(text):
#             end = min(
#                 start + self.chunk_size,
#                 len(text),
#             )

#             chunk = text[start:end].strip()

#             if chunk:
#                 chunks.append(chunk)

#             if end >= len(text):
#                 break

#             start = end - self.overlap

#         return chunks


# # --------------------------------------------------------------------------
# # Embeddings
# # --------------------------------------------------------------------------


# class EmbeddingModel:
#     """Generates embeddings locally."""

#     def __init__(self, model: str) -> None:
#         self.model = SentenceTransformer(model)

#     def encode(self, text: str) -> list[float]:
#         try:
#             embedding = self.model.encode(
#                 text,
#                 normalize_embeddings=True,
#             )
#         except Exception as exc:
#             logger.exception("Local embedding failed")
#             raise RuntimeError(
#                 f"Embedding generation failed: {exc}"
#             ) from exc

#         return embedding.tolist()

# # --------------------------------------------------------------------------
# # Vector store
# # --------------------------------------------------------------------------

# class ChromaManager:
#     """
#     Stores document chunks and embeddings in a persistent Chroma
#     collection and retrieves relevant chunks for questions.
#     """

#     def __init__(self, persist_dir: str) -> None:
#         self.client = chromadb.PersistentClient(
#             path=persist_dir
#         )

#         self.collection = (
#             self.client.get_or_create_collection(
#                 name="technical_documentation"
#             )
#         )

#     def add_chunks(
#         self,
#         chunks: list[str],
#         embeddings: list[list[float]],
#         document_name: str,
#     ) -> int:
#         if not chunks:
#             return 0

#         ids = [
#             f"{uuid.uuid4()}"
#             for _ in chunks
#         ]

#         metadatas = [
#             {
#                 "document": document_name,
#                 "chunk_index": index,
#             }
#             for index in range(len(chunks))
#         ]

#         self.collection.add(
#             ids=ids,
#             documents=chunks,
#             embeddings=embeddings,
#             metadatas=metadatas,
#         )

#         return len(chunks)

#     def search(
#         self,
#         query_embedding: list[float],
#         top_k: int = 5,
#     ) -> list[dict[str, Any]]:
#         total = self.collection.count()

#         if total == 0:
#             return []

#         top_k = min(top_k, total)

#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k,
#         )

#         ids = results.get("ids", [[]])[0]
#         documents = results.get("documents", [[]])[0]
#         metadatas = results.get("metadatas", [[]])[0]
#         distances = results.get("distances", [[]])[0]

#         retrieved: list[dict[str, Any]] = []

#         for doc_id, document, metadata, distance in zip(
#             ids,
#             documents,
#             metadatas,
#             distances,
#         ):
#             retrieved.append(
#                 {
#                     "id": doc_id,
#                     "text": document,
#                     "metadata": metadata,
#                     "distance": distance,
#                 }
#             )

#         return retrieved


# # --------------------------------------------------------------------------
# # RAG prompt
# # --------------------------------------------------------------------------

# RAG_PROMPT = """
# You are a technical documentation assistant.

# Answer the user's question using ONLY the provided documentation context.

# If the answer cannot be found in the context, say:

# "I could not find the answer in the provided documentation."

# Do not invent information or rely on unsupported assumptions.

# Documentation context:
# -----------------------
# {context}
# -----------------------

# User question:
# {question}

# Provide a concise and helpful answer.
# """


# # --------------------------------------------------------------------------
# # LLM client
# # --------------------------------------------------------------------------

# class LLMClient:
#     """OpenAI-compatible chat client pointed at OpenRouter."""

#     def __init__(
#         self,
#         client: OpenAI,
#         model: str,
#     ) -> None:
#         self.client = client
#         self.model = model

#     def generate(self, prompt: str) -> str:
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": prompt,
#                     }
#                 ],
#                 temperature=0.2,
#             )
#         except OpenAIError as exc:
#             logger.exception("LLM call failed")
#             raise RuntimeError(
#                 f"Answer generation failed: {exc}"
#             ) from exc

#         content = response.choices[0].message.content

#         if not content:
#             raise RuntimeError(
#                 "LLM returned an empty response."
#             )

#         return content


# # --------------------------------------------------------------------------
# # RAG service
# # --------------------------------------------------------------------------

# class RAGService:
#     """
#     Coordinates document ingestion, retrieval and answer generation.
#     """

#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         llm_client: LLMClient,
#         chroma_manager: ChromaManager,
#         chunker: TextChunker,
#     ) -> None:
#         self.embedding_model = embedding_model
#         self.llm_client = llm_client
#         self.chroma_manager = chroma_manager
#         self.chunker = chunker

#     def ingest_document(
#         self,
#         document_text: str,
#         document_name: str,
#     ) -> int:
#         chunks = self.chunker.split_text(
#             document_text
#         )

#         if not chunks:
#             raise ValueError(
#                 "Document produced no text chunks."
#             )

#         embeddings = [
#             self.embedding_model.encode(chunk)
#             for chunk in chunks
#         ]

#         return self.chroma_manager.add_chunks(
#             chunks=chunks,
#             embeddings=embeddings,
#             document_name=document_name,
#         )

#     def ask(
#         self,
#         question: str,
#         top_k: int = 5,
#     ) -> dict[str, Any]:
#         query_embedding = self.embedding_model.encode(
#             question
#         )

#         retrieved = self.chroma_manager.search(
#             query_embedding=query_embedding,
#             top_k=top_k,
#         )

#         if not retrieved:
#             raise ValueError(
#                 "No documents have been indexed yet. "
#                 "Upload a PDF document first."
#             )

#         context_parts: list[str] = []

#         for index, item in enumerate(retrieved, start=1):
#             document_name = item["metadata"].get(
#                 "document",
#                 "unknown",
#             )

#             chunk_index = item["metadata"].get(
#                 "chunk_index",
#                 "unknown",
#             )

#             context_parts.append(
#                 f"[Source {index}]\n"
#                 f"Document: {document_name}\n"
#                 f"Chunk: {chunk_index}\n"
#                 f"{item['text']}"
#             )

#         context = "\n\n".join(context_parts)

#         prompt = RAG_PROMPT.format(
#             context=context,
#             question=question,
#         )

#         answer = self.llm_client.generate(prompt)

#         sources = [
#             {
#                 "document": item["metadata"].get(
#                     "document",
#                     "unknown",
#                 ),
#                 "chunk_index": item["metadata"].get(
#                     "chunk_index",
#                     -1,
#                 ),
#                 "distance": item["distance"],
#             }
#             for item in retrieved
#         ]

#         return {
#             "answer": answer,
#             "sources": sources,
#         }


# # --------------------------------------------------------------------------
# # API schemas
# # --------------------------------------------------------------------------

# class AskRequest(BaseModel):
#     question: str = Field(
#         ...,
#         min_length=1,
#         description="Question about the uploaded documentation",
#     )

#     top_k: int = Field(
#         5,
#         ge=1,
#         le=10,
#         description="Number of document chunks to retrieve",
#     )


# class Source(BaseModel):
#     document: str
#     chunk_index: int
#     distance: float


# class AskResponse(BaseModel):
#     answer: str
#     sources: list[Source]


# class UploadResponse(BaseModel):
#     document: str
#     chunks_indexed: int


# # --------------------------------------------------------------------------
# # Dependency wiring
# # --------------------------------------------------------------------------

# @lru_cache
# def get_rag_service() -> RAGService:
#     """
#     Creates and caches the application's heavy dependencies.
#     """

#     settings.require_llm_config()

#     openrouter_client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=settings.openrouter_api_key,
#     )

#     embedding_model = EmbeddingModel(
#         client=openrouter_client,
#         model=settings.openrouter_embed_model,
#     )

#     llm_client = LLMClient(
#         client=openrouter_client,
#         model=settings.openrouter_model,
#     )

#     chroma_manager = ChromaManager(
#         persist_dir=settings.chroma_db_dir,
#     )

#     chunker = TextChunker(
#         chunk_size=1000,
#         overlap=200,
#     )

#     return RAGService(
#         embedding_model=embedding_model,
#         llm_client=llm_client,
#         chroma_manager=chroma_manager,
#         chunker=chunker,
#     )


# # --------------------------------------------------------------------------
# # Application
# # --------------------------------------------------------------------------

# @asynccontextmanager
# async def lifespan(_: FastAPI):
#     """
#     Build the service during application startup.
#     """

#     get_rag_service()

#     yield


# router = APIRouter()


# @router.post(
#     "/documents/upload",
#     response_model=UploadResponse,
# )
# async def upload_document(
#     document: UploadFile = File(...),
# ) -> UploadResponse:
#     """
#     Upload and index a technical PDF document.
#     """

#     if document.content_type not in (
#         "application/pdf",
#         "application/x-pdf",
#     ):
#         raise HTTPException(
#             status_code=415,
#             detail="Only PDF documents are supported.",
#         )

#     service = get_rag_service()

#     with tempfile.NamedTemporaryFile(
#         suffix=".pdf"
#     ) as tmp:
#         tmp.write(await document.read())
#         tmp.flush()

#         try:
#             text = PDFLoader.load_pdf(
#                 tmp.name
#             )
#         except ValueError as exc:
#             raise HTTPException(
#                 status_code=400,
#                 detail=str(exc),
#             ) from exc

#     try:
#         chunks_indexed = service.ingest_document(
#             document_text=text,
#             document_name=document.filename or "unknown.pdf",
#         )
#     except RuntimeError as exc:
#         raise HTTPException(
#             status_code=502,
#             detail=str(exc),
#         ) from exc

#     return UploadResponse(
#         document=document.filename or "unknown.pdf",
#         chunks_indexed=chunks_indexed,
#     )


# @router.post(
#     "/ask",
#     response_model=AskResponse,
# )
# def ask_question(
#     request: AskRequest,
# ) -> AskResponse:
#     """
#     Retrieve relevant documentation and generate an answer.
#     """

#     service = get_rag_service()

#     try:
#         result = service.ask(
#             question=request.question,
#             top_k=request.top_k,
#         )
#     except ValueError as exc:
#         raise HTTPException(
#             status_code=400,
#             detail=str(exc),
#         ) from exc
#     except RuntimeError as exc:
#         raise HTTPException(
#             status_code=502,
#             detail=str(exc),
#         ) from exc

#     return AskResponse(**result)


# app = FastAPI(
#     title="Technical Documentation RAG Assistant",
#     description=(
#         "A Retrieval-Augmented Generation API that indexes "
#         "technical PDF documentation, retrieves relevant "
#         "document chunks, and generates grounded answers "
#         "using an LLM."
#     ),
#     version="1.0.0",
#     lifespan=lifespan,
# )

# app.include_router(router)


# @app.get("/")
# def root() -> dict[str, str]:
#     return {
#         "message": "Technical Documentation RAG Assistant Running"
#     }


# @app.get("/health")
# def health() -> dict[str, str]:
#     return {
#         "status": "ok"
#     }


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(
#         app,
#         host=settings.app_host,
#         port=settings.app_port,
#     )





import logging
import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("technical_rag")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Keep ONLY the secret in .env:
#
# OPENROUTER_API_KEY=your_private_key
#
# Model defaults are deliberately defined here, so they do not need to be
# placed in .env. The embedding model runs locally.
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Settings:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.llm_model = os.getenv("OPENROUTER_MODEL", DEFAULT_LLM_MODEL)
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.chroma_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))
        # Chroma cosine distance: 0 = identical, larger = less similar.
        self.max_distance = float(os.getenv("RAG_MAX_DISTANCE", "0.70"))
        self.host = os.getenv("APP_HOST", "0.0.0.0")
        self.port = int(os.getenv("APP_PORT", "8000"))

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. "
                "Put your private key in .env."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not 1 <= self.top_k <= 10:
            raise RuntimeError("RAG_TOP_K must be between 1 and 10.")


settings = Settings()


# ---------------------------------------------------------------------------
# PDF extraction and chunking
# ---------------------------------------------------------------------------

class PDFLoader:
    @staticmethod
    def load(path: str) -> tuple[str, int]:
        try:
            reader = PdfReader(path)
        except PdfReadError as exc:
            raise ValueError(f"Invalid or unreadable PDF: {exc}") from exc

        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not text:
            raise ValueError(
                "The PDF contains no extractable text. "
                "Scanned/image-only PDFs require OCR."
            )

        return text, len(reader.pages)


class TextChunker:
    def __init__(self, size: int, overlap: int) -> None:
        self.size = size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - self.overlap

        return chunks


# ---------------------------------------------------------------------------
# Local embeddings
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """Local embeddings; document/question text is not sent to OpenRouter."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        try:
            vector = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vector.tolist()
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, directory: str) -> None:
        self.client = chromadb.PersistentClient(path=directory)
        self.collection = self.client.get_or_create_collection(
            name="technical_documentation",
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        filename: str,
    ) -> int:
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [
            {"document": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
        )
        return len(chunks)

    def search(
        self,
        embedding: list[float],
        top_k: int,
        max_distance: float,
    ) -> list[dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        found = []
        for document, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            if float(distance) <= max_distance:
                found.append(
                    {
                        "text": document,
                        "metadata": metadata,
                        "distance": float(distance),
                    }
                )
        return found


# ---------------------------------------------------------------------------
# OpenRouter LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strict technical documentation assistant.

Use ONLY the supplied documentation context.

Rules:
- Never use outside knowledge.
- Never guess or invent an answer.
- If the context does not contain enough information, say exactly:
  "I could not find the answer in the provided documentation."
- Cite supporting excerpts as [Source N].
- Preserve endpoint names, parameters and technical terminology.
- Do not claim an API feature exists unless the documentation supports it.
"""


class LLMClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def answer(self, question: str, context: str) -> str:
        prompt = (
            f"Documentation context:\n---\n{context}\n---\n\n"
            f"Question:\n{question}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")
        return content.strip()


# ---------------------------------------------------------------------------
# RAG service
# ---------------------------------------------------------------------------

class RAGService:
    def __init__(self) -> None:
        self.embedder = EmbeddingModel(settings.embedding_model)
        self.store = VectorStore(settings.chroma_dir)
        self.chunker = TextChunker(
            settings.chunk_size,
            settings.chunk_overlap,
        )
        self.llm = LLMClient(settings.api_key, settings.llm_model)

    def ingest(self, text: str, filename: str) -> int:
        chunks = self.chunker.split(text)
        if not chunks:
            raise ValueError("No usable text chunks were produced.")

        embeddings = [self.embedder.encode(chunk) for chunk in chunks]
        return self.store.add(chunks, embeddings, filename)

    def query(self, question: str, top_k: int) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        matches = self.store.search(
            self.embedder.encode(question),
            top_k,
            settings.max_distance,
        )

        if not matches:
            return {
                "answer": (
                    "I could not find the answer in the "
                    "provided documentation."
                ),
                "sources": [],
            }

        context = "\n\n".join(
            f"[Source {i}]\n"
            f"Document: {item['metadata']['document']}\n"
            f"Chunk: {item['metadata']['chunk_index']}\n"
            f"{item['text']}"
            for i, item in enumerate(matches, 1)
        )

        answer = self.llm.answer(question, context)

        return {
            "answer": answer,
            "sources": [
                {
                    "document": item["metadata"]["document"],
                    "chunk_index": item["metadata"]["chunk_index"],
                    "distance": item["distance"],
                }
                for item in matches
            ],
        }


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class Source(BaseModel):
    document: str
    chunk_index: int
    distance: float = Field(ge=0)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    document: str
    pages: int
    chunks_indexed: int
    status: str


class DocumentInfo(BaseModel):
    document: str
    chunks: int


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@lru_cache
def get_service() -> RAGService:
    settings.validate()
    return RAGService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_service()
    logger.info(
        "RAG ready | LLM=%s | Embeddings=%s",
        settings.llm_model,
        settings.embedding_model,
    )
    yield


app = FastAPI(
    title="Technical Documentation RAG Assistant",
    version="2.0.0",
    description=(
        "Professional grounded RAG API for technical PDF documentation."
    ),
    lifespan=lifespan,
)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    document: UploadFile = File(...),
) -> UploadResponse:
    filename = document.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are supported.")

    data = await document.read()
    if not data:
        raise HTTPException(400, "The uploaded PDF is empty.")

    service = get_service()

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(data)
        tmp.flush()

        try:
            text, pages = PDFLoader.load(tmp.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    try:
        chunks = service.ingest(text, filename)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc

    return UploadResponse(
        document=filename,
        pages=pages,
        chunks_indexed=chunks,
        status="indexed",
    )


@app.get("/query", response_model=QueryResponse)
def query_documentation(
    question: str = Query(
        ...,
        min_length=1,
        description="Question about the indexed documentation",
    ),
    top_k: int = Query(
        settings.top_k,
        ge=1,
        le=10,
    ),
) -> QueryResponse:
    service = get_service()

    try:
        result = service.query(question, top_k)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc

    return QueryResponse(**result)


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    collection = get_service().store.collection
    result = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}

    for metadata in result.get("metadatas", []) or []:
        if metadata:
            name = str(metadata.get("document", "unknown"))
            counts[name] = counts.get(name, 0) + 1

    return [
        DocumentInfo(document=name, chunks=count)
        for name, count in sorted(counts.items())
    ]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Technical Documentation RAG Assistant",
        "version": app.version,
        "docs": "/docs",
        "query": "/query",
        "upload": "/documents/upload",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    service = get_service()
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "indexed_chunks": service.store.collection.count(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
