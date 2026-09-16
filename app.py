import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import chromadb
from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import HTMLResponse
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("university_rag")


class Settings:
    """Centralized application configuration."""

    def __init__(self) -> None:
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

        self.openrouter_model = os.getenv(
            "OPENROUTER_MODEL",
            "nvidia/nemotron-3.5-lightning:free",
        )

        self.openrouter_embed_model = os.getenv(
            "OPENROUTER_EMBED_MODEL",
            "nvidia/nemotron-3-embed-1b:free",
        )

        self.chroma_db_dir = os.getenv(
            "CHROMA_DB_DIR",
            "./chroma_db",
        )

        self.app_host = os.getenv(
            "APP_HOST",
            "0.0.0.0",
        )

        self.app_port = int(
            os.getenv("APP_PORT", "8000")
        )

    def require_llm_config(self) -> None:
        if not self.openrouter_api_key:
            raise RuntimeError(
                "Missing required environment variable: "
                "OPENROUTER_API_KEY. "
                "Set it in your environment or a .env file."
            )


settings = Settings()


# --------------------------------------------------------------------------
# PDF handling
# --------------------------------------------------------------------------

class PDFLoader:
    """Extracts plain text from PDF documents."""

    @staticmethod
    def load_pdf(path: str) -> str:
        try:
            reader = PdfReader(path)
        except PdfReadError as exc:
            raise ValueError(
                f"Could not read PDF file: {exc}"
            ) from exc

        text = " ".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            raise ValueError(
                "No extractable text found in PDF. "
                "The document may be a scanned image."
            )

        return text


# --------------------------------------------------------------------------
# Text chunking
# --------------------------------------------------------------------------

class TextChunker:
    """
    Splits long documents into overlapping text chunks.

    Overlap helps preserve context when important information appears
    near the boundary between two chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list[str]:
        text = text.strip()

        if not text:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = end - self.overlap

        return chunks


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

class EmbeddingModel:
    """Generates embeddings through OpenRouter."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ) -> None:
        self.client = client
        self.model = model

    def encode(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
        except OpenAIError as exc:
            logger.exception("Embedding call failed")
            raise RuntimeError(
                f"Embedding generation failed: {exc}"
            ) from exc

        if not response.data:
            raise RuntimeError(
                "Embedding API returned no data."
            )

        return response.data[0].embedding


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------

class ChromaManager:
    """
    Stores document chunks and embeddings in a persistent Chroma
    collection and retrieves relevant chunks for questions.
    """

    def __init__(self, persist_dir: str) -> None:
        self.client = chromadb.PersistentClient(
            path=persist_dir
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="university_knowledge_cosine",
                metadata={"hnsw:space": "cosine"},
            )
        )

    def add_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        document_name: str,
    ) -> int:
        if not chunks:
            return 0

        ids = [
            f"{uuid.uuid4()}"
            for _ in chunks
        ]

        metadatas = [
            {
                "document": document_name,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        total = self.collection.count()

        if total == 0:
            return []

        top_k = min(top_k, total)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[dict[str, Any]] = []

        for doc_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            retrieved.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return retrieved

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        return self.collection.count()


# --------------------------------------------------------------------------
# RAG prompt
# --------------------------------------------------------------------------

RAG_PROMPT = """
You are the University Knowledge Assistant.

Answer the user's question using ONLY the provided university documentation context.

If the answer cannot be found in the context, say:

"I could not find the answer in the provided university documents."

Do not invent university information or rely on unsupported assumptions.

Documentation context:
-----------------------
{context}
-----------------------

User question:
{question}

Provide a concise and helpful answer.
"""


# --------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------

class LLMClient:
    """OpenAI-compatible chat client pointed at OpenRouter."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ) -> None:
        self.client = client
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
            )
        except OpenAIError as exc:
            logger.exception("LLM call failed")
            raise RuntimeError(
                f"Answer generation failed: {exc}"
            ) from exc

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "LLM returned an empty response."
            )

        return content


# --------------------------------------------------------------------------
# RAG service
# --------------------------------------------------------------------------

class RAGService:
    """
    Coordinates document ingestion, retrieval and answer generation.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        chroma_manager: ChromaManager,
        chunker: TextChunker,
    ) -> None:
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.chroma_manager = chroma_manager
        self.chunker = chunker

    def ingest_document(
        self,
        document_text: str,
        document_name: str,
    ) -> int:
        chunks = self.chunker.split_text(
            document_text
        )

        if not chunks:
            raise ValueError(
                "Document produced no text chunks."
            )

        embeddings = [
            self.embedding_model.encode(chunk)
            for chunk in chunks
        ]

        return self.chroma_manager.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_name=document_name,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        query_embedding = self.embedding_model.encode(
            question
        )

        retrieved = self.chroma_manager.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not retrieved:
            raise ValueError(
                "No documents have been indexed yet. "
                "Upload a PDF document first."
            )

        context_parts: list[str] = []

        for index, item in enumerate(retrieved, start=1):
            document_name = item["metadata"].get(
                "document",
                "unknown",
            )

            chunk_index = item["metadata"].get(
                "chunk_index",
                "unknown",
            )

            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {document_name}\n"
                f"Chunk: {chunk_index}\n"
                f"{item['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = RAG_PROMPT.format(
            context=context,
            question=question,
        )

        answer = self.llm_client.generate(prompt)

        sources = [
            {
                "document": item["metadata"].get(
                    "document",
                    "unknown",
                ),
                "chunk_index": item["metadata"].get(
                    "chunk_index",
                    -1,
                ),
                "distance": item["distance"],
            }
            for item in retrieved
        ]

        return {
            "answer": answer,
            "sources": sources,
        }


# --------------------------------------------------------------------------
# API schemas
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question about the university documents",
    )


class Source(BaseModel):
    document: str
    chunk_index: int
    distance: float
    cosine_similarity: float


class Evaluation(BaseModel):
    """Retrieval and end-to-end evaluation metrics shown in the UI."""

    chunks_indexed: int
    chunks_retrieved: int

    best_distance: float
    average_distance: float

    best_cosine_similarity: float
    average_cosine_similarity: float

    response_time_seconds: float
    sources_included: int


class QueryResponse(BaseModel):
    answer: str
    evaluation: Evaluation
    sources: list[Source]


class UploadResponse(BaseModel):
    document: str
    chunks_indexed: int


# --------------------------------------------------------------------------
# Dependency wiring
# --------------------------------------------------------------------------

@lru_cache
def get_rag_service() -> RAGService:
    """
    Creates and caches the application's heavy dependencies.
    """

    settings.require_llm_config()

    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    embedding_model = EmbeddingModel(
        client=openrouter_client,
        model=settings.openrouter_embed_model,
    )

    llm_client = LLMClient(
        client=openrouter_client,
        model=settings.openrouter_model,
    )

    chroma_manager = ChromaManager(
        persist_dir=settings.chroma_db_dir,
    )

    chunker = TextChunker(
        chunk_size=1000,
        overlap=200,
    )

    return RAGService(
        embedding_model=embedding_model,
        llm_client=llm_client,
        chroma_manager=chroma_manager,
        chunker=chunker,
    )


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Build the service during application startup.
    """

    get_rag_service()

    yield


router = APIRouter()


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
)
async def upload_document(
    document: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload and index a university PDF document.
    """

    if document.content_type not in (
        "application/pdf",
        "application/x-pdf",
    ):
        raise HTTPException(
            status_code=415,
            detail="Only PDF documents are supported.",
        )

    service = get_rag_service()

    file_content = await document.read()

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp.flush()

        try:
            text = PDFLoader.load_pdf(tmp.name)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the PDF.",
        )

    try:
        chunks_indexed = service.ingest_document(
            document_text=text,
            document_name=document.filename or "unknown.pdf",
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        document=document.filename or "unknown.pdf",
        chunks_indexed=chunks_indexed,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query_university(
    request: QueryRequest,
) -> QueryResponse:
    """
    Retrieve relevant university information, generate a grounded answer,
    and calculate retrieval/system evaluation metrics.

    Metrics:
        - Chunks Indexed
        - Chunks Retrieved
        - Best Distance
        - Average Distance
        - Best Cosine Similarity
        - Average Cosine Similarity
        - Response Time
        - Sources Included
    """

    question = request.query.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    service = get_rag_service()

    start_time = time.perf_counter()

    try:
        result = service.ask(
            question=question,
            top_k=5,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    response_time = time.perf_counter() - start_time

    raw_sources = result.get("sources", [])

    sources: list[Source] = []

    for raw_source in raw_sources:
        distance = float(
            raw_source.get("distance", 0.0)
        )

        # IMPORTANT:
        # The Chroma collection is explicitly configured with
        # hnsw:space = cosine. Therefore:
        #
        #     cosine distance = 1 - cosine similarity
        #
        # and:
        #
        #     cosine similarity = 1 - distance
        #
        # We do not use 1/(1+distance), because that is not cosine
        # similarity.
        cosine_similarity = max(
            -1.0,
            min(
                1.0,
                1.0 - distance,
            ),
        )

        metadata = raw_source.get("metadata", {})

        sources.append(
            Source(
                document=str(
                    raw_source.get(
                        "document",
                        metadata.get(
                            "document",
                            "Unknown document",
                        ),
                    )
                ),
                chunk_index=int(
                    raw_source.get(
                        "chunk_index",
                        metadata.get(
                            "chunk_index",
                            -1,
                        ),
                    )
                ),
                distance=round(
                    distance,
                    4,
                ),
                cosine_similarity=round(
                    cosine_similarity,
                    4,
                ),
            )
        )

    distances = [
        source.distance
        for source in sources
    ]

    similarities = [
        source.cosine_similarity
        for source in sources
    ]

    chunks_indexed = service.chroma_manager.count()

    evaluation = Evaluation(
        chunks_indexed=chunks_indexed,
        chunks_retrieved=len(sources),

        best_distance=round(
            min(distances),
            4,
        ) if distances else 0.0,

        average_distance=round(
            sum(distances) / len(distances),
            4,
        ) if distances else 0.0,

        best_cosine_similarity=round(
            max(similarities),
            4,
        ) if similarities else 0.0,

        average_cosine_similarity=round(
            sum(similarities) / len(similarities),
            4,
        ) if similarities else 0.0,

        response_time_seconds=round(
            response_time,
            3,
        ),

        sources_included=len(sources),
    )

    return QueryResponse(
        answer=result.get(
            "answer",
            "I could not find the answer in the provided university documents.",
        ),
        evaluation=evaluation,
        sources=sources,
    )


# --------------------------------------------------------------------------
# FastAPI Application
# --------------------------------------------------------------------------

app = FastAPI(
    title="University Knowledge Assistant",
    description=(
        "A Retrieval-Augmented Generation application that "
        "indexes university documents, retrieves relevant "
        "information, and generates grounded answers with sources."
    ),
    version="1.0.0",
    lifespan=lifespan,

    # Keep the user experience focused on the application,
    # not on Swagger/OpenAPI.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.include_router(router)


# --------------------------------------------------------------------------
# Application Page
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>University Knowledge Assistant</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #f4f6f8;
    color: #202938;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
}

.container {
    width: min(1100px, 94%);
    margin: 0 auto;
    padding: 35px 0 60px;
}

.card {
    background: #fff;
    border: 1px solid #e1e5ea;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 22px;
}

h1 { margin: 0 0 10px; font-size: 38px; }
h2 { margin-top: 0; }
h3 { margin-top: 0; }

p { color: #667085; line-height: 1.6; }

textarea {
    width: 100%;
    min-height: 120px;
    padding: 14px;
    border: 1px solid #ccd3dc;
    border-radius: 10px;
    font: inherit;
    resize: vertical;
}

input[type=file] {
    width: 100%;
    margin: 10px 0 14px;
}

button {
    border: 0;
    border-radius: 9px;
    padding: 12px 18px;
    background: #172033;
    color: white;
    font-weight: 700;
    cursor: pointer;
}

button:disabled {
    opacity: .55;
    cursor: not-allowed;
}

.status {
    margin-top: 12px;
    color: #667085;
}

.error { color: #b42318; }
.success { color: #067647; }

.answer {
    white-space: pre-wrap;
    line-height: 1.75;
    font-size: 17px;
}

.divider {
    border: 0;
    border-top: 1px solid #e5e7eb;
    margin: 28px 0;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.metric {
    background: #f8f9fb;
    border: 1px solid #e4e7eb;
    border-radius: 11px;
    padding: 15px;
}

.label {
    color: #6b7280;
    font-size: 13px;
    line-height: 1.35;
}

.value {
    margin-top: 7px;
    font-size: 20px;
    font-weight: 750;
}

.note {
    margin-top: 5px;
    color: #8a94a3;
    font-size: 11px;
}

.source {
    background: #f8f9fb;
    border: 1px solid #e4e7eb;
    border-radius: 10px;
    padding: 14px;
    margin: 9px 0;
}

.source-title {
    font-weight: 700;
}

.source-meta {
    margin-top: 6px;
    color: #667085;
    font-size: 13px;
}

.explain {
    margin-top: 16px;
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
}

@media (max-width: 800px) {
    .metrics { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 500px) {
    .metrics { grid-template-columns: 1fr; }
    h1 { font-size: 30px; }
}
</style>
</head>

<body>
<div class="container">

<div class="card">
    <h1>🎓 University Knowledge Assistant</h1>
    <p>
        Here is your RAG system to assist you with your education information.
    </p>
</div>

<div class="card">
    <h2>Upload the PDF file courses</h2>
    <p>Upload an official university document to the knowledge base.</p>

    <input id="pdf" type="file" accept=".pdf,application/pdf">

    <button id="uploadButton" onclick="uploadPDF()">
        Upload PDF
    </button>

    <div id="uploadStatus" class="status"></div>
</div>

<div class="card">
    <h2> Ask your Question</h2>

    <textarea
        id="query"
        placeholder="Write Your Question"
    ></textarea>

    <br><br>

    <button id="queryButton" onclick="sendQuery()">
        Send your Question
    </button>

    <div id="queryStatus" class="status"></div>
</div>

<div id="result" class="card" style="display:none;">

    <h2>Answer</h2>
    <div id="answer" class="answer"></div>

    <hr class="divider">

    <h2>Retrieval Evaluation</h2>

    <div class="metrics">

        <div class="metric">
            <div class="label">Chunks Indexed</div>
            <div id="chunksIndexed" class="value">-</div>
            <div class="note">Total chunks in ChromaDB</div>
        </div>

        <div class="metric">
            <div class="label">Chunks Retrieved</div>
            <div id="chunksRetrieved" class="value">-</div>
            <div class="note">Top-k chunks returned</div>
        </div>

        <div class="metric">
            <div class="label">Best Distance</div>
            <div id="bestDistance" class="value">-</div>
            <div class="note">Lower is closer</div>
        </div>

        <div class="metric">
            <div class="label">Average Distance</div>
            <div id="averageDistance" class="value">-</div>
            <div class="note">Across retrieved chunks</div>
        </div>

        <div class="metric">
            <div class="label">Best Cosine Similarity</div>
            <div id="bestSimilarity" class="value">-</div>
            <div class="note">Higher is more similar</div>
        </div>

        <div class="metric">
            <div class="label">Average Cosine Similarity</div>
            <div id="averageSimilarity" class="value">-</div>
            <div class="note">Across retrieved chunks</div>
        </div>

        <div class="metric">
            <div class="label">Response Time</div>
            <div id="responseTime" class="value">-</div>
            <div class="note">End-to-end query time</div>
        </div>

        <div class="metric">
            <div class="label">Sources Included</div>
            <div id="sourcesIncluded" class="value">-</div>
            <div class="note">Retrieved sources shown</div>
        </div>

    </div>

    <hr class="divider">

    <h2>Sources</h2>
    <div id="sources"></div>

    <div class="explain">
        <strong>Evaluation interpretation:</strong>
        Chroma is configured with cosine distance. Therefore,
        cosine similarity = 1 − cosine distance.
        A lower distance indicates a closer retrieved chunk,
        while a higher cosine similarity indicates greater semantic similarity.
    </div>

</div>

</div>

<script>
async function uploadPDF() {
    const input = document.getElementById("pdf");
    const status = document.getElementById("uploadStatus");
    const button = document.getElementById("uploadButton");

    if (!input.files.length) {
        status.className = "status error";
        status.textContent = "Please select a PDF.";
        return;
    }

    const formData = new FormData();
    formData.append("document", input.files[0]);

    button.disabled = true;
    status.className = "status";
    status.textContent = "Indexing university document...";

    try {
        const response = await fetch("/documents/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        status.className = "status success";
        status.textContent =
            data.document +
            " indexed successfully — " +
            data.chunks_indexed +
            " chunks created.";

    } catch (error) {
        status.className = "status error";
        status.textContent = "Error: " + error.message;
    } finally {
        button.disabled = false;
    }
}


async function sendQuery() {
    const query = document.getElementById("query").value.trim();
    const button = document.getElementById("queryButton");
    const status = document.getElementById("queryStatus");

    if (!query) {
        status.className = "status error";
        status.textContent = "Please enter a query.";
        return;
    }

    button.disabled = true;
    status.className = "status";
    status.textContent = "Searching university documents...";

    try {
        const response = await fetch("/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Query failed.");
        }

        document.getElementById("result").style.display = "block";
        document.getElementById("answer").textContent = data.answer;

        const e = data.evaluation;

        document.getElementById("chunksIndexed").textContent =
            e.chunks_indexed;

        document.getElementById("chunksRetrieved").textContent =
            e.chunks_retrieved;

        document.getElementById("bestDistance").textContent =
            Number(e.best_distance).toFixed(4);

        document.getElementById("averageDistance").textContent =
            Number(e.average_distance).toFixed(4);

        document.getElementById("bestSimilarity").textContent =
            Number(e.best_cosine_similarity).toFixed(4);

        document.getElementById("averageSimilarity").textContent =
            Number(e.average_cosine_similarity).toFixed(4);

        document.getElementById("responseTime").textContent =
            Number(e.response_time_seconds).toFixed(2) + " s";

        document.getElementById("sourcesIncluded").textContent =
            e.sources_included;

        const sources = document.getElementById("sources");
        sources.innerHTML = "";

        data.sources.forEach((source, index) => {
            const div = document.createElement("div");
            div.className = "source";

            const title = document.createElement("div");
            title.className = "source-title";
            title.textContent =
                "Source " + (index + 1) +
                " — " + source.document +
                " — Chunk " + source.chunk_index;

            const meta = document.createElement("div");
            meta.className = "source-meta";
            meta.textContent =
                "Distance: " +
                Number(source.distance).toFixed(4) +
                "  •  Cosine Similarity: " +
                Number(source.cosine_similarity).toFixed(4);

            div.appendChild(title);
            div.appendChild(meta);
            sources.appendChild(div);
        });

        status.textContent = "";

        document.getElementById("result").scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {
        status.className = "status error";
        status.textContent = "Error: " + error.message;
    } finally {
        button.disabled = false;
    }
}
</script>

</body>
</html>
"""


# --------------------------------------------------------------------------
# Health Check
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok"
    }


# --------------------------------------------------------------------------
# Run Application
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
    )
