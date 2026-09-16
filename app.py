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
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
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
                name="university_knowledge"
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
    similarity: float


class QueryResponse(BaseModel):
    answer: str
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
    Retrieve relevant university information and generate
    a grounded answer.
    """

    service = get_rag_service()

    try:
        # top_k stays internal so the API remains simple.
        result = service.ask(
            question=request.query.strip(),
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

    sources: list[Source] = []

    for source in result.get("sources", []):
        distance = source.get("distance")

        # Chroma returns distance. This converts it to a simple
        # 0-1 relevance score for presentation.
        if distance is None:
            similarity = 0.0
        else:
            similarity = 1.0 / (1.0 + float(distance))

        sources.append(
            Source(
                document=source.get(
                    "document",
                    "Unknown document",
                ),
                similarity=round(similarity, 3),
            )
        )

    return QueryResponse(
        answer=result.get(
            "answer",
            "I could not find the answer in the provided university documents.",
        ),
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
# Simple Application Page
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
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 50px auto;
            padding: 0 20px;
            background: #f5f7fa;
            color: #1f2937;
        }

        .card {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
        }

        h1 {
            margin-bottom: 8px;
        }

        textarea {
            width: 100%;
            min-height: 100px;
            padding: 12px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 15px;
            box-sizing: border-box;
        }

        input[type="file"] {
            margin: 12px 0;
        }

        button {
            padding: 11px 18px;
            border: 0;
            border-radius: 8px;
            background: #111827;
            color: white;
            cursor: pointer;
        }

        button:disabled {
            opacity: 0.5;
        }

        #answer {
            white-space: pre-wrap;
            line-height: 1.6;
        }

        .source {
            padding: 10px;
            margin-top: 8px;
            background: #f9fafb;
            border-radius: 8px;
        }

        .status {
            margin-top: 12px;
        }
    </style>
</head>

<body>

    <div class="card">
        <h1>🎓 University Knowledge Assistant</h1>
        <p>
            Ask questions using the information contained in your
            indexed university documents.
        </p>
    </div>

    <div class="card">
        <h2>Upload University PDF</h2>

        <input
            type="file"
            id="pdf"
            accept=".pdf,application/pdf"
        >

        <br>

        <button onclick="uploadPDF()">
            Upload & Index
        </button>

        <div id="uploadStatus" class="status"></div>
    </div>

    <div class="card">
        <h2>Ask a Query</h2>

        <textarea
            id="query"
            placeholder="Example: What are the prerequisites for Machine Learning?"
        ></textarea>

        <br><br>

        <button id="queryButton" onclick="sendQuery()">
            Ask University Assistant
        </button>

        <div id="queryStatus" class="status"></div>
    </div>

    <div class="card" id="result" style="display:none;">
        <h2>Answer</h2>

        <div id="answer"></div>

        <h3>Sources</h3>

        <div id="sources"></div>
    </div>

<script>
async function uploadPDF() {
    const fileInput = document.getElementById("pdf");
    const status = document.getElementById("uploadStatus");

    if (!fileInput.files.length) {
        status.textContent = "Please select a PDF.";
        return;
    }

    const formData = new FormData();
    formData.append("document", fileInput.files[0]);

    status.textContent = "Indexing PDF...";

    try {
        const response = await fetch("/documents/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        status.textContent =
            data.document +
            " indexed successfully. " +
            data.chunks_indexed +
            " chunks created.";

    } catch (error) {
        status.textContent = "Error: " + error.message;
    }
}


async function sendQuery() {
    const query = document.getElementById("query").value.trim();
    const button = document.getElementById("queryButton");
    const status = document.getElementById("queryStatus");

    if (!query) {
        status.textContent = "Please enter a query.";
        return;
    }

    button.disabled = true;
    status.textContent = "Searching university documents...";

    try {
        const response = await fetch("/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: query
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Query failed.");
        }

        document.getElementById("result").style.display = "block";
        document.getElementById("answer").textContent = data.answer;

        const sources = document.getElementById("sources");
        sources.innerHTML = "";

        data.sources.forEach(source => {
            const div = document.createElement("div");
            div.className = "source";

            div.textContent =
                source.document +
                " — Similarity: " +
                Number(source.similarity).toFixed(3);

            sources.appendChild(div);
        });

        status.textContent = "";

    } catch (error) {
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
