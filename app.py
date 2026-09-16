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
    """Application configuration."""

    def __init__(self) -> None:
        self.openrouter_api_key = os.getenv(
            "OPENROUTER_API_KEY",
            "",
        )

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
                "Missing OPENROUTER_API_KEY. "
                "Add it to your .env file."
            )


settings = Settings()


# --------------------------------------------------------------------------
# PDF Loader
# --------------------------------------------------------------------------

class PDFLoader:
    """Extract text from PDF files."""

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

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            raise ValueError(
                "No extractable text found in the PDF. "
                "The document may be scanned."
            )

        return text


# --------------------------------------------------------------------------
# Text Chunker
# --------------------------------------------------------------------------

class TextChunker:
    """Splits documents into overlapping chunks."""

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

    def split_text(
        self,
        text: str,
    ) -> list[str]:

        text = text.strip()

        if not text:
            return []

        chunks = []
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
# Embedding Model
# --------------------------------------------------------------------------

class EmbeddingModel:
    """Generates embeddings using OpenRouter."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ) -> None:

        self.client = client
        self.model = model

    def encode(
        self,
        text: str,
    ) -> list[float]:

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )

        except OpenAIError as exc:
            logger.exception(
                "Embedding generation failed"
            )

            raise RuntimeError(
                f"Embedding generation failed: {exc}"
            ) from exc

        if not response.data:
            raise RuntimeError(
                "Embedding API returned no data."
            )

        return response.data[0].embedding


# --------------------------------------------------------------------------
# Chroma Vector Store
# --------------------------------------------------------------------------

class ChromaManager:
    """Persistent vector database for university documents."""

    def __init__(
        self,
        persist_dir: str,
    ) -> None:

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
            str(uuid.uuid4())
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

        top_k = min(
            top_k,
            total,
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
        )

        ids = results.get(
            "ids",
            [[]],
        )[0]

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        retrieved = []

        for (
            doc_id,
            document,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):

            retrieved.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata or {},
                    "distance": float(distance),
                }
            )

        return retrieved


# --------------------------------------------------------------------------
# RAG Prompt
# --------------------------------------------------------------------------

RAG_PROMPT = """
You are the University Knowledge Assistant.

Answer the user's query using ONLY the provided university
documentation.

Do not invent:

- courses
- prerequisites
- university policies
- deadlines
- fees
- regulations
- procedures
- academic requirements

If the answer cannot be found in the provided documents,
respond exactly with:

"I could not find the answer in the provided university documents."

Keep the answer concise, clear, and directly related to the query.

University documentation:
-------------------------
{context}
-------------------------

User query:
{question}

Answer:
"""


# --------------------------------------------------------------------------
# LLM Client
# --------------------------------------------------------------------------

class LLMClient:
    """OpenRouter chat client."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ) -> None:

        self.client = client
        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:

        try:
            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0.2,
                )
            )

        except OpenAIError as exc:
            logger.exception(
                "LLM generation failed"
            )

            raise RuntimeError(
                f"Answer generation failed: {exc}"
            ) from exc

        if not response.choices:
            raise RuntimeError(
                "LLM returned no choices."
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "LLM returned an empty answer."
            )

        return content.strip()


# --------------------------------------------------------------------------
# RAG Service
# --------------------------------------------------------------------------

class RAGService:
    """Coordinates ingestion, retrieval and answer generation."""

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

        if not question.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_embedding = (
            self.embedding_model.encode(
                question.strip()
            )
        )

        retrieved = (
            self.chroma_manager.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        )

        if not retrieved:
            raise ValueError(
                "No documents have been indexed yet. "
                "Upload a university PDF first."
            )

        context_parts = []

        for index, item in enumerate(
            retrieved,
            start=1,
        ):

            document_name = (
                item["metadata"]
                .get(
                    "document",
                    "unknown",
                )
            )

            chunk_index = (
                item["metadata"]
                .get(
                    "chunk_index",
                    "unknown",
                )
            )

            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {document_name}\n"
                f"Chunk: {chunk_index}\n"
                f"{item['text']}"
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = RAG_PROMPT.format(
            context=context,
            question=question,
        )

        answer = self.llm_client.generate(
            prompt
        )

        sources = [
            {
                "document": (
                    item["metadata"]
                    .get(
                        "document",
                        "unknown",
                    )
                ),
                "distance": round(
                    float(
                        item["distance"]
                    ),
                    4,
                ),
            }
            for item in retrieved
        ]

        return {
            "answer": answer,
            "sources": sources,
        }


# --------------------------------------------------------------------------
# API Models
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question about the indexed university documents",
    )


class Source(BaseModel):
    document: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    document: str
    chunks_indexed: int


# --------------------------------------------------------------------------
# RAG Dependency
# --------------------------------------------------------------------------

@lru_cache
def get_rag_service() -> RAGService:

    settings.require_llm_config()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    embedding_model = EmbeddingModel(
        client=client,
        model=settings.openrouter_embed_model,
    )

    llm_client = LLMClient(
        client=client,
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
# Application Startup
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(
    _: FastAPI,
):

    get_rag_service()

    yield


router = APIRouter()


# --------------------------------------------------------------------------
# Upload University PDF
# --------------------------------------------------------------------------

@router.post(
    "/documents/upload",
    response_model=UploadResponse,
)
async def upload_document(
    document: UploadFile = File(...),
) -> UploadResponse:

    if document.content_type not in (
        "application/pdf",
        "application/x-pdf",
    ):

        raise HTTPException(
            status_code=415,
            detail="Only PDF documents are supported.",
        )

    file_content = await document.read()

    if not file_content:

        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    service = get_rag_service()

    with tempfile.NamedTemporaryFile(
        suffix=".pdf"
    ) as tmp:

        tmp.write(file_content)
        tmp.flush()

        try:

            text = PDFLoader.load_pdf(
                tmp.name
            )

        except ValueError as exc:

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    try:

        chunks_indexed = (
            service.ingest_document(
                document_text=text,
                document_name=(
                    document.filename
                    or "unknown.pdf"
                ),
            )
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

    return UploadResponse(
        document=(
            document.filename
            or "unknown.pdf"
        ),
        chunks_indexed=chunks_indexed,
    )


# --------------------------------------------------------------------------
# University Query
# --------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
)
def query_university(
    request: QueryRequest,
) -> QueryResponse:

    service = get_rag_service()

    try:

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

    return QueryResponse(
        answer=result["answer"],
        sources=[
            Source(
                document=source["document"],
                distance=source["distance"],
            )
            for source in result["sources"]
        ],
    )


# --------------------------------------------------------------------------
# FastAPI Application
# --------------------------------------------------------------------------

app = FastAPI(
    title="University Knowledge Assistant",

    description=(
        "A Retrieval-Augmented Generation application "
        "for answering questions using university documents."
    ),

    version="1.0.0",

    lifespan=lifespan,

    # The web interface is the primary user experience.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.include_router(router)


# --------------------------------------------------------------------------
# Web Interface
# --------------------------------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse,
)
def root() -> str:

    return """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
University Knowledge Assistant
</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f5f7fb;

    color: #1f2937;
}

.container {
    max-width: 1000px;

    margin: 0 auto;

    padding: 40px 20px;
}

.header {
    background: white;

    padding: 30px;

    border-radius: 16px;

    margin-bottom: 24px;

    border: 1px solid #e5e7eb;
}

.header h1 {
    margin: 0 0 10px;

    font-size: 30px;
}

.header p {
    margin: 0;

    color: #6b7280;

    line-height: 1.6;
}

.grid {
    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 24px;
}

.card {
    background: white;

    padding: 24px;

    border-radius: 16px;

    border: 1px solid #e5e7eb;
}

.card h2 {
    margin-top: 0;

    font-size: 20px;
}

.card p {
    color: #6b7280;

    line-height: 1.5;
}

input[type="file"] {
    width: 100%;

    margin: 15px 0;
}

textarea {
    width: 100%;

    min-height: 130px;

    padding: 14px;

    border: 1px solid #d1d5db;

    border-radius: 10px;

    resize: vertical;

    font-family: inherit;

    font-size: 15px;
}

button {
    border: none;

    border-radius: 10px;

    padding: 12px 18px;

    font-size: 15px;

    cursor: pointer;

    background: #111827;

    color: white;

    margin-top: 12px;
}

button:hover {
    opacity: 0.9;
}

button:disabled {
    opacity: 0.5;

    cursor: not-allowed;
}

.status {
    margin-top: 15px;

    padding: 12px;

    border-radius: 8px;

    background: #f3f4f6;

    display: none;
}

.answer {
    margin-top: 24px;

    background: white;

    padding: 24px;

    border-radius: 16px;

    border: 1px solid #e5e7eb;
}

.answer h2 {
    margin-top: 0;
}

.answer-text {
    line-height: 1.7;

    white-space: pre-wrap;
}

.source {
    padding: 12px;

    margin-top: 10px;

    background: #f9fafb;

    border-radius: 8px;

    border: 1px solid #e5e7eb;
}

.distance {
    font-size: 13px;

    color: #6b7280;

    margin-top: 5px;
}

.error {
    color: #b91c1c;
}

.success {
    color: #166534;
}

@media (max-width: 700px) {

    .grid {
        grid-template-columns: 1fr;
    }

}

</style>

</head>


<body>

<div class="container">


<div class="header">

<h1>
🎓 University Knowledge Assistant
</h1>

<p>
Ask questions about university courses,
academic information, regulations, and
student services using your indexed documents.
</p>

</div>


<div class="grid">


<!-- Upload -->

<div class="card">

<h2>
University Knowledge Base
</h2>

<p>
Upload a university PDF to add its
information to the knowledge base.
</p>

<input
    type="file"
    id="pdfFile"
    accept=".pdf,application/pdf"
>

<button
    onclick="uploadPDF()"
>
Upload PDF
</button>

<div
    id="uploadStatus"
    class="status"
></div>

</div>


<!-- Query -->

<div class="card">

<h2>
Ask a Question
</h2>

<textarea
    id="query"
    placeholder="Example: What are the prerequisites for Machine Learning?"
></textarea>

<button
    id="queryButton"
    onclick="sendQuery()"
>
Ask University Assistant
</button>

<div
    id="queryStatus"
    class="status"
></div>

</div>


</div>


<!-- Answer -->

<div
    id="answerSection"
    class="answer"
    style="display: none;"
>

<h2>
Answer
</h2>

<div
    id="answerText"
    class="answer-text"
></div>


<h3>
Sources
</h3>

<div id="sources"></div>

</div>


</div>


<script>


async function uploadPDF() {

    const fileInput =
        document.getElementById(
            "pdfFile"
        );

    const status =
        document.getElementById(
            "uploadStatus"
        );


    if (!fileInput.files.length) {

        status.style.display = "block";

        status.className =
            "status error";

        status.textContent =
            "Please select a PDF file.";

        return;
    }


    const formData =
        new FormData();


    formData.append(
        "document",
        fileInput.files[0]
    );


    status.style.display = "block";

    status.className =
        "status";

    status.textContent =
        "Indexing university document...";


    try {

        const response =
            await fetch(
                "/documents/upload",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Upload failed."
            );

        }


        status.className =
            "status success";


        status.textContent =
            data.document +
            " indexed successfully (" +
            data.chunks_indexed +
            " chunks).";


    } catch (error) {

        status.className =
            "status error";

        status.textContent =
            error.message;
    }
}



async function sendQuery() {

    const queryInput =
        document.getElementById(
            "query"
        );


    const button =
        document.getElementById(
            "queryButton"
        );


    const status =
        document.getElementById(
            "queryStatus"
        );


    const query =
        queryInput.value.trim();


    if (!query) {

        status.style.display =
            "block";

        status.className =
            "status error";

        status.textContent =
            "Please enter a question.";

        return;
    }


    button.disabled = true;


    status.style.display =
        "block";

    status.className =
        "status";

    status.textContent =
        "Searching university documents...";


    try {

        const response =
            await fetch(
                "/query",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        query: query
                    })

                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Query failed."
            );

        }


        displayAnswer(data);


        status.style.display =
            "none";


    } catch (error) {

        status.className =
            "status error";

        status.textContent =
            error.message;


    } finally {

        button.disabled = false;
    }

}



function displayAnswer(data) {

    const answerSection =
        document.getElementById(
            "answerSection"
        );


    const answerText =
        document.getElementById(
            "answerText"
        );


    const sources =
        document.getElementById(
            "sources"
        );


    answerSection.style.display =
        "block";


    answerText.textContent =
        data.answer;


    sources.innerHTML = "";


    if (
        !data.sources ||
        data.sources.length === 0
    ) {

        sources.innerHTML =
            "<p>No sources were returned.</p>";

        return;
    }


    data.sources.forEach(
        source => {

            const element =
                document.createElement(
                    "div"
                );


            element.className =
                "source";


            const documentName =
                escapeHTML(
                    source.document
                );


            const distance =
                Number(
                    source.distance
                ).toFixed(4);


            element.innerHTML =
                "<strong>" +
                documentName +
                "</strong>" +

                "<div class='distance'>" +
                "Distance: " +
                distance +
                "</div>";


            sources.appendChild(
                element
            );

        }
    );

}



function escapeHTML(value) {

    return String(value)

        .replaceAll(
            "&",
            "&amp;"
        )

        .replaceAll(
            "<",
            "&lt;"
        )

        .replaceAll(
            ">",
            "&gt;"
        )

        .replaceAll(
            '"',
            "&quot;"
        )

        .replaceAll(
            "'",
            "&#039;"
        );
}

</script>


</body>

</html>
"""


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:

    return {
        "status": "ok"
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
    )
