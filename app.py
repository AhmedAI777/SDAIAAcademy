from contextlib import asynccontextmanager
import tempfile

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag import get_rag_service, PDFLoader, settings


# --------------------------------------------------------------------------
# Request / Response Models
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


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
# Application Startup
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Initialize the RAG service when the application starts.
    """
    get_rag_service()
    yield


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

router = APIRouter()


# --------------------------------------------------------------------------
# Upload University Document
# --------------------------------------------------------------------------

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

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        document=document.filename or "unknown.pdf",
        chunks_indexed=chunks_indexed,
    )


# --------------------------------------------------------------------------
# University Knowledge Query
# --------------------------------------------------------------------------

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

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

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

    sources = []

    for source in result.get("sources", []):

        distance = source.get("distance")

        # Convert cosine distance to similarity.
        if distance is not None:
            similarity = max(
                0.0,
                min(1.0, 1.0 - float(distance)),
            )
        else:
            similarity = 0.0

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
            "I could not find an answer in the provided university documents.",
        ),
        sources=sources,
    )


# --------------------------------------------------------------------------
# FastAPI Application
# --------------------------------------------------------------------------

app = FastAPI(
    title="University Knowledge Assistant",
    description=(
        "A Retrieval-Augmented Generation application for "
        "answering questions using indexed university documents."
    ),
    version="1.0.0",
    lifespan=lifespan,

    # Hide Swagger/ReDoc from the normal user experience.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.include_router(router)


# --------------------------------------------------------------------------
# Web Interface
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>University Knowledge Assistant</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
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
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
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

        .similarity {
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
        <h1>🎓 University Knowledge Assistant</h1>

        <p>
            Ask questions about university courses,
            academic information, regulations, and student services.
        </p>
    </div>


    <div class="grid">

        <!-- Upload -->
        <div class="card">

            <h2>University Knowledge Base</h2>

            <p>
                Upload an official university PDF to add
                its information to the knowledge base.
            </p>

            <input
                type="file"
                id="pdfFile"
                accept=".pdf,application/pdf"
            >

            <button onclick="uploadPDF()">
                Upload PDF
            </button>

            <div id="uploadStatus" class="status"></div>

        </div>


        <!-- Query -->
        <div class="card">

            <h2>Ask a Question</h2>

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

            <div id="queryStatus" class="status"></div>

        </div>

    </div>


    <!-- Answer -->
    <div
        id="answerSection"
        class="answer"
        style="display: none;"
    >

        <h2>Answer</h2>

        <div
            id="answerText"
            class="answer-text"
        ></div>

        <h3>Sources</h3>

        <div id="sources"></div>

    </div>

</div>


<script>

async function uploadPDF() {

    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("uploadStatus");

    if (!fileInput.files.length) {

        status.style.display = "block";
        status.className = "status error";
        status.textContent = "Please select a PDF file.";

        return;
    }

    const formData = new FormData();

    formData.append(
        "document",
        fileInput.files[0]
    );

    status.style.display = "block";
    status.className = "status";
    status.textContent = "Indexing document...";

    try {

        const response = await fetch(
            "/documents/upload",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Upload failed."
            );
        }

        status.className = "status success";

        status.textContent =
            data.document +
            " indexed successfully (" +
            data.chunks_indexed +
            " chunks).";

    } catch (error) {

        status.className = "status error";

        status.textContent = error.message;
    }
}


async function sendQuery() {

    const queryInput = document.getElementById("query");
    const button = document.getElementById("queryButton");

    const status = document.getElementById("queryStatus");

    const query = queryInput.value.trim();

    if (!query) {

        status.style.display = "block";
        status.className = "status error";
        status.textContent = "Please enter a question.";

        return;
    }

    button.disabled = true;

    status.style.display = "block";
    status.className = "status";
    status.textContent = "Searching university documents...";

    try {

        const response = await fetch(
            "/query",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    query: query
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Query failed."
            );
        }

        displayAnswer(data);

        status.style.display = "none";

    } catch (error) {

        status.className = "status error";
        status.textContent = error.message;

    } finally {

        button.disabled = false;
    }
}


function displayAnswer(data) {

    const answerSection =
        document.getElementById("answerSection");

    const answerText =
        document.getElementById("answerText");

    const sources =
        document.getElementById("sources");

    answerSection.style.display = "block";

    answerText.textContent = data.answer;

    sources.innerHTML = "";

    if (!data.sources || data.sources.length === 0) {

        sources.innerHTML =
            "<p>No sources were returned.</p>";

        return;
    }

    data.sources.forEach(source => {

        const element =
            document.createElement("div");

        element.className = "source";

        element.innerHTML =
            "<strong>" +
            escapeHTML(source.document) +
            "</strong>" +
            "<div class='similarity'>" +
            "Similarity: " +
            Number(source.similarity).toFixed(3) +
            "</div>";

        sources.appendChild(element);
    });
}


function escapeHTML(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
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
