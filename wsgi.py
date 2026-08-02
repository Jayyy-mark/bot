import os
import sys
import uuid
import secrets
from collections import deque
from pathlib import Path

# =========================================================
# UTF-8
# =========================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =========================================================
# FLASK
# =========================================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename


# =========================================================
# ENV
# =========================================================

from dotenv import load_dotenv


# =========================================================
# GOOGLE GEMINI
# =========================================================

from google import genai


# =========================================================
# LANGCHAIN
# =========================================================

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# =========================================================
# CHROMA
# =========================================================

import chromadb


# =========================================================
# PDF
# =========================================================

from pypdf import PdfReader


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Set it in Render Environment Variables."
    )

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)




# =========================================================
# CORS
# =========================================================

# Your session ID is stored in the request body.
# Therefore we don't need cookies for chat sessions.

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": [
                "GET",
                "POST",
                "OPTIONS",
            ],
            "allow_headers": [
                "Content-Type",
            ],
        }
    },
)



UPLOAD_FOLDER = Path(app.root_path) / "uploads"
CHROMA_DB_PATH = Path(app.root_path) / "chroma_db"

ALLOWED_EXTENSIONS = {"pdf"}

MAX_FILE_SIZE = 50 * 1024 * 1024

MAX_HISTORY_TURNS = 20

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# =========================================================
# GOOGLE CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)


# =========================================================
# CHROMADB
# =========================================================

print("=" * 60, flush=True)
print(f"[CHROMA] DB PATH = {CHROMA_DB_PATH}", flush=True)
print(f"[CHROMA] PATH EXISTS = {CHROMA_DB_PATH.exists()}", flush=True)
print(f"[CHROMA] PATH IS DIR = {CHROMA_DB_PATH.is_dir()}", flush=True)

print("[CHROMA] Creating PersistentClient...", flush=True)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DB_PATH)
)

print("[CHROMA] PersistentClient created", flush=True)

print("[CHROMA] Getting collection...", flush=True)

collection = chroma_client.get_or_create_collection(
    name="documents"
)

print("[CHROMA] Collection created/opened", flush=True)

print("[CHROMA] Testing count()...", flush=True)

count = collection.count()

print(f"[CHROMA] Initial count = {count}", flush=True)
print("=" * 60, flush=True)




# =========================================================
# IN-MEMORY CHAT HISTORY
# =========================================================

chat_histories: dict[str, deque] = {}


# =========================================================
# HELPERS
# =========================================================


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def get_document_count() -> int:
    """
    Safely get the number of vectors in Chroma.
    """

    try:
        result = collection.count()
        return int(result)

    except Exception as e:
        print(
            f"[CHROMA] Failed to get document count: {e}",
            flush=True,
        )
        raise


# =========================================================
# PDF LOADING
# =========================================================


def load_pdf(pdf_path: str):
    print(
        f"[PDF] Loading: {pdf_path}",
        flush=True,
    )

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:
            text = page.extract_text()

        except Exception as e:
            print(
                f"[PDF] Failed page {page_number}: {e}",
                flush=True,
            )
            continue

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_number,
                },
            )
        )

    print(
        f"[PDF] Pages with text: {len(documents)}",
        flush=True,
    )

    return documents


# =========================================================
# CHUNKING
# =========================================================


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    print(
        f"[CHUNKING] Created {len(chunks)} chunks",
        flush=True,
    )

    return chunks


# =========================================================
# EMBEDDING DOCUMENTS
# =========================================================


def embed_documents(chunks):

    if not chunks:
        return []

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    print(
        f"[EMBEDDING] Creating embeddings for "
        f"{len(texts)} chunks...",
        flush=True,
    )

    embeddings = embedding_model.embed_documents(
        texts
    )

    print(
        "[EMBEDDING] Finished",
        flush=True,
    )

    return embeddings


# =========================================================
# INDEX DOCUMENT
# =========================================================


def index_document(pdf_path: str):

    global collection

    print(
        "\n========== INDEX DOCUMENT ==========",
        flush=True,
    )

    # -----------------------------------------------------
    # Delete previous vectors
    # -----------------------------------------------------

    try:

        existing_count = collection.count()

        print(
            f"[CHROMA] Existing vectors: {existing_count}",
            flush=True,
        )

        if existing_count > 0:

            existing = collection.get(
                include=[]
            )

            existing_ids = existing.get(
                "ids",
                [],
            )

            if existing_ids:

                print(
                    f"[CHROMA] Deleting "
                    f"{len(existing_ids)} vectors...",
                    flush=True,
                )

                collection.delete(
                    ids=existing_ids
                )

                print(
                    "[CHROMA] Old vectors deleted",
                    flush=True,
                )

    except Exception as e:

        print(
            f"[CHROMA] Delete failed: {e}",
            flush=True,
        )

        raise


    # -----------------------------------------------------
    # Load PDF
    # -----------------------------------------------------

    documents = load_pdf(pdf_path)

    if not documents:
        raise ValueError(
            "No readable text was found in the PDF."
        )


    # -----------------------------------------------------
    # Chunk
    # -----------------------------------------------------

    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created from the PDF."
        )


    # -----------------------------------------------------
    # Embeddings
    # -----------------------------------------------------

    embeddings = embed_documents(chunks)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )


    # -----------------------------------------------------
    # Prepare Chroma data
    # -----------------------------------------------------

    ids = []
    texts = []
    metadatas = []

    base_name = os.path.basename(pdf_path)

    for i, chunk in enumerate(chunks):

        ids.append(
            f"{base_name}_{i}"
        )

        texts.append(
            chunk.page_content
        )

        metadatas.append(
            chunk.metadata
        )


    # -----------------------------------------------------
    # Store
    # -----------------------------------------------------

    print(
        "[CHROMA] Storing vectors...",
        flush=True,
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        "[CHROMA] Document indexed successfully.",
        flush=True,
    )

    print(
        f"[CHROMA] Total vectors now: "
        f"{collection.count()}",
        flush=True,
    )

    return len(chunks)


# =========================================================
# VECTOR SEARCH
# =========================================================


def vector_search(
    query: str,
    top_k: int = 3,
):

    print(
        "[RAG] Creating query embedding...",
        flush=True,
    )

    query_embedding = embedding_model.embed_query(
        query
    )

    print(
        "[RAG] Query embedding created",
        flush=True,
    )

    print(
        f"[RAG] Querying Chroma top_k={top_k}...",
        flush=True,
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    print(
        "[RAG] Chroma query completed",
        flush=True,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    retrieved = []

    for document, metadata in zip(
        documents,
        metadatas,
    ):

        if not document:
            continue

        retrieved.append(
            {
                "text": document,
                "metadata": metadata or {},
            }
        )

    print(
        f"[RAG] Retrieved {len(retrieved)} chunks",
        flush=True,
    )

    return retrieved


# =========================================================
# GEMINI GENERATION
# =========================================================


def generate_answer(
    question: str,
    retrieved_docs: list,
    history=None,
):

    print(
        "[GEMINI] Building prompt...",
        flush=True,
    )

    # -----------------------------------------------------
    # History
    # -----------------------------------------------------

    history_block = ""

    if history:

        lines = []

        for turn in history:

            role = (
                "User"
                if turn["role"] == "user"
                else "Assistant"
            )

            lines.append(
                f"{role}: {turn['content']}"
            )

        history_block = (
            "CONVERSATION HISTORY "
            "(most recent last):\n\n"
            + "\n".join(lines)
            + "\n\n"
        )


    # -----------------------------------------------------
    # RAG context
    # -----------------------------------------------------

    if retrieved_docs:

        context_parts = []

        for i, doc in enumerate(
            retrieved_docs,
            start=1,
        ):

            context_parts.append(
                f"--- Context {i} ---\n\n"
                f"{doc['text']}\n"
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are an assistant AI chatbot for
University of Computer Studies Taungoo.

You are helpful, friendly, and concise.

University information:

Rector:
Dr. Ei Ei Hlaing
(ဒေါက်တာအိအိလှိုင်)

Location:
Kanyoe Village
(ကန်ရိုးကျေးရွာ),
Taungoo, Bago Region.

RULES:

1. Use the conversation history when the
   user refers to previous messages.

2. If the user's question can be answered
   from the document context, prioritize
   the document context.

3. Respond naturally to greetings.

4. You may answer general questions about:
   - Computer science
   - Programming
   - Mathematics
   - Technology
   - AI
   - Science

5. If a question is specifically about the
   uploaded document and the answer cannot
   be found in the retrieved context,
   say that the available document context
   does not contain enough information.

6. Do not invent document-specific facts.

7. Do not include source page references.

{history_block}

DOCUMENT CONTEXT:

{context}

CURRENT USER MESSAGE:

{question}
"""

    else:

        prompt = f"""
You are DocMind AI, a helpful and friendly
assistant with memory of the current
conversation.

{history_block}

Answer the user's message naturally.

You can answer general questions about:

- Computer science
- Programming
- Mathematics
- Technology
- AI
- Science

Be concise, helpful, and accurate.

CURRENT USER MESSAGE:

{question}
"""


    # -----------------------------------------------------
    # Gemini
    # -----------------------------------------------------

    print(
        "[GEMINI] Sending request...",
        flush=True,
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    print(
        "[GEMINI] Response received",
        flush=True,
    )

    if not response:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    answer = getattr(
        response,
        "text",
        None,
    )

    if not answer:
        raise RuntimeError(
            "Gemini response did not contain text."
        )

    return answer.strip()


# =========================================================
# RAG PIPELINE
# =========================================================


def ask(
    question: str,
    has_documents: bool = False,
    history=None,
):

    print(
        "\n"
        + "=" * 60,
        flush=True,
    )

    print(
        f"[ASK] Question: {question}",
        flush=True,
    )

    retrieved = []

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    if has_documents:

        print(
            "[ASK] Documents available",
            flush=True,
        )

        retrieved = vector_search(
            query=question,
            top_k=3,
        )

    else:

        print(
            "[ASK] No documents. "
            "Using general knowledge.",
            flush=True,
        )


    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    answer = generate_answer(
        question=question,
        retrieved_docs=retrieved,
        history=history,
    )

    return answer


# =========================================================
# ROUTE: HOME
# =========================================================


@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# ROUTE: HEALTH CHECK
# =========================================================


@app.route("/health", methods=["GET"])
def health():

    try:

        count = collection.count()

        return jsonify(
            {
                "status": "ok",
                "vectors": count,
            }
        ), 200

    except Exception as e:

        return jsonify(
            {
                "status": "error",
                "error": str(e),
            }
        ), 500


# =========================================================
# ROUTE: UPLOAD
# =========================================================


@app.route(
    "/upload",
    methods=["POST"],
)
def upload():

    print(
        "[UPLOAD] Request received",
        flush=True,
    )

    if "file" not in request.files:

        return jsonify(
            {
                "error":
                "No file part in the request."
            }
        ), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify(
            {
                "error":
                "No file selected."
            }
        ), 400


    if not allowed_file(
        file.filename
    ):

        return jsonify(
            {
                "error":
                "Only PDF files are allowed."
            }
        ), 400


    filename = secure_filename(
        file.filename
    )

    unique_name = (
        f"{uuid.uuid4().hex}_"
        f"{filename}"
    )

    save_path = (
        UPLOAD_FOLDER
        / unique_name
    )


    try:

        file.save(
            str(save_path)
        )

        print(
            f"[UPLOAD] Saved: {save_path}",
            flush=True,
        )

        chunk_count = index_document(
            str(save_path)
        )

        return jsonify(
            {
                "success": True,
                "message":
                    "Document indexed successfully.",
                "filename":
                    filename,
                "chunks":
                    chunk_count,
            }
        ), 200


    except Exception as e:

        print(
            f"[UPLOAD ERROR] {repr(e)}",
            flush=True,
        )

        return jsonify(
            {
                "error": str(e)
            }
        ), 500


# =========================================================
# ROUTE: ASK
# =========================================================


@app.route(
    "/ask",
    methods=["POST"],
)
def ask_question():

    print(
        "\n========== /ask START ==========",
        flush=True,
    )


    # -----------------------------------------------------
    # Parse JSON
    # -----------------------------------------------------

    data = request.get_json(
        silent=True
    )

    print(
        f"[ASK ROUTE] JSON: {data}",
        flush=True,
    )


    if not data or "question" not in data:

        print(
            "[ASK ROUTE] Missing question",
            flush=True,
        )

        return jsonify(
            {
                "error":
                "Missing 'question' in request body."
            }
        ), 400


    question = data["question"]


    if not isinstance(
        question,
        str,
    ):

        return jsonify(
            {
                "error":
                "'question' must be a string."
            }
        ), 400


    question = question.strip()


    if not question:

        return jsonify(
            {
                "error":
                "Question cannot be empty."
            }
        ), 400


    # -----------------------------------------------------
    # Session
    # -----------------------------------------------------

    sid = (
        data.get("session_id")
        or secrets.token_hex(16)
    )


    if not isinstance(
        sid,
        str,
    ):

        sid = str(sid)


    if sid not in chat_histories:

        chat_histories[sid] = deque(
            maxlen=MAX_HISTORY_TURNS
        )


    history = list(
        chat_histories[sid]
    )


    print(
        f"[ASK ROUTE] Session: {sid}",
        flush=True,
    )


    # -----------------------------------------------------
    # Check documents
    # -----------------------------------------------------

    try:

        print(
            "[ASK ROUTE] Checking Chroma...",
            flush=True,
        )

        vector_count = collection.count()

        has_documents = (
            vector_count > 0
        )

        print(
            f"[ASK ROUTE] Chroma vectors: "
            f"{vector_count}",
            flush=True,
        )


    except Exception as e:

        print(
            f"[ASK ROUTE] Chroma error: "
            f"{repr(e)}",
            flush=True,
        )

        return jsonify(
            {
                "error":
                    "Vector database error.",
                "details":
                    str(e),
            }
        ), 500


    # -----------------------------------------------------
    # Ask
    # -----------------------------------------------------

    try:

        print(
            "[ASK ROUTE] Calling ask()...",
            flush=True,
        )

        answer = ask(
            question=question,
            has_documents=has_documents,
            history=history,
        )

        print(
            "[ASK ROUTE] ask() completed",
            flush=True,
        )


        # -------------------------------------------------
        # Save conversation
        # -------------------------------------------------

        chat_histories[sid].append(
            {
                "role": "user",
                "content": question,
            }
        )

        chat_histories[sid].append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


        print(
            "[ASK ROUTE] Returning response",
            flush=True,
        )


        return jsonify(
            {
                "answer": answer,
                "session_id": sid,
            }
        ), 200


    except Exception as e:

        print(
            "\n[ASK ERROR]",
            repr(e),
            flush=True,
        )

        return jsonify(
            {
                "error":
                    str(e)
            }
        ), 500


# =========================================================
# ROUTE: CLEAR HISTORY
# =========================================================


@app.route(
    "/clear-history",
    methods=["POST"],
)
def clear_history():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    sid = data.get(
        "session_id"
    )


    if sid and sid in chat_histories:

        chat_histories[sid].clear()

        print(
            f"[SESSION] Cleared "
            f"{sid[:8]}...",
            flush=True,
        )


    return jsonify(
        {
            "success": True
        }
    ), 200


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "5001",
            )
        ),
        debug=True,
    )