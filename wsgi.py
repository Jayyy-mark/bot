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
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


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

load_dotenv()


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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set."
    )


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# PATHS
# =========================================================

# app.root_path is safe here because app has already been created.
#
# Render:
# /opt/render/project/src/uploads
# /opt/render/project/src/chroma_db
#
# Local:
# D:\Internship\bot\uploads
# D:\Internship\bot\chroma_db

BASE_DIR = Path(app.root_path)

UPLOAD_FOLDER = BASE_DIR / "uploads"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

CHROMA_DB_PATH.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 70, flush=True)
print("[PATH] BASE_DIR =", BASE_DIR, flush=True)
print("[PATH] UPLOAD_FOLDER =", UPLOAD_FOLDER, flush=True)
print("[PATH] CHROMA_DB_PATH =", CHROMA_DB_PATH, flush=True)
print("[PATH] UPLOAD EXISTS =", UPLOAD_FOLDER.exists(), flush=True)
print("[PATH] CHROMA EXISTS =", CHROMA_DB_PATH.exists(), flush=True)
print("=" * 70, flush=True)


# =========================================================
# FLASK CONFIG
# =========================================================

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# =========================================================
# CORS
# =========================================================

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": [
                "GET",
                "POST",
                "OPTIONS"
            ],
            "allow_headers": [
                "Content-Type"
            ]
        }
    }
)


# =========================================================
# CONSTANTS
# =========================================================

ALLOWED_EXTENSIONS = {"pdf"}

MAX_HISTORY_TURNS = 20


# =========================================================
# CHAT MEMORY
# =========================================================

# session_id -> deque
#
# Example:
#
# {
#     "abc123": [
#         {"role": "user", "content": "hello"},
#         {"role": "assistant", "content": "Hi!"},
#     ]
# }

chat_histories: dict[str, deque] = {}


# =========================================================
# GEMINI CLIENT
# =========================================================

print("[GEMINI] Creating Gemini client...", flush=True)

client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("[GEMINI] Gemini client created", flush=True)


# =========================================================
# EMBEDDING MODEL
# =========================================================

print(
    "[EMBEDDING] Creating Gemini embedding model...",
    flush=True
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)

print(
    "[EMBEDDING] Embedding model created",
    flush=True
)


# =========================================================
# CHROMA INITIALIZATION
# =========================================================

print("=" * 70, flush=True)
print("[CHROMA] Initializing Chroma...", flush=True)
print(
    f"[CHROMA] Path = {CHROMA_DB_PATH}",
    flush=True
)

try:

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    print(
        "[CHROMA] PersistentClient created",
        flush=True
    )

    collection = chroma_client.get_or_create_collection(
        name="documents"
    )

    print(
        "[CHROMA] Collection opened",
        flush=True
    )

    # Test ONCE during startup.
    startup_count = collection.count()

    print(
        f"[CHROMA] Startup vector count = {startup_count}",
        flush=True
    )

    print(
        "[CHROMA] Initialization successful",
        flush=True
    )

except Exception as e:

    print(
        "[CHROMA] INITIALIZATION FAILED",
        flush=True
    )

    print(
        repr(e),
        flush=True
    )

    raise

print("=" * 70, flush=True)


# =========================================================
# HELPERS
# =========================================================

def allowed_file(filename: str) -> bool:

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# PDF LOADING
# =========================================================

def load_pdf(pdf_path: str):

    print(
        f"[PDF] Opening: {pdf_path}",
        flush=True
    )

    reader = PdfReader(pdf_path)

    documents = []

    total_pages = len(reader.pages)

    print(
        f"[PDF] Total pages: {total_pages}",
        flush=True
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text()

        except Exception as e:

            print(
                f"[PDF] Page {page_number} failed: {repr(e)}",
                flush=True
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
                    "source": os.path.basename(pdf_path),
                    "page": page_number
                }
            )
        )

    print(
        f"[PDF] Pages with text: {len(documents)}",
        flush=True
    )

    return documents


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(documents):

    print(
        "[CHUNKING] Starting...",
        flush=True
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"[CHUNKING] Created {len(chunks)} chunks",
        flush=True
    )

    return chunks


# =========================================================
# EMBEDD DOCUMENTS
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
        flush=True
    )

    embeddings = embedding_model.embed_documents(
        texts
    )

    print(
        "[EMBEDDING] Embeddings completed",
        flush=True
    )

    return embeddings


# =========================================================
# CLEAR VECTOR DATABASE
# =========================================================

def clear_collection():

    print(
        "[CHROMA] Clearing existing vectors...",
        flush=True
    )

    try:

        existing = collection.get(
            include=[]
        )

        existing_ids = existing.get(
            "ids",
            []
        )

        print(
            f"[CHROMA] Existing IDs: {len(existing_ids)}",
            flush=True
        )

        if existing_ids:

            collection.delete(
                ids=existing_ids
            )

            print(
                "[CHROMA] Existing vectors deleted",
                flush=True
            )

        else:

            print(
                "[CHROMA] Nothing to delete",
                flush=True
            )

    except Exception as e:

        print(
            f"[CHROMA] Clear failed: {repr(e)}",
            flush=True
        )

        raise


# =========================================================
# INDEX DOCUMENT
# =========================================================

def index_document(pdf_path: str):

    print(
        "\n========== INDEX DOCUMENT ==========",
        flush=True
    )

    # -----------------------------------------------------
    # Clear old document
    # -----------------------------------------------------

    clear_collection()

    # -----------------------------------------------------
    # Load PDF
    # -----------------------------------------------------

    documents = load_pdf(
        pdf_path
    )

    if not documents:

        raise ValueError(
            "No readable text was found in the PDF."
        )

    # -----------------------------------------------------
    # Create chunks
    # -----------------------------------------------------

    chunks = create_chunks(
        documents
    )

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )

    # -----------------------------------------------------
    # Embeddings
    # -----------------------------------------------------

    embeddings = embed_documents(
        chunks
    )

    if len(embeddings) != len(chunks):

        raise RuntimeError(
            "Embedding count does not match "
            "chunk count."
        )

    # -----------------------------------------------------
    # Prepare Chroma data
    # -----------------------------------------------------

    ids = []
    texts = []
    metadatas = []

    base_name = os.path.basename(
        pdf_path
    )

    for i, chunk in enumerate(chunks):

        ids.append(
            f"{base_name}_{i}"
        )

        texts.append(
            chunk.page_content
        )

        metadata = dict(
            chunk.metadata
        )

        metadata["source"] = base_name

        metadatas.append(
            metadata
        )

    # -----------------------------------------------------
    # Store
    # -----------------------------------------------------

    print(
        "[CHROMA] Storing vectors...",
        flush=True
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        "[CHROMA] Vectors stored successfully",
        flush=True
    )

    print(
        f"[CHROMA] Indexed chunks = {len(chunks)}",
        flush=True
    )

    print(
        "========== INDEX COMPLETE ==========\n",
        flush=True
    )

    return len(chunks)


# =========================================================
# VECTOR SEARCH
# =========================================================

def vector_search(
    query: str,
    top_k: int = 3
):

    print(
        "[RAG] Creating query embedding...",
        flush=True
    )

    query_embedding = embedding_model.embed_query(
        query
    )

    print(
        "[RAG] Query embedding created",
        flush=True
    )

    print(
        f"[RAG] Searching Chroma top_k={top_k}...",
        flush=True
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print(
        "[RAG] Chroma search completed",
        flush=True
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    retrieved = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        if not document:
            continue

        retrieved.append(
            {
                "text": document,
                "metadata": metadata or {}
            }
        )

    print(
        f"[RAG] Retrieved {len(retrieved)} chunks",
        flush=True
    )

    return retrieved


# =========================================================
# SIMPLE QUESTION CLASSIFICATION
# =========================================================

def should_use_rag(question: str) -> bool:

    q = question.lower().strip()

    # -----------------------------------------------------
    # Greetings
    # -----------------------------------------------------

    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "မင်္ဂလာပါ"
    }

    if q in greetings:

        print(
            "[RAG] Greeting detected -> no RAG",
            flush=True
        )

        return False

    # -----------------------------------------------------
    # Very short conversational messages
    # -----------------------------------------------------

    conversational = {
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "bye",
        "goodbye"
    }

    if q in conversational:

        print(
            "[RAG] Conversation detected -> no RAG",
            flush=True
        )

        return False

    return True


# =========================================================
# GEMINI GENERATION
# =========================================================

def generate_answer(
    question: str,
    retrieved_docs: list,
    history=None
):

    print(
        "[GEMINI] Building prompt...",
        flush=True
    )

    # -----------------------------------------------------
    # Conversation history
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
            start=1
        ):

            context_parts.append(
                f"--- Context {i} ---\n"
                f"{doc['text']}\n"
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are an AI assistant for
University of Computer Studies Taungoo.

You are helpful, friendly, concise,
and accurate.

UNIVERSITY INFORMATION:

Rector:
Dr. Ei Ei Hlaing
(ဒေါက်တာအိအိလှိုင်)

Location:
Kanyoe Village
(ကန်ရိုးကျေးရွာ),
Taungoo, Bago Region.

RULES:

1. Use conversation history to understand
   references to previous messages.

2. If the user's question is answered by
   the document context, prioritize the
   document context.

3. Do not invent facts from the document.

4. If the question is specifically about
   the uploaded document and the supplied
   context does not contain the answer,
   say that the available document context
   does not contain enough information.

5. General questions about programming,
   computer science, mathematics,
   technology, AI, and science may be
   answered using general knowledge.

6. Respond naturally to greetings and
   normal conversation.

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
AI assistant.

You have memory of the current conversation.

{history_block}

Answer the user's message naturally.

You may answer questions about:

- Computer science
- Programming
- Mathematics
- Technology
- AI
- Science
- General knowledge

Be helpful, concise, and accurate.

CURRENT USER MESSAGE:

{question}
"""

    # -----------------------------------------------------
    # Gemini request
    # -----------------------------------------------------

    print(
        "[GEMINI] Sending request...",
        flush=True
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    print(
        "[GEMINI] Response received",
        flush=True
    )

    if not response:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    answer = getattr(
        response,
        "text",
        None
    )

    if not answer:

        raise RuntimeError(
            "Gemini response did not contain text."
        )

    return answer.strip()


# =========================================================
# ASK PIPELINE
# =========================================================

def ask(
    question: str,
    history=None
):

    print(
        "\n"
        + "=" * 60,
        flush=True
    )

    print(
        f"[ASK] Question: {question}",
        flush=True
    )

    retrieved = []

    # -----------------------------------------------------
    # Decide whether RAG is needed
    # -----------------------------------------------------

    use_rag = should_use_rag(
        question
    )

    if use_rag:

        print(
            "[ASK] RAG required",
            flush=True
        )

        # IMPORTANT:
        #
        # We don't call collection.count()
        # here.
        #
        # Instead, directly try the search.
        #
        # If there are no documents, Chroma will
        # return an empty result or raise an error
        # that we can handle.

        try:

            retrieved = vector_search(
                query=question,
                top_k=3
            )

        except Exception as e:

            print(
                f"[RAG] Search failed: {repr(e)}",
                flush=True
            )

            # Don't crash the entire chatbot.
            #
            # Fall back to Gemini without RAG.

            retrieved = []

    else:

        print(
            "[ASK] RAG not required",
            flush=True
        )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    answer = generate_answer(
        question=question,
        retrieved_docs=retrieved,
        history=history
    )

    return answer


# =========================================================
# HOME
# =========================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "status": "ok"
        }
    ), 200


# =========================================================
# CHROMA HEALTH
# =========================================================

@app.route(
    "/health/chroma",
    methods=["GET"]
)
def chroma_health():

    try:

        count = collection.count()

        return jsonify(
            {
                "status": "ok",
                "vectors": count,
                "path": str(CHROMA_DB_PATH)
            }
        ), 200

    except Exception as e:

        return jsonify(
            {
                "status": "error",
                "error": str(e),
                "path": str(CHROMA_DB_PATH)
            }
        ), 500


# =========================================================
# UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    print(
        "\n========== UPLOAD START ==========",
        flush=True
    )

    try:

        # -------------------------------------------------
        # Check file
        # -------------------------------------------------

        if "file" not in request.files:

            return jsonify(
                {
                    "error":
                        "No file part in request."
                }
            ), 400

        file = request.files["file"]

        if not file.filename:

            return jsonify(
                {
                    "error":
                        "No file selected."
                }
            ), 400

        # -------------------------------------------------
        # Validate extension
        # -------------------------------------------------

        if not allowed_file(
            file.filename
        ):

            return jsonify(
                {
                    "error":
                        "Only PDF files are allowed."
                }
            ), 400

        # -------------------------------------------------
        # Secure filename
        # -------------------------------------------------

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

        print(
            f"[UPLOAD] Saving to: {save_path}",
            flush=True
        )

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        file.save(
            str(save_path)
        )

        print(
            "[UPLOAD] File saved",
            flush=True
        )

        # -------------------------------------------------
        # Index
        # -------------------------------------------------

        chunk_count = index_document(
            str(save_path)
        )

        print(
            "[UPLOAD] Indexing complete",
            flush=True
        )

        print(
            "========== UPLOAD COMPLETE ==========\n",
            flush=True
        )

        return jsonify(
            {
                "success": True,
                "message":
                    "Document indexed successfully.",
                "filename":
                    filename,
                "chunks":
                    chunk_count
            }
        ), 200

    except Exception as e:

        print(
            "[UPLOAD ERROR]",
            flush=True
        )

        print(
            repr(e),
            flush=True
        )

        return jsonify(
            {
                "error": str(e)
            }
        ), 500


# =========================================================
# ASK
# =========================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask_question():

    print(
        "\n========== /ask START ==========",
        flush=True
    )

    try:

        # -------------------------------------------------
        # JSON
        # -------------------------------------------------

        data = request.get_json(
            silent=True
        )

        print(
            f"[ASK ROUTE] JSON = {data}",
            flush=True
        )

        if not data:

            return jsonify(
                {
                    "error":
                        "Request body must be JSON."
                }
            ), 400

        if "question" not in data:

            return jsonify(
                {
                    "error":
                        "Missing 'question'."
                }
            ), 400

        question = data["question"]

        if not isinstance(
            question,
            str
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

        # -------------------------------------------------
        # Session
        # -------------------------------------------------

        sid = (
            data.get("session_id")
            or secrets.token_hex(16)
        )

        sid = str(sid)

        if sid not in chat_histories:

            chat_histories[sid] = deque(
                maxlen=MAX_HISTORY_TURNS
            )

        history = list(
            chat_histories[sid]
        )

        print(
            f"[ASK ROUTE] Session = {sid}",
            flush=True
        )

        print(
            f"[ASK ROUTE] History messages = "
            f"{len(history)}",
            flush=True
        )

        # -------------------------------------------------
        # IMPORTANT
        #
        # NO collection.count() HERE
        # -------------------------------------------------

        print(
            "[ASK ROUTE] Starting ask()...",
            flush=True
        )

        answer = ask(
            question=question,
            history=history
        )

        print(
            "[ASK ROUTE] ask() completed",
            flush=True
        )

        # -------------------------------------------------
        # Save conversation
        # -------------------------------------------------

        chat_histories[sid].append(
            {
                "role": "user",
                "content": question
            }
        )

        chat_histories[sid].append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        print(
            f"[ASK ROUTE] History now contains "
            f"{len(chat_histories[sid])} messages",
            flush=True
        )

        print(
            "[ASK ROUTE] Returning response",
            flush=True
        )

        print(
            "========== /ask END ==========\n",
            flush=True
        )

        return jsonify(
            {
                "answer": answer,
                "session_id": sid
            }
        ), 200

    except Exception as e:

        print(
            "\n========== /ask ERROR ==========",
            flush=True
        )

        print(
            repr(e),
            flush=True
        )

        print(
            "================================\n",
            flush=True
        )

        return jsonify(
            {
                "error": str(e)
            }
        ), 500


# =========================================================
# CLEAR CHAT HISTORY
# =========================================================

@app.route(
    "/clear-history",
    methods=["POST"]
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

    if sid:

        sid = str(sid)

        if sid in chat_histories:

            chat_histories[sid].clear()

            print(
                f"[SESSION] Cleared {sid[:8]}...",
                flush=True
            )

    return jsonify(
        {
            "success": True
        }
    ), 200


# =========================================================
# DEBUG ROUTE
# =========================================================

@app.route(
    "/debug/paths",
    methods=["GET"]
)
def debug_paths():

    return jsonify(
        {
            "base_dir": str(BASE_DIR),
            "upload_folder": str(UPLOAD_FOLDER),
            "chroma_db": str(CHROMA_DB_PATH),
            "upload_exists": UPLOAD_FOLDER.exists(),
            "chroma_exists": CHROMA_DB_PATH.exists()
        }
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5001"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )