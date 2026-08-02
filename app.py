import os
import sys
import uuid
import secrets
from collections import deque

# ── Force UTF-8 output on Windows (fixes Myanmar / Unicode charmap errors) ──
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

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
# RERANKER
# =========================================================

# from flashrank import Ranker, RerankRequest


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# GOOGLE GENAI CLIENT
# =========================================================

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# LANGCHAIN GEMINI EMBEDDING
# =========================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", google_api_key=GEMINI_API_KEY
)


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(name="documents")


# =========================================================
# RERANKER
# =========================================================

# ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./reranker_cache")


# =========================================================
# HELPERS
# =========================================================


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================================================
# PDF LOADING
# =========================================================


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text, metadata={"source": pdf_path, "page": page_number}
            )
        )

    return documents


# =========================================================
# CHUNKING
# =========================================================


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


# =========================================================
# EMBEDDING
# =========================================================


def embed_documents(chunks):
    texts = [chunk.page_content for chunk in chunks]
    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.embed_documents(texts)
    return embeddings


# =========================================================
# INDEX DOCUMENT
# Clears existing vector DB, then re-indexes the new file.
# =========================================================


def index_document(pdf_path):
    global collection

    # -------------------------------------------------
    # Delete all existing vectors (reset the collection)
    # -------------------------------------------------
    existing_ids = collection.get(include=[])["ids"]
    if existing_ids:
        print(f"Deleting {len(existing_ids)} existing vectors...")
        collection.delete(ids=existing_ids)

    print("\nLoading PDF...")
    documents = load_pdf(pdf_path)
    print(f"Pages loaded: {len(documents)}")

    print("\nCreating chunks...")
    chunks = create_chunks(documents)
    print(f"Chunks created: {len(chunks)}")

    embeddings = embed_documents(chunks)

    ids = []
    documents_text = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{os.path.basename(pdf_path)}_{i}")
        documents_text.append(chunk.page_content)
        metadatas.append(chunk.metadata)

    collection.upsert(
        ids=ids, documents=documents_text, embeddings=embeddings, metadatas=metadatas
    )

    print("\nDocument indexed successfully.")
    return len(chunks)


# =========================================================
# VECTOR RETRIEVAL
# =========================================================


def vector_search(query, top_k=10):
    print("\nCreating query embedding...")
    query_embedding = embedding_model.embed_query(query)

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved = []
    for document, metadata in zip(documents, metadatas):
        retrieved.append({"text": document, "metadata": metadata})

    return retrieved


# =========================================================
# RERANK
# =========================================================


# def rerank_documents(query, documents, top_k=3):
#     if not documents:
#         return []

#     passages = [{"id": str(i), "text": doc["text"]} for i, doc in enumerate(documents)]

#     rerank_request = RerankRequest(query=query, passages=passages)
#     results = ranker.rerank(rerank_request)

#     reranked = []
#     for result in results[:top_k]:
#         index = int(result["id"])
#         original = documents[index]
#         reranked.append(
#             {
#                 "text": original["text"],
#                 "metadata": original["metadata"],
#                 "score": result.get("score", 0),
#             }
#         )

#     return reranked


# =========================================================
# GEMINI GENERATION
# =========================================================


def generate_answer(question, retrieved_docs, history=None):
    """
    Dual-mode answering with optional conversation history.
    history: list of {"role": "user"|"assistant", "content": str}
    """

    # ── Build conversation history block ──────────────────
    history_block = ""
    if history:
        lines = []
        for turn in history:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {turn['content']}")
        history_block = (
            "CONVERSATION HISTORY (most recent last):\n\n" + "\n".join(lines) + "\n\n"
        )

    if retrieved_docs:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_parts.append(f"--- Context {i} ---\n\n{doc['text']}\n")
        context = "\n".join(context_parts)

        prompt = f"""
You are an assistant ai chat bot for university of computer studies taungoo, a helpful and friendly assistant with memory of the current conversation.

You have access to a user-uploaded document context AND your general knowledge.

Rector of the university is Dr. Ei Ei Hlaing (ဒေါက်တာအိအိလှိုင်)

University Location: Kanyoe Village (ကန်ရိုးကျေးရွာ) , Taungoo, Bago Region
RULES:

1. Use the CONVERSATION HISTORY to understand context from previous turns.
   If the user refers to something mentioned earlier (e.g. "explain more",
   "what about that?"), use the history to understand what they mean.

2. If the user's question is directly answerable from the DOCUMENT CONTEXT,
   answer using the document. Keep the answer clear and concise.

3. Respond warmly to greetings.

4. Answer general topics (CS, programming, math, technology, AI, science)
   using your own knowledge. Do NOT say "not in the document".

5. Only say you don't have information if the question is clearly specific
   to the uploaded document but the answer is missing from the context.

6. Do NOT include source page references.

{history_block}DOCUMENT CONTEXT (use when relevant):

{context}

CURRENT USER MESSAGE:

{question}
"""
    else:
        prompt = f"""
You are DocMind AI, a helpful and friendly assistant with memory of the current conversation.

Answer the user's message naturally.

- Use the CONVERSATION HISTORY to understand context from previous turns.
- Respond warmly to greetings.
- Answer computer science, programming, mathematics, technology, AI, and
  general knowledge questions clearly and concisely.
- Be friendly, helpful, and accurate.

{history_block}CURRENT USER MESSAGE:

{question}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", contents=prompt
    )

    return response.text


# =========================================================
# RAG PIPELINE
# =========================================================


def ask(question, has_documents=True, history=None):
    print("\n" + "=" * 60)
    print("QUESTION:")
    print(question)

    if has_documents:
        retrieved = vector_search(query=question, top_k=3)
        print(f"\nVector retrieval: {len(retrieved)} chunks")

        # reranked = rerank_documents(query=question, documents=retrieved, top_k=3)
        # print(f"After reranking: {len(reranked)} chunks")
    # else:
    #     reranked = []
    #     print("No document loaded — using general knowledge.")

    answer = generate_answer(question, retrieved, history=history)
    return answer


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
# Allow any origin — session state is carried in the request body, not cookies.
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

# ── In-memory conversation histories (session_id -> deque of turns) ──────────
# Each turn: {"role": "user"|"assistant", "content": str}
# Lost on server restart — intentionally non-persistent.
MAX_HISTORY_TURNS = 20  # keep last 20 turns per session
chat_histories: dict[str, deque] = {}


@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# ROUTE: /upload
# Accepts a PDF, resets the vector DB, re-indexes it.
# =========================================================


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        chunk_count = index_document(save_path)
        return (
            jsonify(
                {
                    "success": True,
                    "message": f"Document indexed successfully with {chunk_count} chunks.",
                    "filename": filename,
                    "chunks": chunk_count,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# ROUTE: /ask
# Accepts a JSON body { "question": "..." }
# =========================================================


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json(silent=True)

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    question = data["question"].strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    # ── Session identity (cross-origin safe — carried in request body, not cookie) ──
    # The caller provides their own session_id (or we generate a fresh one).
    # The server returns session_id in every response so the caller can store it
    # (e.g. localStorage) and resend it on the next request.
    sid = data.get("session_id") or secrets.token_hex(16)

    if sid not in chat_histories:
        chat_histories[sid] = deque(maxlen=MAX_HISTORY_TURNS)

    history = list(chat_histories[sid])  # snapshot for this request

    # ── RAG or general knowledge ──────────────────────────────────────────────
    existing_ids = collection.get(include=[])["ids"]
    has_documents = len(existing_ids) > 0

    try:
        answer = ask(question, has_documents=has_documents, history=history)

        # ── Append this turn to history ───────────────────────────────────────
        chat_histories[sid].append({"role": "user", "content": question})
        chat_histories[sid].append({"role": "assistant", "content": answer})

        # Return session_id so the caller can persist and resend it
        return jsonify({"answer": answer, "session_id": sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# ROUTE: /clear-history
# Clears the in-memory conversation history for this session.
# =========================================================


@app.route("/clear-history", methods=["POST"])
def clear_history():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if sid and sid in chat_histories:
        chat_histories[sid].clear()
        print(f"Cleared history for session {sid[:8]}...")
    return jsonify({"success": True}), 200


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, port=5001)
