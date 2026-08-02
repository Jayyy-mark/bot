from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai

from dotenv import load_dotenv
import os


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured")


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

CORS(app)


# =========================================================
# GEMINI
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# EMBEDDINGS
# =========================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)


# =========================================================
# HEALTH
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok"
    })


# =========================================================
# ASK GEMINI
# =========================================================

@app.route("/ask", methods=["POST"])
def ask_llm():

    data = request.get_json(silent=True) or {}

    prompt = data.get("prompt")

    if not prompt:

        return jsonify({
            "error": "Missing prompt"
        }), 400

    try:

        print(
            "[BOT SERVER] Gemini request...",
            flush=True
        )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

        answer = response.text.strip()

        print(
            "[BOT SERVER] Gemini response received",
            flush=True
        )

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print(
            f"[BOT SERVER] Gemini error: {repr(e)}",
            flush=True
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# DOCUMENT EMBEDDINGS
# =========================================================

@app.route("/embedding/documents", methods=["POST"])
def get_document_embeddings():

    data = request.get_json(silent=True) or {}

    texts = data.get("texts")

    if not texts:

        return jsonify({
            "error": "Missing texts"
        }), 400

    if not isinstance(texts, list):

        return jsonify({
            "error": "texts must be a list"
        }), 400

    try:

        print(
            f"[BOT SERVER] Embedding "
            f"{len(texts)} documents...",
            flush=True
        )

        embeddings = embedding_model.embed_documents(
            texts
        )

        print(
            "[BOT SERVER] Document embeddings completed",
            flush=True
        )

        return jsonify({
            "embeddings": embeddings
        })

    except Exception as e:

        print(
            f"[BOT SERVER] Embedding error: {repr(e)}",
            flush=True
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# QUERY EMBEDDING
# =========================================================

@app.route("/embeddings/userQuery", methods=["POST"])
def get_query_embedding():

    data = request.get_json(silent=True) or {}

    text = data.get("text")

    if not text:

        return jsonify({
            "error": "Missing text"
        }), 400

    try:

        print(
            "[BOT SERVER] Creating query embedding...",
            flush=True
        )

        embedding = embedding_model.embed_query(
            text
        )

        print(
            "[BOT SERVER] Query embedding completed",
            flush=True
        )

        return jsonify({
            "embedding": embedding
        })

    except Exception as e:

        print(
            f"[BOT SERVER] Query embedding error: "
            f"{repr(e)}",
            flush=True
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )