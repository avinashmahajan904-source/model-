import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, abort, send_from_directory
from typing import List, Dict, Any

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── Load .env FIRST so GROQ_API_KEY is available to every route ──
from dotenv import load_dotenv
_env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_file, override=True)

# Confirm key loaded at startup
_groq_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
if _groq_key and _groq_key != "your_new_groq_api_key_here":
    print("✅ GROQ_API_KEY loaded from .env")
else:
    print("⚠️  WARNING: GROQ_API_KEY not set or still placeholder — edit your .env file!")

# Import the ChatChunkSaver from the sibling module
from chat_save import ChatChunkSaver
import requests
from urllib.parse import urlparse

app = Flask(__name__)

# Default path for storing important chat chunks
DEFAULT_CHUNK_FILE = os.path.join(os.path.dirname(__file__), "important_chat_chunks.json")

# Initialize a saver instance (will load existing data if present)
chunk_saver = ChatChunkSaver()
if os.path.exists(DEFAULT_CHUNK_FILE):
    try:
        chunk_saver.load_from_file(DEFAULT_CHUNK_FILE)
    except Exception as e:
        # If loading fails, start with an empty list but log the error
        print(f"⚠️ Failed to load existing chunks: {e}")

# Serve the main UI page at the root URL
@app.route("/", methods=["GET"])
def index():
    """Return the HTML UI for the plant disease analyzer."""
    # The HTML file is expected to be in the same directory as server.py
    return send_from_directory(os.path.dirname(__file__), "plant_disease_analyzer.html")

@app.route("/api/key", methods=["GET"])
def get_api_key():
    """Serve the Groq API key to the frontend securely (only available locally)."""
    key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not key or key == "your_new_groq_api_key_here":
        abort(404, description="GROQ_API_KEY not configured in .env — add your real key!")
    return jsonify({"key": key})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Proxy route: receive image + query from UI, call AI via converter, return result.
    Accept optional RAG control flags (use_rag, rag_top_k) in the JSON payload.
    """
    from converter import analyze_input
    if not request.is_json:
        abort(400, description="Request body must be JSON")
    data = request.get_json()
    result = analyze_input(
        user_query=data.get("custom_query"),
        image_base64=data.get("image_base64"),
        conversation_history=data.get("conversation_history", []),
        use_rag=data.get("use_rag", True),
        rag_top_k=data.get("rag_top_k", 2)
    )
    return jsonify(result)

@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    """Perform live web research search via WebResearchRetriever."""
    from rag_knowledge import retrieve_relevant_knowledge
    if request.method == "POST" and request.is_json:
        payload = request.get_json() or {}
        query = payload.get("query", "")
        top_k = payload.get("top_k", 3)
    else:
        query = request.args.get("query", "")
        top_k = int(request.args.get("top_k", 3))

    if not query:
        abort(400, description="Missing 'query' parameter")

    results = retrieve_relevant_knowledge(query, top_k=top_k)
    return jsonify({"query": query, "count": len(results), "results": results})

@app.route("/api/env-status", methods=["GET"])
def env_status():
    """Return whether the API key, WebResearchRetriever, and vector memory are configured."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    configured = bool(key) and key != "your_new_groq_api_key_here"
    masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
    
    from rag_knowledge import is_web_research_ready
    web_ready = is_web_research_ready()
    
    chroma_dir = Path(__file__).resolve().parent / "chroma_db"
    vector_ready = chroma_dir.exists() and any(chroma_dir.iterdir())
    
    return jsonify({
        "configured": configured,
        "masked_key": masked if configured else "",
        "web_research_ready": web_ready,
        "vector_db_ready": vector_ready or web_ready,
        "storage": "Live Web Research (WebResearchRetriever) + Chroma DB"
    })

# Existing routes follow

@app.route("/chunks", methods=["GET"])
def get_chunks():
    """Return all stored important chat chunks as JSON."""
    return jsonify(chunk_saver.to_list())

@app.route("/chunks", methods=["POST"])
def add_chunks():
    """Accept a list of chat messages and store the ones marked important.

    Expected payload format:
    [{"role": "user", "content": "...", "important": true}, ...]
    """
    if not request.is_json:
        abort(400, description="Request body must be JSON")
    messages: List[Dict[str, Any]] = request.get_json()
    for msg in messages:
        # Guard against malformed entries
        role = msg.get("role")
        content = msg.get("content")
        important = bool(msg.get("important"))
        if role and content:
            chunk_saver.add_message(role=role, content=content, important=important)
    # Persist after adding
    chunk_saver.save_to_file(DEFAULT_CHUNK_FILE)
    return jsonify({"status": "saved", "count": len(chunk_saver.to_list())})

@app.route("/fetch", methods=["GET"])
def fetch_file():
    """Read an arbitrary file from the project directory and return its content.

    Query parameter ``file`` should be a path relative to the current working
    directory (the server's root folder). For security, the path is normalized
    and constrained to stay within the project directory.
    """
    rel_path = request.args.get("file")
    if not rel_path:
        abort(400, description="Missing 'file' query parameter")
    # Resolve to absolute path and ensure it's inside the workspace folder
    base_dir = os.path.abspath(os.path.dirname(__file__))
    target_path = os.path.abspath(os.path.join(base_dir, rel_path))
    if not target_path.startswith(base_dir):
        abort(403, description="Access to the requested path is forbidden")
    if not os.path.isfile(target_path):
        abort(404, description="File not found")
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({"file": rel_path, "content": content})
    except Exception as e:
        abort(500, description=f"Failed to read file: {e}")

@app.route("/fetch_url", methods=["GET"])
def fetch_url():
    """Fetch the content of an external URL safely.

    * ``url`` – required, absolute HTTP/HTTPS URL.
    * Enforces a 5‑second timeout and a configurable maximum response size
      (default 5 MiB). Larger responses are aborted with HTTP 413.
    * Detects the response's charset via ``response.encoding`` and decodes
      accordingly. If decoding fails, the payload is returned Base64‑encoded.
    """
    url = request.args.get("url")
    if not url:
        abort(400, description="Missing 'url' query parameter")
    # Basic URL validation – allow only http/https schemes
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        abort(400, description="Only http and https URLs are allowed")
    # Optional custom limit (in bytes) via query param, fallback to 5 MiB
    try:
        max_bytes = int(request.args.get("max_bytes", 5 * 1024 * 1024))
    except ValueError:
        abort(400, description="Invalid max_bytes value")
    if max_bytes <= 0:
        abort(400, description="max_bytes must be positive")
    try:
        # Stream the response to enforce size limit and avoid loading huge bodies
        resp = requests.get(url, timeout=5, stream=True)
        resp.raise_for_status()
        data = b""
        for chunk in resp.iter_content(chunk_size=8192):
            data += chunk
            if len(data) > max_bytes:
                abort(413, description="Response too large (exceeds max_bytes)")
        # Use the charset reported by the server, falling back to utf‑8
        encoding = resp.encoding if resp.encoding else "utf-8"
        try:
            content = data.decode(encoding)
            return jsonify({"url": url, "content": content, "encoding": encoding})
        except UnicodeDecodeError:
            # Binary or unknown charset – return base64 representation
            import base64
            b64 = base64.b64encode(data).decode("ascii")
            return jsonify({"url": url, "content_base64": b64, "encoding": "base64"})
    except requests.RequestException as e:
        abort(502, description=f"Failed to fetch URL: {e}")

@app.route("/groq_vision", methods=["POST"])
def groq_vision():
    """Forward an image to Groq Vision API and return the result."""
    # Ensure an image file is provided
    if "image" not in request.files:
        abort(400, description="Missing 'image' file in request")
    image_file = request.files["image"]
    # Read image bytes
    image_bytes = image_file.read()
    # Prepare request to Groq Vision
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        abort(500, description="Server misconfiguration: GROQ_API_KEY not set")
    uri = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    import base64
    b64 = base64.b64encode(image_bytes).decode("ascii")
    mime = image_file.mimetype or "image/jpeg"
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]
        }]
    }
    try:
        resp = requests.post(uri, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.HTTPError as e:
        if resp.status_code == 401:
            abort(401, description="Invalid Groq API key")
        abort(resp.status_code, description=f"Groq Vision request failed: {e}")
    except requests.RequestException as e:
        abort(502, description=f"Failed to contact Groq Vision API: {e}")

if __name__ == "__main__":
    try:
        from data import get_vectorstore
        print("⚡ Pre-loading local laptop vector memory (Chroma DB)...")
        get_vectorstore()
        print("✅ Local laptop vector memory ready!")
    except Exception as e:
        print(f"⚠️ Vector DB initialization notice: {e}")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    print(f"🌱 AgroVision server starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

