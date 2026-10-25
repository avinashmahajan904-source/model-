import os
import sys
import time
import webbrowser
import threading
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load .env
env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_file, override=True)

HOST = "127.0.0.1"
PORT = 5000
UI_URL = f"http://{HOST}:{PORT}"


def open_browser_delayed(url: str, delay: float = 1.2):
    """Wait for server to start and open UI automatically in default browser."""
    time.sleep(delay)
    print(f"🌐 Opening AgroVision AI in your web browser: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"⚠️ Could not automatically open browser: {e}")
        print(f"👉 Please open manually: {url}")


def main():
    print("=" * 65)
    print("🌱 AGROVISION AI & PLANT PATHOLOGY ASSISTANT")
    print("   Powered by Live WebResearchRetriever, Chroma DB & Groq LLM")
    print("=" * 65)

    # 1. Check API Key
    groq_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not groq_key or groq_key == "your_new_groq_api_key_here":
        print("⚠️  WARNING: GROQ_API_KEY is not configured in .env!")
        print("   Please edit your .env file and add your key: GROQ_API_KEY=gsk_...")
    else:
        masked = f"{groq_key[:8]}...{groq_key[-4:]}" if len(groq_key) > 12 else "***"
        print(f"🔑 GROQ_API_KEY: Configured ({masked})")

    # 2. Check / Initialize WebResearchRetriever & Vector Memory
    chroma_dir = Path(__file__).resolve().parent / "chroma_db"
    print(f"\n🌐 Initializing Live WebResearchRetriever & Vector Memory...")
    try:
        from rag_knowledge import get_web_research_retriever
        retriever = get_web_research_retriever()
        if retriever:
            print("✅ WebResearchRetriever Ready! (Live Web Search + Chroma DB)")
        else:
            print("ℹ️ WebResearchRetriever initialized in fallback mode.")
    except Exception as e:
        print(f"⚠️ WebResearchRetriever initialization notice: {e}")

    # 3. Import and launch Flask Server
    print(f"\n🚀 Starting AgroVision AI Server on: {UI_URL}")
    print("   Press Ctrl+C anytime to stop the server.\n")

    # Open browser in a separate background thread
    threading.Thread(target=open_browser_delayed, args=(UI_URL, 1.2), daemon=True).start()

    from server import app
    try:
        app.run(host=HOST, port=PORT, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 AgroVision server stopped. Goodbye!")


if __name__ == "__main__":
    main()
