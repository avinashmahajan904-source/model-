import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Vector Store Imports (supports both langchain_chroma and langchain_community)
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Paths Configuration ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"

# Local model cache location specified by user
LOCAL_CACHE_BASE = Path(r"C:\Users\nikit\OneDrive\Desktop\AviProgram\GenAI\Document loader")
LOCAL_CACHE_DIR = LOCAL_CACHE_BASE / "models--sentence-transformers--all-MiniLM-L6-v2"

# ── Global Singletons ─────────────────────────────────────────────────────────
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFaceEmbeddings model.
    Prioritizes the local cached model in the user's laptop memory,
    with an automatic fallback to sentence-transformers if the path changes.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    # 1. Check local project directory first (if model was cached in workspace)
    project_snapshots = BASE_DIR / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    if project_snapshots.exists():
        p_dirs = list(project_snapshots.glob("*"))
        if p_dirs:
            print(f"📦 Loading embedding model from workspace cache: {p_dirs[0]}")
            _embedding_model = HuggingFaceEmbeddings(
                model_name=str(p_dirs[0]),
                model_kwargs={"local_files_only": True}
            )
            return _embedding_model

    # 2. Check local snapshot directory on user's laptop
    if LOCAL_CACHE_DIR.exists():
        snapshots_dir = LOCAL_CACHE_DIR / "snapshots"
        if snapshots_dir.exists():
            snapshot_dirs = list(snapshots_dir.glob("*"))
            if snapshot_dirs:
                model_dir = snapshot_dirs[0]
                print(f"📦 Loading embedding model from local laptop cache: {model_dir}")
                _embedding_model = HuggingFaceEmbeddings(
                    model_name=str(model_dir),
                    model_kwargs={"local_files_only": True}
                )
                return _embedding_model

    # 3. Cloud / Server Fallback: auto-downloads and caches on any cloud server or new machine
    print("🌐 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    _embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return _embedding_model


def get_initial_documents() -> List[Document]:
    """
    Builds structured documents for agricultural knowledge base, plant diseases,
    pest control, and crop nutrition to store in laptop memory.
    """
    from rag_knowledge import AGRI_KNOWLEDGE_BASE

    documents: List[Document] = []
    for item in AGRI_KNOWLEDGE_BASE:
        text = (
            f"Crop: {item.get('plane_name', '')}\n"
            f"Disease / Topic: {item.get('disease', '')}\n"
            f"Keywords: {', '.join(item.get('keywords', []))}\n"
            f"Details & Treatment: {item.get('content', '')}"
        )
        metadata = {
            "id": item.get("id", ""),
            "plane_name": item.get("plane_name", ""),
            "disease": item.get("disease", ""),
            "keywords": ", ".join(item.get("keywords", [])),
            "source": "agri_knowledge_base"
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def store_data(
    documents: Optional[List[Document]] = None,
    persist_directory: Optional[Path] = None
) -> Chroma:
    """
    Stores documents into the laptop memory (persistent Chroma vector database on disk).
    """
    global _vectorstore
    save_path = persist_directory or PERSIST_DIR
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    emb = get_embedding_model()

    if documents is None:
        documents = get_initial_documents()

    print(f"💾 Vectorizing and saving {len(documents)} documents to laptop memory at: {save_path}")
    _vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=emb,
        persist_directory=str(save_path)
    )
    print("✅ Successfully stored in laptop memory (Chroma DB)!")
    return _vectorstore


def get_vectorstore(persist_directory: Optional[Path] = None) -> Chroma:
    """
    Retrieves the persistent Chroma vectorstore from laptop storage.
    If it does not exist yet, initializes and populates it.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    save_path = persist_directory or PERSIST_DIR
    save_path = Path(save_path)

    emb = get_embedding_model()

    # Check if existing Chroma DB files exist in the directory
    if save_path.exists() and any(save_path.iterdir()):
        try:
            _vectorstore = Chroma(
                persist_directory=str(save_path),
                embedding_function=emb
            )
            return _vectorstore
        except Exception as e:
            print(f"⚠️ Re-indexing vector database: {e}")

    # Initialize with default documents
    return store_data(persist_directory=save_path)


def query_vectorstore(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Queries the laptop memory vector store using semantic similarity search.
    Returns matched documents with metadata and similarity scores.
    """
    if not query or not query.strip():
        return []

    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(query, k=top_k)
        formatted: List[Dict[str, Any]] = []
        for doc, score in results:
            formatted.append({
                "id": doc.metadata.get("id", "doc"),
                "plane_name": doc.metadata.get("plane_name", ""),
                "disease": doc.metadata.get("disease", ""),
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            })
        return formatted
    except Exception as e:
        print(f"⚠️ Vector search error: {e}")
        return []


def search_relevant_context(query: str, top_k: int = 2) -> str:
    """
    Retrieves relevant context using WebResearchRetriever (live web research),
    falling back to local vector memory if offline.
    """
    try:
        from rag_knowledge import retrieve_relevant_knowledge, format_rag_context
        docs = retrieve_relevant_knowledge(query, top_k=top_k)
        if docs:
            return format_rag_context(docs)
    except Exception:
        pass

    results = query_vectorstore(query, top_k=top_k)
    if not results:
        return ""

    lines = ["\n[स्थानीय मेमोरी से प्राप्त संदर्भ (Retrieved from Local Vector Memory)]:"]
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. {item.get('content', '')}")
    return "\n".join(lines) + "\n"


# ── Direct Run Execution & Verification ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Initializing HuggingFace Embeddings & Chroma Vector Store")
    print("=" * 60)

    # 1. Initialize Embeddings
    embedding_model = get_embedding_model()
    test_vec = embedding_model.embed_query("hello How are you")
    print(f"✅ Embedding created successfully! Dimension: {len(test_vec)}")
    print(f"Sample values: {test_vec[:5]} ...\n")

    # 2. Store documents in laptop memory
    vs = store_data()

    # 3. Test Retrieval
    test_query = "गेहूँ में पीला रतुआ और टमाटर का रोग का इलाज"
    print(f"\n🔍 Testing Vector Search for: '{test_query}'")
    hits = query_vectorstore(test_query, top_k=2)

    for idx, hit in enumerate(hits, 1):
        print(f"\n--- Match #{idx} (Score: {hit['score']:.4f}) ---")
        print(f"Crop: {hit['plane_name']} | Topic: {hit['disease']}")
        print(f"Content:\n{hit['content'][:180]}...")

    print("\n" + "=" * 60)
    print("🎉 All data successfully embedded and stored in laptop memory!")
    print(f"📁 Storage location: {PERSIST_DIR.resolve()}")
    print("=" * 60)