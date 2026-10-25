import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Ensure .env is always loaded
env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_file, override=True)

# LangChain & WebResearchRetriever imports
try:
    from langchain_community.retrievers.web_research import WebResearchRetriever
except ImportError:
    from langchain.retrievers.web_research import WebResearchRetriever

try:
    from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
except ImportError:
    from langchain.utilities.google_search import GoogleSearchAPIWrapper

from langchain_groq import ChatGroq

# Comprehensive Agricultural & Plant Pathology Knowledge Base (Retained for offline fallback)
AGRI_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "wheat_yellow_rust",
        "keywords": ["wheat", "gehu", "gehun", "yellow rust", "peela ratua", "ratua", "puccinia striiformis"],
        "plane_name": "Wheat (गेहूँ)",
        "disease": "Yellow Rust / Peela Ratua (पीला रतुआ)",
        "content": "गेहूँ में पीला रतुआ (Yellow Rust) एक फफूंदजनित रोग है जो Puccinia striiformis के कारण होता है। इसमें पत्तियों पर समानांतर पीली धारियों के रूप में पाउडर जैसे धब्बे बनते हैं।\nसमाधान: प्रतिरोधी किस्में (जैसे HD 2967, PBW 550) लगाएं। लक्षण दिखने पर प्रोपिकोनाजोल 25% EC (Propiconazole - Tilt) 1 मिली प्रति लीटर पानी में या टेबुकोनाजोल (Tebuconazole) 25.9% EC का छिड़काव 15 दिन के अंतराल पर करें। संतुलित नाइट्रोजन दें।"
    },
    {
        "id": "wheat_karnal_bunt",
        "keywords": ["wheat", "karnal bunt", "gehun", "badbu", "kala dana", "tilletia indica"],
        "plane_name": "Wheat (गेहूँ)",
        "disease": "Karnal Bunt (करनाल बंट)",
        "content": "करनाल बंट कवक टिलेशिया इंडिका से होता है। दानों में सड़ी मछली जैसी गंध आती है और दाने काले चूर्ण में बदल जाते हैं।\nसमाधान: बीजोपचार कार्बेन्डाजिम या थीरम 2.5 ग्राम प्रति किलो बीज से करें। बाली निकलने के समय प्रोपिकोनाजोल (0.1%) का स्प्रे करें।"
    },
    {
        "id": "tomato_early_late_blight",
        "keywords": ["tomato", "tamatar", "blight", "early blight", "late blight", "jhulsa", "patti sukhi"],
        "plane_name": "Tomato (टमाटर)",
        "disease": "Late Blight / Early Blight (अगेती व पछेती झुलसा)",
        "content": "टमाटर में झुलसा रोग (Blight) पत्तियों और तनों पर काले-भूरे धब्बे बनाता है और फल सड़ने लगते हैं। यह फाइटोफ्थोरा या अल्टरनेरिया कवक से होता है।\nसमाधान: मैंकोजेब (Mancozeb) 2.5 ग्राम प्रति लीटर या मेटालैक्सिल + मैंकोजेब (Ridomil Gold) 2 ग्राम प्रति लीटर पानी में घोलकर 10-12 दिन के अंतराल पर छिड़कें। खेत में जलभराव न होने दें।"
    },
    {
        "id": "tomato_leaf_curl",
        "keywords": ["tomato", "leaf curl", "tamatar", "patti mudna", "whitefly", "safed makhi"],
        "plane_name": "Tomato (टमाटर)",
        "disease": "Leaf Curl Virus (पर्ण कुंचन विषाणु)",
        "content": "टमाटर में पत्ती मुड़ना (Leaf Curl) एक वायरस जनित रोग है जो सफेद मक्खी (Whitefly) द्वारा फैलता है। पत्तियां ऊपर की ओर मुड़ जाती हैं और पौधे की वृद्धि रुक जाती है।\nसमाधान: सफेद मक्खी के नियंत्रण के लिए पीले चिपचिपे कार्ड (Yellow Sticky Traps) लगाएं और इमिडाक्लोप्रिड 17.8% SL (0.5 मिली/लीटर) या एसिटामिप्रिड 20% SP (0.5 ग्राम/लीटर) का छिड़काव करें।"
    },
    {
        "id": "apple_scab",
        "keywords": ["apple", "seb", "scab", "apple scab", "venturia inaequalis", "kaale dhabbe"],
        "plane_name": "Apple (सेब)",
        "disease": "Apple Scab (सेब का स्कैब / पपड़ी रोग)",
        "content": "एप्पल स्कैब (Venturia inaequalis) सेब की पत्तियों और फलों पर जैतून-हरे से काले, मखमली धब्बे बनाता है जिससे फल फट जाते हैं।\nसमाधान: गिरे हुए रोगग्रस्त पत्तों को नष्ट करें। कली खिलने (Pink bud) और पंखुड़ी झड़ने के समय कैप्टन 50% WP (2.5 ग्राम/लीटर) या डोडीन (Dodine) 65% WP या हेक्साकोनाजोल 5% EC का छिड़काव करें।"
    },
    {
        "id": "rice_blast_sheath_blight",
        "keywords": ["rice", "dhan", "blast", "sheath blight", "jhulka", "pyricularia"],
        "plane_name": "Rice (धान)",
        "disease": "Rice Blast / शीथ ब्लास्ट (ब्लास्ट रोग)",
        "content": "धान में ब्लास्ट रोग पत्तियों पर आंख या नाव के आकार के धब्बे बनाता है।\nसमाधान: ट्राइसाइक्लाजोल 75% WP (Tricyclazole - Baan) 0.6 ग्राम प्रति लीटर पानी या आइसोप्रोथिओलेन का छिड़काव करें। नाइट्रोजन की अधिक मात्रा न डालें।"
    },
    {
        "id": "cotton_bollworm_whitefly",
        "keywords": ["cotton", "kapas", "whitefly", "pink bollworm", "gulabi sundi", "keeda"],
        "plane_name": "Cotton (कपास)",
        "disease": "Pink Bollworm / Whitefly (गुलाबी सुंडी / सफेद मक्खी)",
        "content": "कपास में गुलाबी सुंडी गूलर को अंदर से खाती है और सफेद मक्खी रस चूसती है।\nसमाधान: फेरोमोन ट्रैप (Pheromone Traps) प्रति एकड़ 8-10 लगाएं। प्रोफेनोफॉस 50% EC (2 मिली/लीटर) या इमामेक्टिन बेंजोएट 5% SG (0.4 ग्राम/लीटर) का छिड़काव करें।"
    },
    {
        "id": "potato_late_blight",
        "keywords": ["potato", "aaloo", "aalu", "late blight", "pacheti jhulsa"],
        "plane_name": "Potato (आलू)",
        "disease": "Late Blight of Potato (आलू का पछेती झुलसा)",
        "content": "आलू की पत्तियों के किनारों पर पानी से भीगे भूरे-काले धब्बे बनते हैं और निचली सतह पर सफेद फफूंद दिखती है।\nसमाधान: साइमोक्सानिल + मैंकोजेब (Curzate) 2.5 ग्राम/लीटर या सिमोक्सानिल या कॉपर ऑक्सीक्लोराइड 50% WP (3 ग्राम/लीटर) का समय पर छिड़काव करें।"
    },
    {
        "id": "general_fertilizer_npk",
        "keywords": ["fertilizer", "khad", "npk", "urea", "dap", "poshan", "mitti", "soil"],
        "plane_name": "Fasal Poshan (फसल पोषण)",
        "disease": "Nutrient Deficiency (पोषक तत्वों की कमी)",
        "content": "फसलों में संतुलित पोषण के लिए NPK (नाइट्रोजन, फास्फोरस, पोटाश) का अनुपात 4:2:1 अनुशंसित है। पीली पत्तियों के लिए यूरिया या 19:19:19 घुलनशील खाद का 1% स्प्रे करें। जिंक की कमी पर जिंक सल्फेट (21%) 5 ग्राम/लीटर का छिड़काव उपयोगी होता है।"
    },
    {
        "id": "pest_management_organic",
        "keywords": ["neem", "jaivik", "organic", "kitnashak", "pesticide", "keeda", "desi"],
        "plane_name": "Jaivik Kheti (जैविक कृषि)",
        "disease": "Organic Pest Control (जैविक कीट नियंत्रण)",
        "content": "प्राकृतिक कीट नियंत्रण के लिए नीम तेल (Neem Oil 1500 to 10000 PPM) 3-5 मिली प्रति लीटर पानी में शैम्पू के साथ मिलाकर छिड़काव करें। ट्राइकोडर्मा विरिडी (Trichoderma viride) 5 ग्राम प्रति किलो बीज उपचार या गोबर खाद में मिलाकर भूमि में दें।"
    }
]


class ResilientWebSearch(GoogleSearchAPIWrapper):
    """
    Search provider compatible with GoogleSearchAPIWrapper for WebResearchRetriever.
    - If GOOGLE_API_KEY and GOOGLE_CSE_ID exist, uses Google Custom Search API.
    - Otherwise, or if Google limits/quotas fail, seamlessly conducts web search
      so live web data is retrieved reliably without requiring paid Google Cloud credentials.
    """
    def __init__(self, **kwargs):
        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        google_cse_id = os.getenv("GOOGLE_CSE_ID", "").strip() or os.getenv("GOOGLE_SEARCH_CX", "").strip()
        if google_api_key and google_cse_id:
            try:
                super().__init__(google_api_key=google_api_key, google_cse_id=google_cse_id, **kwargs)
                self._has_google_creds = True
            except Exception:
                self._has_google_creds = False
        else:
            self._has_google_creds = False

    def _clean_query(self, query: str) -> str:
        # Strip leading numbers or 'Q1:' prefixes generated by search prompt LLM
        q = re.sub(r"^\s*(?:\d+[\.\)]|Q\d+:?)\s*", "", query)
        q = q.strip(" ?\"'")
        # Filter out isolated boolean keywords that cause zero matches in raw search engines
        words = [w for w in q.split() if w.lower() not in ["or", "and"]]
        return " ".join(words) if words else q

    def _web_fallback_search(self, query: str, num_results: int = 2) -> List[Dict[str, str]]:
        import urllib.parse
        import requests
        from bs4 import BeautifulSoup

        clean_q = self._clean_query(query)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        results: List[Dict[str, str]] = []

        # Strategy 1: DuckDuckGo HTML search
        try:
            url = "https://html.duckduckgo.com/html/"
            resp = requests.post(url, headers=headers, data={"q": clean_q}, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", class_="result__url"):
                    href = a.get("href", "")
                    if href and "uddg=" in href:
                        part = href.split("uddg=")[1]
                        actual = urllib.parse.unquote(part.split("&")[0])
                        if actual.startswith("http") and not actual.endswith(".pdf"):
                            results.append({
                                "link": actual,
                                "title": a.get_text().strip(),
                                "snippet": ""
                            })
                            if len(results) >= num_results:
                                break
        except Exception:
            pass

        if results:
            return results

        # Strategy 2: DuckDuckGo Lite search
        try:
            url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote_plus(clean_q)}"
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", class_="result-link"):
                    href = a.get("href", "")
                    if href and href.startswith("http") and not href.endswith(".pdf"):
                        results.append({
                            "link": href,
                            "title": a.get_text().strip(),
                            "snippet": ""
                        })
                        if len(results) >= num_results:
                            break
        except Exception:
            pass

        return results

    def results(self, query: str, num_results: int = 2) -> List[Dict[str, str]]:
        if getattr(self, "_has_google_creds", False):
            try:
                res = super().results(query, num_results)
                if res:
                    return res
            except Exception as e:
                print(f"⚠️ Google API search failed, falling back to open web search: {e}")

        return self._web_fallback_search(query, num_results=num_results)


# Global singleton retriever instance
_retriever_instance: Optional[WebResearchRetriever] = None


def get_web_research_retriever(num_search_results: int = 1) -> Optional[WebResearchRetriever]:
    """
    Initializes and returns the WebResearchRetriever singleton with:
    - Chroma vectorstore for storing web page splits
    - Groq LLM (openai/gpt-oss-20b) for generating search questions
    - Resilient search provider (Google API / live web search fallback)
    """
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    try:
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        if not groq_api_key:
            return None

        # Load embedding model and vectorstore
        from data import get_embedding_model
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        emb = get_embedding_model()
        persist_path = Path(__file__).resolve().parent / "chroma_db"
        vectorstore = Chroma(
            persist_directory=str(persist_path),
            embedding_function=emb,
            collection_name="web_research_cache"
        )

        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=groq_api_key,
            temperature=0.0
        )

        search = ResilientWebSearch()

        _retriever_instance = WebResearchRetriever.from_llm(
            vectorstore=vectorstore,
            llm=llm,
            search=search,
            num_search_results=num_search_results,
            allow_dangerous_requests=True
        )
        return _retriever_instance
    except Exception as e:
        print(f"⚠️ Could not initialize WebResearchRetriever: {e}")
        return None


def is_web_research_ready() -> bool:
    """Checks whether the environment has keys ready for WebResearchRetriever."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(groq_key) and groq_key != "your_new_groq_api_key_here"


def retrieve_relevant_knowledge(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """
    Live Web Research RAG Retriever via WebResearchRetriever:
    1. Generates targeted search queries and fetches real-time agricultural data from the web.
    2. Extracts, cleans, splits, and embeds live web documents into vector memory.
    3. Returns top relevant documents matching the query.
    4. Seamlessly falls back to local knowledge base in case of complete network offline mode.
    """
    if not query or not query.strip():
        return []

    search_query = query.strip()

    # 1. Primary: Retrieve live knowledge from the web using WebResearchRetriever
    try:
        retriever = get_web_research_retriever(num_search_results=1)
        if retriever:
            print(f"🌐 [WebResearchRetriever] Searching web for: '{search_query[:70]}'")
            if hasattr(retriever, "invoke"):
                raw_docs = retriever.invoke(search_query)
            else:
                raw_docs = retriever.get_relevant_documents(search_query)

            if raw_docs:
                results = []
                for i, doc in enumerate(raw_docs[:top_k], 1):
                    source_url = doc.metadata.get("source", "")
                    title = doc.metadata.get("title") or doc.metadata.get("description", "")
                    results.append({
                        "id": f"web_research_{i}",
                        "plane_name": title or source_url or "कृषि अनुसंधान वेब स्रोत",
                        "disease": title or search_query,
                        "content": doc.page_content.strip(),
                        "source": source_url,
                        "title": title,
                        "metadata": doc.metadata
                    })
                if results:
                    print(f"✅ [WebResearchRetriever] Successfully retrieved {len(results)} live web document(s)!")
                    return results
    except Exception as e:
        print(f"⚠️ WebResearchRetriever execution error: {e}")

    # 2. Resilient Fallback: Local knowledge base (offline resilience)
    print("ℹ️ Using fallback knowledge base reference (offline mode)...")
    clean_query = search_query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', clean_query))
    scored_items: List[Tuple[float, Dict[str, Any]]] = []

    for item in AGRI_KNOWLEDGE_BASE:
        score = 0.0
        for kw in item["keywords"]:
            if kw in clean_query:
                score += 5.0
            else:
                kw_tokens = set(kw.split())
                overlap = query_tokens.intersection(kw_tokens)
                if overlap:
                    score += len(overlap) * 2.0

        content_tokens = set(re.findall(r'\b\w+\b', item["content"].lower()))
        content_overlap = len(query_tokens.intersection(content_tokens))
        score += content_overlap * 0.5

        if score > 0:
            scored_items.append((score, item))

    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored_items[:top_k]]


def format_rag_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format retrieved web research knowledge into prompt context."""
    if not retrieved_docs:
        return ""

    lines = ["\n[विश्वसनीय वेब अनुसंधान संदर्भ (Retrieved Web Research Knowledge & Facts)]:"]
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("source") or doc.get("plane_name") or "Web Research"
        subject = doc.get("title") or doc.get("disease") or doc.get("plane_name") or "विषय (Topic)"
        content = doc.get("content", "").strip()
        # Truncate per document if extremely long to maintain prompt efficiency
        if len(content) > 1200:
            content = content[:1200] + "..."
        lines.append(f"{i}. स्रोत: {source} | विषय: {subject}\n   तथ्य: {content}")
    return "\n".join(lines) + "\n"
