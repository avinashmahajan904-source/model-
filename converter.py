import os
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from dotenv import load_dotenv

# Ensure .env is always reliably located and loaded
env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_file, override=True)

from rag_knowledge import retrieve_relevant_knowledge, format_rag_context
from json_parser_utils import extract_and_parse_json

def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_image_mime(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".png"):
        return "image/png"
    if path_lower.endswith(".webp"):
        return "image/webp"
    if path_lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"

def analyze_input(
    user_query: Optional[str] = None,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    use_rag: bool = True,
    rag_top_k: int = 2
) -> Dict[str, Any]:
    """
    RAG-Augmented Dual-Mode Agricultural Advisory:
    - Vision model analyzes image and outputs JSON (parsed reliably without server-side validation aborts).
    - Text doubts answer in detailed Hindi paragraph.
    """
    if not api_key:
        load_dotenv(dotenv_path=env_file, override=True)
    api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")

    if not api_key:
        return {
            "type": "error",
            "plane_name": "Unknown",
            "disease_inshowing_image_": "API Key Missing",
            "problem": "GROQ_API_KEY .env file mein nahi mili.",
            "reason": "Kripya .env file check karein.",
            "solution": ".env file mein GROQ_API_KEY=gsk_... add karein.",
            "other_points": []
        }

    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    b64_data = ""
    mime_type = "image/jpeg"
    if image_base64:
        b64_data = image_base64
        if "data:" in b64_data and ";base64," in b64_data:
            mime_type = b64_data.split(";base64,")[0].replace("data:", "")
            b64_data = b64_data.split(";base64,")[1]
    elif image_path and os.path.exists(image_path):
        b64_data = encode_image_to_base64(image_path)
        mime_type = get_image_mime(image_path)

    has_image = bool(b64_data)

    # Web Research RAG Retrieval via WebResearchRetriever
    if use_rag:
        search_query = user_query or (image_path if isinstance(image_path, str) else "")
        retrieved_docs = retrieve_relevant_knowledge(search_query, top_k=rag_top_k)
        rag_context = format_rag_context(retrieved_docs)
    else:
        retrieved_docs = []
        rag_context = ""

    # Format previous conversation history
    history_messages: List[Dict[str, str]] = []
    if conversation_history:
        for turn in conversation_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content and role in ["user", "assistant"]:
                history_messages.append({"role": role, "content": str(content)})

    # ══════════════════════════════════════════════════════════════════════════════
    # CASE 1: ONLY TEXT (Agriculture doubt) -> Fast text model (openai/gpt-oss-20b)
    # ══════════════════════════════════════════════════════════════════════════════
    if not has_image:
        text_model = model or "qwen/qwen3.8-27b"
        system_instruction = f"""Aap ek anubhavi krishi visheshagya aur gyanwan sahayak (expert advisor and intelligent assistant) hain.
Aap user ke sawal ka sahi, vistrit aur to-the-point jawab dein.
Chahe sawal kheti, fasal, dawai, mausam, vigyan ya kisi bhi samanya vishay (general knowledge) se juda ho, upalabdh web research sandarbh ka upyog karke satik uttar dein.
Pichli batchit (conversation history) ko dhyan mein rakh kar hi aage ka jawab dein.
{rag_context}
Nirdesh:
1. Pichli batchit ke sandarbh ko samajhkar lagatar judaa hua jawab dein.
2. Agar sawal kheti se juda hai to dawai, khad ya upchar ki spasht jankari dein; agar koi anya vishay hai to spasht aur satik jankari dein.
3. Hamesha saral, spasht aur to-the-point paragraph mein uttar dein.
4. Koi anavashyak JSON ya code block mat banayein, seedha paragraph likhein.
"""
        messages = [{"role": "system", "content": system_instruction}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_query or "Kheti ke baare mein sujhav dijiye"})

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "model": text_model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 600
                },
                timeout=60
            )
            response.raise_for_status()
            paragraph_text = (response.json()["choices"][0]["message"]["content"] or "").strip()
            return {
                "type": "paragraph",
                "content": paragraph_text,
                "rag_used": bool(retrieved_docs)
            }
        except Exception as e:
            return {
                "type": "error",
                "content": f"Error: {str(e)}"
            }

    # ══════════════════════════════════════════════════════════════════════════════
    # CASE 2: IMAGE PROVIDED -> Vision model (qwen/qwen3.8-27b)
    # ══════════════════════════════════════════════════════════════════════════════
    vision_model = model or "qwen/qwen3.8-27b"
    
    json_prompt = f"""Ye image ek paudhe (plant) ya patti ki hai. Iska dhyan se vishleshan (analysis) karo aur samasya ka pata lagao.
{f'Kisan dwara pucha gaya sawal / lakshan: "{user_query}"' if user_query else ''}

{rag_context}
HAMESHA PURE HINDI (हिंदी / Hinglish / Hindi) MEIN HI JAWAB DO.

ZAROORI NIRDESH: Apna pura uttar keval aur keval ek valid JSON object ke roop mein dein.
Structure bilkul is prakar hona chahiye:
{{
  "plane_name": "<paudhe ya fasal ka naam jaise Apple, Tomato, Rice, Aam, etc.>",
  "disease_inshowing_image_": "<image mein dikh rahi bimari ya rog ka naam>",
  "problem": "<ek line mein kya samasya ya lakshan dikh raha hai>",
  "reason": "<yeh bimari/samasya kyun ho rahi hai>",
  "solution": "<isko kaise theek karein ya kaunsi dawai/upchar use karein>",
  "other_points": ["<sujhav 1>", "<sujhav 2>", "<sujhav 3>"]
}}
"""

    image_content = [
        {"type": "text", "text": json_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64_data}"
            }
        }
    ]

    # Build messages: include recent history then the image prompt
    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": image_content})

    try:
        # NOTE: Omitting response_format={'type':'json_object'} prevents the Groq
        # "Failed to validate JSON (json_validate_failed)" 400 error when model outputs reasoning or markdown fences.
        response = requests.post(
            api_url,
            headers=headers,
            json={
                "model": vision_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 512
            },
            timeout=60
        )
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"] or ""
        parsed = extract_and_parse_json(raw_content)
        parsed["type"] = "json"
        parsed["rag_used"] = bool(retrieved_docs)
        return parsed

    except Exception as e:
        return {
            "type": "json",
            "plane_name": "अज्ञात पौधा",
            "disease_inshowing_image_": "त्रुटि",
            "problem": f"Vision API Error: {str(e)}",
            "reason": "छवि के विश्लेषण में समस्या आई।",
            "solution": "कृपया पुनः प्रयास करें।",
            "other_points": []
        }

# Backward compatibility alias
def analyze_image(prompt_text: str, image_path: Optional[str] = None, image_base64: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
    return analyze_input(user_query=prompt_text, image_path=image_path, image_base64=image_base64, api_key=api_key, model=model)
