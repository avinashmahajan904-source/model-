import re
import json
from typing import Dict, Any

def extract_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Safely extracts valid JSON from AI generation even if wrapped in markdown
    fences, conversational text, or prefixes.
    """
    if not raw_text or not raw_text.strip():
        return {
            "plane_name": "अज्ञात पौधा",
            "disease_inshowing_image_": "पहचान नहीं हो सकी",
            "problem": "छवि में लक्षण स्पष्ट नहीं हैं।",
            "reason": "छवि धुंधली या पौधे से संबंधित नहीं हो सकती है।",
            "solution": "कृपया पौधे या पत्ती की स्पष्ट और नज़दीक से ली गई फोटो अपलोड करें।",
            "other_points": ["अच्छी रोशनी में फोटो लें", "पत्ती के प्रभावित हिस्से को केंद्र में रखें"]
        }

    cleaned = raw_text.strip()
    
    # 1. Remove ```json and ``` code blocks if present
    if "```" in cleaned:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            # Strip outermost backticks
            cleaned = re.sub(r'^```(?:json)?', '', cleaned).strip()
            cleaned = re.sub(r'```$', '', cleaned).strip()

    # 2. Try direct JSON parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Search for first { and last }
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = cleaned[first_brace:last_brace+1]
        try:
            return json.loads(json_candidate)
        except Exception:
            pass

    # 4. Fallback structured response if model produced plain text
    return {
        "plane_name": "पौधा विश्लेषण",
        "disease_inshowing_image_": "निरीक्षण परिणाम",
        "problem": raw_text[:200] if len(raw_text) > 200 else raw_text,
        "reason": "मॉडल द्वारा प्राप्त विश्लेषण।",
        "solution": "सलाह के अनुसार उचित उपचार व सावधानी अपनाएं।",
        "other_points": ["नियमित रूप से फसल की जांच करें"]
    }
