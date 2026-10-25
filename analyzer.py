import os
import json
import sys
from pathlib import Path
import importlib.util
from dotenv import load_dotenv

# Ensure UTF-8 printing in Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Automatically load environment variables from .env file
env_path = Path(__file__).resolve().with_name(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
if not groq_api_key:
    print("Warning: GROQ_API_KEY .env file mein nahi mili ya set nahi hai.")

converter_path = Path(__file__).resolve().with_name("converter.py")
spec = importlib.util.spec_from_file_location("converter_module", converter_path)
converter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter_module)

# User input: Either image path, text query, or both
image_path = None
user_query = None

if len(sys.argv) > 1:
    arg = sys.argv[1].strip()
    if os.path.exists(arg):
        image_path = arg
    else:
        user_query = " ".join(sys.argv[1:])

if not image_path and not user_query:
    user_input = input("Image path ya Krishi sawal (doubt) likhiye: ").strip()
    if os.path.exists(user_input):
        image_path = user_input
    else:
        user_query = user_input

result = converter_module.analyze_input(
    user_query=user_query,
    image_path=image_path,
    api_key=groq_api_key,
)

print(f"Connected using path: {converter_path}")

if result.get("type") == "paragraph":
    print("\n[कृषि सलाह - विस्तृत पैराग्राफ (Hindi Paragraph Answer)]:\n")
    print(result.get("content"))
else:
    print("\n[पौधा एवं रोग निदान - JSON Output]:\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
