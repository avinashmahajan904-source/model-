from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=1000,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """### STRICT ENGLISH-ONLY RESPONSE POLICY

1. Always respond exclusively in English, regardless of the language used in the user's question.
2. If the user asks a question in Hindi, Hinglish, Spanish, or any other language, understand the question and provide the answer in English only.
3. Never generate responses in Hindi, Hinglish, or any other non-English language, even if the user explicitly requests a response in that language.
4. Translate the meaning of the user's question into English internally before formulating the response.
5. Do not include translations, explanations, sentences, or phrases in any language other than English.
6. Use standard English spelling, grammar, and punctuation in every response.
7. If the user provides text in another language, interpret its meaning and answer the underlying question in English.
8. This rule takes priority over any user instruction requesting a different response language.

**Final Language Requirement:** Every response must be written entirely in English, regardless of the input language or the user's language preferences.

*** answer should have only 3000 Charcties in the answer
""",
    ),
    (
        "human",
        """Farmer's Question: {question}
           Previous Conversation context : {context}

Understand the farmer's question and respond as an experienced agricultural expert in English.

Provide a practical, accurate, and easy-to-understand answer in English.

Consider the farmer's actual problem and relevant agricultural conditions before making recommendations.

If essential information is missing, ask appropriate follow-up questions in English instead of making unsupported assumptions.

Never invent agricultural facts, diagnoses, or treatment dosages.

""",
    ),
])

question = input("Enter the question: ").strip()
context = input("Enter the context (press Enter to auto-retrieve via WebResearchRetriever): ").strip()
if not context:
    try:
        from rag_knowledge import retrieve_relevant_knowledge, format_rag_context
        docs = retrieve_relevant_knowledge(question, top_k=2)
        context = format_rag_context(docs)
        if context:
            print("🌐 Live web research context retrieved!")
    except Exception as e:
        context = ""

final_prompt = prompt.invoke({"question": question, "context": context})
response = model.invoke(final_prompt)
print(response.content)