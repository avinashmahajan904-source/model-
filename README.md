# 🌱 AgroVision AI — Plant Disease Diagnosis & Agricultural Advisory

> **AI-Powered Plant Pathology & Farmer Advisory System** with Multi-Turn Conversation Memory, Live Web Research (WebResearchRetriever), Chroma Vector Storage, and Groq Vision & Text LLMs.

---

## 🌟 Key Features

* **🌿 AI Plant Disease Diagnosis**: Upload a photo of an infected leaf or plant, and receive an instant structured diagnosis (crop name, identified disease, symptoms, causes, chemical & organic solutions).
* **🌐 Live Web Research (WebResearchRetriever)**: Uses LangChain's **`WebResearchRetriever`** to dynamically search the web, scrape up-to-date plant pathology literature, split and index relevant articles into a **Chroma vector database** in real-time instead of relying solely on static local data.
* **💬 Multi-Turn Conversation Memory**: Maintains conversation context so farmers and researchers can ask continuous follow-up questions in natural Hindi or English.
* **🎙️ Voice / Audio Input**: Built-in Hindi Speech-to-Text (`SpeechRecognition`) for hands-free queries.
* **⚡ Ultra-Fast Inference**: Uses Groq Cloud LLMs (`qwen/qwen3.8-27b` and `openai/gpt-oss-20b`) for near-instant responses.
* **🚀 Ready for Cloud & Local**: Runs out-of-the-box on your personal computer, or can be deployed to online platforms (Render, Railway, Fly.io, Hugging Face Spaces).

---

## 📁 Project Architecture

```
├── data.py                    # Embedding model & Chroma vector store
├── rag_knowledge.py           # WebResearchRetriever live web research RAG engine
├── converter.py               # Vision & Text inference engine with prompt structuring
├── server.py                  # Flask web backend (serves UI & REST API endpoints)
├── run.py                     # One-command launcher (warms WebResearchRetriever & auto-opens browser)
├── run.bat                    # Windows batch launcher (double-click to run)
├── analyzer.py                # Command-line interface for terminal users
├── plant_disease_analyzer.html# Modern responsive web UI
├── requirements.txt           # Project dependencies
├── Procfile                   # Cloud PaaS entrypoint (Render / Railway)
├── .env.example               # Template for environment configuration
└── .gitignore                 # Protects API keys and local vector caches
```

---

## 🚀 Quickstart: Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/agrovision-ai.git
cd agrovision-ai
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Your Free API Key
1. Get a free Groq API key from [Groq Console](https://console.groq.com/keys).
2. Copy the template file:
```bash
cp .env.example .env
```
3. Open `.env` and paste your key:
```env
GROQ_API_KEY="gsk_your_real_key_here"
```

### 4. Start the Application
Simply run:
```bash
python run.py
```
*(On Windows, you can also double-click `run.bat`)*

Your browser will automatically open to `http://127.0.0.1:5000` with the complete AgroVision interface!

---

## ☁️ How to Deploy Online (Free)

You can host this project online so anyone in the world can access it from their phone or browser.

### Option A: Deploy on Render.com (Recommended)
1. Push this repository to your GitHub account.
2. Sign in to [Render.com](https://render.com) and click **New + > Web Service**.
3. Connect your GitHub repository.
4. Set the following settings:
   * **Runtime**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn server:app --bind 0.0.0.0:$PORT`
5. In **Environment Variables**, add:
   * `GROQ_API_KEY` = your Groq API key
   * `PYTHON_VERSION` = `3.10.12`
6. Click **Deploy Web Service**. Render will give you a public URL (e.g., `https://agrovision.onrender.com`).

### Option B: Deploy on Railway.app
1. Go to [Railway.app](https://railway.app/) and create a **New Project > Deploy from GitHub Repo**.
2. Railway will automatically detect the `Procfile` and `requirements.txt`.
3. Add `GROQ_API_KEY` under the **Variables** tab.
4. Generate a public domain in **Settings > Domains**.

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the interactive web UI. |
| `/api/analyze` | `POST` | Primary analysis endpoint. Accepts `{ "image_base64": "...", "custom_query": "...", "conversation_history": [...] }`. |
| `/api/search` | `POST` | Live web research search via WebResearchRetriever & Chroma DB. Accepts `{ "query": "wheat yellow rust", "top_k": 3 }`. |
| `/api/env-status` | `GET` | Health check returning status of vector database and API key configuration. |

---

## 🔒 Security Best Practices
* **Never commit `.env`**: Your real API key must remain confidential. The repository includes `.gitignore` to prevent secret leakage.
* **Client-side Privacy**: Uploaded photos are processed in memory and never permanently written to public storage.

---

## 📜 License
This project is open-source under the MIT License.
