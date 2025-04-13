from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# === Load FAISS index and text chunks ===
index = faiss.read_index("/Users/yashdusane/Desktop/demo_1/course_index.faiss")
with open("/Users/yashdusane/Desktop/demo_1/course_texts.pkl", "rb") as f:
    texts = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === FastAPI app ===
app = FastAPI()

class Query(BaseModel):
    question: str

# === Groq API ===
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-8b-8192"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def search_context(query: str, k: int = 10) -> str:
    q_embed = embedder.encode([query])
    _, indices = index.search(q_embed, k)
    return "\n---\n".join(texts[i] for i in indices[0])

@app.post("/chat")
async def chat(query: Query):
    top_chunks = search_context(query.question)

    prompt = f"""
You are a helpful assistant answering questions about university courses and professors using the context below.

Context:
{top_chunks}

Question: {query.question}
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Answer accurately using the context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GROQ_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return {"response": result['choices'][0]['message']['content']}
        except Exception as e:
            return {"error": str(e)}
