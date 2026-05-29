# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load vector DB
client = chromadb.PersistentClient(path="./api_news_db")
collection = client.get_collection("news_articles")
model = SentenceTransformer('all-MiniLM-L6-V2')

class Query(BaseModel):
    question: str
    n_results: int = 5

@app.post("/ask")
def ask(query: Query):
    q_embedding = model.encode([query.question])[0]
    results = collection.query(
        query_embeddings=[q_embedding.tolist()],
        n_results=query.n_results
    )
    return {
        "question": query.question,
        "answers": [
            {
                "content": doc[:300],
                "source": meta.get("source", "Unknown"),
                "category": meta.get("category", "Unknown")
            }
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    }

@app.get("/")
def root():
    return {"message": "RAG News API is running!"}