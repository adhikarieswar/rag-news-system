from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

app = FastAPI()

# Initialize (runs once when server starts)
model = SentenceTransformer('all-MiniLM-L6-V2')
client_db = chromadb.PersistentClient(path="./news_vector_db")
collection = client_db.get_collection(name="news_articles")
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    # Embed question
    q_embedding = model.encode([query.question])[0]
    
    # Search vector DB
    results = collection.query(
        query_embeddings=[q_embedding.tolist()],
        n_results=5
    )
    
    # Build context
    context = "\n\n".join(results['documents'][0])
    
    # Get LLM answer
    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer based ONLY on provided news articles."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.question}"}
        ]
    )
    
    return {"answer": response.choices[0].message.content}

@app.get("/")
def root():
    return {"message": "RAG News System is running!"}