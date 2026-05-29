# app.py - PRODUCTION READY VERSION
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="RAG News System",
    description="Retrieval Augmented Generation for News",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# INITIALIZATION
# ============================================

logger.info("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-V2')
logger.info("✅ Model loaded")

logger.info("Connecting to ChromaDB...")
client_db = chromadb.PersistentClient(path="./news_vector_db")
collection = client_db.get_collection(name="news_articles")
logger.info(f"✅ ChromaDB connected: {collection.count()} documents")

# Get Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY not set! LLM responses will not work.")
else:
    groq = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq client initialized")

# ============================================
# PYDANTIC MODELS
# ============================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User question")
    n_results: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    processing_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    documents_count: int
    model: str
    timestamp: str

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "message": "RAG News System is running!",
        "docs": "/docs",
        "health": "/health",
        "version": "2.0.0"
    }

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        documents_count=collection.count(),
        model="all-MiniLM-L6-v2",
        timestamp=datetime.now().isoformat()
    )

@app.post("/ask", response_model=QueryResponse)
def ask(query: QueryRequest):
    """Ask a question and get an answer based on news articles"""
    
    start_time = time.time()
    
    try:
        # 1. Embed the question
        q_embedding = model.encode([query.question])[0]
        
        # 2. Search vector DB
        results = collection.query(
            query_embeddings=[q_embedding.tolist()],
            n_results=query.n_results
        )
        
        # 3. Build context from retrieved documents
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context = "\n\n---\n\n".join([
            f"Source: {meta.get('source', 'Unknown')}\nCategory: {meta.get('category', 'Unknown')}\nContent: {doc}"
            for doc, meta in zip(documents, metadatas)
        ])
        
        # 4. Prepare sources for response
        sources = [
            {
                "content": doc[:300],
                "source": meta.get('source', 'Unknown'),
                "category": meta.get('category', 'Unknown'),
                "date": meta.get('date', 'Unknown')
            }
            for doc, meta in zip(documents, metadatas)
        ]
        
        # 5. Get LLM answer (only if API key exists)
        if GROQ_API_KEY:
            response = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful news assistant. Answer based ONLY on the provided news articles. If the answer cannot be found, say 'I don't have information about that.'"},
                    {"role": "user", "content": f"Based on these news articles:\n\n{context}\n\nQuestion: {query.question}"}
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content
        else:
            answer = "GROQ_API_KEY not configured. Please add your API key."
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            question=query.question,
            answer=answer,
            sources=sources,
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def stats():
    """System statistics"""
    return {
        "documents_count": collection.count(),
        "model": "all-MiniLM-L6-V2",
        "embedding_dimension": 384,
        "groq_configured": bool(GROQ_API_KEY)
    }