# etl_api_to_vector.py - Complete ETL from API to Vector DB
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
import os

# ============================================
# CONFIGURATION
# ============================================
JSON_FILE = "latest_news.json"  # Your fetched news file
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_news_data():
    """Load news from JSON file"""
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    df = pd.DataFrame(articles)
    print(f"📊 Loaded {len(df)} articles from {JSON_FILE}")
    
    # Create text field from title and description
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = df["title"] + " " + df["description"]
    
    # Add date field
    df["date"] = pd.to_datetime(df["pubDate"], errors='coerce')
    
    # Filter out empty text
    df = df[df["text"].str.len() > 20]
    print(f"✅ Valid articles after filtering: {len(df)}")
    
    return df

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    # FIX: Handle empty text
    if not text or len(text.strip()) == 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += step
    
    return chunks

def create_chunks(df):
    """Create chunks from all articles"""
    all_chunks = []
    all_metadata = []
    
    for idx, row in df.iterrows():
        text = row["text"]
        if len(text) > 20:
            chunks = chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                # FIX: Safe date handling
                date_value = ""
                if "date" in row and pd.notna(row.get("date")):
                    date_value = str(row["date"])
                
                all_metadata.append({
                    "source": row.get("source_name", "Unknown"),
                    "category": str(row.get("category", ["Unknown"])[0]) if row.get("category") else "Unknown",
                    "date": date_value,
                    "title": row["title"][:100] if row["title"] else "",
                    "article_id": row.get("article_id", ""),
                    "chunk_id": chunk_idx
                })
    
    print(f"📝 Created {len(all_chunks)} chunks from {len(df)} articles")
    return all_chunks, all_metadata

def generate_embeddings(chunks):
    """Generate embeddings using SentenceTransformer"""
    print("🔄 Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"✅ Generated {len(embeddings)} embeddings")
    return embeddings, model

def load_to_chromadb(chunks, embeddings, metadata, db_path="./api_news_db"):
    """Load everything into ChromaDB"""
    print("💾 Loading into ChromaDB...")
    
    # Clear existing DB if needed
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print("   Removed existing database")
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection(name="news_articles")
    
    # Prepare IDs
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            embeddings=embeddings[i:end].tolist(),
            documents=chunks[i:end],
            metadatas=metadata[i:end],
            ids=ids[i:end]
        )
        print(f"   Loaded {end}/{len(chunks)} chunks")
    
    print(f"✅ Loaded {len(chunks)} chunks into ChromaDB")
    return collection

def test_query(collection, query_text, model, n_results=3):
    """Test the RAG system with a query"""
    query_embedding = model.encode([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    print(f"\n🔍 Query: '{query_text}'")
    print(f"📚 Top {n_results} results:\n")
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"{i+1}. {doc[:150]}...")
        print(f"   Source: {meta.get('source', 'Unknown')}")
        print(f"   Category: {meta.get('category', 'Unknown')}")
        print()

# ============================================
# MAIN ETL PIPELINE
# ============================================

def run_etl():
    print("=" * 60)
    print("🚀 ETL PIPELINE: API News to Vector Database")
    print("=" * 60)
    
    start = datetime.now()
    
    # 1. Extract
    print("\n📥 STEP 1: EXTRACT - Loading news data")
    df = load_news_data()
    
    # 2. Transform
    print("\n🔄 STEP 2: TRANSFORM - Chunking text")
    chunks, metadata = create_chunks(df)
    
    print("\n🔄 Generating embeddings")
    embeddings, model = generate_embeddings(chunks)
    
    # 3. Load
    print("\n💾 STEP 3: LOAD - Storing in vector database")
    collection = load_to_chromadb(chunks, embeddings, metadata)
    
    # Summary
    duration = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 60)
    print("🎉 ETL PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Articles processed: {len(df)}")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Vector DB: ./api_news_db")
    print(f"   Time taken: {duration:.2f} seconds")
    print("=" * 60)
    
    return collection, model

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    collection, model = run_etl()
    
    # Test with sample queries
    if collection:
        print("\n" + "=" * 60)
        print("🔍 TESTING RAG SYSTEM")
        print("=" * 60)
        
        test_queries = [
            "What is the latest technology news?",
            "Business and finance updates",
            "Latest news summary"
        ]
        
        for query in test_queries:
            test_query(collection, query, model, n_results=2)