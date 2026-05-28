# fast_etl.py - EXTREMELY FAST VERSION (5-10 minutes)
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import numpy as np

DATA_PATH = r"C:\Users\Office\.cache\kagglehub\datasets\rmisra\news-category-dataset\versions\3"

print("=" * 50)
print("STEP 1: Loading data")
print("=" * 50)
df = pd.read_json(f"{DATA_PATH}/News_Category_Dataset_v3.json", lines=True)
df["text"] = df["headline"] + " " + df["short_description"]
df = df[df["text"].notna()]
print(f"✅ Loaded {len(df)} articles")

print("\n" + "=" * 50)
print("STEP 2: Creating embeddings (FAST BATCH MODE)")
print("=" * 50)
model = SentenceTransformer('all-MiniLM-L6-V2')
texts = df["text"].tolist()

# This is the fast part - processes in large batches
embeddings = model.encode(texts, batch_size=512, show_progress_bar=True)
print(f"✅ Created {len(embeddings)} embeddings")

print("\n" + "=" * 50)
print("STEP 3: Loading into ChromaDB")
print("=" * 50)
client = chromadb.PersistentClient(path="./news_vector_db")
collection = client.get_or_create_collection(name="news_articles")

# Prepare metadata (simple strings only)
metadatas = []
for _, row in df.iterrows():
    meta = {}
    if "category" in row:
        meta["category"] = str(row["category"])[:100]  # Limit length
    if "date" in row:
        meta["date"] = str(row["date"])[:20]
    metadatas.append(meta)

ids = [f"doc_{i}" for i in range(len(df))]

# Add in large batches
batch_size = 2000
for i in tqdm(range(0, len(df), batch_size), desc="Adding to database"):
    batch_end = min(i + batch_size, len(df))
    collection.add(
        embeddings=embeddings[i:batch_end].tolist(),
        documents=texts[i:batch_end],
        metadatas=metadatas[i:batch_end],
        ids=ids[i:batch_end]
    )

print("\n" + "=" * 50)
print(f"🎉 COMPLETE! Loaded {len(df)} articles")
print("=" * 50)