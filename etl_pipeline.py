# etl_pipeline.py - COMPLETE CORRECTED VERSION
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

DATA_PATH = r"C:\Users\Office\.cache\kagglehub\datasets\rmisra\news-category-dataset\versions\3"

# 1. EXTRACT
df = pd.read_json(f"{DATA_PATH}/News_Category_Dataset_v3.json", lines=True)
print(f"✅ Extracted {len(df)} articles")

# 2. TRANSFORM
df["text"] = df["headline"] + " " + df["short_description"]
df = df[df["text"].notna()]

loader = DataFrameLoader(df, page_content_column="text")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print(f"✅ Embeddings ready")

# Filter complex metadata
cleaned_chunks = filter_complex_metadata(chunks)
print(f"✅ Metadata cleaned")

# 3. LOAD
vectordb = Chroma.from_documents(
    documents=cleaned_chunks,
    embedding=embeddings,
    persist_directory="./news_vector_db"
)
print(f"✅ Loaded into vector database at ./news_vector_db")

print("\n🎉 ETL Pipeline Complete!")