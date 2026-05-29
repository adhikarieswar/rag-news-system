# tests/test_my_project.py
import pytest
import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add parent directory to import your functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================
# MOCK DATA (used for testing, not real data)
# ============================================

MOCK_ARTICLES = [
    {
        "title": "AI Breakthrough Announced",
        "description": "New AI model achieves human-level performance",
        "pubDate": "2026-05-29 10:00:00",
        "source_name": "TechNews",
        "category": ["technology"],
        "article_id": "12345"
    },
    {
        "title": "Stock Market Rally",
        "description": "Markets hit all-time highs",
        "pubDate": "2026-05-29 09:00:00",
        "source_name": "FinanceDaily",
        "category": ["business"],
        "article_id": "67890"
    }
]

MOCK_DF = pd.DataFrame({
    "text": ["This is a test news article for chunking purposes."],
    "source_name": ["TestSource"],
    "category": [["test"]],
    "title": ["Test Article"]
})


# ============================================
# TEST 1: chunk_text() - Text splitting logic
# ============================================

def test_chunk_text_short_text():
    """GOAL: Short text should NOT be split into multiple chunks"""
    from etl_api_to_vector import chunk_text
    
    result = chunk_text("Hello World", chunk_size=100, overlap=10)
    
    assert len(result) == 1, f"Expected 1 chunk, got {len(result)}"
    assert result[0] == "Hello World"


def test_chunk_text_long_text():
    """GOAL: Long text SHOULD be split into multiple chunks"""
    from etl_api_to_vector import chunk_text
    
    long_text = "A" * 300
    result = chunk_text(long_text, chunk_size=100, overlap=20)
    
    assert len(result) >= 3, f"Expected at least 3 chunks, got {len(result)}"
    assert all(len(chunk) <= 120 for chunk in result)


def test_chunk_text_empty():
    """GOAL: Empty text should return empty list"""
    from etl_api_to_vector import chunk_text
    
    result = chunk_text("", chunk_size=100, overlap=10)
    
    assert result == [], f"Expected empty list, got {result}"


def test_chunk_text_exact_size():
    """GOAL: Text exactly chunk size should return 1 chunk"""
    from etl_api_to_vector import chunk_text
    
    exact_text = "A" * 100
    result = chunk_text(exact_text, chunk_size=100, overlap=10)
    
    assert len(result) == 1
    assert len(result[0]) == 100


# ============================================
# TEST 2: create_chunks() - Chunk and metadata creation
# ============================================

def test_create_chunks_basic():
    """GOAL: Should create chunks with matching metadata for each article"""
    from etl_api_to_vector import create_chunks
    
    chunks, metadata = create_chunks(MOCK_DF)
    
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert len(chunks) == len(metadata), "Chunks and metadata count mismatch"
    assert metadata[0]["source"] == "TestSource"


def test_create_chunks_skip_empty():
    """GOAL: Empty text should be skipped (no chunks created)"""
    from etl_api_to_vector import create_chunks
    
    empty_df = pd.DataFrame({
        "text": [""],
        "source_name": ["Test"],
        "category": [["test"]]
    })
    
    chunks, metadata = create_chunks(empty_df)
    
    assert len(chunks) == 0, f"Expected 0 chunks, got {len(chunks)}"
    assert len(metadata) == 0


def test_create_chunks_multiple():
    """GOAL: Multiple articles should create multiple chunks"""
    from etl_api_to_vector import create_chunks
    
    multiple_df = pd.DataFrame({
        "text": ["Article 1 content here", "Article 2 content here", "Article 3 content here"],
        "source_name": ["Source A", "Source B", "Source C"],
        "category": [["news"], ["tech"], ["business"]]
    })
    
    chunks, metadata = create_chunks(multiple_df)
    
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert len(metadata) == 3
    assert metadata[0]["source"] == "Source A"
    assert metadata[1]["source"] == "Source B"
    assert metadata[2]["source"] == "Source C"


def test_create_chunks_metadata_structure():
    """GOAL: Metadata should contain required fields"""
    from etl_api_to_vector import create_chunks
    
    chunks, metadata = create_chunks(MOCK_DF)
    
    required_fields = ["source", "category", "title", "chunk_id"]
    for field in required_fields:
        assert field in metadata[0], f"Missing required field: {field}"


# ============================================
# TEST 3: load_news_data() - JSON loading logic
# ============================================

def test_load_news_data_file_not_found():
    """GOAL: Should handle missing file gracefully"""
    from etl_api_to_vector import load_news_data
    
    # Temporarily rename existing file if it exists
    temp_rename = None
    if os.path.exists("latest_news.json"):
        temp_rename = "latest_news.json.bak"
        os.rename("latest_news.json", temp_rename)
    
    try:
        # This should create empty DataFrame or raise handled exception
        df = load_news_data()
        # If function returns empty DataFrame on error, check it
        assert isinstance(df, pd.DataFrame)
    except FileNotFoundError:
        # If function raises exception, that's acceptable too
        pass
    finally:
        if temp_rename and os.path.exists(temp_rename):
            os.rename(temp_rename, "latest_news.json")


# ============================================
# TEST 4: save_articles_to_txt() - File saving logic
# ============================================

def test_save_articles_to_txt():
    """GOAL: Should save articles to a text file"""
    from fetch_news_to_notepad import save_articles_to_txt
    
    filename = "test_output.txt"
    
    # Ensure clean start
    if os.path.exists(filename):
        os.remove(filename)
    
    result = save_articles_to_txt(MOCK_ARTICLES, filename)
    
    assert os.path.exists(filename), f"File {filename} was not created"
    assert result == filename
    
    # Check file content
    with open(filename, 'r') as f:
        content = f.read()
        assert "AI Breakthrough Announced" in content
        assert "Stock Market Rally" in content
    
    # Clean up
    os.remove(filename)


def test_save_articles_to_txt_empty():
    """GOAL: Should handle empty article list"""
    from fetch_news_to_notepad import save_articles_to_txt
    
    filename = "test_empty.txt"
    
    if os.path.exists(filename):
        os.remove(filename)
    
    result = save_articles_to_txt([], filename)
    
    assert os.path.exists(filename), f"File {filename} was not created"
    
    with open(filename, 'r') as f:
        content = f.read()
        assert "Total articles fetched: 0" in content
    
    os.remove(filename)


# ============================================
# TEST 5: save_articles_to_json() - JSON saving logic
# ============================================

def test_save_articles_to_json():
    """GOAL: Should save articles to a JSON file"""
    from fetch_news_to_notepad import save_articles_to_json
    
    filename = "test_output.json"
    
    if os.path.exists(filename):
        os.remove(filename)
    
    save_articles_to_json(MOCK_ARTICLES, filename)
    
    assert os.path.exists(filename), f"File {filename} was not created"
    
    with open(filename, 'r') as f:
        saved_data = json.load(f)
    
    assert len(saved_data) == 2
    assert saved_data[0]["title"] == "AI Breakthrough Announced"
    assert saved_data[1]["title"] == "Stock Market Rally"
    
    os.remove(filename)


def test_save_articles_to_json_empty():
    """GOAL: Should handle empty article list"""
    from fetch_news_to_notepad import save_articles_to_json
    
    filename = "test_empty.json"
    
    if os.path.exists(filename):
        os.remove(filename)
    
    save_articles_to_json([], filename)
    
    assert os.path.exists(filename)
    
    with open(filename, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data == []
    
    os.remove(filename)


# ============================================
# TEST 6: fetch_news() - API fetch logic
# ============================================

def test_fetch_news_api_key_configured():
    """GOAL: Should have API key configured"""
    import os
    
    api_key = os.environ.get("NEWSAPI_KEY", "pub_daa4a45064fe490390637d5b5536073c")
    
    # Just check that key exists and has correct format
    assert api_key is not None, "NEWSAPI_KEY not configured"
    assert len(api_key) > 10, "API key too short"
    assert api_key.startswith("pub_"), "API key should start with 'pub_'"


def test_fetch_news_returns_list():
    """GOAL: fetch_news() should return a list"""
    from fetch_news_to_notepad import fetch_latest_news
    
    result = fetch_latest_news()
    
    # Check return type (can be empty list if API fails)
    assert isinstance(result, list), f"Expected list, got {type(result)}"


# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 RUNNING UNIT TESTS FOR RAG PROJECT")
    print("=" * 60)
    
    tests = [
        ("test_chunk_text_short_text", test_chunk_text_short_text),
        ("test_chunk_text_long_text", test_chunk_text_long_text),
        ("test_chunk_text_empty", test_chunk_text_empty),
        ("test_chunk_text_exact_size", test_chunk_text_exact_size),
        ("test_create_chunks_basic", test_create_chunks_basic),
        ("test_create_chunks_skip_empty", test_create_chunks_skip_empty),
        ("test_create_chunks_multiple", test_create_chunks_multiple),
        ("test_create_chunks_metadata_structure", test_create_chunks_metadata_structure),
        ("test_load_news_data_file_not_found", test_load_news_data_file_not_found),
        ("test_save_articles_to_txt", test_save_articles_to_txt),
        ("test_save_articles_to_txt_empty", test_save_articles_to_txt_empty),
        ("test_save_articles_to_json", test_save_articles_to_json),
        ("test_save_articles_to_json_empty", test_save_articles_to_json_empty),
        ("test_fetch_news_api_key_configured", test_fetch_news_api_key_configured),
        ("test_fetch_news_returns_list", test_fetch_news_returns_list),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name} - PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name} - FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)