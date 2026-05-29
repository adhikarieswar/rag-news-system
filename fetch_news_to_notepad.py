# fetch_news_to_notepad.py
import requests
import json
from datetime import datetime
import os

# Your NewsData.io API key
API_KEY = "pub_daa4a45064fe490390637d5b5536073c"

def fetch_latest_news():
    """Fetch the latest news from NewsData.io API (last 48 hours)"""
    url = "https://newsdata.io/api/1/latest"
    params = {
        "apikey": API_KEY,
        "language": "en"
        # size parameter removed - defaults to 10 for free tier
    }
    
    print("📡 Fetching latest news from NewsData.io API...")
    print(f"   URL: {url}")
    print(f"   Parameters: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                articles = data.get("results", [])
                print(f"✅ Successfully fetched {len(articles)} articles.")
                
                # Print date range if available
                if articles:
                    dates = [a.get('pubDate', '') for a in articles if a.get('pubDate')]
                    if dates:
                        print(f"📅 Date range: {min(dates)} to {max(dates)}")
                return articles
            else:
                print(f"❌ API returned error: {data.get('message', 'Unknown error')}")
                return []
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return []

def save_articles_to_txt(articles, filename="latest_news.txt"):
    """Save articles to a formatted text file"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"LATEST NEWS - Fetched on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total articles fetched: {len(articles)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, article in enumerate(articles, 1):
            f.write(f"[{i}] TITLE: {article.get('title', 'No title')}\n")
            f.write(f"    DESCRIPTION: {article.get('description', 'No description')}\n")
            f.write(f"    PUBLISHED: {article.get('pubDate', 'Unknown date')}\n")
            f.write(f"    SOURCE: {article.get('source_name', 'Unknown source')}\n")
            f.write(f"    CATEGORY: {article.get('category', ['Unknown'])[0] if article.get('category') else 'Unknown'}\n")
            f.write("-" * 60 + "\n\n")
    
    print(f"✅ Saved {len(articles)} articles to {filename}")
    return filename

def save_articles_to_json(articles, filename="latest_news.json"):
    """Save articles to JSON file for backup"""
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved JSON backup to {filename}")

def open_in_notepad(filename):
    """Open the text file in Notepad"""
    try:
        os.system(f"notepad {filename}")
        print(f"📝 Opened {filename} in Notepad")
    except Exception as e:
        print(f"⚠️ Could not open Notepad automatically: {e}")
        print(f"   Manually open {filename}")

def main():
    print("=" * 60)
    print("FETCHING NEWS FROM NEWSDATA.IO API")
    print("=" * 60)
    print()
    
    # Fetch news
    articles = fetch_latest_news()
    
    if articles:
        # Save to files
        txt_file = save_articles_to_txt(articles, "latest_news.txt")
        save_articles_to_json(articles, "latest_news.json")
        
        print()
        print("=" * 60)
        print("✅ COMPLETE!")
        print("=" * 60)
        print(f"📄 Text file: {txt_file}")
        print(f"📦 JSON backup: latest_news.json")
        
        # Open in Notepad
        open_in_notepad(txt_file)
    else:
        print()
        print("=" * 60)
        print("❌ FAILED: No articles fetched")
        print("=" * 60)
        print("\nPossible issues:")
        print("1. Check your API key")
        print("2. Check internet connection")
        print("3. Free tier only returns last 48 hours of news")
        print("4. Rate limit: 30 credits per 15 minutes")

if __name__ == "__main__":
    main()