# tests/test_data_quality_2021_2022.py
import pytest
import pandas as pd
import requests
import os
from datetime import datetime
import time
DATA_PATH = r"C:\Users\Office\.cache\kagglehub\datasets\rmisra\news-category-dataset\versions\3"

# Your NewsAPI key
NEWSAPI_KEY = "1a99bb152f731460ab70429b26298ea9"

def load_filtered_data():
    """Load only 2021 and 2022 data"""
    df = pd.read_json(f"{DATA_PATH}/News_Category_Dataset_v3.json", lines=True)
    df["date_parsed"] = pd.to_datetime(df["date"], errors='coerce')
    
    # KEEP ONLY 2021 and 2022
    df_filtered = df[(df["date_parsed"] >= "2021-01-01") & (df["date_parsed"] <= "2022-12-31")]
    
    print(f"\n📊 Data Filtering Summary:")
    print(f"   Original records: {len(df)}")
    print(f"   Records kept (2021-2022): {len(df_filtered)}")
    print(f"   Records removed: {len(df) - len(df_filtered)}")
    
    return df_filtered


# ============================================
# 1. ACCURACY TEST (2021-2022 data only)
# ============================================
def test_accuracy_against_newsapi():
    """Check accuracy of 2021-2022 news against NewsData.io API"""
    
    df = load_filtered_data()
    sample = df.sample(min(15, len(df)))
    
    verified = 0
    results = []
    
    print("\n🔍 Validating 2021-2022 articles against NewsData.io API...")
    
    for idx, row in sample.iterrows():
        headline = row["headline"]
        year = row["date_parsed"].year
        category = row["category"]
        
        # CHANGED: NewsData.io endpoint
        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": NEWSAPI_KEY,
            "q": headline,
            "language": "en"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # CHANGED: NewsData.io uses 'results' key
                if data.get("results") and len(data.get("results", [])) > 0:
                    verified += 1
                    results.append({"headline": headline, "status": "✅ VERIFIED", "year": year, "category": category})
                else:
                    results.append({"headline": headline, "status": "❌ NOT FOUND", "year": year, "category": category})
            else:
                results.append({"headline": headline, "status": f"⚠️ API ERROR {response.status_code}", "year": year, "category": category})
        except Exception as e:
            results.append({"headline": headline, "status": "⚠️ REQUEST FAILED", "year": year, "category": category})
        
        time.sleep(0.5)  # Respect rate limit
    
    accuracy_rate = (verified / len(sample)) * 100 if len(sample) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 ACCURACY VALIDATION REPORT (2021-2022 Data)")
    print(f"{'='*60}")
    
    for r in results:
        print(f"{r['status']} | {r['year']} | {r['category']} | {r['headline'][:50]}...")
    
    print(f"\n📈 Accuracy rate: {accuracy_rate:.1f}% ({verified}/{len(sample)})")
    
    # Accept 40% accuracy for 2021-2022 data (older news harder to verify)
    assert accuracy_rate > 40, f"Accuracy too low: {accuracy_rate:.1f}%"


# ============================================
# 2. COMPLETENESS TEST
# ============================================

def test_completeness():
    """Check for missing values in 2021-2022 data only"""
    
    df = load_filtered_data()
    
    print(f"\n{'='*60}")
    print(f"📊 COMPLETENESS REPORT (2021-2022 Data)")
    print(f"{'='*60}")
    
    for column in ["headline", "short_description", "category", "date"]:
        missing = df[column].isnull().sum()
        status = "✅" if missing == 0 else "❌"
        print(f"{status} {column}: {missing} missing")
        assert missing == 0, f"Column {column} has {missing} missing values"


# ============================================
# 3. CONSISTENCY TEST
# ============================================

def test_consistency():
    """Check consistency in 2021-2022 data"""
    
    df = load_filtered_data()
    
    print(f"\n{'='*60}")
    print(f"📊 CONSISTENCY REPORT (2021-2022 Data)")
    print(f"{'='*60}")
    
    # Year distribution
    year_counts = df["date_parsed"].dt.year.value_counts().sort_index()
    print(f"📅 Year distribution:")
    for year, count in year_counts.items():
        print(f"   {year}: {count:,} articles ({count/len(df)*100:.1f}%)")
    
    # Date format consistency
    date_format_ok = df["date"].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$').all()
    print(f"{'✅' if date_format_ok else '❌'} Date format: All YYYY-MM-DD")
    
    # Category consistency
    print(f"✅ Categories: {df['category'].nunique()} unique values")
    
    # Duplicate check
    duplicates = df["headline"].duplicated().sum()
    print(f"{'✅' if duplicates < len(df)*0.01 else '⚠️'} Duplicates: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    assert date_format_ok, "Date format inconsistent"


# ============================================
# 4. TIMELINESS TEST (Modified for 2021-2022)
# ============================================

def test_timeliness():
    """Check if 2021-2022 data meets requirements"""
    
    df = load_filtered_data()
    
    print(f"\n{'='*60}")
    print(f"📊 TIMELINESS REPORT (2021-2022 Data)")
    print(f"{'='*60}")
    
    latest_date = df["date_parsed"].max()
    earliest_date = df["date_parsed"].min()
    current_date = datetime.now()
    
    age_days = (current_date - latest_date).days
    
    print(f"📅 Date range: {earliest_date.date()} to {latest_date.date()}")
    print(f"📅 Latest news: {latest_date.date()} ({age_days} days ago)")
    print(f"📅 Years covered: {latest_date.year - earliest_date.year + 1} years")
    
    # Accept 2021-2022 data (even if older, it's intentional)
    print(f"✅ Data range is 2021-2022 (acceptable for this project)")
    
    # Future dates check
    future_dates = df[df["date_parsed"] > current_date]
    print(f"{'✅' if len(future_dates) == 0 else '❌'} Future dates: {len(future_dates)} found")
    
    assert len(future_dates) == 0, f"Found {len(future_dates)} future dates"


# ============================================
# 5. VALIDITY TEST (Modified for 2021-2022)
# ============================================

def test_validity():
    """Check if 2021-2022 data follows rules"""
    
    df = load_filtered_data()
    
    print(f"\n{'='*60}")
    print(f"📊 VALIDITY REPORT (2021-2022 Data)")
    print(f"{'='*60}")
    
    issues = []
    
    # Headline length
    short_headlines = df[df["headline"].str.len() < 10]
    if len(short_headlines) > 0:
        issues.append(f"⚠️ {len(short_headlines)} headlines too short")
    else:
        print(f"✅ All {len(df)} headlines have valid length")
    
    # Description length
    short_desc = df[df["short_description"].str.len() < 20]
    if len(short_desc) > 0:
        issues.append(f"⚠️ {len(short_desc)} descriptions too short")
    else:
        print(f"✅ All {len(df)} descriptions have valid length")
    
    # Category validity (2021-2022 relevant categories)
    expected_categories = {
        "POLITICS", "BUSINESS", "TECH", "ENTERTAINMENT", "SPORTS", 
        "WORLD NEWS", "SCIENCE", "HEALTH", "TRAVEL", "FOOD"
    }
    invalid_cats = set(df["category"].unique()) - expected_categories
    if invalid_cats:
        print(f"ℹ️ Additional categories present: {invalid_cats}")
    else:
        print(f"✅ All categories are standard")
    
    for issue in issues:
        print(issue)
    
    # Validity should pass with minimal issues
    assert len(short_headlines) < len(df) * 0.01, "Too many short headlines"


# ============================================
# RUN ALL TESTS
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA QUALITY VALIDATION - 2021-2022 DATA ONLY")
    print("=" * 60)
    
    try:
        test_completeness()
        test_consistency()
        test_timeliness()
        test_validity()
        
        if NEWSAPI_KEY:
            test_accuracy_against_newsapi()
        else:
            print("\n⚠️ Skipping accuracy test: No NEWSAPI_KEY set")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED (2021-2022 Data)")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise