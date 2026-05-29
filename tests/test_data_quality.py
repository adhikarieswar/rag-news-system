# tests/test_data_quality.py
import pytest
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import time

DATA_PATH = r"C:\Users\Office\.cache\kagglehub\datasets\rmisra\news-category-dataset\versions\3"

# Your NewsAPI key
NEWSAPI_KEY = "1a99bb152f731460ab70429b26298ea9"

# ============================================
# 1. ACCURACY TEST (Compare against NewsAPI)
# ============================================

def test_accuracy_against_newsapi():
    """Check if news articles are accurate by comparing with NewsAPI"""
    
    df = pd.read_json(f"{DATA_PATH}/News_Category_Dataset_v3.json", lines=True)
    
    # Test 15 random articles (to stay within free tier limits)
    sample = df.sample(min(15, len(df)))
    
    verified = 0
    results = []
    
    print("\n🔍 Validating articles against NewsAPI...")
    
    for idx, row in sample.iterrows():
        headline = row["headline"]
        category = row["category"]
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": headline,
            "apiKey": NEWSAPI_KEY,
            "pageSize": 1,
            "sortBy": "relevancy"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("totalResults", 0) > 0:
                    verified += 1
                    results.append({"headline": headline, "status": "✅ VERIFIED", "category": category})
                    print(f"   ✅ Verified: {headline[:50]}...")
                else:
                    results.append({"headline": headline, "status": "❌ NOT FOUND", "category": category})
                    print(f"   ❌ Not found: {headline[:50]}...")
            else:
                results.append({"headline": headline, "status": f"⚠️ API ERROR {response.status_code}", "category": category})
        except Exception as e:
            results.append({"headline": headline, "status": f"⚠️ REQUEST FAILED", "category": category})
            print(f"   ⚠️ Failed: {headline[:50]}...")
        
        time.sleep(0.5)  # Respect rate limit
    
    accuracy_rate = (verified / len(sample)) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 ACCURACY VALIDATION REPORT")
    print(f"{'='*60}")
    
    for r in results:
        print(f"{r['status']} | {r['category']} | {r['headline'][:60]}...")
    
    print(f"\n📈 Accuracy rate: {accuracy_rate:.1f}% ({verified}/{len(sample)})")
    
    # Accept 50% accuracy for free API (real news may not match exactly)
    assert accuracy_rate > 50, f"Accuracy too low: {accuracy_rate:.1f}%"


# ============================================
# 2. COMPLETENESS TEST
# ============================================

def test_completeness():
    """Check for missing values in all critical fields"""
    
    df = pd.read_json(f"{DATA_PATH}/News_Category_Dataset_v3.json", lines=True)
    
    print(f"\n{'='*60}")
    print(f"📊 COMPLETENESS REPORT")
    print(f"{'='*60}")
    
    total_rows = len(df)
    completeness_results = []
    
    for column in ["headline", "short_description", "category", "date"]:
        missing = df[column].isnull().sum()
        missing_pct = (missing / total_rows) * 100
        status = "PASS" if missing == 0 else "FAIL"
        completeness_results.append({"column": column, "missing": missing, "pct": missing_pct, "status": status})
        
        icon = "✅" if missing == 0 else "❌"
        print(f"{icon} {column}: {missing} missing ({missing_pct:.2f}%)")
        assert missing == 0, f"Column {column} has {missing} missing values"
    
    return completeness_results


# ============================================
# 3. CONSISTENCY TEST
# ============================================

def test_consistency():
    """Check for consistent data formats across all records"""
    
    df = pd.read_json(f"{DATA_PATH}/News_Category_Dataset_v3.json", lines=True)
    
    print(f"\n{'='*60}")
    print(f"📊 CONSISTENCY REPORT")
    print(f"{'='*60}")
    
    # 1. Date format consistency
    date_format_consistent = df["date"].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$').all()
    print(f"{'✅' if date_format_consistent else '❌'} Date format: All dates in YYYY-MM-DD format")
    
    # 2. Category name consistency (no case variations)
    categories = df["category"].unique()
    category_count = len(categories)
    print(f"✅ Categories: {category_count} unique categories")
    
    # 3. Check for duplicate headlines
    duplicates = df["headline"].duplicated().sum()
    duplicate_pct = (duplicates / len(df)) * 100
    print(f"{'✅' if duplicate_pct < 1 else '⚠️'} Duplicates: {duplicates} ({duplicate_pct:.2f}%)")
    
    # 4. Check category distribution consistency
    category_counts = df["category"].value_counts()
    top_category = category_counts.index[0]
    top_pct = (category_counts.iloc[0] / len(df)) * 100
    print(f"📊 Top category: {top_category} ({top_pct:.1f}% of data)")
    
    assert date_format_consistent, "Date format inconsistent"
    assert duplicate_pct < 5, f"Too many duplicates: {duplicate_pct:.2f}%"


# ============================================
# 4. TIMELINESS TEST (Current Data)
# ============================================

def test_timeliness():
    """Check if data is current and relevant"""
    
    df = pd.read_json(f"{DATA_PATH}/News_Category_Dataset_v3.json", lines=True)
    
    print(f"\n{'='*60}")
    print(f"📊 TIMELINESS REPORT")
    print(f"{'='*60}")
    
    # Convert to datetime
    df["date_parsed"] = pd.to_datetime(df["date"], errors='coerce')
    valid_dates = df[df["date_parsed"].notna()]
    
    if len(valid_dates) == 0:
        pytest.fail("No valid dates found")
    
    latest_date = valid_dates["date_parsed"].max()
    earliest_date = valid_dates["date_parsed"].min()
    current_date = datetime.now()
    
    age_days = (current_date - latest_date).days
    age_years = age_days / 365
    
    print(f"📅 Date range: {earliest_date.date()} to {latest_date.date()}")
    print(f"📅 Latest news: {latest_date.date()} ({age_days} days ago)")
    print(f"📅 Total timespan: {(latest_date - earliest_date).days} days")
    print(f"📅 Years covered: {age_years:.1f} years")
    
    # Check if data is too old (news should be within last 3 years)
    is_recent = age_days < 1095  # 3 years
    print(f"{'✅' if is_recent else '⚠️'} Data recency: {'Current' if is_recent else f'Stale ({age_days} days old)'}")
    
    # Check for future dates
    future_dates = df[df["date_parsed"] > current_date]
    has_future = len(future_dates) == 0
    print(f"{'✅' if has_future else '❌'} Future dates: {len(future_dates)} found")
    
    # Check data coverage across years
    years = df["date_parsed"].dt.year.dropna().unique()
    year_count = len(years)
    print(f"✅ Year coverage: {year_count} years ({sorted(years)[0]} to {sorted(years)[-1]})")
    
    assert age_days < 1095, f"Data is stale: {age_days} days old"
    assert has_future, f"Found {len(future_dates)} future dates"


# ============================================
# 5. VALIDITY TEST (Data follows rules)
# ============================================

def test_validity():
    """Check if data follows defined business rules"""
    
    df = pd.read_json(f"{DATA_PATH}/News_Category_Dataset_v3.json", lines=True)
    
    print(f"\n{'='*60}")
    print(f"📊 VALIDITY REPORT")
    print(f"{'='*60}")
    
    issues = []
    passes = []
    
    # 1. Headline length validity (minimum 10 characters)
    short_headlines = df[df["headline"].str.len() < 10]
    if len(short_headlines) > 0:
        issues.append(f"⚠️ {len(short_headlines)} headlines too short (<10 chars)")
    else:
        passes.append(f"✅ All {len(df)} headlines have valid length")
    
    # 2. Description validity (minimum 20 characters)
    short_desc = df[df["short_description"].str.len() < 20]
    if len(short_desc) > 0:
        issues.append(f"⚠️ {len(short_desc)} descriptions too short (<20 chars)")
    else:
        passes.append(f"✅ All {len(df)} descriptions have valid length")
    
    # 3. Category validity (must be in expected list)
    expected_categories = {
        "POLITICS", "BUSINESS", "TECH", "ENTERTAINMENT", "SPORTS", 
        "WORLD NEWS", "SCIENCE", "HEALTH", "TRAVEL", "FOOD",
        "STYLE", "PARENTING", "QUEER VOICES", "WEDDINGS", "COMEDY",
        "ARTS", "GREEN", "BLACK VOICES", "LATINO VOICES"
    }
    invalid_cats = set(df["category"].unique()) - expected_categories
    if invalid_cats:
        issues.append(f"⚠️ Invalid categories found: {invalid_cats}")
    else:
        passes.append(f"✅ All categories are valid")
    
    # 4. Date validity (must be parseable)
    invalid_dates = []
    for date_str in df["date"].dropna().sample(min(1000, len(df))):
        try:
            pd.to_datetime(date_str)
        except:
            invalid_dates.append(date_str)
    
    if invalid_dates:
        issues.append(f"⚠️ {len(invalid_dates)} invalid date formats in sample")
    else:
        passes.append(f"✅ Date formats are valid")
    
    # 5. Text readability (letter ratio > 0.5)
    sample = df.sample(min(500, len(df)))
    low_readability = 0
    for _, row in sample.iterrows():
        text = str(row["headline"]) + " " + str(row["short_description"])
        if len(text) > 0:
            letter_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
            if letter_ratio < 0.5:
                low_readability += 1
    
    readability_pct = (1 - low_readability/len(sample)) * 100
    if readability_pct < 90:
        issues.append(f"⚠️ Readability: {readability_pct:.1f}% text readable")
    else:
        passes.append(f"✅ Readability: {readability_pct:.1f}% text readable")
    
    # Report results
    for p in passes:
        print(p)
    for i in issues:
        print(i)
    
    # Validity checks should pass with minimal issues
    assert len(invalid_cats) == 0, f"Invalid categories found: {invalid_cats}"
    assert readability_pct > 80, f"Readability too low: {readability_pct:.1f}%"


# ============================================
# RUN ALL TESTS
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA QUALITY VALIDATION - 5 DIMENSIONS")
    print("=" * 60)
    
    try:
        # Run all tests
        test_completeness()
        test_consistency()
        test_timeliness()
        test_validity()
        
        # Accuracy test (requires API key)
        if NEWSAPI_KEY:
            test_accuracy_against_newsapi()
        else:
            print("\n⚠️ Skipping accuracy test: No NEWSAPI_KEY set")
            print("   Get free key from https://newsapi.org")
        
        print("\n" + "=" * 60)
        print("✅ ALL DATA QUALITY TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise