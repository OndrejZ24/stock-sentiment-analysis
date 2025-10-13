#!/usr/bin/env python3
"""
Test script na to ze mas vsechno rdy. Od Simona s laskou.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        from utils import check_dependencies, detect_tickers_in_text, normalize_text_for_sentiment
        print("✅ utils imported successfully")
    except ImportError as e:
        print(f"❌ utils import failed: {e}")
        return False
    
    return True

def test_utils_functions():
    """Test core utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils import detect_tickers_in_text, normalize_text_for_sentiment
        
        # Test ticker detection
        test_text = "Buy $AAPL and GOOGL now! TSLA to the moon! 🚀"
        ticker_set = {"AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"}
        
        tickers = detect_tickers_in_text(test_text, ticker_set)
        expected_tickers = ["AAPL", "GOOGL", "TSLA"]
        
        if set(tickers) == set(expected_tickers):
            print(f"✅ Ticker detection working: {tickers}")
        else:
            print(f"❌ Ticker detection failed. Expected {expected_tickers}, got {tickers}")
            return False
        
        # Test text normalization
        normalized = normalize_text_for_sentiment(test_text, keep_tickers=True)
        if "$AAPL" in normalized and "GOOGL" in normalized:
            print(f"✅ Text normalization working: '{normalized[:50]}...'")
        else:
            print(f"❌ Text normalization failed: '{normalized}'")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        return False

def test_file_structure():
    """Test that required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "utils.py",
        "preprocessing.ipynb", 
        "data-import.ipynb"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def main():
    """Run all tests"""
    print("Stock Sentiment Analysis - System Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test utility functions 
    if not test_utils_functions():
        all_tests_passed = False
        
    # Test file structure
    if not test_file_structure():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Set up your .env file with database and Reddit API credentials")
        print("2. Run the data-import notebook to fetch Reddit data")
        print("3. Run the preprocessing notebook to clean and enrich the data")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())