#!/usr/bin/env python3
"""
Test script to verify NLP functionality works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.text_preprocessing import TextPreprocessor

def test_text_preprocessing():
    """Test the text preprocessing functionality"""
    print("Testing text preprocessing...")
    
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "I have experience with machine learning and deep learning frameworks!",
        "Python programming is essential for data science projects.",
        "I'm not sure about this technology stack, it seems complicated."
    ]
    
    for i, text in enumerate(sample_texts):
        cleaned = preprocessor.clean_text(text)
        keywords = preprocessor.extract_keywords(text)
        metrics = preprocessor.calculate_clarity_metrics(text)
        
        print(f"\nSample {i+1}:")
        print(f"Original: {text}")
        print(f"Cleaned: {cleaned}")
        print(f"Keywords: {keywords}")
        print(f"Metrics: {metrics}")
    
    print("\nText preprocessing test completed successfully! ✅")

if __name__ == "__main__":
    test_text_preprocessing()