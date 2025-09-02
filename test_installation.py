#!/usr/bin/env python3
"""
Test script to verify all required packages are installed correctly.
"""

import importlib
import sys

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError:
        print(f"✗ {module_name} not found")
        return False

def main():
    print("Testing required packages...")
    
    required_modules = [
        'torch',
        'torchvision',
        'transformers',
        'streamlit',
        'cv2',
        'pandas',
        'numpy',
        'sklearn',
        'nltk',
        'matplotlib',
        'seaborn',
        'tqdm',
        'dotenv'
    ]
    
    success_count = 0
    for module in required_modules:
        if test_import(module):
            success_count += 1
    
    print(f"\nResults: {success_count}/{len(required_modules)} packages installed correctly")
    
    if success_count == len(required_modules):
        print("All packages installed successfully! 🎉")
        return 0
    else:
        print("Some packages are missing. Please install them with: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())