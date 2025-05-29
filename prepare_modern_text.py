"""
Script to prepare modern text dataset for IMM benchmark.
"""

import os
import json
import numpy as np
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import re
import pickle

def clean_text(text):
    """Clean and normalize text"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_modern_text():
    """Fetch modern text from various sources"""
    modern_texts = []
    
    # Example sources (you can add more)
    sources = [
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/1661/1661-0.txt",  # The Adventures of Sherlock Holmes
        "https://www.gutenberg.org/files/98/98-0.txt",      # A Tale of Two Cities
    ]
    
    for url in tqdm(sources, desc="Fetching modern texts"):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            text = clean_text(text)
            modern_texts.append(text)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    
    return modern_texts

def encode_text(text, stoi):
    """Encode text using the same encoding as Shakespeare dataset"""
    return [stoi.get(c, 0) for c in text]

def prepare_dataset():
    """Prepare modern text dataset"""
    # Create data directory if it doesn't exist
    os.makedirs('data/modern_text', exist_ok=True)
    
    # Load encoder from Shakespeare dataset
    meta_path = os.path.join('data', 'shakespeare_char', 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    
    # Fetch modern texts
    modern_texts = fetch_modern_text()
    
    # Combine all texts
    all_text = ' '.join(modern_texts)
    
    # Encode text
    encoded = encode_text(all_text, stoi)
    
    # Split into train/val
    n = len(encoded)
    train_data = encoded[:int(0.9*n)]
    val_data = encoded[int(0.9*n):]
    
    # Save as binary files
    train_data = np.array(train_data, dtype=np.uint16)
    val_data = np.array(val_data, dtype=np.uint16)
    
    train_data.tofile('data/modern_text/train.bin')
    val_data.tofile('data/modern_text/val.bin')
    
    # Save metadata
    meta = {
        'vocab_size': len(stoi),
        'stoi': stoi,
        'itos': itos
    }
    with open('data/modern_text/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

if __name__ == '__main__':
    prepare_dataset() 