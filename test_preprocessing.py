"""
PHASE 2 - PREPROCESSING (Sample Version for Testing)
Test with smaller sample dataset
"""

import sys
sys.path.insert(0, '/src')

from src.preprocessing import preprocess_and_save

# Run with sample (2000 questions per split)
results = preprocess_and_save(
    train_path='data/raw/train.csv',
    val_path='data/raw/dev.csv',
    test_path='data/raw/test.csv',
    output_dir='data/processed',
    onehot_features=5000,
    tfidf_features=5000
)

print("\n✓ Preprocessing complete!")
