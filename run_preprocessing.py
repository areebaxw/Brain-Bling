import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_and_save

# Set up paths for local environment
base_dir = os.path.join(os.path.dirname(__file__), 'data')

print("Starting preprocessing...")
results = preprocess_and_save(
    train_path=os.path.join(base_dir, 'raw', 'train.csv'),
    val_path=os.path.join(base_dir, 'raw', 'dev.csv'),
    test_path=os.path.join(base_dir, 'raw', 'test.csv'),
    output_dir=os.path.join(base_dir, 'processed'),
    onehot_features=1000,
    tfidf_features=1000,
    sample_size=1000  # Use much smaller sample for faster processing
)

print("\n[OK] Preprocessing complete!")
