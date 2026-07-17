import os
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import load_and_clean_data
from features import preprocess_data
from models import train_and_evaluate_models

def main():
    print("==================================================")
    print("Starting Customer Churn & LTV ML Pipeline")
    print("==================================================")
    
    # 1. Load Data
    print("\n[Step 1/3] Loading and cleaning raw data...")
    df = load_and_clean_data()
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # 2. Preprocess Data
    print("\n[Step 2/3] Performing feature engineering & preprocessing...")
    X_trans, y, metadata = preprocess_data(df, is_training=True)
    print("Features preprocessed and scaled.")
    
    # 3. Train Models
    print("\n[Step 3/3] Training and evaluating machine learning models...")
    best_model, results = train_and_evaluate_models(X_trans, y)
    print("\n==================================================")
    print("Pipeline training completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
