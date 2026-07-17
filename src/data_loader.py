import os
import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")

def download_data():
    """Downloads the IBM Telco Customer Churn dataset if not already present."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        response = requests.get(DATA_URL)
        response.raise_for_status()
        with open(DATA_PATH, "wb") as f:
            f.write(response.content)
        print(f"Dataset downloaded successfully and saved to {DATA_PATH}")
    else:
        print(f"Dataset already exists at {DATA_PATH}")

def load_and_clean_data():
    """Loads the dataset, handles basic cleaning (types, missing values), and returns a DataFrame."""
    download_data()
    
    df = pd.read_csv(DATA_PATH)
    
    # Clean TotalCharges: replace spaces with NaN, convert to numeric
    df['TotalCharges'] = df['TotalCharges'].replace(' ', pd.NA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Fill TotalCharges NaNs with 0 (since they represent new customers with 0 tenure)
    # Check if tenure is 0 for these rows
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Map target Churn variable to binary 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(f"Loaded DataFrame with shape: {df.shape}")
    print(df.head())
