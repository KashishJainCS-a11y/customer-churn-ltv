import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def engineer_features(df):
    """Engineers new features from the raw DataFrame."""
    df_copy = df.copy()
    
    # List of service columns to count
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Calculate Total Services active for the customer
    total_services = np.zeros(len(df_copy))
    for col in service_cols:
        if col in df_copy.columns:
            # InternetService is categorical (DSL, Fiber optic, No). Check if not 'No'
            if col == 'InternetService':
                total_services += (df_copy[col] != 'No').astype(int)
            # Other service columns are Yes, No, or No internet service
            else:
                total_services += (df_copy[col] == 'Yes').astype(int)
                
    df_copy['TotalServices'] = total_services
    
    # Calculate Average Charge per active service (add 1 to avoid division by zero)
    df_copy['AvgChargePerService'] = df_copy['MonthlyCharges'] / (df_copy['TotalServices'] + 1)
    
    return df_copy

def get_preprocessor(df):
    """Fits a preprocessing pipeline on the features and returns the ColumnTransformer."""
    # Ensure directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Columns to drop
    drop_cols = ['customerID']
    if 'Churn' in df.columns:
        drop_cols.append('Churn')
        
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # Identify numeric and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'AvgChargePerService']
    
    categorical_features = [col for col in feature_cols if col not in numeric_features]
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_features, categorical_features

def preprocess_data(df, is_training=True, preprocessor_path=None):
    """Engineers features, then fits or applies the ColumnTransformer preprocessor."""
    if preprocessor_path is None:
        preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
        
    # Engineer custom features first
    df_engineered = engineer_features(df)
    
    # Prepare features and target
    X = df_engineered.drop(columns=['customerID', 'Churn'], errors='ignore')
    
    if 'Churn' in df_engineered.columns:
        y = df_engineered['Churn']
    else:
        y = None
        
    if is_training:
        preprocessor, num_features, cat_features = get_preprocessor(df_engineered)
        X_trans = preprocessor.fit_transform(X)
        
        # Get output feature names
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features_encoded = cat_encoder.get_feature_names_out(cat_features).tolist()
        feature_names = num_features + cat_features_encoded
        
        # Save preprocessor along with feature metadata
        metadata = {
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'num_features': num_features,
            'cat_features': cat_features,
            'service_cols': [
                'PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
        }
        joblib.dump(metadata, preprocessor_path)
        print(f"Saved preprocessor and metadata to {preprocessor_path}")
    else:
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Train the model first.")
        
        metadata = joblib.load(preprocessor_path)
        preprocessor = metadata['preprocessor']
        feature_names = metadata['feature_names']
        
        X_trans = preprocessor.transform(X)
        
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
    return X_trans_df, y, metadata

if __name__ == "__main__":
    from data_loader import load_and_clean_data
    df = load_and_clean_data()
    X_trans, y, metadata = preprocess_data(df, is_training=True)
    print(f"Preprocessed features shape: {X_trans.shape}")
    print(f"Feature names: {X_trans.columns.tolist()[:10]}...")
