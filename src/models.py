import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def train_and_evaluate_models(X, y, model_path=None):
    """Trains Logistic Regression, Random Forest, and XGBoost models.
    Selects the best performing model based on ROC-AUC and saves it.
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
        
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, max_depth=4, learning_rate=0.1)
    }
    
    best_auc = 0.0
    best_model = None
    best_name = None
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_val, y_prob)
        report = classification_report(y_val, y_pred, output_dict=True)
        cm = confusion_matrix(y_val, y_pred)
        
        print(f"ROC-AUC: {auc:.4f}")
        print(f"F1-Score (Class 1): {report['1']['f1-score']:.4f}")
        print(f"Accuracy: {report['accuracy']:.4f}")
        
        results[name] = {
            'model': model,
            'auc': auc,
            'f1_class_1': report['1']['f1-score'],
            'accuracy': report['accuracy'],
            'confusion_matrix': cm.tolist()
        }
        
        # Select best model based on ROC-AUC (standard for imbalanced/balanced churn models)
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with ROC-AUC {best_auc:.4f}")
    
    # Save the best model
    joblib.dump({
        'model_name': best_name,
        'model': best_model,
        'results': results
    }, model_path)
    print(f"Saved best model to {model_path}")
    
    return best_model, results

def calculate_customer_ltv(monthly_charges, total_charges, churn_prob, max_tenure_months=60):
    """Calculates the estimated Customer Lifetime Value (LTV).
    
    Formula:
    Expected Remaining Tenure (months) = (1 - churn_prob) / (churn_prob + epsilon)
    We cap remaining tenure to a maximum of max_tenure_months to keep LTV projections realistic.
    
    LTV = TotalCharges (historic spent) + (Expected Remaining Tenure * MonthlyCharges) (predicted future spend)
    """
    epsilon = 1e-5
    
    # Expected remaining tenure based on probability of churn
    expected_remaining_tenure = (1 - churn_prob) / (churn_prob + epsilon)
    
    # Cap to prevent unrealistically large values
    expected_remaining_tenure = np.minimum(expected_remaining_tenure, max_tenure_months)
    
    # Forecast future spend
    future_spend = expected_remaining_tenure * monthly_charges
    
    # Total LTV is historical spend plus future predicted spend
    ltv = total_charges + future_spend
    
    return np.round(ltv, 2), np.round(expected_remaining_tenure, 1)

if __name__ == "__main__":
    # Test LTV calculation
    print(calculate_customer_ltv(70.0, 1400.0, 0.05)) # Low churn risk
    print(calculate_customer_ltv(70.0, 1400.0, 0.85)) # High churn risk
