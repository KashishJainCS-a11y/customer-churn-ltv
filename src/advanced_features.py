"""
Advanced features for batch prediction, cohort analysis, and reporting.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import io

def predict_batch(df_input, preprocessor, model, features_metadata):
    """
    Predict churn and LTV for multiple customers.
    
    Args:
        df_input: DataFrame with customer data
        preprocessor: Fitted preprocessor
        model: Trained model
        features_metadata: Metadata about features
        
    Returns:
        DataFrame with predictions and LTV
    """
    from features import preprocess_data
    from models import calculate_customer_ltv
    
    # Preprocess
    X_input_trans, _, _ = preprocess_data(df_input, is_training=False, preprocessor_path=None)
    
    # Manually set preprocessor for prediction
    X_trans = preprocessor['preprocessor'].transform(df_input.drop(columns=['customerID', 'Churn'], errors='ignore'))
    
    # Predict
    churn_probs = model.predict_proba(X_trans)[:, 1]
    
    # Calculate LTV
    ltv_values = []
    remaining_tenures = []
    for idx, row in df_input.iterrows():
        ltv, tenure = calculate_customer_ltv(
            row['MonthlyCharges'],
            row['TotalCharges'],
            churn_probs[idx]
        )
        ltv_values.append(ltv)
        remaining_tenures.append(tenure)
    
    # Create results dataframe
    results_df = df_input.copy()
    results_df['Churn_Probability'] = churn_probs
    results_df['Churn_Risk'] = pd.cut(churn_probs, bins=[0, 0.3, 0.7, 1.0], 
                                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
    results_df['Predicted_LTV'] = ltv_values
    results_df['Remaining_Tenure_Months'] = remaining_tenures
    
    return results_df


def segment_customers_rfm(df):
    """
    Segment customers using RFM analysis + Churn Risk.
    
    Args:
        df: DataFrame with customer data and churn status
        
    Returns:
        DataFrame with segmentation
    """
    df_seg = df.copy()
    
    # RFM: Recency (tenure), Frequency (services), Monetary (monthly charges)
    df_seg['Recency'] = pd.cut(df_seg['tenure'], bins=5, labels=['1', '2', '3', '4', '5'])
    
    # Count services
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    service_count = 0
    for col in service_cols:
        if col in df_seg.columns:
            if col == 'InternetService':
                service_count += (df_seg[col] != 'No').astype(int)
            else:
                service_count += (df_seg[col] == 'Yes').astype(int)
    
    df_seg['Services_Count'] = service_count
    df_seg['Frequency'] = pd.cut(df_seg['Services_Count'], bins=[0, 2, 4, 6, 10], 
                                  labels=['1', '2', '3', '4'])
    df_seg['Monetary'] = pd.cut(df_seg['MonthlyCharges'], bins=5, labels=['1', '2', '3', '4', '5'])
    
    # Combine RFM
    df_seg['RFM_Score'] = (df_seg['Recency'].astype(str) + 
                           df_seg['Frequency'].astype(str) + 
                           df_seg['Monetary'].astype(str))
    
    # Create segments
    segments = []
    for score in df_seg['RFM_Score']:
        r, f, m = int(score[0]), int(score[1]), int(score[2])
        avg_score = (r + f + m) / 3
        
        if avg_score >= 4:
            segments.append('Champions')
        elif avg_score >= 3.5:
            segments.append('Loyal Customers')
        elif avg_score >= 3:
            segments.append('Potential Loyalists')
        elif avg_score >= 2:
            segments.append('At Risk')
        else:
            segments.append('Lost')
    
    df_seg['Segment'] = segments
    
    # Add LTV buckets
    if 'MonthlyCharges' in df_seg.columns and 'tenure' in df_seg.columns:
        df_seg['Estimated_LTV'] = df_seg['MonthlyCharges'] * df_seg['tenure']
        df_seg['LTV_Bucket'] = pd.cut(df_seg['Estimated_LTV'], 
                                       bins=[0, 500, 1500, 3000, float('inf')],
                                       labels=['Low', 'Medium', 'High', 'Premium'])
    
    return df_seg


def generate_csv_report(df_predictions):
    """Generate CSV report for predictions."""
    buffer = io.StringIO()
    df_predictions.to_csv(buffer, index=False)
    return buffer.getvalue()


def generate_pdf_report(df_predictions, title="Churn & LTV Predictions Report"):
    """
    Generate a simple text-based PDF report.
    Note: For production, consider using reportlab or similar.
    """
    report = f"""
{'='*80}
{title}
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
{'-'*80}
Total Customers Analyzed: {len(df_predictions)}
Average Churn Probability: {df_predictions['Churn_Probability'].mean():.2%}
Average Predicted LTV: ${df_predictions['Predicted_LTV'].mean():,.2f}

CHURN RISK DISTRIBUTION
{'-'*80}
"""
    
    if 'Churn_Risk' in df_predictions.columns:
        risk_dist = df_predictions['Churn_Risk'].value_counts()
        for risk, count in risk_dist.items():
            pct = (count / len(df_predictions)) * 100
            report += f"{risk}: {count} customers ({pct:.1f}%)\n"
    
    report += f"""
TOP 10 HIGHEST CHURN RISK CUSTOMERS
{'-'*80}
"""
    
    top_churn = df_predictions.nlargest(10, 'Churn_Probability')[
        ['customerID', 'Churn_Probability', 'Predicted_LTV', 'Contract']
    ]
    
    for idx, row in top_churn.iterrows():
        report += f"  {row['customerID']}: {row['Churn_Probability']:.1%} churn risk, LTV: ${row['Predicted_LTV']:,.0f}\n"
    
    report += f"""
RETENTION RECOMMENDATIONS
{'-'*80}
Focus retention efforts on:
- Customers with HIGH RISK (>70% churn probability)
- Customers with Month-to-month contracts
- Customers without security/backup services
- New customers (tenure < 12 months)

{'='*80}
End of Report
{'='*80}
"""
    
    return report


def get_segment_insights(df_segmented):
    """Get insights from customer segments."""
    insights = {}
    
    for segment in df_segmented['Segment'].unique():
        seg_data = df_segmented[df_segmented['Segment'] == segment]
        
        if 'Churn' in df_segmented.columns:
            churn_rate = seg_data['Churn'].mean() * 100
        else:
            churn_rate = None
        
        insights[segment] = {
            'count': len(seg_data),
            'avg_monthly_charges': seg_data['MonthlyCharges'].mean(),
            'avg_tenure': seg_data['tenure'].mean(),
            'churn_rate': churn_rate
        }
    
    return insights


if __name__ == "__main__":
    print("Advanced features module loaded successfully")
