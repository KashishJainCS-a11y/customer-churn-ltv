# 🎯 Customer Churn & LTV Prediction

A machine learning application that predicts customer churn probability and calculates customer lifetime value (LTV) using Scikit-learn and Streamlit.

##  Features

### 1.  Single Customer Predictor
- Interactive form to input customer demographics and service details
- Real-time churn probability prediction
- Customer LTV estimation
- Explainability factors (contract type, fiber optic, security features, tenure)

### 2.  Batch Prediction
- Upload CSV files with customer data
- Bulk predictions for multiple customers
- Churn risk classification (Low/Medium/High)
- Download results as CSV

### 3.  Customer Segments
- RFM (Recency, Frequency, Monetary) segmentation
- 5 customer segments: Champions, Loyal, Potential, At Risk, Lost
- LTV bucketing (Low/Medium/High/Premium)
- Segment-wise analytics and insights

### 4.  Analytics & Insights
- Global KPIs: Churn rate, average monthly charges, customer tenure
- Feature importance analysis
- Churn rate by contract type
- Customer tenure vs. monthly charges visualization
- Churn probability distribution

##  Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/customer_churn_ltv.git
   cd customer_churn_ltv
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate virtual environment
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

5. Train the model (optional - auto-trains on first run)
   ```bash
   python train.py
   ```

6. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

7. Access the app
   - Local: http://localhost:8501
   - Network: http://192.168.29.139:8501 (replace IP with your machine IP)

##  Project Structure

```
customer_churn_ltv/
├── app.py                      # Main Streamlit dashboard
├── train.py                    # ML training pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source package
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Dataset download and cleaning
│   ├── features.py            # Feature engineering and preprocessing
│   ├── models.py              # Model training and LTV calculation
│   └── advanced_features.py   # Batch prediction and segmentation
│
├── models/                     # Trained ML models
│   ├── churn_model.pkl        # Trained Logistic Regression model
│   └── preprocessor.pkl       # Feature preprocessor
│
└── data/                       # Dataset
    └── Telco-Customer-Churn.csv
```

##  Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation and analysis |
| numpy | Latest | Numerical computations |
| scikit-learn | Latest | ML model training and preprocessing |
| xgboost | Latest | Gradient boosting (model comparison) |
| streamlit | Latest | Web framework for dashboard |
| plotly | Latest | Interactive visualizations |
| requests | Latest | Dataset downloading |
| joblib | Latest | Model serialization |

##  Model Performance

- Algorithm: Logistic Regression
- Best ROC-AUC Score: 0.8423
- F1-Score (Churn Class): 0.6061
- Accuracy: 0.8062
- Dataset: IBM Telco Customer Churn (7,043 customers, 21 features)
- Train/Validation Split: 80/20 with stratification

##  Usage Guide

### Single Customer Predictor
1. Go to the 🔮 Single Customer Predictor tab
2. Fill in customer demographics (age, contract type, services, charges)
3. Click Predict
4. View churn probability, LTV, and risk factors

### Batch Prediction
1. Go to the  Batch Prediction tab
2. Upload a CSV file with customer data (matching training features)
3. View predictions table
4. Download results as CSV

### Customer Segments
1. Go to the  Customer Segments tab
2. View overall segment distribution and LTV buckets
3. Select a segment from dropdown
4. View segment-specific metrics and insights

### Analytics & Insights
1. Go to the  Analytics & Insights tab
2. View global KPIs and 4 interactive charts
3. Hover over charts for detailed information

##  Input Features

The model expects the following customer features:
- Demographics: Age, Senior Citizen status, Gender
- Account Info: Tenure, Monthly Charges, Total Charges
- Services: Phone, Internet, Online Security, Tech Support, etc.
- Contract: Month-to-month, One year, Two year

##  How LTV is Calculated

```
Expected Remaining Tenure = (1 - Churn_Probability) / (Churn_Probability + 1e-5)
LTV = Total_Charges_to_Date + (Remaining_Tenure * Monthly_Charges)
```

##  Data Source

- Dataset: IBM Telco Customer Churn
- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- Auto-download: Dataset is automatically downloaded on first run

##  Development

### Training the Model
```bash
python train.py
```
This will:
1. Download the dataset
2. Clean and preprocess data
3. Train multiple models (Logistic Regression, Random Forest, XGBoost)
4. Save the best model and preprocessor

### Running Tests
```bash
python -m py_compile app.py  # Syntax check
```

##  Deployment

To deploy to cloud:

### Streamlit Cloud
1. Push code to GitHub
2. Sign up at https://streamlit.io/cloud
3. Connect your GitHub repo
4. Deploy in one click

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

##  License

This project is open source and available under the MIT License.

##  Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


---

Built with ❤️ using Python, Scikit-learn, and Streamlit
