import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import sys
from datetime import datetime
# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
# Set page config
st.set_page_config(
    page_title="Customer Churn & LTV Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom Premium Styling (Dark Theme & Glassmorphism UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global style overrides */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: #0d111c;
        color: #f3f4f6;
    }
    
    /* Glassmorphism containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.15);
    }
    
    /* Custom Headers & Text */
    .gradient-text {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin-top: 10px;
        letter-spacing: -0.02em;
    }
    .value-blue {
        color: #3b82f6;
        text-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
    }
    .value-purple {
        color: #a855f7;
        text-shadow: 0 0 20px rgba(168, 85, 247, 0.4);
    }
    .value-green {
        color: #10b981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    .value-red {
        color: #ef4444;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    /* Recommendations & Alerts */
    .rec-box {
        border-left: 4px solid;
        border-radius: 4px;
        padding: 12px 16px;
        background: rgba(255, 255, 255, 0.02);
        margin-top: 15px;
    }
    .rec-low {
        border-left-color: #10b981;
        color: #a7f3d0;
    }
    .rec-medium {
        border-left-color: #f59e0b;
        color: #fde68a;
    }
    .rec-high {
        border-left-color: #ef4444;
        color: #fca5a5;
    }
    
    /* Streamlit widget tweaks */
    div[data-testid="stSidebar"] {
        background-color: #080b13;
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }
    
</style>
""", unsafe_allow_html=True)
# Paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
# Helper function to auto-run pipeline if models are missing
@st.cache_resource
def load_ml_assets():
    if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
        st.info("🔄 First run: Auto-training machine learning models on IBM customer dataset...")
        try:
            import train
            train.main()
            st.success("🎉 Models trained successfully and saved!")
        except Exception as e:
            st.error(f"Failed to auto-train models: {e}")
            return None, None
            
    # Load preprocessor and model
    try:
        metadata = joblib.load(PREPROCESSOR_PATH)
        model_data = joblib.load(MODEL_PATH)
        return metadata, model_data
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None
metadata, model_data = load_ml_assets()
# Read the raw data (for visualizations)
@st.cache_data
def get_raw_data():
    from data_loader import load_and_clean_data
    return load_and_clean_data()
try:
    df_raw = get_raw_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    df_raw = None
# Title area
st.markdown("<h1 style='margin-bottom:0px;'><span class='gradient-text'>Customer Churn & LTV Forecasting</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6b7280; margin-bottom: 30px;'>An AI-powered data science platform predicting subscriber attrition and customer lifetime value (LTV).</p>", unsafe_allow_html=True)
if metadata is None or model_data is None:
    st.stop()

# Tab setup
tab_predict, tab_batch, tab_segments, tab_analytics = st.tabs([
    "🔮 Single Customer Predictor", 
    "📤 Batch Prediction",
    "📊 Customer Segments",
    "📈 Analytics & Insights"
])
# --- TAB 1: PREDICTOR ---
with tab_predict:
    st.markdown("### Predict Customer Health & Value")
    
    col_input, col_result = st.columns([2, 3])
    
    with col_input:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Customer Characteristics")
        
        # Demographics
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        # Subscription Details
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
        # Calculate expected total charges initially based on tenure * monthly charges
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))
        
        st.markdown("---")
        st.subheader("Services Subscribed")
        
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        # Conditional selections based on internet service
        if internet_service != "No":
            security = st.selectbox("Online Security", ["No", "Yes"])
            backup = st.selectbox("Online Backup", ["No", "Yes"])
            protection = st.selectbox("Device Protection", ["No", "Yes"])
            support = st.selectbox("Tech Support", ["No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            security = "No internet service"
            backup = "No internet service"
            protection = "No internet service"
            support = "No internet service"
            tv = "No internet service"
            movies = "No internet service"
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_result:
        # Create input dict matching raw data columns
        input_data = {
            'customerID': ['TEMP'],
            'gender': [gender],
            'SeniorCitizen': [1 if senior == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [security],
            'OnlineBackup': [backup],
            'DeviceProtection': [protection],
            'TechSupport': [support],
            'StreamingTV': [tv],
            'StreamingMovies': [movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless],
            'PaymentMethod': [payment],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        
        df_input = pd.DataFrame(input_data)
        
        # Preprocess input using the helper function
        from features import preprocess_data
        X_input_trans, _, _ = preprocess_data(df_input, is_training=False, preprocessor_path=PREPROCESSOR_PATH)
        
        # Predict churn probability
        model = model_data['model']
        churn_prob = model.predict_proba(X_input_trans)[0, 1]
        
        # Calculate LTV
        from models import calculate_customer_ltv
        ltv_val, remaining_tenure = calculate_customer_ltv(monthly_charges, total_charges, churn_prob)
        
        # Predict category color & alert
        if churn_prob < 0.3:
            risk_class = "value-green"
            risk_label = "Low Risk"
            rec_html = """
            <div class='rec-box rec-low'>
                <strong>Customer Health: Solid</strong><br/>
                • Recommendation: Keep engaged with standard marketing campaigns.<br/>
                • Action: Pitch loyalty rewards or long-term upgrade programs to lock them in further.
            </div>
            """
        elif churn_prob < 0.7:
            risk_class = "value-purple"
            risk_label = "Medium Risk"
            rec_html = """
            <div class='rec-box rec-medium'>
                <strong>Customer Health: Warning</strong><br/>
                • Recommendation: Customer is exhibiting warning signs.<br/>
                • Action: Provide targeted promotional discounts or contact them via tech support channels to resolve any open disputes.
            </div>
            """
        else:
            risk_class = "value-red"
            risk_label = "High Risk"
            rec_html = """
            <div class='rec-box rec-high'>
                <strong>Customer Health: Critical</strong><br/>
                • Recommendation: Immediate retention action required!<br/>
                • Action: Send a direct email discount or trigger an automated high-priority customer support call to renew their contract.
            </div>
            """
        # Display Key Metrics
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("AI Analysis Results")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"<div class='metric-label'>Churn Risk Assessment</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value {risk_class}'>{churn_prob*100:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 1.1rem; font-weight: 600; margin-top:5px;'>Status: {risk_label}</div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-label'>Forecasted Lifetime Value (LTV)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value value-blue'>${ltv_val:,.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.9rem; color: #9ca3af; margin-top:5px;'>Expected Remaining Tenure: {remaining_tenure} months</div>", unsafe_allow_html=True)
        
        # Recommendations
        st.markdown(rec_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Churn Probability Gauge
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Churn Probability Meter")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = churn_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#f3f4f6"},
                'bar': {'color': "#6366f1"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 3},
                    'thickness': 0.75,
                    'value': churn_prob * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f3f4f6", 'family': "Outfit"},
            height=200,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Local Feature Contribution
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Why this score? (Explainable AI)")
        
        # Approximate feature contribution by comparing input features against mean values
        # or plotting the model coefficients for Logistic Regression or tree importances.
        # Since we want something interactive for this customer, let's show the top features
        # and highlight their impact.
        st.write("This customer's risk profile is influenced by their account contract, services, and charges. Here are the top factors:")
        
        # Calculate feature contributions based on coefficients (Logistic Regression) or SHAP proxy
        model_type = model_data['model_name']
        feature_names = metadata['feature_names']
        
        contributions = []
        
        # We can construct a simple feature contribution visualization
        # For simplicity and visual impact, let's fetch the global feature importances
        # and explain the customer's specific values relative to them.
        
        has_fiber = "Yes" if internet_service == "Fiber optic" else "No"
        m2m_contract = "Yes" if contract == "Month-to-month" else "No"
        no_security = "Yes" if security == "No" else "No"
        
        factors = [
            ("Contract type is Month-to-month", m2m_contract == "Yes", "🔴 Increases churn risk (low loyalty)", "🟢 Lowers churn risk (high loyalty)"),
            ("Internet is Fiber optic", has_fiber == "Yes", "🔴 Increases churn risk (higher complaint rate in data)", "🟢 Neutral or lower churn risk"),
            ("No Online Security active", no_security == "Yes", "🔴 Increases churn risk (easier to switch)", "🟢 Lowers churn risk (higher stickiness)"),
            ("Tenure length", tenure < 12, "🔴 High risk (New customer under 1 year)", "🟢 Low risk (Established customer)")
        ]
        
        for name, condition, positive_text, negative_text in factors:
            if condition:
                st.markdown(f"<span style='color:#ef4444;'>• {positive_text}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:#10b981;'>• {negative_text}</span>", unsafe_allow_html=True)
                
        st.markdown("</div>", unsafe_allow_html=True)
# --- TAB 2: BATCH PREDICTION ---
with tab_batch:
    st.markdown("### Batch Prediction from CSV")
    st.write("Upload a CSV file with multiple customers to predict churn and LTV for all of them at once.")
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"📋 Loaded {len(df_batch)} customers")
            st.write(f"Columns: {', '.join(df_batch.columns.tolist())}")
            
            # Show preview
            st.subheader("Preview of uploaded data")
            st.dataframe(df_batch.head())
            
            if st.button("🚀 Predict Churn & LTV for all customers"):
                with st.spinner("Processing predictions..."):
                    from advanced_features import predict_batch
                    
                    # Prepare data
                    df_pred = df_batch.copy()
                    if 'Churn' not in df_pred.columns:
                        df_pred['Churn'] = 0
                    
                    # Run predictions
                    df_results = predict_batch(df_pred, metadata, model_data['model'], metadata)
                    
                    # Display results
                    st.success(f"✅ Predictions complete for {len(df_results)} customers!")
                    
                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_churn = df_results['Churn_Probability'].mean()
                        st.metric("Avg Churn Risk", f"{avg_churn:.1%}")
                    with col2:
                        high_risk_count = (df_results['Churn_Probability'] > 0.7).sum()
                        st.metric("High Risk Customers", f"{high_risk_count} ({high_risk_count/len(df_results):.1%})")
                    with col3:
                        avg_ltv = df_results['Predicted_LTV'].mean()
                        st.metric("Avg Predicted LTV", f"${avg_ltv:,.0f}")
                    
                    # Show results table
                    st.subheader("Detailed Predictions")
                    display_cols = ['customerID', 'Churn_Probability', 'Churn_Risk', 'Predicted_LTV', 
                                   'MonthlyCharges', 'Contract']
                    st.dataframe(df_results[display_cols].sort_values('Churn_Probability', ascending=False))
                    
                    # Download CSV
                    csv_data = df_results.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv_data,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Template download
    st.subheader("📥 Download CSV Template")
    template_df = pd.DataFrame({
        'customerID': ['CUST001', 'CUST002'],
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'tenure': [12, 24],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['Yes', 'No'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['Yes', 'No'],
        'StreamingTV': ['No', 'Yes'],
        'StreamingMovies': ['Yes', 'No'],
        'Contract': ['Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)'],
        'MonthlyCharges': [65.0, 85.0],
        'TotalCharges': [780.0, 2040.0],
        'Churn': [0, 0]
    })
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="Download Template CSV",
        data=csv_template,
        file_name="customer_batch_template.csv",
        mime="text/csv"
    )

# --- TAB 3: CUSTOMER SEGMENTS ---
with tab_segments:
    st.markdown("### Customer Segmentation & RFM Analysis")
    
    if df_raw is not None:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        from advanced_features import segment_customers_rfm, get_segment_insights
        
        # Segment customers
        df_segmented = segment_customers_rfm(df_raw)
        
        # Display segmentation overview
        st.subheader("Customer Segments Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            segment_dist = df_segmented['Segment'].value_counts()
            fig_seg = px.pie(
                values=segment_dist.values,
                names=segment_dist.index,
                title="Customer Distribution by Segment",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_seg.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"}
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        
        with col2:
            ltv_dist = df_segmented['LTV_Bucket'].value_counts()
            fig_ltv = px.pie(
                values=ltv_dist.values,
                names=ltv_dist.index,
                title="Customer Distribution by LTV Bucket",
                color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981', '#3b82f6']
            )
            fig_ltv.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"}
            )
            st.plotly_chart(fig_ltv, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Segment insights
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Segment Insights")
        
        insights = get_segment_insights(df_segmented)
        
        seg_cols = st.columns(len(insights))
        for idx, (segment, data) in enumerate(insights.items()):
            with seg_cols[idx]:
                st.metric(f"{segment}", f"{data['count']} customers")
                st.caption(f"Avg Tenure: {data['avg_tenure']:.0f}m")
                if data['churn_rate'] is not None:
                    st.caption(f"Churn: {data['churn_rate']:.1f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Detailed segment analysis
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Detailed Segment Analysis")
        
        selected_segment = st.selectbox("Select Segment", df_segmented['Segment'].unique())
        seg_data = df_segmented[df_segmented['Segment'] == selected_segment]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Count", len(seg_data))
        with col2:
            st.metric("Avg Monthly Charges", f"${seg_data['MonthlyCharges'].mean():.2f}")
        with col3:
            st.metric("Avg Tenure", f"{seg_data['tenure'].mean():.0f} months")
        with col4:
            if 'Churn' in seg_data.columns:
                churn_rate = seg_data['Churn'].mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Contract type in segment
        if 'Contract' in seg_data.columns:
            st.subheader(f"Contract Type Distribution - {selected_segment}")
            contract_dist = seg_data['Contract'].value_counts()
            fig_contract = px.bar(
                x=contract_dist.index,
                y=contract_dist.values,
                color=contract_dist.index,
                color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981'],
                labels={'x': 'Contract Type', 'y': 'Count'}
            )
            fig_contract.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"},
                showlegend=False
            )
            st.plotly_chart(fig_contract, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.warning("No data available for segmentation")

# --- TAB 4: ANALYTICS ---
with tab_analytics:
    st.markdown("### Churn and LTV Cohort Analytics")
    
    if df_raw is not None:
        # Key Global Metrics
        total_customers = len(df_raw)
        overall_churn = df_raw['Churn'].mean() * 100
        avg_monthly = df_raw['MonthlyCharges'].mean()
        avg_tenure = df_raw['tenure'].mean()
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Total Customers Analysed</div><div class='metric-value value-blue'>{total_customers:,}</div></div>", unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Overall Churn Rate</div><div class='metric-value value-red'>{overall_churn:.1f}%</div></div>", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Avg Monthly Charges</div><div class='metric-value value-purple'>${avg_monthly:.2f}</div></div>", unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Avg Customer Tenure</div><div class='metric-value value-green'>{avg_tenure:.1f} Mo</div></div>", unsafe_allow_html=True)
            
        # Plotly Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Churn Rate by Contract Type")
            # Calculate churn rate by contract
            contract_churn = df_raw.groupby('Contract')['Churn'].mean().reset_index()
            contract_churn['Churn Rate (%)'] = contract_churn['Churn'] * 100
            
            fig1 = px.bar(
                contract_churn,
                x='Contract',
                y='Churn Rate (%)',
                color='Contract',
                color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981'],
                text='Churn Rate (%)'
            )
            fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"},
                showlegend=False,
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Global Feature Importance (Best Model)")
            # Pull global feature importances or coefficients
            model_info = model_data['model']
            
            if hasattr(model_info, 'feature_importances_'):
                importances = model_info.feature_importances_
            elif hasattr(model_info, 'coef_'):
                importances = np.abs(model_info.coef_[0])
            else:
                importances = np.ones(len(metadata['feature_names']))
                
            fi_df = pd.DataFrame({
                'Feature': metadata['feature_names'],
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            # Make the feature names clean for presentation
            fi_df['Feature'] = fi_df['Feature'].str.replace('num__', '').str.replace('cat__', '')
            
            fig3 = px.bar(
                fi_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"},
                coloraxis_showscale=False,
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with chart_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Tenure & Monthly Charges vs. Churn Status")
            
            # Sample data to speed up plotting and keep it clean
            sample_df = df_raw.sample(n=min(1500, len(df_raw)), random_state=42)
            sample_df['Churn Status'] = sample_df['Churn'].map({1: 'Churned', 0: 'Retained'})
            
            fig2 = px.scatter(
                sample_df,
                x='tenure',
                y='MonthlyCharges',
                color='Churn Status',
                color_discrete_map={'Churned': '#ef4444', 'Retained': '#10b981'},
                opacity=0.6,
                labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)'}
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"},
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Churn Density by Monthly Charges")
            
            fig4 = px.histogram(
                df_raw,
                x='MonthlyCharges',
                color=df_raw['Churn'].map({1: 'Churned', 0: 'Retained'}),
                barmode='overlay',
                color_discrete_map={'Churned': 'rgba(239, 68, 68, 0.6)', 'Retained': 'rgba(16, 185, 129, 0.6)'},
                labels={'MonthlyCharges': 'Monthly Charges ($)', 'count': 'Number of Customers'}
            )
            fig4.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f3f4f6", 'family': "Outfit"},
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.warning("No data available to display dashboard analytics. Run the training script first.")
