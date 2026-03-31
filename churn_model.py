import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
st.set_page_config(page_title="Mahi Fashion | Churn AI", layout="wide")

# --- STEP 1: DATA & MODELING ---
@st.cache_data
def load_and_prep():
    df = pd.read_csv('data/Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # We use these specific features for the UI
    cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
    X = pd.get_dummies(df[cols], columns=['Contract', 'InternetService'])
    y = df['Churn']
    return X, y, X.columns.tolist()

X, y, feature_names = load_and_prep()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_pro_model(X_t, y_t):
    model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
    model.fit(X_t, y_t)
    return model

model = train_pro_model(X_train, y_train)

# --- STEP 2: INTERACTIVE SIDEBAR ---
st.sidebar.header("🔍 Predict Individual Customer")
def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.sidebar.slider("Monthly Charges ($)", 18, 120, 70)
    total = st.sidebar.number_input("Total Charges ($)", 0, 9000, 500)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Matching the One-Hot Encoding format
    data = {'tenure': tenure, 'MonthlyCharges': monthly, 'TotalCharges': total}
    # Initialize all dummy columns to 0
    for col in feature_names:
        if col not in data: data[col] = 0
    
    # Set the selected dummies to 1
    if f"Contract_{contract}" in data: data[f"Contract_{contract}"] = 1
    if f"InternetService_{internet}" in data: data[f"InternetService_{internet}"] = 1
    
    return pd.DataFrame([data])[feature_names]

input_df = user_input_features()
# Add this to see what the model is actually reading
st.subheader("Debug: Processed Input Data")
st.write(input_df)

# --- STEP 3: MAIN DASHBOARD ---
st.title("📊 Telco Customer Churn Prediction")
st.markdown("---")

# Row 1: Prediction Results
prediction_proba = model.predict_proba(input_df)[0][1]
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Prediction Result")
    risk_color = "red" if prediction_proba > 0.5 else "green"
    st.markdown(f"### Churn Risk: <span style='color:{risk_color}'>{prediction_proba:.1%}</span>", unsafe_allow_html=True)
    st.progress(prediction_proba)

with col_b:
    st.subheader("Action Recommendation")
    if prediction_proba > 0.6:
        st.warning("High Risk! Offer a loyalty discount or contract upgrade.")
    elif prediction_proba > 0.3:
        st.info("Moderate Risk. Send a personalized engagement email.")
    else:
        st.success("Low Risk. Customer is likely satisfied.")

st.markdown("---")

# Row 2: Model Analytics
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig2, ax2 = plt.subplots()
    importances.plot(kind='barh', color='#4B8BBE', ax=ax2)
    st.pyplot(fig2)

with col_d:
    st.subheader("Model Confidence (ROC)")
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, color='orange', label=f'AUC: {auc(fpr, tpr):.2f}')
    ax1.plot([0,1],[0,1], linestyle='--')
    ax1.legend()
    st.pyplot(fig1)