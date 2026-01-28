import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split

# ----------------- App Title -----------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a Stacking Ensemble Machine Learning model to predict "
    "whether a loan will be approved by combining multiple ML models for better decision making."
)

# ----------------- Load Dataset (house.csv) -----------------
@st.cache_data
def load_house_data():
    df = pd.read_csv("house.csv")
    return df

house_df = load_house_data()
st.subheader("🏠 House Dataset Preview (Reference Only)")
st.dataframe(house_df.head())
st.write(f"Dataset Shape: {house_df.shape}")

# ----------------- Sidebar Inputs -----------------
st.sidebar.header("Applicant Details")

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=1000)
coapplicant_income = st.sidebar.number_input("Co-Applicant Income", min_value=0, step=500)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, step=100)
loan_term = st.sidebar.number_input("Loan Amount Term (in months)", min_value=12, step=12)
credit_history = st.sidebar.radio("Credit History", options=["Yes", "No"])
employment_status = st.sidebar.selectbox("Employment Status", options=["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", options=["Urban", "Semi-Urban", "Rural"])

# ----------------- Create Mock Loan Dataset -----------------
@st.cache_data
def create_loan_dataset(n_samples=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "Applicant_Income": np.random.randint(2000, 15000, n_samples),
        "Coapplicant_Income": np.random.randint(0, 8000, n_samples),
        "Loan_Amount": np.random.randint(50, 600, n_samples),
        "Loan_Amount_Term": np.random.choice([120, 180, 240, 360], n_samples),
        "Credit_History": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "Employment_Status": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "Property_Area": np.random.choice([0, 1, 2], n_samples),  # 0=Rural,1=Semi-Urban,2=Urban
    })
    # Target: Loan Approved (1) / Rejected (0)
    df['Loan_Status'] = (
        (df['Applicant_Income'] + df['Coapplicant_Income'] > 5000) &
        (df['Credit_History'] == 1)
    ).astype(int)
    return df

loan_df = create_loan_dataset()

# ----------------- Train Stacking Model -----------------
X = loan_df.drop("Loan_Status", axis=1)
y = loan_df["Loan_Status"]

# Scale all features (numeric + categorical)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]
meta_model = LogisticRegression(max_iter=1000)

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=False
)
stacking_clf.fit(X_train, y_train)

# ----------------- Model Architecture -----------------
st.subheader("🔹 Stacking Model Architecture")
st.write("Base Models Used:")
st.write("- Logistic Regression")
st.write("- Decision Tree")
st.write("- Random Forest")
st.write("Meta Model Used:")
st.write("- Logistic Regression")

# ----------------- Input Preprocessing -----------------
def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # Encode categorical inputs
    df['Credit_History'] = df['Credit_History'].map({'Yes': 1, 'No': 0})
    df['Employment_Status'] = df['Employment_Status'].map({'Salaried': 1, 'Self-Employed': 0})
    df['Property_Area'] = df['Property_Area'].map({"Urban": 2, "Semi-Urban": 1, "Rural": 0})
    
    # Scale ALL features together (numeric + encoded categorical)
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    
    return df_scaled

# ----------------- Prediction -----------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):
    input_data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Loan_Amount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Employment_Status": employment_status,
        "Property_Area": property_area
    }
    
    processed_data = preprocess_input(input_data)
    
    # Base model predictions from fitted stacking classifier
    base_preds = {}
    for name, model in stacking_clf.named_estimators_.items():
        pred = model.predict(processed_data)[0]
        base_preds[name] = "Approved" if pred == 1 else "Rejected"
    
    # Stacking final prediction
    final_pred = stacking_clf.predict(processed_data)[0]
    final_result = "✅ Loan Approved" if final_pred == 1 else "❌ Loan Rejected"
    color = "green" if final_pred == 1 else "red"
    
    # Display results
    st.subheader("📊 Base Model Predictions")
    for name, pred in base_preds.items():
        st.write(f"{name} → {pred}")
    
    st.subheader("🧠 Final Stacking Decision")
    st.markdown(f"<h2 style='color:{color}'>{final_result}</h2>", unsafe_allow_html=True)
    
    # Confidence score
    if hasattr(stacking_clf, "predict_proba"):
        conf_score = stacking_clf.predict_proba(processed_data)[0][final_pred]*100
        st.write(f"📈 Confidence Score: {conf_score:.2f}%")
    
    # Business explanation
    st.subheader("💡 Business Explanation")
    explanation = (
        "Based on income, credit history, employment status, property area, and combined predictions "
        "from multiple models, the applicant is likely / unlikely to repay the loan.\n\n"
        f"Therefore, the stacking model predicts {final_result.split()[1]}."
    )
    st.info(explanation)
