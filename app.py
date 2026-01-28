import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# ----------------- Load Dataset -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("house.csv")
    
    # Create target variable for demo (High price vs Low price)
    df['price_category'] = (df['price'] > df['price'].median()).astype(int)
    
    # Drop unnecessary columns
    df = df.drop(['id','date','price'], axis=1)
    
    # Fill missing values (no inplace to avoid warnings)
    numerical_features = [
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
        'waterfront','view','condition','grade','sqft_above',
        'sqft_basement','yr_built','yr_renovated','lat','long',
        'sqft_living15','sqft_lot15'
    ]
    categorical_features = ['zipcode']
    
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].mean())
    
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode categorical features
    le = LabelEncoder()
    df['zipcode'] = le.fit_transform(df['zipcode'])
    
    X = df.drop('price_category', axis=1)
    y = df['price_category']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

X_train, X_test, y_train, y_test, scaler, feature_cols = load_data()

# ----------------- Sidebar Inputs -----------------
st.sidebar.header("Applicant Details")

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=1000)
coapplicant_income = st.sidebar.number_input("Co-Applicant Income", min_value=0, step=500)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=100)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=12, step=12)
credit_history = st.sidebar.radio("Credit History", options=["Yes", "No"])
employment_status = st.sidebar.selectbox("Employment Status", options=["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", options=["Urban", "Semi-Urban", "Rural"])

# ----------------- Preprocessing Function -----------------
def preprocess_input(data, scaler, feature_cols):
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    df['Credit_History'] = df['Credit_History'].map({'Yes': 1, 'No': 0})
    df['Employment_Status'] = df['Employment_Status'].map({'Salaried': 1, 'Self-Employed': 0})
    area_mapping = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
    df['Property_Area'] = df['Property_Area'].map(area_mapping)
    
    # Scale numeric features
    numeric_cols = ['Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Amount_Term']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # Make sure columns order matches training
    df = df[feature_cols]
    return df

# ----------------- Stacking Model -----------------
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(max_depth=6)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]
meta_model = LogisticRegression(max_iter=1000)

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=False
)

# ----------------- Train Stacking -----------------
stacking_clf.fit(X_train, y_train)

# ----------------- Model Architecture Display -----------------
st.subheader("🔹 Stacking Model Architecture")
st.write("Base Models Used:")
st.write("- Logistic Regression")
st.write("- Decision Tree")
st.write("- Random Forest")
st.write("Meta Model Used:")
st.write("- Logistic Regression")

# ----------------- Prediction -----------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):
    input_data = {
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Loan_Amount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Employment_Status': employment_status,
        'Property_Area': property_area
    }
    
    processed_data = preprocess_input(input_data, scaler, feature_cols)
    
    # Base model predictions
    base_preds = {}
    for name, model in base_models:
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
    
    # Business Explanation
    st.subheader("💡 Business Explanation")
    explanation = (
        "Based on income, credit history, employment status, property area, and combined predictions "
        "from multiple models, the applicant is likely / unlikely to repay the loan.\n\n"
        f"Therefore, the stacking model predicts {final_result.split()[1]}."
    )
    st.info(explanation)
