import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AI Bank Reconciliation", layout="wide")

st.title("🏦 AI-Based Bank Statement Reconciliation")

# ----------------------------------
# Upload Files
# ----------------------------------

col1, col2 = st.columns(2)

with col1:
    bank_file = st.file_uploader("Upload Bank Statement CSV", type=["csv"])

with col2:
    company_file = st.file_uploader("Upload Company Transactions CSV", type=["csv"])

if bank_file and company_file:

    bank_df = pd.read_csv(bank_file)
    company_df = pd.read_csv(company_file)

    st.subheader("📂 Uploaded Data Preview")
    st.write("Bank Data")
    st.dataframe(bank_df.head())

    st.write("Company Data")
    st.dataframe(company_df.head())

    # ----------------------------------
    # MATCHING LOGIC
    # ----------------------------------

    st.subheader("🔎 Transaction Matching")

    bank_df["Amount"] = bank_df["Debit"] + bank_df["Credit"]

    merged = pd.merge(
        bank_df,
        company_df,
        on=["Date", "Amount"],
        how="outer",
        indicator=True
    )

    merged["Match_Status"] = merged["_merge"].map({
        "both": "Matched",
        "left_only": "Only in Bank",
        "right_only": "Only in Company"
    })

    match_counts = merged["Match_Status"].value_counts()

    st.write("Matching Summary")
    st.write(match_counts)

    fig1, ax1 = plt.subplots()
    ax1.pie(match_counts, labels=match_counts.index, autopct='%1.1f%%')
    st.pyplot(fig1)

    # ----------------------------------
    # LOAD MODEL & ENCODERS
    # ----------------------------------

    model = joblib.load("fraud_model.pkl")
    le_desc = joblib.load("desc_encoder.pkl")
    le_type = joblib.load("type_encoder.pkl")

    # Safe Encoding Function
    def safe_transform(encoder, values):
        return [
            encoder.transform([v])[0] if v in encoder.classes_
            else 0
            for v in values
        ]

    company_df["Description"] = safe_transform(le_desc, company_df["Description"])
    company_df["Type"] = safe_transform(le_type, company_df["Type"])

    # ----------------------------------
    # FRAUD PREDICTION
    # ----------------------------------

    st.subheader("🤖 Fraud Detection")

    X = company_df[["Description", "Amount", "Type"]]

    if "Fraud_Label" in company_df.columns:
        y_true = company_df["Fraud_Label"]
        y_pred = model.predict(X)

        accuracy = accuracy_score(y_true, y_pred)
        st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

        cm = confusion_matrix(y_true, y_pred)
        fig2, ax2 = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax2)
        st.pyplot(fig2)

    else:
        y_pred = model.predict(X)
        st.write("Fraud Predictions Generated")

    company_df["Fraud_Prediction"] = y_pred

    st.write("Total Fraud Detected:", sum(y_pred))

    # ----------------------------------
    # ANOMALY DETECTION
    # ----------------------------------

    st.subheader("⚠️ Anomaly Detection")

    iso = IsolationForest(contamination=0.05, random_state=42)
    company_df["Anomaly"] = iso.fit_predict(company_df[["Amount"]])
    company_df["Anomaly"] = company_df["Anomaly"].apply(lambda x: 1 if x == -1 else 0)

    st.write("Total Anomalies:", company_df["Anomaly"].sum())

    # ----------------------------------
    # FLAGGED TRANSACTIONS
    # ----------------------------------

    st.subheader("🚨 Flagged Transactions")

    flagged = company_df[
        (company_df["Fraud_Prediction"] == 1) |
        (company_df["Anomaly"] == 1)
    ]

    st.dataframe(flagged)

    # ----------------------------------
    # FINAL DASHBOARD METRICS
    # ----------------------------------

    st.subheader("📊 Dashboard Summary")

    col3, col4, col5 = st.columns(3)

    col3.metric("Matched Transactions", match_counts.get("Matched", 0))
    col4.metric("Fraud Detected", sum(y_pred))
    col5.metric("Anomalies", company_df["Anomaly"].sum())