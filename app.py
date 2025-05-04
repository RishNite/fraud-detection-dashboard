import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io

# ---------------------------
# ✅ STREAMLIT CONFIG
# ---------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
)

# ---------------------------
# 🎯 TITLE & SUBTITLE
# ---------------------------
st.markdown(
    "<h1 style='text-align: center;'>💳 Credit Card Fraud Detection Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>Powered by Random Forest, Scikit-learn & Streamlit</h4>",
    unsafe_allow_html=True,
)

# ---------------------------
# 📤 FILE UPLOADER
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload a CSV file with transaction data", type=["csv"]
)

if uploaded_file is None:
    st.warning("⚠️ Please upload a CSV file to begin.")
    st.stop()

# ✅ Decode the uploaded CSV (fix for Streamlit Cloud)
stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
df = pd.read_csv(stringio)

# ---------------------------
# 📄 RAW DATA PREVIEW
# ---------------------------
st.subheader("📄 Raw Uploaded Data")
st.dataframe(df.head(10))

# ---------------------------
# 🔍 LOAD MODEL + SCALER + FEATURES
# ---------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# ---------------------------
# 🎯 PREPROCESSING
# ---------------------------
categorical_cols = [
    "Customer Location",
    "Product Category",
    "Payment Method",
    "Device Used",
]
df_processed = pd.get_dummies(
    df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True
)

# Add any missing cols that the model expects, then reorder
for col in feature_names:
    if col not in df_processed.columns:
        df_processed[col] = 0
df_processed = df_processed[feature_names]

# ---------------------------
# 🧮 SCALING
# ---------------------------
scaled_data = scaler.transform(df_processed)

# ---------------------------
# 🔮 PREDICTIONS
# ---------------------------
probabilities = model.predict_proba(scaled_data)[:, 1]

# ---------------------------
# 🎚️ THRESHOLD SLIDER
# ---------------------------
threshold = st.slider(
    "Set fraud probability threshold", min_value=0.0, max_value=1.0, value=0.50
)
labels = (probabilities >= threshold).astype(int)

# ---------------------------
# 📊 DONUT CHART SUMMARY
# ---------------------------
st.subheader("📊 Prediction Summary")
fraud_pct = labels.mean()
legit_pct = 1 - fraud_pct

fig1, ax1 = plt.subplots()
wedges, _ = ax1.pie(
    [fraud_pct, legit_pct],
    colors=["crimson", "limegreen"],
    startangle=90,
    wedgeprops=dict(width=0.4, edgecolor="white"),
)
ax1.text(
    0,
    0,
    f"{fraud_pct*100:.1f}%\nFraud",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="crimson",
)
ax1.legend(
    wedges,
    ["Fraud", "Legit"],
    title="Class",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
)
ax1.axis("equal")
plt.tight_layout()
st.pyplot(fig1)

# ---------------------------
# 📈 PROBABILITY DISTRIBUTION
# ---------------------------
st.subheader("📈 Fraud Probability Distribution")
st.caption("How confident is the model across all predictions?")
fig2, ax2 = plt.subplots()
ax2.hist(probabilities, bins=25, color="skyblue", edgecolor="black")
ax2.set_xlabel("Fraud Probability")
ax2.set_ylabel("Number of Transactions")
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------
# 📌 FEATURE IMPORTANCE
# ---------------------------
st.subheader("📌 Feature Importance (What matters most for fraud detection)")
importances = model.feature_importances_
feat_imp = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values("Importance", ascending=False)
# Exclude Customer Location dummy columns
feat_imp = feat_imp[~feat_imp["Feature"].str.startswith("Customer Location")].head(20)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.barh(feat_imp["Feature"], feat_imp["Importance"], color="purple")
ax3.set_xlabel("Importance")
ax3.set_ylabel("Feature")
ax3.set_title("Top 20 Important Features")
ax3.invert_yaxis()
plt.tight_layout()
st.pyplot(fig3)

# ---------------------------
# ⚠️ FILTERED TRANSACTIONS
# ---------------------------
st.subheader(f"⚠️ Transactions with Fraud Probability ≥ {threshold}")
filtered = df.loc[probabilities >= threshold]
st.dataframe(filtered)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Filtered Transactions",
    data=csv,
    file_name="filtered_fraud_transactions.csv",
    mime="text/csv",
)
