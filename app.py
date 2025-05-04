import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# 🎯 Title
st.markdown("<h1 style='text-align: center;'>💳 Credit Card Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Powered by Random Forest, Scikit-learn & Streamlit</h4>", unsafe_allow_html=True)

# 📤 File Upload
uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file.getvalue().decode("utf-8"))
    import io
    df = pd.read_csv(io.StringIO(df))

    st.subheader("📄 Raw Uploaded Data")
    st.dataframe(df.head(10))

    # 🔍 Load model + features + scaler
    model = joblib.load("model.pkl")
    feature_names = joblib.load("features.pkl")
    scaler = joblib.load("scaler.pkl")

    # 🎯 Preprocessing (don't drop Customer Location yet)
    categorical_cols = ["Customer Location", "Product Category", "Payment Method", "Device Used"]
    df_processed = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)

    # ✅ Add missing columns and reorder to match training features
    for col in feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[feature_names]

    # 🧮 Scale
    scaled_data = scaler.transform(df_processed)

    # 🔮 Predict
    probabilities = model.predict_proba(scaled_data)[:, 1]

    # 🎚️ Threshold
    threshold = st.slider("Set fraud probability threshold", 0.0, 1.0, 0.5)
    labels = (probabilities >= threshold).astype(int)

    # 📊 Summary Pie Chart
    # 📊 Summary Pie Chart (Improved Visibility)
    # 📊 Summary Pie Chart (Final Clean Fix)
    # 📊 Summary Pie Chart (Improved Label Visibility)
    # 📊 Summary Donut Chart (with central Fraud %)
    st.subheader("📊 Prediction Summary")

    fraud_pct = np.mean(labels)
    legit_pct = 1 - fraud_pct

    fig1, ax1 = plt.subplots()
    # Draw a donut
    wedges, _ = ax1.pie(
        [fraud_pct, legit_pct],
        colors=["crimson", "limegreen"],
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white")
    )

    # Center text for fraud percentage
    ax1.text(
        0, 0,
        f"{fraud_pct*100:.1f}%\nFraud",
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        color="crimson"
    )

    # Legend to the right
    ax1.legend(
        wedges,
        ["Fraud", "Legit"],
        title="Class",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    ax1.axis("equal")  # Keep it circular
    plt.tight_layout()
    st.pyplot(fig1)





    # 📈 Probability Distribution
    st.subheader("📈 Fraud Probability Distribution")
    st.caption("How confident is the model across all predictions?")
    fig2, ax2 = plt.subplots()
    ax2.hist(probabilities, bins=25, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Fraud Probability")
    ax2.set_ylabel("Number of Transactions")
    st.pyplot(fig2)

    # 📌 Feature Importance (Top 20, hide Customer Location just in plot)
    st.subheader("📌 Feature Importance (What matters most for fraud detection)")
    importances = model.feature_importances_
    top_features_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top_features_df = top_features_df[~top_features_df["Feature"].str.startswith("Customer Location")]
    top_features_df = top_features_df.sort_values("Importance", ascending=False).head(20)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(top_features_df["Feature"], top_features_df["Importance"], color="purple")
    ax3.set_xlabel("Importance")
    ax3.set_ylabel("Feature")
    ax3.set_title("Top 20 Important Features")
    ax3.invert_yaxis()
    st.pyplot(fig3)

    # ⚠️ Filtered Transactions
    st.subheader(f"⚠️ Transactions with Fraud Probability ≥ {threshold}")
    filtered = df[probabilities >= threshold]
    st.dataframe(filtered)
    st.download_button("📥 Download Filtered Transactions", filtered.to_csv(index=False), "filtered_fraud.csv", "text/csv")
