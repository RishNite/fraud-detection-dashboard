import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io

# ───── STREAMLIT SETUP ─────
st.set_page_config("Credit Card Fraud Detection", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>💳 Credit Card Fraud Detection Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align:center;color:gray;'>Powered by Random Forest, Scikit-learn & Streamlit</h4>",
    unsafe_allow_html=True,
)

# ───── FILE UPLOAD ─────
uploaded = st.file_uploader("Upload a CSV file", type="csv")
if not uploaded:
    st.warning("⚠️ Please upload a CSV file to begin.")
    st.stop()

# decode bytes to str, then load into pandas
s = io.StringIO(uploaded.getvalue().decode("utf-8"))
df = pd.read_csv(s)

st.subheader("📄 Raw Data Preview")
st.dataframe(df.head(10))

# ───── LOAD ARTIFACTS ─────
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ───── PREPROCESS ─────
cats = ["Customer Location","Product Category","Payment Method","Device Used"]
df_enc = pd.get_dummies(df, columns=[c for c in cats if c in df.columns], drop_first=True)

# ensure every trained feature in place
for col in features:
    if col not in df_enc:
        df_enc[col] = 0
df_enc = df_enc[features]

# ───── SCALE & PREDICT ─────
X_scaled = scaler.transform(df_enc)
probs = model.predict_proba(X_scaled)[:,1]

# ───── THRESHOLD SLIDER ─────
th = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5)
labels = (probs >= th).astype(int)

# ───── DONUT CHART ─────
st.subheader("📊 Prediction Summary")
pct = labels.mean()
fig, ax = plt.subplots()
w, _ = ax.pie(
    [pct,1-pct],
    colors=["crimson","limegreen"],
    startangle=90,
    wedgeprops=dict(width=0.4,edgecolor="white")
)
ax.text(0,0,f"{pct*100:.1f}%\nFraud",ha="center",va="center",fontsize=18,color="crimson")
ax.legend(w,["Fraud","Legit"],loc="center left",bbox_to_anchor=(1,0,0.5,1))
ax.axis("equal")
plt.tight_layout()
st.pyplot(fig)

# ───── PROB DISTRIBUTION ─────
st.subheader("📈 Fraud Probability Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(probs, bins=25, color="skyblue", edgecolor="black")
ax2.set_xlabel("Fraud Probability"); ax2.set_ylabel("Count")
plt.tight_layout()
st.pyplot(fig2)

# ───── FEATURE IMPORTANCE ─────
st.subheader("📌 Feature Importance (Top 20)")
imp = model.feature_importances_
df_imp = pd.DataFrame({"Feature":features,"Importance":imp})
df_imp = df_imp[~df_imp["Feature"].str.startswith("Customer Location")].nlargest(20,"Importance")
fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.barh(df_imp["Feature"],df_imp["Importance"],color="purple")
ax3.invert_yaxis(); ax3.set_xlabel("Importance")
plt.tight_layout()
st.pyplot(fig3)

# ───── FILTERED TRANSACTIONS ─────
st.subheader(f"⚠️ Transactions with Probability ≥ {th}")
filtered = df.loc[probs>=th]
st.dataframe(filtered)
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", data=csv, mime="text/csv")
