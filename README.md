# Fraud Detection Dashboard

A web-based interactive dashboard built with **Streamlit** that detects potentially fraudulent credit card transactions using a machine learning model trained on real-world e-commerce data.

🔗 **Live App**: [fraud-detection-dashboard-rishik.streamlit.app](https://fraud-detection-dashboard-rishik.streamlit.app)

---

## Features

- Upload your own transaction CSV file
- Preprocessing with one-hot encoding & scaling
- Model predicts fraud probability for each transaction
- Adjustable fraud probability threshold
- Visualizations:
  - Fraud vs Legit pie/donut chart
  - Fraud probability histogram
  - Top feature importances

---

## Machine Learning

- **Model**: Random Forest Classifier with hyperparameter tuning
- **Preprocessing**:
  - One-hot encoding of categorical variables
  - Standardization via `StandardScaler`
- **SMOTE** used to address class imbalance

---

## File Structure

| File             | Description                          |
|------------------|--------------------------------------|
| `app.py`         | Streamlit frontend app code          |
| `model.pkl`      | Trained Random Forest model          |
| `scaler.pkl`     | Scaler for feature normalization     |
| `features.pkl`   | List of training feature names       |
| `requirements.txt` | Python dependencies for deployment |

---

## Run Locally

```bash
git clone https://github.com/RishNite/fraud-detection-dashboard.git
cd fraud-detection-dashboard
pip install -r requirements.txt
streamlit run app.py
