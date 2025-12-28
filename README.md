# Credit Card Fraud Detection System

Every digital payment carries risk. Behind the scenes, intelligent systems must decide—within milliseconds—whether a transaction should be trusted, challenged, or blocked. This project recreates that decision-making process using machine learning.

The Credit Card Fraud Detection System is an end-to-end fraud risk engine that combines supervised learning, anomaly detection, and rule-based logic to identify suspicious transactions. It not only predicts fraud probability but also converts model outputs into meaningful risk levels and system actions through an interactive Streamlit dashboard.

This project demonstrates how real-world financial institutions design fraud detection pipelines that balance accuracy, interpretability, and operational decision-making.

---

## Project Overview

The system simulates a realistic fraud detection workflow used in banking and fintech environments. It processes transaction attributes, extracts behavioral risk signals, evaluates them using multiple machine learning models, and produces actionable outcomes.

Key capabilities include:
- Fraud probability prediction using multiple ML models  
- Anomaly detection for unseen or rare patterns  
- Risk-level classification (Low / Medium / High)  
- Automated decision logic for transaction handling  
- Interactive user interface for real-time experimentation  

---

## Key Features

- End-to-end fraud detection pipeline  
- Multiple supervised and unsupervised models  
- Risk-based decision engine  
- Interactive Streamlit dashboard  
- Preprocessed and feature-engineered dataset  
- Model persistence and reuse  
- Modular and scalable project structure  

---

## Models Used

### Supervised Learning Models
- Logistic Regression (calibrated)
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Multi-Layer Perceptron (Neural Network)

### Unsupervised Model
- Isolation Forest for anomaly detection

---

## Project Structure
<pre>
Credit Card Fraud Detection/
│
├── data/
│ └── credit_card_fraud_dataset.xlsx
│
├── models/
│ ├── logistic.pkl
│ ├── random_forest.pkl
│ ├── svm.pkl
│ ├── xgboost.pkl
│ ├── lightgbm.pkl
│ ├── mlp.pkl
│ └── isolation_forest.pkl
│
├── src/
│ └── train_models.py
│
├── app.py
├── requirements.txt
└── README.md
</pre>

---

## Dataset Description

The dataset represents transaction-level information with engineered behavioral features commonly used in fraud detection systems.

### Numerical Features
- amount  
- avg_transaction_amount  
- transaction_count_24h  
- account_age_days  
- amount_to_avg_ratio  
- high_velocity_flag  
- foreign_tx_flag  
- new_account_flag  

### Categorical Features
- transaction_type  
- merchant_category  
- currency  
- card_type  
- customer_region  
- transaction_country  
- billing_country  
- device_type  
- browser  

### Target Variable
- `is_fraud`  
  - `1` → Fraudulent transaction  
  - `0` → Legitimate transaction  

---

## Feature Engineering Strategy

To improve detection performance and realism, the dataset includes engineered behavioral indicators such as:

- Transaction velocity signals  
- Cross-border activity indicators  
- New-account risk flags  
- Spending deviation ratios  
- Behavioral anomaly markers  

These features help models learn abnormal transaction behavior rather than relying only on raw values.

---

## Model Training Pipeline

Implemented in `src/train_models.py`, the training workflow includes:

- Train–test split  
- Feature scaling for numerical variables  
- One-hot encoding of categorical features  
- Class imbalance handling using SMOTE  
- Model training and calibration  
- Serialization of trained models using Joblib  

All trained models are saved in the `models/` directory and reused during inference.

---

## Risk Scoring and Decision Logic

### Risk Level Mapping

| Fraud Probability | Risk Level |
|------------------|------------|
| ≥ 0.75           | HIGH       |
| 0.40 – 0.74      | MEDIUM     |
| < 0.40           | LOW        |

### System Actions

| Risk Level | System Decision |
|-----------|------------------|
| HIGH      | Block transaction and send for manual review |
| MEDIUM    | Require step-up authentication |
| LOW       | Approve transaction |

This separation between prediction and decision logic reflects how real financial systems operate.

---

## Streamlit Application

The Streamlit interface enables:

- Model selection at runtime  
- Interactive transaction input  
- Real-time fraud probability prediction  
- Risk classification display  
- Automated decision output  
- Clean, user-friendly dashboard layout  

---

## How to Run the Project

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate the environment
- For Windows
```bash
venv\Scripts\activate
```

-For macOS / Linux
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the models
```bash 
cd src
python train_models.py
```
This step generates trained model files inside the models/ directory.

### 5. Launch the Streamlit app
```bash
cd ..
streamlit run app.py
```

Open in your browser:
http://localhost:8501

---

## Use Cases

This system is designed to resemble how fraud detection logic is applied in real-world financial platforms. It can be adapted or extended for the following practical scenarios:

### 1. Real-Time Transaction Risk Screening
Used at the payment gateway level to evaluate incoming card transactions and classify them as low, medium, or high risk before authorization.

### 2. Fraud Analyst Decision Support
Acts as a decision-support tool for fraud analysts by providing:
- Fraud probability scores  
- Risk categorization  
- Suggested system actions  
This helps reduce manual effort and speeds up investigation workflows.

### 3. Behavioral Fraud Pattern Analysis
Helps analyze user spending behavior and detect abnormal patterns such as:
- Sudden spikes in transaction amounts  
- Unusual transaction velocity  
- Cross-border or location-inconsistent activity  

### 4. Risk Policy Prototyping for Fintech Systems
Can be used to experiment with different threshold values and rule-based logic to design and test fraud policies before deploying them in production systems.

### 5. Portfolio-Grade Machine Learning Demonstration
Serves as a strong portfolio project showcasing:
- End-to-end ML pipeline design  
- Model training and evaluation  
- Decision logic integration  
- Interactive deployment using Streamlit  

---

## Future Enhancements

- Add model explainability using SHAP or LIME  
- Improve risk scoring with weighted and dynamic thresholds  
- Expose prediction APIs using FastAPI or Flask  
- Store transactions and predictions in a database  
- Enable model monitoring and drift detection  
- Deploy using Docker and cloud platforms (AWS / GCP / Azure)  
- Add dashboards, batch uploads, and improved UI visualizations  

---

## Collaborators
- Tapasya Patel 
- Varun Patel

---
