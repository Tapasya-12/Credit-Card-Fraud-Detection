import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = "../data/credit_card_fraud_dataset.xlsx"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_excel(DATA_PATH)
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

numeric_features = [
    "amount",
    "avg_transaction_amount",
    "transaction_count_24h",
    "account_age_days",
    "amount_to_avg_ratio",
    "high_velocity_flag",
    "foreign_tx_flag",
    "new_account_flag"
]

categorical_features = [
    "transaction_type",
    "merchant_category",
    "currency",
    "card_type",
    "customer_region",
    "transaction_country",
    "billing_country",
    "device_type",
    "browser"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

models = {
    "logistic": CalibratedClassifierCV(
        LogisticRegression(max_iter=1000),
        method="sigmoid"
    ),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "svm": SVC(probability=True),
    "xgboost": XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=10,
        random_state=42
    ),
    "lightgbm": LGBMClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
}

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    print(f"{name} ROC-AUC: {auc:.4f}")
    joblib.dump(pipeline, f"{MODEL_DIR}/{name}.pkl")


iso_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", IsolationForest(contamination=0.03, random_state=42))
])

iso_pipeline.fit(X_train)
joblib.dump(iso_pipeline, f"{MODEL_DIR}/isolation_forest.pkl")

print("\nAll models trained successfully")
