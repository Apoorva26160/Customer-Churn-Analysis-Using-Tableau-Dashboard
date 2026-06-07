"""
Customer Churn Analysis
Feature engineering and ML model to predict churn, with exports for Tableau dashboards.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    # Tenure segments
    df["tenure_segment"] = pd.cut(
        df["tenure_months"],
        bins=[0, 6, 12, 24, 48, float("inf")],
        labels=["0-6m", "6-12m", "1-2yr", "2-4yr", "4yr+"],
    )

    # Monthly charge per product
    df["charge_per_product"] = df["monthly_charges"] / (df["num_products"].clip(lower=1))

    # Revenue at risk
    df["revenue_at_risk"] = df["monthly_charges"] * df["tenure_months"]

    # Engagement score (0–1)
    engagement_cols = [c for c in ["has_online_backup", "has_tech_support", "has_streaming"] if c in df.columns]
    if engagement_cols:
        df["engagement_score"] = df[engagement_cols].mean(axis=1)

    # Contract type encoding
    if "contract_type" in df.columns:
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        df["contract_encoded"] = df["contract_type"].map(contract_map).fillna(0)

    return df


def encode_categoricals(df, cat_cols):
    df = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


# ── MODEL ─────────────────────────────────────────────────────────────────────

def train_churn_model(df, target="churned", test_size=0.2):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = encode_categoricals(df, cat_cols)

    drop_cols = [target, "customer_id"] if "customer_id" in df.columns else [target]
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Churn Model Results ===")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== Top 10 Churn Drivers ===")
    print(importance.head(10))

    return model, X_test, y_test, y_pred, y_prob


# ── TABLEAU EXPORT ────────────────────────────────────────────────────────────

def export_for_tableau(df, model, feature_cols, output_path="churn_tableau_export.csv"):
    """Add churn probability scores to the dataset for Tableau visualisation."""
    df = df.copy()
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["churn_risk_segment"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    df.to_csv(output_path, index=False)
    print(f"Tableau-ready export saved to {output_path}")
    return df


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "customer_id": range(n),
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 120, n),
        "num_products": np.random.randint(1, 6, n),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "has_online_backup": np.random.randint(0, 2, n),
        "has_tech_support": np.random.randint(0, 2, n),
        "has_streaming": np.random.randint(0, 2, n),
        "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "churned": np.random.choice([0, 1], n, p=[0.73, 0.27]),
    })

    df = engineer_features(df)
    model, X_test, y_test, y_pred, y_prob = train_churn_model(df)
