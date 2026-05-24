# models/churn_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

BASE        = Path(__file__).resolve().parent.parent
PROCESSED   = BASE / "data" / "processed"
MODELS_DIR  = BASE / "models"
SCREENSHOTS = BASE / "screenshots"

def load_features():
    df = pd.read_csv(PROCESSED / "customer_features.csv")
    print(f"  Loaded customer features: {df.shape}")
    print(f"  Churn rate: {df['is_churned'].mean()*100:.1f}%")
    return df

def prepare_features(df):
    features = [
        "total_orders",
        "total_spent",
        "avg_order_value",
        "avg_review_score",
        "recency_days",
        "customer_lifetime_days",
    ]
    df[features] = df[features].fillna(df[features].median())
    X = df[features]
    y = df["is_churned"]
    print(f"\n  Features: {features}")
    print(f"  X shape: {X.shape}")
    print(f"  Class balance: {y.value_counts().to_dict()}")
    return X, y, features

def train_models(X_train, X_test, y_train, y_test):
    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    }

    print("\n" + "="*55)
    print("  MODEL COMPARISON")
    print("="*55)
    print(f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model": model, "scaler": scaler if name == "Logistic Regression" else None,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "y_pred": y_pred, "y_prob": y_prob,
        }
        print(f"  {name:<25} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f} {auc:>8.3f}")

    return results

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(feat_df["feature"], feat_df["importance"], color="#185FA5")
    ax.set_title("XGBoost Feature Importance for Churn Prediction", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar, val in zip(bars, feat_df["importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "feature_importance.png", dpi=150)
    plt.show()
    print("  Saved feature_importance.png")

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active", "Churned"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "confusion_matrix.png", dpi=150)
    plt.show()
    print("  Saved confusion_matrix.png")

def save_model(model, feature_names):
    model_data = {"model": model, "feature_names": feature_names, "model_type": "XGBoost"}
    path = MODELS_DIR / "churn_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  Model saved to {path}")

def score_all_customers(model, feature_names, df):
    X = df[feature_names].fillna(df[feature_names].median())
    df = df.copy()
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["churn_risk"] = pd.cut(df["churn_probability"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    df.to_csv(PROCESSED / "customer_churn_scores.csv", index=False)
    print(f"\n  Churn Risk Distribution:")
    print(df["churn_risk"].value_counts().to_string())
    print(f"\n  Saved customer_churn_scores.csv")
    return df

if __name__ == "__main__":
    print("="*55)
    print("  RETAILX - CHURN PREDICTION MODEL")
    print("="*55)

    df = load_features()
    X, y, feature_names = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    results = train_models(X_train, X_test, y_train, y_test)

    best = results["XGBoost"]
    best_model = best["model"]

    print("\n" + "="*55)
    print("  XGBOOST - DETAILED REPORT")
    print("="*55)
    print(classification_report(y_test, best["y_pred"], target_names=["Active", "Churned"]))

    plot_feature_importance(best_model, feature_names)
    plot_confusion_matrix(y_test, best["y_pred"], "XGBoost")
    save_model(best_model, feature_names)
    scored = score_all_customers(best_model, feature_names, df)

    print("\nChurn model complete. Ready for forecasting.\n")
