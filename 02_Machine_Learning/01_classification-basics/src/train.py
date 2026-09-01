from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
RESULTS, ASSETS, MODELS = ROOT / "results", ROOT / "assets", ROOT / "models"
SEED = 42

def load_and_split():
    data = pd.read_csv(RAW)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    y = data.pop("Churn").eq("Yes").astype(int)
    x = data.drop(columns="customerID")
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=.4, stratify=y, random_state=SEED)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=.5, stratify=y_temp, random_state=SEED)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def preprocess(x):
    numeric = x.select_dtypes(include="number").columns
    categorical = x.select_dtypes(exclude="number").columns
    return ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])

def metrics(name, model, x, y):
    pred, prob = model.predict(x), model.predict_proba(x)[:, 1]
    return {"model": name, "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, prob)}

def main():
    if not RAW.exists(): raise FileNotFoundError("Run download_data.py first.")
    for path in (RESULTS, ASSETS, MODELS): path.mkdir(exist_ok=True)
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_and_split()
    candidates = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=8, class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    trained, rows = {}, []
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocess(x_train)), ("model", estimator)]).fit(x_train, y_train)
        trained[name], row = pipe, metrics(name, pipe, x_valid, y_valid)
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(["recall", "roc_auc"], ascending=False)
    selected_name = comparison.iloc[0]["model"]
    final = Pipeline([("preprocess", preprocess(pd.concat([x_train, x_valid]))), ("model", candidates[selected_name])]).fit(pd.concat([x_train, x_valid]), pd.concat([y_train, y_valid]))
    test = metrics(selected_name, final, x_test, y_test)
    joblib.dump(final, MODELS / "churn_model.joblib")
    comparison.to_csv(RESULTS / "validation-model-comparison.csv", index=False)
    (RESULTS / "test-metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    (RESULTS / "split-summary.json").write_text(json.dumps({"train":len(x_train),"validation":len(x_valid),"test":len(x_test)}), encoding="utf-8")
    RocCurveDisplay.from_estimator(final, x_test, y_test); plt.tight_layout(); plt.savefig(ASSETS / "roc-curve.png", dpi=160); plt.close()
    features = final.named_steps["preprocess"].get_feature_names_out()
    estimator = final.named_steps["model"]
    values = estimator.coef_[0] if hasattr(estimator, "coef_") else estimator.feature_importances_
    top = pd.DataFrame({"feature":features,"importance":values}).assign(abs=lambda d:d.importance.abs()).nlargest(15,"abs").sort_values("importance")
    plt.figure(figsize=(9,6)); plt.barh(top.feature, top.importance); plt.axvline(0,color="gray"); plt.tight_layout(); plt.savefig(ASSETS / "feature-importance.png",dpi=160); plt.close()
    print(json.dumps(test))

if __name__ == "__main__": main()
