"""Compare baselines with the Explainable Boosting Machine on the validation year,
all tracked in MLflow. This is exploration, separate from the production pipeline in dvc.yaml.
"""

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import tracking
from src.config import ROOT, load_params
from src.metrics import score
from src.train import make_ebm

# Money and count columns are heavily skewed, so linear models see them on a log scale.
SKEWED = ["value", "value_prev", "value_prev2", "market_value", "world_value"]
SKEWED += ["market_share", "price_ratio", "market_suppliers", "n_destinations"]
SKEWED += ["n_products_to_destination"]


class NewFlowRule:
    """Baseline with no learning: a flow that was not exported last year is at risk."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        risk = (X["years_exported"] == 1).astype(float).to_numpy()
        return np.column_stack([1 - risk, risk])


def build_models(params: dict, columns: list[str]) -> dict:
    cats = params["model"]["cat_features"]
    numeric = [c for c in columns if c not in cats]
    other = [c for c in numeric if c not in SKEWED]
    encode = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
            ("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), SKEWED),
            ("num", StandardScaler(), other),
        ]
    )
    ordinal = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats)],
        remainder="passthrough",
    )
    return {
        "rule_new_flow": NewFlowRule(),
        "logistic_regression": make_pipeline(encode, LogisticRegression(max_iter=2000)),
        "decision_tree": make_pipeline(
            ordinal, DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=42)
        ),
        "random_forest": make_pipeline(
            ordinal,
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            categorical_features="from_dtype", max_iter=400, learning_rate=0.05, random_state=42
        ),
        "ebm": make_ebm(params["model"]["ebm"], columns, cats),
    }


def main() -> None:
    tracking.configure()
    params = load_params()
    cats = params["model"]["cat_features"]
    k = params["evaluate"]["top_k_fraction"]
    data_dir = ROOT / params["data"]["processed_dir"]
    train, val = pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "val.csv")
    X_tr, y_tr = train.drop(columns=["y"]), train["y"]
    X_val, y_val = val.drop(columns=["y"]), val["y"]

    mlflow.set_experiment(params["model"]["experiment"] + "-experiments")
    results = []
    for name, model in build_models(params, list(X_tr.columns)).items():
        X_fit, X_eval = X_tr, X_val
        if name == "hist_gradient_boosting":  # needs pandas 'category' columns
            X_fit = X_tr.astype({c: "category" for c in cats})
            X_eval = X_val.astype({c: pd.CategoricalDtype(X_fit[c].cat.categories) for c in cats})
        with mlflow.start_run(run_name=name):
            started = time.time()
            model.fit(X_fit, y_tr)
            metrics = score(y_val, model.predict_proba(X_eval)[:, 1], k)
            metrics["fit_seconds"] = round(time.time() - started, 1)
            mlflow.log_param("model_family", name)
            mlflow.log_metrics(metrics)
        results.append({"model": name, **metrics})
        print(f"{name}: PR-AUC {metrics['pr_auc']:.3f} in {metrics['fit_seconds']:.0f}s")

    table = pd.DataFrame(results).set_index("model").round(3)
    columns = ["pr_auc", "roc_auc", "recall_top_k", "fit_seconds"]
    print(table[columns].sort_values("pr_auc", ascending=False))


if __name__ == "__main__":
    main()
