"""Score the trained model on the evaluation set, plot what it learned, log both to MLflow."""

import json

import matplotlib

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from src import tracking
from src.config import ROOT, load_params  # noqa: E402
from src.metrics import score  # noqa: E402
from src.train import MODEL_PATH  # noqa: E402


def plot_importance(model, path, top: int = 15) -> None:
    """EBM's global importance: the average absolute contribution of each term to the score."""
    terms = pd.Series(model.term_importances(), index=model.term_names_).nlargest(top)
    plt.figure(figsize=(8, 5))
    terms.iloc[::-1].plot.barh(color="#3a6ea5")
    plt.xlabel("mean absolute contribution (log-odds of stopping)")
    plt.title("What the EBM relies on most")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close("all")


def main() -> None:
    tracking.configure()
    params = load_params()
    k = params["evaluate"]["top_k_fraction"]
    df = pd.read_csv(ROOT / params["data"]["processed_dir"] / "eval.csv")
    X, y = df.drop(columns=["y"]), df["y"]

    model = joblib.load(MODEL_PATH)
    metrics = score(y, model.predict_proba(X)[:, 1], k)
    metrics["eval_rows"] = len(df)
    (ROOT / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    plot_path = ROOT / "plots" / "importance.png"
    plot_path.parent.mkdir(exist_ok=True)
    plot_importance(model, plot_path)

    run_id = json.loads((ROOT / "run_info.json").read_text())["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_{name}": value for name, value in metrics.items()})
        mlflow.log_artifact(str(plot_path))


if __name__ == "__main__":
    main()
