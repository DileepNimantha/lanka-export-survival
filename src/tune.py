"""Tune the EBM on the validation year, one hyperparameter at a time, tracked in MLflow.

Each setting is varied around the current params.yaml values while the others stay fixed,
which gives one validation curve per hyperparameter for the report. Two things are scored:
PR-AUC (the gate metric) and roughness, how jagged the learned shape functions are.
Rule: best PR-AUC, but a smoother setting within TOLERANCE of it is preferred.
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import tracking
from src.config import ROOT, load_params  # noqa: E402
from src.metrics import score  # noqa: E402
from src.train import make_ebm  # noqa: E402

GRID = {
    "interactions": [0, 10, "3x"],
    "smoothing_rounds": [75, 500, 2000],
    "min_samples_leaf": [4, 50, 200],
    "max_bins": [64, 256, 1024],
    "learning_rate": [0.005, 0.015, 0.05],
}
DEFAULTS = {"smoothing_rounds": 75, "min_samples_leaf": 4}  # EBM defaults not in params.yaml
TOLERANCE = 0.002


def candidates(base: dict) -> list[tuple[str, str, object, dict]]:
    """The base settings plus every one-at-a-time change from them."""
    runs = [("base", "base", None, base)]
    for name, values in GRID.items():
        for value in values:
            if value != base[name]:
                runs.append((f"{name}={value}", name, value, {**base, name: value}))
    return runs


def roughness(model, cats: list[str]) -> float:
    """Mean total variation of the numeric shape functions, in log-odds. Lower = smoother."""
    explanation = model.explain_global()
    totals = []
    for i, term in enumerate(model.term_names_):
        if " & " in term or term in cats:
            continue
        totals.append(np.abs(np.diff(explanation.data(i)["scores"])).sum())
    return float(np.mean(totals))


METRICS = ["pr_auc", "roc_auc", "recall_top_k", "positive_rate", "mean_prediction"]
METRICS += ["roughness", "fit_seconds"]


def finished_runs(experiment: str) -> dict:
    """Runs that already finished, so an interrupted tuning session can resume."""
    runs = mlflow.search_runs(
        experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'"
    )
    if runs.empty or "tags.mlflow.runName" not in runs:
        return {}
    newest = runs.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")
    return {row["tags.mlflow.runName"]: row for _, row in newest.iterrows()}


def same_settings(run: pd.Series, settings: dict) -> bool:
    return all(run.get(f"params.{key}") == str(value) for key, value in settings.items())


def choose(results: pd.DataFrame) -> pd.Series:
    best = results.pr_auc.max()
    close = results[results.pr_auc >= best - TOLERANCE]
    return close.sort_values(["roughness", "pr_auc"], ascending=[True, False]).iloc[0]


def plot_curves(results: pd.DataFrame, base: dict, path) -> None:
    fig, axes = plt.subplots(1, len(GRID), figsize=(4 * len(GRID), 3.4), sharey=True)
    for ax, name in zip(axes, GRID, strict=True):
        rows = results[(results.param == name) | (results.run == "base")].copy()
        rows["value"] = rows["value"].where(rows.run != "base", base[name]).astype(str)
        order = [str(v) for v in GRID[name]]
        rows = rows.set_index("value").loc[order]
        ax.plot(order, rows.pr_auc, "o-", color="#3a6ea5")
        ax.set_title(name)
        twin = ax.twinx()
        twin.plot(order, rows.roughness, "s--", color="#c0392b", alpha=0.7)
        twin.tick_params(axis="y", colors="#c0392b")
    axes[0].set_ylabel("validation PR-AUC (blue)")
    twin.set_ylabel("roughness (red)", color="#c0392b")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    tracking.configure()
    params = load_params()
    m, k = params["model"], params["evaluate"]["top_k_fraction"]
    data_dir = ROOT / params["data"]["processed_dir"]
    train, val = pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "val.csv")
    X_tr, y_tr = train.drop(columns=["y"]), train["y"]
    X_val, y_val = val.drop(columns=["y"]), val["y"]
    base = {**DEFAULTS, **m["ebm"]}

    experiment = m["experiment"] + "-tuning"
    mlflow.set_experiment(experiment)
    done = finished_runs(experiment)
    results = []
    for run, name, value, settings in candidates(base):
        cached = done.get(run)
        if cached is not None and same_settings(cached, settings):
            metrics = {key: float(cached[f"metrics.{key}"]) for key in METRICS}
            results.append({"run": run, "param": name, "value": value, **metrics})
            print(f"{run}: PR-AUC {metrics['pr_auc']:.4f}  (already done, from MLflow)")
            continue
        with mlflow.start_run(run_name=run):
            started = time.time()
            model = make_ebm(settings, list(X_tr.columns), m["cat_features"]).fit(X_tr, y_tr)
            metrics = score(y_val, model.predict_proba(X_val)[:, 1], k)
            metrics["roughness"] = roughness(model, m["cat_features"])
            metrics["fit_seconds"] = round(time.time() - started, 1)
            mlflow.log_params({key: str(v) for key, v in settings.items()})
            mlflow.log_metrics(metrics)
        results.append({"run": run, "param": name, "value": value, **metrics})
        print(f"{run}: PR-AUC {metrics['pr_auc']:.4f}  roughness {metrics['roughness']:.2f}")

    table = pd.DataFrame(results)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    table.round(4).to_csv(out / "tuning.csv", index=False)
    plot_curves(table, base, ROOT / "plots" / "tuning.png")
    best = choose(table)
    chosen = base if best.run == "base" else {**base, best.param: best.value}
    print(table[["run", "pr_auc", "roc_auc", "roughness", "fit_seconds"]].round(4).to_string())
    print(f"chosen: {best.run} -> {json.dumps(chosen)}")


if __name__ == "__main__":
    main()
