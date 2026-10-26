"""Explain the trained EBM three ways and check its behaviour against trade research.

1. EBM shape functions: the model's own, exact explanation (it is additive in log-odds).
2. SHAP (permutation, model-agnostic): does an outside method rank the features the same way?
3. Partial dependence: the average effect of each top feature on the predicted risk.
Plus local explanations of three test-year flows, and the direction of each feature's effect
compared with the trade-duration literature (e.g. Besedes and Prusa, 2006).
"""

import json

import matplotlib

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay  # noqa: E402

from src import tracking
from src.config import ROOT, load_params  # noqa: E402
from src.train import MODEL_PATH  # noqa: E402

OUT = ROOT / "plots" / "explain"
# Expected sign of each effect on the risk of a flow stopping: -1 = higher value, lower risk.
EXPECTED = {
    "value": -1,  # bigger flows last longer
    "value_prev": -1,
    "years_exported": -1,  # the longer a flow has lasted, the more likely it continues
    "n_destinations": -1,  # products sold to many countries are more established
    "market_share": -1,  # a bigger foothold in the market is harder to lose
}
LOG_SCALE = {"value", "value_prev", "value_prev2", "market_value", "world_value", "market_share"}


def main_effects(model) -> pd.Series:
    """EBM importance of each single-feature term (interaction terms left out)."""
    importance = pd.Series(model.term_importances(), index=model.term_names_)
    return importance[[t for t in model.term_names_ if " & " not in t]]


def plot_shapes(model, features: list[str], cats: list[str], path) -> None:
    """The learned shape function of each feature: its contribution to the log-odds of stopping."""
    explanation = model.explain_global()
    names = list(model.term_names_)
    fig, axes = plt.subplots(2, (len(features) + 1) // 2, figsize=(13, 7))
    for ax, feature in zip(axes.flat, features, strict=False):
        data = explanation.data(names.index(feature))
        scores = np.asarray(data["scores"])
        if feature in cats:  # the 6 lowest- and 6 highest-risk categories
            top = pd.Series(scores, index=data["names"]).sort_values().iloc[np.r_[0:6, -6:0]]
            top.plot.barh(ax=ax, color=np.where(top > 0, "#c0392b", "#2e86c1"))
        else:
            edges = np.asarray(data["names"], dtype=float)
            ax.step(edges[:-1], scores, where="post", color="#3a6ea5")
            if feature in LOG_SCALE:
                ax.set_xscale("log")
            ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(feature)
    for ax in list(axes.flat)[len(features) :]:
        ax.axis("off")
    fig.supylabel("contribution to log-odds of stopping")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def shap_values(model, sample: pd.DataFrame, background: pd.DataFrame, cats: list[str]):
    """Permutation SHAP on the log-odds. SHAP needs numbers, so categories are coded and
    decoded again inside the prediction function."""
    codes = {c: pd.Index(sorted(set(sample[c]) | set(background[c]))) for c in cats}

    def encode(df):
        return df.assign(**{c: codes[c].get_indexer(df[c]) for c in cats}).astype(float)

    def log_odds(values):
        df = pd.DataFrame(values, columns=sample.columns)
        df = df.assign(**{c: codes[c][df[c].round().astype(int)] for c in cats})
        p = np.clip(model.predict_proba(df)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    masker = shap.maskers.Independent(encode(background).to_numpy(), max_samples=len(background))
    explainer = shap.PermutationExplainer(log_odds, masker)
    return explainer(encode(sample).to_numpy()), encode(sample)


def directions(model, sample: pd.DataFrame, features: list[str]) -> list[dict]:
    """Sign of each feature's effect: rank correlation between its value and its contribution."""
    contributions = pd.DataFrame(model.eval_terms(sample), columns=model.term_names_)
    rows = []
    for feature in features:
        rho = spearmanr(sample[feature], contributions[feature]).statistic
        observed = int(np.sign(rho)) if abs(rho) > 0.1 else 0
        expected = EXPECTED.get(feature)
        rows.append(
            {
                "feature": feature,
                "spearman": round(float(rho), 3),
                "observed": {1: "higher -> more risk", -1: "higher -> less risk", 0: "flat"}[
                    observed
                ],
                "expected": None if expected is None else "higher -> less risk",
                "agrees": None if expected is None else observed == expected,
            }
        )
    return rows


def local_examples(model, eval_df: pd.DataFrame, path) -> list[dict]:
    """Three flows: a correct high-risk call, a correct low-risk call and a confident miss,
    taken from the 30 most common destinations so the examples are typical, not exotic."""
    X = eval_df.drop(columns=["y"])
    p = model.predict_proba(X)[:, 1]
    common = X.destination.isin(X.destination.value_counts().index[:30]).to_numpy()
    stopped, continued = common & (eval_df.y == 1), common & (eval_df.y == 0)
    picks = {
        "stopped, predicted high risk": np.where(stopped, p, -1).argmax(),
        "continued, predicted low risk": np.where(continued, p, 2).argmin(),
        "continued, but predicted high risk": np.where(continued, p, -1).argmax(),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    examples = []
    for ax, (label, i) in zip(axes, picks.items(), strict=True):
        row = X.iloc[[i]]
        terms = pd.Series(model.eval_terms(row)[0], index=model.term_names_)
        top = terms[terms.abs().nlargest(8).index].iloc[::-1]
        top.plot.barh(ax=ax, color=np.where(top > 0, "#c0392b", "#2e86c1"))
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(
            f"{label}\nrisk {p[i]:.0%}, {row.hs_chapter.iat[0]} to {row.destination.iat[0]}"
        )
        examples.append(
            {
                "case": label,
                "risk": round(float(p[i]), 3),
                "stopped": int(eval_df.y.iat[i]),
                "features": json.loads(row.iloc[0].to_json()),
                "top_terms": {k: round(float(v), 3) for k, v in top.iloc[::-1].items()},
            }
        )
    fig.supxlabel("contribution to log-odds of stopping (red = raises risk)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return examples


def main() -> None:
    tracking.configure()
    params = load_params()
    e, cats = params["explain"], params["model"]["cat_features"]
    eval_df = pd.read_csv(ROOT / params["data"]["processed_dir"] / "eval.csv")
    X = eval_df.drop(columns=["y"])
    model = joblib.load(MODEL_PATH)
    OUT.mkdir(parents=True, exist_ok=True)

    ebm_importance = main_effects(model).sort_values(ascending=False)
    top = list(ebm_importance.index[: e["top_features"]])
    plot_shapes(model, top, cats, OUT / "ebm_shapes.png")

    sample = X.sample(e["sample_rows"], random_state=e["seed"])
    background = X.sample(e["background_rows"], random_state=e["seed"] + 1)
    values, coded = shap_values(model, sample, background, cats)
    shap_importance = pd.Series(np.abs(values.values).mean(0), index=X.columns)
    shap.summary_plot(values.values, coded, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=120)
    plt.close("all")
    agreement = spearmanr(ebm_importance[X.columns], shap_importance[X.columns]).statistic

    numeric = [f for f in top if f not in cats][:4]
    as_float = sample.astype({f: float for f in numeric})  # PDP refuses integer columns
    fig, ax = plt.subplots(figsize=(15, 3.5))
    PartialDependenceDisplay.from_estimator(
        model,
        as_float,
        numeric,
        kind="both",
        subsample=50,
        random_state=e["seed"],
        ax=ax,
        n_cols=len(numeric),
    )
    fig.tight_layout()
    fig.savefig(OUT / "pdp.png", dpi=120)
    plt.close(fig)

    summary = {
        "ebm_importance": ebm_importance.round(3).to_dict(),
        "shap_importance": shap_importance.sort_values(ascending=False).round(3).to_dict(),
        "ebm_vs_shap_rank_correlation": round(float(agreement), 3),
        "directions": directions(model, sample, [c for c in X.columns if c not in cats]),
        "examples": local_examples(model, eval_df, OUT / "local_examples.png"),
    }
    (ROOT / "explain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ("ebm_vs_shap_rank_correlation", "directions")}, indent=2
        )
    )

    run_id = json.loads((ROOT / "run_info.json").read_text())["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("ebm_vs_shap_rank_correlation", summary["ebm_vs_shap_rank_correlation"])
        mlflow.log_artifacts(str(OUT), artifact_path="explain")
        mlflow.log_artifact(str(ROOT / "explain_summary.json"))


if __name__ == "__main__":
    main()
