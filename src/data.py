"""Turn Sri Lanka's raw export flows into a model table, validate it against a contract,
and split it by year into train, validation, test and "future" sets.

One row = one export flow (an HS6 product sent to one destination country) in one year.
Target y = 1 if that flow stops: Sri Lanka does not export that product to that country
the following year. Features only use information available at the end of the year.
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.config import ROOT, load_params

CAT_FEATURES = ["hs_chapter", "destination"]

# The data contract: every feature the model uses, with its type and allowed values.
FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "value": pa.Column(float, pa.Check.gt(0)),  # thousand USD, this year
        "value_prev": pa.Column(float, pa.Check.ge(0)),  # last year, 0 = not exported
        "value_prev2": pa.Column(float, pa.Check.ge(0)),  # two years ago
        "years_exported": pa.Column(int, pa.Check.in_range(1, 3)),  # in a row, 3 = 3 or more
        "market_value": pa.Column(float, pa.Check.gt(0)),  # destination's imports of the product
        "market_share": pa.Column(float, pa.Check.in_range(0, 1, include_min=False)),
        "market_suppliers": pa.Column(int, pa.Check.ge(1)),  # countries selling it there
        "world_value": pa.Column(float, pa.Check.gt(0)),  # world trade in the product
        "price_ratio": pa.Column(float, pa.Check.in_range(0.01, 100)),  # unit price vs world
        "n_destinations": pa.Column(int, pa.Check.ge(1)),  # countries Sri Lanka sells it to
        "n_products_to_destination": pa.Column(int, pa.Check.ge(1)),
        "hs_chapter": pa.Column(str, pa.Check.str_matches(r"^HS\d{2}$")),
        "destination": pa.Column(str, pa.Check.str_matches(r"^[A-Z0-9]{3}$")),  # ISO3
    },
    coerce=True,
)
FEATURES = list(FEATURE_SCHEMA.columns)


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"product": str, "destination": str})


def build_features(raw: pd.DataFrame, first_year: int) -> pd.DataFrame:
    """Add history, market and label columns. The last year has no label, so it is dropped."""
    key = ["destination", "product"]
    df = raw.copy()

    def shifted(years_back: int, name: str) -> pd.DataFrame:
        past = df[["year", *key, "value"]].rename(columns={"value": name})
        return past.assign(year=past.year + years_back)

    df = df.merge(shifted(1, "value_prev"), on=["year", *key], how="left")
    df = df.merge(shifted(2, "value_prev2"), on=["year", *key], how="left")
    df = df.merge(shifted(-1, "value_next"), on=["year", *key], how="left")
    df[["value_prev", "value_prev2"]] = df[["value_prev", "value_prev2"]].fillna(0.0)

    exported_last_year = df.value_prev > 0
    df["years_exported"] = 1 + exported_last_year + (exported_last_year & (df.value_prev2 > 0))
    df["market_share"] = (df.value / df.market_value).clip(upper=1.0)
    unit_price = df.value / df.quantity
    world_price = df.world_value / df.world_quantity
    ratio = (unit_price / world_price).replace([np.inf, -np.inf], np.nan)
    df["price_ratio"] = ratio.fillna(1.0).clip(0.01, 100)  # unknown quantity -> neutral 1.0
    df["n_destinations"] = df.groupby(["year", "product"]).destination.transform("size")
    df["n_products_to_destination"] = df.groupby(["year", "destination"]).product.transform("size")
    df["hs_chapter"] = "HS" + df["product"].str[:2]
    df["y"] = df.value_next.isna().astype(int)

    last_year = df.year.max()
    df = df[(df.year >= first_year) & (df.year < last_year)]
    return df.drop(columns=["value_next"]).sort_values(["year", *key]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the features against the contract. Raises pandera.errors.SchemaError on bad data."""
    checked = FEATURE_SCHEMA.validate(df[FEATURES])
    return checked.assign(y=df["y"].astype(int).to_numpy())


def split_years(table: pd.DataFrame, p: dict):
    """Split by year, so every model is judged on years it has never seen."""
    return tuple(
        validate(table[table.year.isin(p[f"{part}_years"])]).reset_index(drop=True)
        for part in ("train", "val", "test", "future")
    )


def build_sets(table: pd.DataFrame, production: pd.DataFrame, p: dict):
    train, val, test, future = split_years(table, p)
    if len(production) == 0:
        # First model: train on the training years, tune on val, judge it on the test years.
        return train, val, test, future
    # Retraining: the newest labelled production rows become the evaluation set,
    # and everything older (all history + older production rows) is used for training.
    production = production.reset_index(drop=True)
    cut = int(len(production) * (1 - p["holdout_frac"]))
    older, holdout = production.iloc[:cut], production.iloc[cut:]
    history = pd.concat([train, val, test, older], ignore_index=True)
    return history, val, holdout, future


def main() -> None:
    p = load_params()["data"]
    table = build_features(load_raw(ROOT / p["raw"]), p["first_year"])
    leaked = set(p["drop_columns"]) & set(FEATURES)
    if leaked:
        raise ValueError(f"leakage columns reached the model data: {leaked}")
    production = pd.read_csv(ROOT / p["production"])
    if len(production):
        production = validate(production.sort_values("timestamp"))

    train, val, evaluation, future = build_sets(table, production, p)
    out = ROOT / p["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("eval", evaluation), ("future", future)):
        part.to_csv(out / f"{name}.csv", index=False)
    print(
        f"train={len(train)}  val={len(val)}  eval={len(evaluation)}  future={len(future)}  "
        f"production rows used={len(production)}"
    )
    print(
        f"stop rate: train={train.y.mean():.3f}  val={val.y.mean():.3f}  "
        f"eval={evaluation.y.mean():.3f}  future={future.y.mean():.3f}"
    )


if __name__ == "__main__":
    main()
