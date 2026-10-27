"""Build the front end's lookup table: every export flow in the newest year, with its
features and readable product and country names.

These are the flows whose future is still unknown, so they are what a user would ask
about: "will Sri Lanka still export this product to this country next year?"
"""

import pandas as pd

from src.config import ROOT, load_params
from src.data import FEATURE_SCHEMA, FEATURES, add_features, load_raw


def build_catalog(raw: pd.DataFrame, products: pd.DataFrame, countries: pd.DataFrame):
    df = add_features(raw)
    df = df[df.year == df.year.max()]
    checked = FEATURE_SCHEMA.validate(df[FEATURES])
    names = products.set_index("code")["description"]
    countries = countries.set_index("iso3")["name"]
    catalog = pd.DataFrame(
        {
            "year": df.year.to_numpy(),
            "product": df["product"].to_numpy(),
            "product_name": df["product"].map(names).fillna("(no description)").to_numpy(),
            "destination_name": df.destination.map(countries).fillna(df.destination).to_numpy(),
        }
    )
    catalog = pd.concat([catalog, checked.reset_index(drop=True)], axis=1)
    return catalog.sort_values("value", ascending=False).reset_index(drop=True)


def main() -> None:
    p = load_params()["data"]
    products = pd.read_csv(ROOT / p["products"], dtype={"code": str})
    countries = pd.read_csv(ROOT / p["countries"])
    catalog = build_catalog(load_raw(ROOT / p["raw"]), products, countries)
    out = ROOT / p["catalog"]
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out, index=False)
    print(
        f"Saved {out}: {len(catalog):,} flows in {catalog.year.iat[0]}, "
        f"{catalog['product'].nunique():,} products to {catalog.destination.nunique()} countries"
    )


if __name__ == "__main__":
    main()
