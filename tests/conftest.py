"""Shared test fixtures. Tests use small synthetic data so they run without DVC or MLflow."""

import random

import pandas as pd
import pytest

CHAPTERS = ["HS09", "HS40", "HS61", "HS62", "HS08", "HS03"]  # tea, rubber, garments, ...
DESTINATIONS = ["USA", "GBR", "IND", "DEU", "ARE", "CHN", "JPN", "S19"]


def make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        value = round(rnd.uniform(0.01, 5000), 3)
        market = value * rnd.uniform(1, 200)
        years = rnd.randint(1, 3)
        rows.append(
            {
                "value": value,
                "value_prev": round(rnd.uniform(0.01, 5000), 3) if years > 1 else 0.0,
                "value_prev2": round(rnd.uniform(0.01, 5000), 3) if years > 2 else 0.0,
                "years_exported": years,
                "market_value": round(market, 3),
                "market_share": value / market,
                "market_suppliers": rnd.randint(1, 120),
                "world_value": round(market * rnd.uniform(5, 500), 3),
                "price_ratio": round(rnd.uniform(0.2, 5), 3),
                "n_destinations": rnd.randint(1, 80),
                "n_products_to_destination": rnd.randint(1, 900),
                "hs_chapter": rnd.choice(CHAPTERS),
                "destination": rnd.choice(DESTINATIONS),
                "y": int(rnd.random() < 0.3),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def rows() -> pd.DataFrame:
    return make_rows(200)


@pytest.fixture
def raw() -> pd.DataFrame:
    """A tiny raw extract: three products to two countries over four years."""
    records = []
    for year in (2012, 2013, 2014, 2015):
        for destination, product, years_present in [
            ("USA", "090240", (2012, 2013, 2014, 2015)),  # tea, exported every year
            ("USA", "401110", (2013, 2014)),  # tyres, starts 2013 and stops after 2014
            ("GBR", "610910", (2012, 2014, 2015)),  # t-shirts, gap in 2013
        ]:
            if year in years_present:
                records.append(
                    {
                        "year": year,
                        "destination": destination,
                        "product": product,
                        "value": 100.0,
                        "quantity": 10.0,
                        "market_value": 1000.0,
                        "market_suppliers": 5,
                        "world_value": 50000.0,
                        "world_quantity": 2500.0,
                    }
                )
    return pd.DataFrame(records)
