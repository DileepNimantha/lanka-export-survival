import pandas as pd

from src.catalog import build_catalog
from src.data import FEATURES


def test_catalog_lists_the_newest_flows_with_names(raw):
    products = pd.DataFrame({"code": ["090240", "610910"], "description": ["Tea", "T-shirts"]})
    countries = pd.DataFrame({"iso3": ["USA", "GBR"], "name": ["USA", "United Kingdom"]})
    catalog = build_catalog(raw, products, countries)
    assert set(catalog.year) == {2015}  # tyres stopped after 2014, so only tea and t-shirts
    assert set(catalog["product"]) == {"090240", "610910"}
    assert set(catalog.destination_name) == {"USA", "United Kingdom"}
    assert set(FEATURES) <= set(catalog.columns)
    tea = catalog.set_index("product").loc["090240"]
    assert tea.years_exported == 3 and tea.hs_chapter == "HS09"
