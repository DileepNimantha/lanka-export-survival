import pandas as pd
import pandera.errors
import pytest

from src.config import load_params
from src.data import FEATURES, build_features, build_sets, validate

SPLIT = {
    "train_years": [2014, 2015],
    "val_years": [2019],
    "test_years": [2020],
    "future_years": [2022],
    "holdout_frac": 0.3,
}


def test_contract_accepts_valid_rows(rows):
    assert len(validate(rows)) == len(rows)


def test_contract_rejects_share_given_as_percent(rows):
    rows.loc[0, "market_share"] = 35.0  # an upstream system sends 35 instead of 0.35
    with pytest.raises(pandera.errors.SchemaError):
        validate(rows)


def test_contract_rejects_bad_destination_code(rows):
    rows.loc[0, "destination"] = "Sri Lanka"
    with pytest.raises(pandera.errors.SchemaError):
        validate(rows)


def test_leakage_column_is_never_a_feature():
    for column in load_params()["data"]["drop_columns"]:
        assert column not in FEATURES


def test_label_marks_flows_that_stop_next_year(raw):
    table = build_features(raw, first_year=2013).set_index(["year", "product"])
    assert table.loc[(2013, "401110"), "y"] == 0  # tyres still exported in 2014
    assert table.loc[(2014, "401110"), "y"] == 1  # tyres stop after 2014
    assert table.loc[(2014, "090240"), "y"] == 0
    assert 2015 not in table.index.get_level_values("year")  # last year has no label


def test_history_features_only_look_back(raw):
    table = build_features(raw, first_year=2013).set_index(["year", "product"])
    tea = table.loc[(2014, "090240")]
    assert tea.years_exported == 3 and tea.value_prev == 100.0 and tea.value_prev2 == 100.0
    assert tea.price_ratio == pytest.approx(0.5)  # 10 per tonne vs world 20 per tonne
    shirts = table.loc[(2014, "610910")]
    assert shirts.years_exported == 1 and shirts.value_prev == 0.0  # gap in 2013
    assert shirts.hs_chapter == "HS61"


def test_first_model_splits_by_year(rows):
    table = rows.assign(year=[2014, 2015, 2019, 2020, 2022] * 40)
    train, val, evaluation, future = build_sets(table, pd.DataFrame(), SPLIT)
    assert (len(train), len(val), len(evaluation), len(future)) == (80, 40, 40, 40)
    assert "year" not in train.columns  # the year is never a feature


def test_retraining_evaluates_on_newest_production_rows(rows):
    table = rows.assign(year=[2014, 2015, 2019, 2020, 2022] * 40)
    production = validate(rows.head(100))
    train, _, evaluation, _ = build_sets(table, production, SPLIT)
    assert len(evaluation) == 30  # newest 30% of production rows
    assert len(train) == 80 + 40 + 40 + 70  # train, val and test years plus older production
