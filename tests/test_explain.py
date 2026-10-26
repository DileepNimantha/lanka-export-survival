import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

from src.data import CAT_FEATURES, FEATURES
from src.explain import directions, main_effects
from tests.conftest import make_rows


@pytest.fixture(scope="module")
def data_and_model():
    # Synthetic flows where bigger export values make stopping much less likely.
    df = make_rows(600, seed=3)
    rnd = np.random.default_rng(3)
    df["y"] = (rnd.random(len(df)) < 1 / (1 + np.exp(np.log(df["value"]) - 4))).astype(int)
    feature_types = ["nominal" if c in CAT_FEATURES else "auto" for c in FEATURES]
    model = ExplainableBoostingClassifier(feature_types=feature_types, interactions=2, outer_bags=1)
    return df[FEATURES], model.fit(df[FEATURES], df["y"])


def test_main_effects_leave_out_interaction_terms(data_and_model):
    _, model = data_and_model
    effects = main_effects(model)
    assert set(effects.index) == set(FEATURES)
    assert not any(" & " in term for term in effects.index)


def test_directions_find_the_known_effect(data_and_model):
    X, model = data_and_model
    rows = {r["feature"]: r for r in directions(model, X, ["value"])}
    assert rows["value"]["observed"] == "higher -> less risk"
    assert rows["value"]["agrees"] is True
