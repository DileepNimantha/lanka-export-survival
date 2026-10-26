from src.config import ROOT, load_params
from src.data import CAT_FEATURES


def test_params_file_exists():
    assert (ROOT / "params.yaml").exists()


def test_next_year_value_is_dropped():
    assert "value_next" in load_params()["data"]["drop_columns"]


def test_split_years_do_not_overlap_and_run_forward():
    data = load_params()["data"]
    parts = [data[f"{part}_years"] for part in ("train", "val", "test", "future")]
    assert all(max(a) < min(b) for a, b in zip(parts, parts[1:], strict=False))
    assert min(parts[0]) >= data["first_year"]


def test_model_categorical_features_match_the_contract():
    assert load_params()["model"]["cat_features"] == CAT_FEATURES
