"""Front end: pick a Sri Lankan export flow by name, see the risk that it stops next year
and why. Feature values come from the catalog (data/catalog/flows.csv); predictions and
explanations come from the API, so the app shows exactly what the deployed model says.

Run locally:  make serve  (API)  and  make ui  (this app)
"""

import math
import os
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CATALOG_PATH = Path(os.getenv("CATALOG_PATH", ROOT / "data" / "catalog" / "flows.csv"))
FEATURES = [
    "value", "value_prev", "value_prev2", "years_exported", "market_value", "market_share",
    "market_suppliers", "world_value", "price_ratio", "n_destinations",
    "n_products_to_destination", "hs_chapter", "destination",
]  # fmt: skip
LABELS = {
    "value": "Export value this year",
    "value_prev": "Export value last year",
    "value_prev2": "Export value two years ago",
    "years_exported": "Years exported in a row",
    "market_value": "Size of this market",
    "market_share": "Sri Lanka's share of this market",
    "market_suppliers": "Countries supplying this market",
    "world_value": "World trade in the product",
    "price_ratio": "Price vs world average",
    "n_destinations": "Countries Sri Lanka sells the product to",
    "n_products_to_destination": "Products Sri Lanka sells to this country",
    "hs_chapter": "Product group (HS chapter)",
    "destination": "Destination country",
}
DEFAULT_PRODUCT = "090230"  # black tea in small packs, Sri Lanka's signature export
MONEY = {"value", "value_prev", "value_prev2", "market_value", "world_value"}  # thousand USD


def usd(thousands: float) -> str:
    amount = thousands * 1000
    for limit, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if amount >= limit:
            return f"${amount / limit:.1f}{unit}"
    return f"${amount:.0f}"


def show(feature: str, value) -> str:
    if feature in MONEY:
        return usd(value) if value > 0 else "not exported"
    if feature == "market_share":
        return f"{value:.2%}"
    if feature == "price_ratio":
        return f"{value:.2f}x"
    if feature == "years_exported":
        return "3 or more" if value >= 3 else str(int(value))
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return str(value)


def term_label(term: str, features: dict) -> str:
    parts = term.split(" & ")
    return " × ".join(f"{LABELS[p]} = {show(p, features[p])}" for p in parts)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@st.cache_data
def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_PATH, dtype={"product": str})


def explain(features: dict) -> dict:
    response = httpx.post(f"{API_URL}/explain", json=features, timeout=120)
    response.raise_for_status()
    return response.json()


def contribution_chart(result: dict, features: dict, top: int = 8) -> alt.Chart:
    terms = pd.DataFrame(result["contributions"])
    shown, rest = terms.iloc[:top].copy(), terms.iloc[top:]
    shown["label"] = [term_label(t, features) for t in shown.term]
    if len(rest):
        other = {"label": f"{len(rest)} smaller effects", "contribution": rest.contribution.sum()}
        shown = pd.concat([shown, pd.DataFrame([other])], ignore_index=True)
    shown["effect"] = shown.contribution.map(lambda c: "raises risk" if c > 0 else "lowers risk")
    return (
        alt.Chart(shown)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="effect on the risk (log-odds)"),
            y=alt.Y(
                "label:N",
                sort=None,
                title=None,
                # never drop a label: every bar must say which fact it is
                axis=alt.Axis(labelLimit=520, labelOverlap=False, labelFontSize=12),
            ),
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(
                    domain=["raises risk", "lowers risk"], range=["#c0392b", "#2e8b57"]
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["label", alt.Tooltip("contribution:Q", format="+.3f")],
        )
        .properties(height=alt.Step(34))  # a fixed height per bar
    )


st.set_page_config(page_title="Sri Lanka export survival", page_icon="🇱🇰", layout="wide")
st.title("Will this export survive next year?")
st.caption(
    "Pick one of Sri Lanka's export flows (a product sold to a country). An Explainable "
    "Boosting Machine trained on CEPII BACI trade data estimates the chance that Sri Lanka "
    "stops exporting it next year, and shows exactly which facts drive that estimate."
)

catalog = load_catalog()
year = int(catalog.year.iat[0])

with st.sidebar:
    st.header("Choose an export flow")
    totals = catalog.groupby(["product", "product_name"]).value.sum().sort_values(ascending=False)
    product, product_name = st.selectbox(
        "Product (type to search)",
        totals.index,
        index=next((i for i, p in enumerate(totals.index) if p[0] == DEFAULT_PRODUCT), 0),
        format_func=lambda p: f"{p[0]} · {p[1][:80]}",
    )
    flows = catalog[catalog["product"] == product]
    row_id = st.selectbox(
        f"Destination ({len(flows)} countries in {year})",
        flows.index,
        format_func=lambda i: f"{flows.at[i, 'destination_name']} · {usd(flows.at[i, 'value'])}",
    )
    flow = flows.loc[row_id]

    st.header("What if?")
    st.caption("Change the export history to see how the risk responds.")
    value = st.number_input(
        "Export value this year (thousand USD)", min_value=0.001, value=float(flow.value)
    )
    value_prev = st.number_input(
        "Last year (0 = not exported)", min_value=0.0, value=float(flow.value_prev)
    )
    value_prev2 = st.number_input(
        "Two years ago (0 = not exported)", min_value=0.0, value=float(flow.value_prev2)
    )
    price_ratio = st.slider(
        "Price vs world average", 0.01, max(10.0, float(flow.price_ratio)),
        float(flow.price_ratio), step=0.01,
    )  # fmt: skip
    st.divider()
    st.caption(f"Model API: {API_URL}")

features = {f: flow[f] for f in FEATURES}
features.update(value=value, value_prev=value_prev, value_prev2=value_prev2)
features["price_ratio"] = price_ratio
features["years_exported"] = 1 + (value_prev > 0) + (value_prev > 0 and value_prev2 > 0)
features["market_share"] = min(value / flow.market_value, 1.0)
features = {k: v.item() if hasattr(v, "item") else v for k, v in features.items()}

st.subheader(f"{product_name[:120]} → {flow.destination_name}")
st.caption(f"HS {product} · {usd(flow.value)} exported in {year}")

try:
    with st.spinner("Asking the model (the first request can take ~30 s while the API wakes up)"):
        result = explain(features)
except httpx.HTTPError as error:
    st.error(f"Could not reach the model API at {API_URL}: {error}")
    st.stop()

risk, typical = result["probability"], sigmoid(result["baseline"])
band = "High" if risk >= 0.5 else "Medium" if risk >= 0.2 else "Low"
left, middle, right = st.columns(3)
left.metric(f"Risk it stops in {year + 1}", f"{risk:.0%}" if risk >= 0.01 else f"{risk:.1%}")
middle.metric("Risk level", band)
right.metric("Starting point (before this flow's facts)", f"{typical:.0%}")
st.progress(risk)

reasons = [c for c in result["contributions"] if abs(c["contribution"]) > 0.05][:3]
if reasons:
    lines = [
        f"- {term_label(c['term'], features)} **{'raises' if c['contribution'] > 0 else 'lowers'}**"
        " the risk"
        for c in reasons
    ]
    # "$" starts a formula in Streamlit markdown, so dollar amounts are escaped
    st.markdown("Main reasons:\n" + "\n".join(lines).replace("$", "\\$"))

st.markdown("##### Why: each fact's effect on the risk")
st.altair_chart(contribution_chart(result, features), width="stretch")
st.caption(
    "The model adds these effects to the starting point. Red bars push the risk up, "
    "green bars pull it down. Labels with × are pairs of facts the model learned together."
)
with st.expander("The facts the model used", expanded=True):
    table = pd.DataFrame(
        {"Fact": [LABELS[f] for f in FEATURES], "Value": [show(f, features[f]) for f in FEATURES]}
    )
    st.table(table.set_index("Fact"))

st.caption(
    f"Model version {result['model_version']} · data: CEPII BACI HS12 (Etalab 2.0 licence) · "
    "the estimate is a probability from past patterns, not a forecast of policy or events."
)
