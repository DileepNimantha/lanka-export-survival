"""Download the CEPII BACI trade data and keep Sri Lanka's export flows.

BACI has one CSV per year with every exporter -> importer -> product flow in the world.
We keep Sri Lanka's rows plus, for each of them, the size of that destination market
and of the world market, which can only be measured from the full global data.
The 1.27 GB zip is cached in data/cache/ (not tracked) so it is downloaded only once.
"""

import re
import shutil
import urllib.request
import zipfile

import pandas as pd

from src.config import ROOT, load_params

YEAR_FILE = re.compile(r"BACI_HS\d+_Y(\d{4})_V\d+\.csv$")


def fetch(url: str) -> zipfile.ZipFile:
    path = ROOT / "data" / "cache" / url.rsplit("/", 1)[1]
    if not (path.exists() and zipfile.is_zipfile(path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} (about 1.3 GB)")
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as resp, open(partial, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        partial.rename(path)
    return zipfile.ZipFile(path)


def exporter_flows(year: pd.DataFrame, exporter: int) -> pd.DataFrame:
    """One exporter's flows, with the destination market and world market for each product."""
    own = year[year.i == exporter]
    pairs = own[["j", "k"]].drop_duplicates()
    market = (
        year.merge(pairs, on=["j", "k"])
        .groupby(["j", "k"])
        .agg(market_value=("v", "sum"), market_suppliers=("i", "nunique"))
        .reset_index()
    )
    world = (
        year[year.k.isin(own.k.unique())]
        .groupby("k")
        .agg(world_value=("v", "sum"), world_quantity=("q", "sum"))
        .reset_index()
    )
    return own.merge(market, on=["j", "k"]).merge(world, on="k")


def main() -> None:
    p = load_params()["data"]
    archive = fetch(p["url"])
    names = archive.namelist()
    countries = pd.read_csv(archive.open(next(n for n in names if n.startswith("country_codes"))))
    products = pd.read_csv(
        archive.open(next(n for n in names if n.startswith("product_codes"))), dtype={"code": str}
    )

    flows = []
    for name in sorted(n for n in names if YEAR_FILE.search(n)):
        year = pd.read_csv(archive.open(name), dtype={"k": str}, engine="pyarrow")
        year.columns = year.columns.str.strip()
        flows.append(exporter_flows(year, p["exporter_code"]))
        print(f"{name}: {len(year):,} world flows, {len(flows[-1]):,} from Sri Lanka")

    iso3 = countries.set_index("country_code")["country_iso3"]
    raw = pd.concat(flows, ignore_index=True)
    # pyarrow ignores dtype=str and reads HS codes as numbers, dropping the leading zero
    # (tea 090240 -> 90240), so the 6-digit code is restored here.
    raw = raw.assign(destination=raw.j.map(iso3), k=raw.k.astype(str).str.zfill(6)).rename(
        columns={"t": "year", "k": "product", "v": "value", "q": "quantity"}
    )
    columns = ["year", "destination", "product", "value", "quantity"]
    columns += ["market_value", "market_suppliers", "world_value", "world_quantity"]
    out = ROOT / p["raw"]
    out.parent.mkdir(parents=True, exist_ok=True)
    raw[columns].to_csv(out, index=False)
    products.to_csv(ROOT / p["products"], index=False)
    print(f"Saved {out} ({len(raw):,} rows, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
