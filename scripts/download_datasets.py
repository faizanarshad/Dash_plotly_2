from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DownloadResult:
    name: str
    path: Path | None
    ok: bool
    note: str = ""


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, out_path: Path, *, force: bool = False, timeout_s: int = 60) -> DownloadResult:
    if out_path.exists() and not force:
        return DownloadResult(name=out_path.name, path=out_path, ok=True, note="exists (skipped)")

    r = requests.get(url, stream=True, timeout=timeout_s)
    r.raise_for_status()
    _ensure_dir(out_path.parent)
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    return DownloadResult(name=out_path.name, path=out_path, ok=True, note=f"downloaded from {url}")


def download_world_happiness(*, force: bool = False) -> DownloadResult:
    # World Happiness Report provides free downloads for Figure 2.1 on their data-sharing page.
    url = "https://files.worldhappiness.report/WHR26_Data_Figure_2.1.xlsx"
    out_path = RAW_DIR / "world_happiness" / "WHR26_Data_Figure_2.1.xlsx"
    return _download_file(url, out_path, force=force)


def _sdmx_json_v2_to_df(payload: dict) -> pd.DataFrame:
    """
    Convert SDMX-JSON v2.0 data message to a tidy DataFrame.

    OECD's legacy endpoint currently returns `application/vnd.sdmx.data+json; ... version=2`.
    In practice, the OECD response wraps content under `data`, with:
    - structure under `data.structures[0].dimensions.{series,observation}`
    - observations under `data.dataSets[0].series[<series-key>].observations[<time-index>]`
    """
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    structures = (data.get("structures") or []) if isinstance(data, dict) else []
    if not structures:
        raise RuntimeError("SDMX payload missing data.structures[0]")
    structure = structures[0]

    dims = structure.get("dimensions") or {}
    series_dims = dims.get("series") or []
    obs_dims = dims.get("observation") or []
    if not obs_dims:
        raise RuntimeError("SDMX payload missing observation dimensions")

    def _dim_codes(dim: dict) -> list[str]:
        vals = dim.get("values") or []
        return [v.get("id") or v.get("name") or "" for v in vals]

    obs_dim_ids = [d.get("id") or "" for d in obs_dims]
    obs_dim_codes = [_dim_codes(d) for d in obs_dims]

    data_sets = data.get("dataSets") or []
    if not data_sets:
        raise RuntimeError("SDMX payload missing data.dataSets[0]")
    ds0 = data_sets[0]

    rows: list[dict] = []

    # Case A: dataset uses series[...] with per-series observations (common for many OECD datasets)
    series = ds0.get("series")
    if isinstance(series, dict) and series:
        series_dim_ids = [d.get("id") or "" for d in series_dims]
        series_dim_codes = [_dim_codes(d) for d in series_dims]

        for series_key, series_obj in series.items():
            idxs = [int(x) for x in series_key.split(":")] if series_key else []
            base = {}
            for dim_id, codes, idx in zip(series_dim_ids, series_dim_codes, idxs):
                base[dim_id] = codes[idx] if 0 <= idx < len(codes) else None

            observations = (series_obj or {}).get("observations") or {}
            for obs_key, obs in observations.items():
                # Observation key is typically the time index (single dimension), but keep it generic.
                oidxs = [int(x) for x in str(obs_key).split(":")] if obs_key is not None else []
                row = dict(base)
                for dim_id, codes, idx in zip(obs_dim_ids, obs_dim_codes, oidxs):
                    row[dim_id] = codes[idx] if 0 <= idx < len(codes) else None
                row["value"] = obs[0] if isinstance(obs, list) and obs else obs
                rows.append(row)

        return pd.DataFrame(rows)

    # Case B: dataset stores everything directly under observations with multi-dim keys (e.g. BLI)
    observations = ds0.get("observations")
    if isinstance(observations, dict) and observations:
        for obs_key, obs in observations.items():
            oidxs = [int(x) for x in str(obs_key).split(":")] if obs_key is not None else []
            row = {}
            for dim_id, codes, idx in zip(obs_dim_ids, obs_dim_codes, oidxs):
                row[dim_id] = codes[idx] if 0 <= idx < len(codes) else None
            row["value"] = obs[0] if isinstance(obs, list) and obs else obs
            rows.append(row)
        return pd.DataFrame(rows)

    raise RuntimeError("SDMX payload missing usable observations (series or observations)")


def download_oecd_dataset(dataset_id: str, *, start_year: int, end_year: int, force: bool = False) -> DownloadResult:
    """
    Downloads an OECD dataset via SDMX-JSON (legacy OECD.Stat endpoint) and saves as CSV.

    Notes:
    - Endpoint: https://stats.oecd.org/SDMX-JSON/data/<DATASET>/all/all
    - Some datasets are large; keep the year window tight (or expect a big CSV).
    """
    url = (
        f"https://stats.oecd.org/SDMX-JSON/data/{dataset_id}/all/all"
        f"?startTime={start_year}&endTime={end_year}"
    )
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:  # noqa: BLE001
        return DownloadResult(
            name=dataset_id,
            path=None,
            ok=False,
            note=f"OECD download failed for {dataset_id}: {e}",
        )

    try:
        df = _sdmx_json_v2_to_df(payload)
    except Exception as e:  # noqa: BLE001
        return DownloadResult(name=dataset_id, path=None, ok=False, note=f"Failed to parse SDMX-JSON for {dataset_id}: {e}")

    out_path = RAW_DIR / "oecd" / f"{dataset_id}_{start_year}-{end_year}.csv"
    if out_path.exists() and not force:
        return DownloadResult(name=dataset_id, path=out_path, ok=True, note="exists (skipped)")

    _ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    return DownloadResult(name=dataset_id, path=out_path, ok=True, note=f"downloaded via OECD SDMX-JSON ({url})")


def _wdi_fetch_indicator(indicator: str, *, per_page: int = 20000, timeout_s: int = 60) -> list[dict]:
    """
    Fetch all pages from World Bank v2 API for one indicator.
    """
    page = 1
    rows: list[dict] = []
    while True:
        url = (
            "https://api.worldbank.org/v2/country/all/indicator/"
            f"{indicator}?format=json&per_page={per_page}&page={page}"
        )
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"Unexpected World Bank response for {indicator}: {payload!r}")
        meta, data = payload[0], payload[1]
        rows.extend([x for x in data if x])
        pages = int(meta.get("pages", 1))
        if page >= pages:
            break
        page += 1
    return rows


def download_world_bank_wdi(*, start_year: int, end_year: int, force: bool = False) -> DownloadResult:
    """
    Downloads key World Bank WDI indicators needed in the docx plan, and saves a tidy CSV.

    Indicators:
    - GDP per capita, PPP (constant 2021 intl $): NY.GDP.PCAP.PP.KD
    - Gini index: SI.POV.GINI
    - Unemployment, total (% of total labor force): SL.UEM.TOTL.ZS
    """
    indicators = {
        "NY.GDP.PCAP.PP.KD": "gdp_per_capita_ppp_constant_2021_intl_dollars",
        "SI.POV.GINI": "gini_index",
        "SL.UEM.TOTL.ZS": "unemployment_rate_pct",
    }
    out_path = RAW_DIR / "world_bank_wdi" / f"wdi_{start_year}-{end_year}.csv"
    if out_path.exists() and not force:
        return DownloadResult(name="WDI", path=out_path, ok=True, note="exists (skipped)")

    frames: list[pd.DataFrame] = []
    for code, colname in indicators.items():
        rows = _wdi_fetch_indicator(code)
        df = pd.json_normalize(rows)
        df = df.rename(
            columns={
                "country.id": "country_iso2",
                "country.value": "country_name",
                "countryiso3code": "country_iso3",
                "date": "year",
                "value": colname,
            }
        )
        df = df[["country_iso3", "country_name", "year", colname]]
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
        frames.append(df)

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["country_iso3", "country_name", "year"], how="outer")

    merged = merged.sort_values(["country_name", "year"])
    _ensure_dir(out_path.parent)
    merged.to_csv(out_path, index=False)
    return DownloadResult(name="WDI", path=out_path, ok=True, note="downloaded via World Bank API")


def _print_summary(results: Iterable[DownloadResult]) -> None:
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    print("\n=== Download summary ===")
    for r in ok:
        print(f"[OK]  {r.name}: {r.path} ({r.note})")
    for r in bad:
        print(f"[ERR] {r.name}: {r.note}")
    if bad:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets referenced in Group 6 Phase II dashboard plan.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite outputs.")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    _ensure_dir(RAW_DIR)

    results: list[DownloadResult] = []

    # 1) World Happiness Report (free data file for Figure 2.1)
    results.append(download_world_happiness(force=args.force))

    # 2) OECD Social Protection & Welfare (SOCX_AGG)
    results.append(download_oecd_dataset("SOCX_AGG", start_year=args.start_year, end_year=args.end_year, force=args.force))

    # 3) OECD Better Life Index (BLI)
    results.append(download_oecd_dataset("BLI", start_year=args.start_year, end_year=args.end_year, force=args.force))

    # 4) World Bank WDI indicators
    results.append(download_world_bank_wdi(start_year=args.start_year, end_year=args.end_year, force=args.force))

    _print_summary(results)

    print("\nNext:")
    print("- Raw files are under: data/raw/")
    print("- World Happiness file is Excel; you can read it with pandas.read_excel().")
    print("- OECD + WDI outputs are CSV.")
    print("\nNote:")
    print("- Gallup World Poll microdata is not freely downloadable; WHR links to Gallup access options.")


if __name__ == "__main__":
    main()

