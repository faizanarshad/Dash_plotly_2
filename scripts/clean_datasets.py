from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CLEAN_DIR = REPO_ROOT / "data" / "cleaned"


# WHR country labels that are not matched to WDI naming by default.
WHR_TO_WDI_NAME = {
    "Congo": "Congo, Rep.",
    "Côte d’Ivoire": "Cote d'Ivoire",
    "DR Congo": "Congo, Dem. Rep.",
    "Egypt": "Egypt, Arab Rep.",
    "Gambia": "Gambia, The",
    "Hong Kong SAR of China": "Hong Kong SAR, China",
    "Iran": "Iran, Islamic Rep.",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Republic of Korea": "Korea, Rep.",
    "Republic of Moldova": "Moldova",
    "Slovakia": "Slovak Republic",
    "State of Palestine": "West Bank and Gaza",
    "Swaziland": "Eswatini",
    "Syria": "Syrian Arab Republic",
    "Türkiye": "Turkiye",
    "Venezuela": "Venezuela, RB",
    "Yemen": "Yemen, Rep.",
}


def clean_wdi() -> pd.DataFrame:
    wdi_path = PROCESSED_DIR / "phase2_inputs" / "world_bank_wdi_controls_2015_2023.csv"
    wdi = pd.read_csv(wdi_path)

    # Keep analysis horizon.
    wdi["year"] = pd.to_numeric(wdi["year"], errors="coerce")
    wdi = wdi[wdi["year"].between(2015, 2023)].copy()

    # Drop aggregate groups and malformed keys (these create duplicate join rows).
    wdi = wdi.dropna(subset=["country_iso3"]).copy()
    wdi["country_iso3"] = wdi["country_iso3"].astype(str).str.strip()
    wdi = wdi[wdi["country_iso3"].str.len() == 3].copy()

    # Resolve any remaining duplicate country-year keys using first non-null per column.
    wdi = (
        wdi.sort_values(["country_iso3", "year"])
        .groupby(["country_iso3", "year"], as_index=False)
        .agg(
            {
                "country_name": "first",
                "gdp_per_capita_ppp_constant_2021_intl_dollars": "first",
                "gini_index": "first",
                "unemployment_rate_pct": "first",
            }
        )
    )
    return wdi


def clean_socx() -> pd.DataFrame:
    socx_path = PROCESSED_DIR / "phase2_inputs" / "oecd_socx_social_spending_2015_2023.csv"
    socx = pd.read_csv(socx_path)
    socx["year"] = pd.to_numeric(socx["year"], errors="coerce")
    socx = socx[socx["year"].between(2015, 2023)].copy()
    socx = socx.dropna(subset=["country_iso3"]).copy()
    socx["country_iso3"] = socx["country_iso3"].astype(str).str.strip()
    socx = socx[socx["country_iso3"].str.len() == 3].copy()
    socx = socx.drop_duplicates(subset=["country_iso3", "year"], keep="first")
    return socx


def clean_whr() -> pd.DataFrame:
    whr_path = PROCESSED_DIR / "phase2_inputs" / "whr_happiness_2015_2023.csv"
    whr = pd.read_csv(whr_path)
    whr["year"] = pd.to_numeric(whr["year"], errors="coerce")
    whr = whr[whr["year"].between(2015, 2023)].copy()
    whr = whr.rename(columns={"country_name": "country_name_whr"})
    whr = whr.drop_duplicates(subset=["country_name_whr", "year"], keep="first")
    return whr


def clean_bli() -> pd.DataFrame:
    bli_path = PROCESSED_DIR / "phase2_inputs" / "oecd_bli_dimensions_country_level.csv"
    bli = pd.read_csv(bli_path)
    bli = bli.dropna(subset=["country_iso3"]).copy()
    bli["country_iso3"] = bli["country_iso3"].astype(str).str.strip()
    bli = bli[bli["country_iso3"].str.len() == 3].copy()
    bli = bli.drop_duplicates(subset=["country_iso3"], keep="first")
    return bli


def build_clean_combined(whr: pd.DataFrame, socx: pd.DataFrame, wdi: pd.DataFrame, bli: pd.DataFrame) -> pd.DataFrame:
    # Bridge WHR names to ISO3 via cleaned WDI mapping.
    whr["country_name_wdi_key"] = whr["country_name_whr"].replace(WHR_TO_WDI_NAME)
    name_to_iso = wdi[["country_name", "country_iso3"]].drop_duplicates().rename(columns={"country_name": "country_name_wdi_key"})
    whr = whr.merge(name_to_iso, on="country_name_wdi_key", how="left")

    merged = (
        whr.merge(socx, on=["country_iso3", "year"], how="left")
        .merge(wdi, on=["country_iso3", "year"], how="left", suffixes=("", "_wdi"))
        .merge(bli, on="country_iso3", how="left")
    )

    merged["country_name"] = merged["country_name"].fillna(merged["country_name_whr"])
    merged = merged.drop(columns=["country_name_wdi_key"], errors="ignore")

    # Keep rows with valid country key and deduplicate hard.
    merged = merged.dropna(subset=["country_iso3"]).copy()
    merged["country_iso3"] = merged["country_iso3"].astype(str).str.strip()
    merged = merged[merged["country_iso3"].str.len() == 3].copy()
    merged = merged.drop_duplicates(subset=["country_iso3", "year"], keep="first")

    # Standardize numeric dtypes and mild sanity clipping for bounded percentages.
    pct_cols = ["social_spending_pct_gdp", "gini_index", "unemployment_rate_pct"]
    for col in pct_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            merged.loc[(merged[col] < 0) | (merged[col] > 100), col] = pd.NA

    merged["happiness_score"] = pd.to_numeric(merged["happiness_score"], errors="coerce")
    merged.loc[(merged["happiness_score"] < 0) | (merged["happiness_score"] > 10), "happiness_score"] = pd.NA

    merged = merged.sort_values(["country_name", "year"]).reset_index(drop=True)
    return merged


def make_analysis_ready(df: pd.DataFrame) -> pd.DataFrame:
    # Keep rows with all core variables present for modeling/charting.
    core = [
        "happiness_score",
        "social_spending_pct_gdp",
        "gdp_per_capita_ppp_constant_2021_intl_dollars",
        "gini_index",
        "unemployment_rate_pct",
    ]
    return df.dropna(subset=core).copy()


def round_numeric_columns(df: pd.DataFrame, decimals: int = 1) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        out[col] = out[col].round(decimals)
    return out


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    whr = clean_whr()
    socx = clean_socx()
    wdi = clean_wdi()
    bli = clean_bli()
    clean_combined = build_clean_combined(whr, socx, wdi, bli)
    analysis_ready = make_analysis_ready(clean_combined)

    # Save cleaned source inputs.
    (CLEAN_DIR / "phase2_inputs").mkdir(parents=True, exist_ok=True)
    whr.to_csv(CLEAN_DIR / "phase2_inputs" / "whr_happiness_2015_2023_clean.csv", index=False)
    socx.to_csv(CLEAN_DIR / "phase2_inputs" / "oecd_socx_social_spending_2015_2023_clean.csv", index=False)
    wdi.to_csv(CLEAN_DIR / "phase2_inputs" / "world_bank_wdi_controls_2015_2023_clean.csv", index=False)
    bli.to_csv(CLEAN_DIR / "phase2_inputs" / "oecd_bli_dimensions_country_level_clean.csv", index=False)

    # Enforce one digit after decimal for numeric fields in exported cleaned datasets.
    whr_out = round_numeric_columns(whr, decimals=1)
    socx_out = round_numeric_columns(socx, decimals=1)
    wdi_out = round_numeric_columns(wdi, decimals=1)
    bli_out = round_numeric_columns(bli, decimals=1)
    clean_combined_out = round_numeric_columns(clean_combined, decimals=1)
    analysis_ready_out = round_numeric_columns(analysis_ready, decimals=1)

    whr_out.to_csv(CLEAN_DIR / "phase2_inputs" / "whr_happiness_2015_2023_clean.csv", index=False)
    socx_out.to_csv(CLEAN_DIR / "phase2_inputs" / "oecd_socx_social_spending_2015_2023_clean.csv", index=False)
    wdi_out.to_csv(CLEAN_DIR / "phase2_inputs" / "world_bank_wdi_controls_2015_2023_clean.csv", index=False)
    bli_out.to_csv(CLEAN_DIR / "phase2_inputs" / "oecd_bli_dimensions_country_level_clean.csv", index=False)

    clean_combined_out.to_csv(CLEAN_DIR / "combined_wellbeing_policy_2015_2023_clean.csv", index=False)
    analysis_ready_out.to_csv(CLEAN_DIR / "combined_wellbeing_policy_2015_2023_analysis_ready.csv", index=False)
    print("Saved cleaned files in:", CLEAN_DIR)
    print("combined_clean shape:", clean_combined_out.shape)
    print("analysis_ready shape:", analysis_ready_out.shape)
    print("duplicate country-year in clean combined:", clean_combined_out.duplicated(["country_iso3", "year"]).sum())
    print("missing in core columns:")
    for col in [
        "happiness_score",
        "social_spending_pct_gdp",
        "gdp_per_capita_ppp_constant_2021_intl_dollars",
        "gini_index",
        "unemployment_rate_pct",
    ]:
        print(f"  {col}: {int(clean_combined_out[col].isna().sum())}")


if __name__ == "__main__":
    main()

