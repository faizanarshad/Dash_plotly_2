from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"


# Map WHR country labels to World Bank naming used in WDI file.
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


def load_whr() -> pd.DataFrame:
    path = RAW / "world_happiness" / "WHR26_Data_Figure_2.1.xlsx"
    df = pd.read_excel(path)
    df = df.rename(
        columns={
            "Country name": "country_name_whr",
            "Year": "year",
            "Life evaluation (3-year average)": "happiness_score",
        }
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(2015, 2023)]
    return df[["country_name_whr", "year", "happiness_score"]].copy()


def load_wdi() -> pd.DataFrame:
    path = RAW / "world_bank_wdi" / "wdi_2015-2023.csv"
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    keep = [
        "country_iso3",
        "country_name",
        "year",
        "gdp_per_capita_ppp_constant_2021_intl_dollars",
        "gini_index",
        "unemployment_rate_pct",
    ]
    return df[keep].copy()


def load_socx() -> pd.DataFrame:
    """
    Extract social spending as % GDP from SOCX.
    Uses: UNIT_MEASURE=PT_B1GQ, SPENDING_TYPE=_T, PROGRAMME_TYPE=_T,
    EXPEND_SOURCE=ES30 (public social expenditure), annual data.
    """
    path = RAW / "oecd" / "SOCX_AGG_2015-2023.csv"
    df = pd.read_csv(path)
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
    filt = (
        (df["UNIT_MEASURE"] == "PT_B1GQ")
        & (df["SPENDING_TYPE"] == "_T")
        & (df["PROGRAMME_TYPE"] == "_T")
        & (df["EXPEND_SOURCE"] == "ES30")
        & (df["FREQ"] == "A")
    )
    out = df.loc[filt, ["REF_AREA", "TIME_PERIOD", "value"]].copy()
    out = out.rename(
        columns={
            "REF_AREA": "country_iso3",
            "TIME_PERIOD": "year",
            "value": "social_spending_pct_gdp",
        }
    )
    return out


def load_bli() -> pd.DataFrame:
    """
    Extract BLI dimensions requested in the proposal, INEQUALITY=TOT:
    - housing_score -> HO_HISH (housing expenditure burden)
    - job_security_score -> JE_LMIS (labour market insecurity; inverse interpretation)
    - health_score -> HS_LEB (life expectancy)
    - civic_engagement_score -> CG_SENG (stakeholder engagement)
    - safety_score -> PS_FSAFEN (feeling safe at night)
    - work_life_balance_score -> WL_EWLH (employees working very long hours)

    Note: current BLI extract has no year dimension in this download, so these
    values are country-level and are joined by country only.
    """
    path = RAW / "oecd" / "BLI_2015-2023.csv"
    df = pd.read_csv(path)
    indicator_map = {
        "HO_HISH": "housing_score",
        "JE_LMIS": "job_security_score",
        "HS_LEB": "health_score",
        "CG_SENG": "civic_engagement_score",
        "PS_FSAFEN": "safety_score",
        "WL_EWLH": "work_life_balance_score",
    }
    filt = df["INDICATOR"].isin(indicator_map.keys()) & (df["MEASURE"] == "L") & (df["INEQUALITY"] == "TOT")
    out = df.loc[filt, ["LOCATION", "INDICATOR", "value"]].copy()
    out["metric"] = out["INDICATOR"].map(indicator_map)
    out = out.pivot_table(index="LOCATION", columns="metric", values="value", aggfunc="first").reset_index()
    out = out.rename(columns={"LOCATION": "country_iso3"})
    return out


def combine() -> pd.DataFrame:
    whr = load_whr()
    wdi = load_wdi()
    socx = load_socx()
    bli = load_bli()

    # Bridge WHR country names to ISO3 using WDI naming.
    whr["country_name_wdi_key"] = whr["country_name_whr"].replace(WHR_TO_WDI_NAME)
    wdi_name_iso = (
        wdi[["country_name", "country_iso3"]]
        .dropna()
        .drop_duplicates(subset=["country_name"])
        .rename(columns={"country_name": "country_name_wdi_key"})
    )
    whr = whr.merge(wdi_name_iso, on="country_name_wdi_key", how="left")

    # Base on WHR country-year observations, then bring in policy and controls.
    merged = (
        whr.merge(socx, on=["country_iso3", "year"], how="left")
        .merge(wdi, on=["country_iso3", "year"], how="left", suffixes=("", "_wdi"))
        .merge(bli, on="country_iso3", how="left")
    )

    # Keep a clear and compact schema.
    merged["country_name"] = merged["country_name"].fillna(merged["country_name_whr"])
    out_cols = [
        "country_iso3",
        "country_name",
        "country_name_whr",
        "year",
        "happiness_score",
        "social_spending_pct_gdp",
        "gdp_per_capita_ppp_constant_2021_intl_dollars",
        "gini_index",
        "unemployment_rate_pct",
        "housing_score",
        "job_security_score",
        "health_score",
        "civic_engagement_score",
        "safety_score",
        "work_life_balance_score",
    ]
    merged = merged[out_cols].sort_values(["country_name", "year"]).reset_index(drop=True)
    return merged


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    phase2_inputs = PROCESSED / "phase2_inputs"
    phase2_inputs.mkdir(parents=True, exist_ok=True)

    # Export cleaned key-variable inputs per source.
    whr = load_whr().rename(columns={"country_name_whr": "country_name"})
    whr.to_csv(phase2_inputs / "whr_happiness_2015_2023.csv", index=False)

    socx = load_socx()
    socx.to_csv(phase2_inputs / "oecd_socx_social_spending_2015_2023.csv", index=False)

    wdi = load_wdi()
    wdi.to_csv(phase2_inputs / "world_bank_wdi_controls_2015_2023.csv", index=False)

    bli = load_bli()
    bli.to_csv(phase2_inputs / "oecd_bli_dimensions_country_level.csv", index=False)

    df = combine()
    out_path = PROCESSED / "combined_wellbeing_policy_2015_2023.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Saved cleaned source extracts in: {phase2_inputs}")
    print("Missingness:")
    for col in [
        "happiness_score",
        "social_spending_pct_gdp",
        "gdp_per_capita_ppp_constant_2021_intl_dollars",
        "gini_index",
        "unemployment_rate_pct",
        "housing_score",
        "job_security_score",
        "health_score",
        "civic_engagement_score",
        "safety_score",
        "work_life_balance_score",
    ]:
        miss = int(df[col].isna().sum())
        print(f"  {col}: {miss} missing")


if __name__ == "__main__":
    main()

