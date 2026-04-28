from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "combined_wellbeing_policy_2015_2023.csv"
OUT_DIR = REPO_ROOT / "outputs" / "phase2_graphs"


def add_spending_tiers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    country_spend = (
        out.groupby("country_iso3", as_index=False)["social_spending_pct_gdp"]
        .mean()
        .dropna()
        .rename(columns={"social_spending_pct_gdp": "avg_social_spending_pct_gdp"})
    )
    country_spend["spending_tier"] = pd.qcut(
        country_spend["avg_social_spending_pct_gdp"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )
    out = out.merge(country_spend[["country_iso3", "spending_tier"]], on="country_iso3", how="left")
    return out


def add_regions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gap = px.data.gapminder()[["iso_alpha", "continent"]].drop_duplicates()
    gap = gap.rename(columns={"iso_alpha": "country_iso3", "continent": "region"})
    out = out.merge(gap, on="country_iso3", how="left")
    out["region"] = out["region"].fillna("Other")
    return out


def viz1_choropleth_2022(df: pd.DataFrame) -> go.Figure:
    d = df[df["year"] == 2022].copy()
    d = d.dropna(subset=["happiness_score"])
    d["happiness_quartile"] = pd.qcut(
        d["happiness_score"],
        q=4,
        labels=["Q1: Lowest", "Q2", "Q3", "Q4: Highest"],
        duplicates="drop",
    )
    fig = px.choropleth(
        d,
        locations="country_iso3",
        color="happiness_quartile",
        hover_name="country_name",
        category_orders={"happiness_quartile": ["Q1: Lowest", "Q2", "Q3", "Q4: Highest"]},
        color_discrete_map={
            "Q1: Lowest": "#deebf7",
            "Q2": "#9ecae1",
            "Q3": "#3182bd",
            "Q4: Highest": "#08519c",
        },
        hover_data={"happiness_score": ":.2f", "happiness_quartile": True},
        title="VIZ 1 · GEOSPATIAL MAP — Happiness Score by Country (2022)",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def viz2_time_series_by_tier(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["spending_tier"]).copy()
    ts = (
        d.groupby(["year", "spending_tier"], as_index=False, observed=False)["happiness_score"]
        .mean()
        .sort_values(["year", "spending_tier"])
    )
    order = ["low", "medium", "high"]
    ts["spending_tier"] = pd.Categorical(ts["spending_tier"], categories=order, ordered=True)
    ts = ts.sort_values(["spending_tier", "year"])

    line_dash_map = {"low": "dot", "medium": "dash", "high": "solid"}
    fig = px.line(
        ts,
        x="year",
        y="happiness_score",
        color="spending_tier",
        line_dash="spending_tier",
        line_dash_map=line_dash_map,
        markers=True,
        title="VIZ 2 · TIME SERIES — Happiness Score Trends by Spending Tier (2015-2023)",
        labels={"happiness_score": "Average Happiness Score", "spending_tier": "Social Spending Tier"},
    )
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Average Happiness Score")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def viz3_scatter_spending_vs_happiness(df: pd.DataFrame) -> go.Figure:
    d = (
        df.groupby(["country_iso3", "country_name"], as_index=False)
        .agg(
            social_spending_pct_gdp=("social_spending_pct_gdp", "mean"),
            happiness_score=("happiness_score", "mean"),
            gdp_per_capita_ppp_constant_2021_intl_dollars=(
                "gdp_per_capita_ppp_constant_2021_intl_dollars",
                "mean",
            ),
            region=("region", "first"),
        )
        .dropna(subset=["social_spending_pct_gdp", "happiness_score"])
    )

    fig = px.scatter(
        d,
        x="social_spending_pct_gdp",
        y="happiness_score",
        color="region",
        hover_name="country_name",
        size="gdp_per_capita_ppp_constant_2021_intl_dollars",
        title="VIZ 3 · SCATTERPLOT — Social Spending (% GDP) vs. Happiness Score",
        labels={
            "social_spending_pct_gdp": "Social Spending (% GDP)",
            "happiness_score": "Average Happiness Score",
            "region": "Region",
        },
    )

    # OLS trend line (manual) to avoid extra dependency.
    x = d["social_spending_pct_gdp"].to_numpy(dtype=float)
    y = d["happiness_score"].to_numpy(dtype=float)
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trend line",
                line=dict(color="black", dash="dash"),
            )
        )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def viz4_grouped_bar_bli_dimensions(df: pd.DataFrame) -> go.Figure:
    dims = [
        "housing_score",
        "job_security_score",
        "health_score",
        "civic_engagement_score",
        "safety_score",
        "work_life_balance_score",
    ]
    d = df.dropna(subset=["spending_tier"]).copy()
    d = d[d["spending_tier"].isin(["high", "low"])]

    # Collapse to country-level first (BLI is country-level in current extract).
    country_level = d.groupby(["country_iso3", "spending_tier"], as_index=False, observed=False)[dims].mean()
    melted = country_level.melt(
        id_vars=["country_iso3", "spending_tier"], value_vars=dims, var_name="dimension", value_name="score"
    )
    grp = melted.groupby(["dimension", "spending_tier"], as_index=False, observed=False)["score"].mean()

    label_map = {
        "housing_score": "Housing",
        "job_security_score": "Jobs",
        "health_score": "Health",
        "civic_engagement_score": "Civic Engagement",
        "safety_score": "Safety",
        "work_life_balance_score": "Work-Life Balance",
    }
    grp["dimension"] = grp["dimension"].map(label_map)

    fig = px.bar(
        grp,
        x="score",
        y="dimension",
        color="spending_tier",
        barmode="group",
        orientation="h",
        title="VIZ 4 · GROUPED BAR CHART — Wellbeing Dimensions: High vs. Low Spending Countries",
        labels={"score": "OECD Better Life Index Score", "spending_tier": "Spending Tier", "dimension": "Dimension"},
    )
    fig.update_xaxes(title="OECD Better Life Index Score (0-10)")
    fig.update_yaxes(title="Wellbeing Dimension")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = add_regions(add_spending_tiers(df))

    figs = {
        "viz1_choropleth_2022.html": viz1_choropleth_2022(df),
        "viz2_time_series_tiers.html": viz2_time_series_by_tier(df),
        "viz3_scatter_spending_vs_happiness.html": viz3_scatter_spending_vs_happiness(df),
        "viz4_grouped_bar_bli_dimensions.html": viz4_grouped_bar_bli_dimensions(df),
    }

    for name, fig in figs.items():
        fig.write_html(OUT_DIR / name, include_plotlyjs="cdn")
        print(f"Saved {OUT_DIR / name}")


if __name__ == "__main__":
    main()

