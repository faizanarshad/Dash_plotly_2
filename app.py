from __future__ import annotations

from pathlib import Path

import dash
from dash import Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "combined_wellbeing_policy_2015_2023.csv"


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
    return out.merge(country_spend[["country_iso3", "spending_tier"]], on="country_iso3", how="left")


def choropleth(df: pd.DataFrame, year: int) -> go.Figure:
    d = df[df["year"] == year].copy()
    fig = px.choropleth(
        d,
        locations="country_iso3",
        color="happiness_score",
        hover_name="country_name",
        color_continuous_scale="Blues",
        title=f"VIZ 1: Happiness Score by Country ({year})",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=460)
    return fig


def timeseries(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["spending_tier"]).copy()
    ts = d.groupby(["year", "spending_tier"], as_index=False, observed=False)["happiness_score"].mean()
    order = ["low", "medium", "high"]
    ts["spending_tier"] = pd.Categorical(ts["spending_tier"], categories=order, ordered=True)
    ts = ts.sort_values(["spending_tier", "year"])
    fig = px.line(
        ts,
        x="year",
        y="happiness_score",
        color="spending_tier",
        line_dash="spending_tier",
        line_dash_map={"low": "dot", "medium": "dash", "high": "solid"},
        markers=True,
        title="VIZ 2: Happiness Trends by Spending Tier (2015-2023)",
        labels={"happiness_score": "Average Happiness Score", "spending_tier": "Spending Tier"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
    return fig


def scatter(df: pd.DataFrame) -> go.Figure:
    d = (
        df.groupby(["country_iso3", "country_name"], as_index=False)
        .agg(
            social_spending_pct_gdp=("social_spending_pct_gdp", "mean"),
            happiness_score=("happiness_score", "mean"),
            gdp=("gdp_per_capita_ppp_constant_2021_intl_dollars", "mean"),
            spending_tier=("spending_tier", "first"),
        )
        .dropna(subset=["social_spending_pct_gdp", "happiness_score"])
    )
    fig = px.scatter(
        d,
        x="social_spending_pct_gdp",
        y="happiness_score",
        color="spending_tier",
        size="gdp",
        hover_name="country_name",
        title="VIZ 3: Social Spending (% GDP) vs Happiness Score",
        labels={
            "social_spending_pct_gdp": "Social Spending (% GDP)",
            "happiness_score": "Average Happiness Score",
            "spending_tier": "Spending Tier",
        },
    )
    x = d["social_spending_pct_gdp"].to_numpy(dtype=float)
    y = d["happiness_score"].to_numpy(dtype=float)
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(x=x_line, y=y_line, mode="lines", name="Trend line", line=dict(color="black", dash="dash"))
        )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
    return fig


def grouped_bars(df: pd.DataFrame) -> go.Figure:
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
    country_level = d.groupby(["country_iso3", "spending_tier"], as_index=False, observed=False)[dims].mean()
    melted = country_level.melt(
        id_vars=["country_iso3", "spending_tier"],
        value_vars=dims,
        var_name="dimension",
        value_name="score",
    )
    grp = melted.groupby(["dimension", "spending_tier"], as_index=False, observed=False)["score"].mean()
    grp["dimension"] = grp["dimension"].map(
        {
            "housing_score": "Housing",
            "job_security_score": "Jobs",
            "health_score": "Health",
            "civic_engagement_score": "Civic Engagement",
            "safety_score": "Safety",
            "work_life_balance_score": "Work-Life Balance",
        }
    )
    fig = px.bar(
        grp,
        x="score",
        y="dimension",
        color="spending_tier",
        barmode="group",
        orientation="h",
        title="VIZ 4: Wellbeing Dimensions - High vs Low Spending",
        labels={"score": "Average BLI Score", "spending_tier": "Spending Tier", "dimension": "Dimension"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
    return fig


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing combined dataset at {DATA_PATH}. Run scripts/combine_four_datasets.py first."
    )

BASE_DF = add_spending_tiers(pd.read_csv(DATA_PATH))
YEARS = sorted([int(y) for y in BASE_DF["year"].dropna().unique()])
TIER_OPTIONS = ["all", "low", "medium", "high"]

app = dash.Dash(__name__)
app.title = "Social Safety Nets Dashboard"

app.layout = html.Div(
    [
        html.H2("Social Safety Nets and Subjective Wellbeing"),
        html.P("Interactive Phase II dashboard with 4 required visualizations."),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Map Year"),
                        dcc.Dropdown(
                            id="year-dropdown",
                            options=[{"label": str(y), "value": int(y)} for y in YEARS],
                            value=2022 if 2022 in YEARS else YEARS[-1],
                            clearable=False,
                        ),
                    ],
                    style={"width": "220px"},
                ),
                html.Div(
                    [
                        html.Label("Spending Tier Filter"),
                        dcc.Dropdown(
                            id="tier-dropdown",
                            options=[{"label": t.title(), "value": t} for t in TIER_OPTIONS],
                            value="all",
                            clearable=False,
                        ),
                    ],
                    style={"width": "220px"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "marginBottom": "14px"},
        ),
        dcc.Graph(id="map-fig"),
        dcc.Graph(id="ts-fig"),
        dcc.Graph(id="scatter-fig"),
        dcc.Graph(id="bar-fig"),
    ],
    style={"maxWidth": "1300px", "margin": "0 auto", "padding": "16px"},
)


@app.callback(
    Output("map-fig", "figure"),
    Output("ts-fig", "figure"),
    Output("scatter-fig", "figure"),
    Output("bar-fig", "figure"),
    Input("year-dropdown", "value"),
    Input("tier-dropdown", "value"),
)
def update_figures(selected_year: int, selected_tier: str):
    d = BASE_DF.copy()
    if selected_tier != "all":
        d = d[d["spending_tier"] == selected_tier]
    return (
        choropleth(d, selected_year),
        timeseries(d),
        scatter(d),
        grouped_bars(d),
    )


if __name__ == "__main__":
    app.run(debug=True)

