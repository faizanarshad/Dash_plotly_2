from __future__ import annotations

from pathlib import Path

import dash
from dash import Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "cleaned" / "combined_wellbeing_policy_2015_2023_clean.csv"


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


def add_regions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gap = px.data.gapminder()[["iso_alpha", "continent"]].drop_duplicates()
    gap = gap.rename(columns={"iso_alpha": "country_iso3", "continent": "region"})
    out = out.merge(gap, on="country_iso3", how="left")
    out["region"] = out["region"].fillna("Other")
    return out


def choropleth(df: pd.DataFrame, year: int) -> go.Figure:
    d = df[df["year"] == year].copy()
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
        hover_data={"happiness_score": ":.1f", "happiness_quartile": True},
        title=f"VIZ 1 · GEOSPATIAL MAP — Happiness Score by Country ({year})",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=460)
    return fig


def timeseries(df: pd.DataFrame, selected_country_iso3: str | None = None) -> go.Figure:
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
        title="VIZ 2 · TIME SERIES — Happiness Score Trends by Spending Tier (2015-2023)",
        labels={"happiness_score": "Average Happiness Score", "spending_tier": "Social Spending Tier"},
    )
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Average Happiness Score", tickformat=".1f")

    # Phase III link: map click adds selected country's trend line.
    if selected_country_iso3:
        country_ts = (
            df[df["country_iso3"] == selected_country_iso3]
            .dropna(subset=["happiness_score"])
            .sort_values("year")
        )
        if not country_ts.empty:
            country_name = country_ts["country_name"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=country_ts["year"],
                    y=country_ts["happiness_score"],
                    mode="lines+markers",
                    name=f"{country_name} trend",
                    line=dict(color="black", width=3),
                    marker=dict(symbol="diamond", size=7),
                )
            )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
    return fig


def build_country_level(df: pd.DataFrame) -> pd.DataFrame:
    dims = [
        "housing_score",
        "job_security_score",
        "health_score",
        "civic_engagement_score",
        "safety_score",
        "work_life_balance_score",
    ]
    return (
        df.groupby(["country_iso3", "country_name"], as_index=False)
        .agg(
            social_spending_pct_gdp=("social_spending_pct_gdp", "mean"),
            happiness_score=("happiness_score", "mean"),
            gdp=("gdp_per_capita_ppp_constant_2021_intl_dollars", "mean"),
            region=("region", "first"),
            spending_tier=("spending_tier", "first"),
            **{c: (c, "mean") for c in dims},
        )
        .dropna(subset=["social_spending_pct_gdp", "happiness_score"])
    )


def scatter(
    df: pd.DataFrame,
    selected_country_iso3: str | None = None,
    hovered_bar_countries: list[str] | None = None,
) -> go.Figure:
    d = build_country_level(df).sort_values(["region", "country_name"]).reset_index(drop=True)
    fig = px.scatter(
        d,
        x="social_spending_pct_gdp",
        y="happiness_score",
        color="region",
        size="gdp",
        hover_name="country_name",
        hover_data={
            "social_spending_pct_gdp": ":.1f",
            "happiness_score": ":.1f",
            "gdp": ":.1f",
        },
        title="VIZ 3 · SCATTERPLOT — Social Spending (% GDP) vs. Happiness Score",
        labels={
            "social_spending_pct_gdp": "Social Spending (% GDP)",
            "happiness_score": "Average Happiness Score",
            "region": "Region",
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

    # Phase III link: bar hover highlights corresponding countries in scatter.
    if hovered_bar_countries:
        h = d[d["country_iso3"].isin(hovered_bar_countries)]
        if not h.empty:
            fig.add_trace(
                go.Scatter(
                    x=h["social_spending_pct_gdp"],
                    y=h["happiness_score"],
                    mode="markers",
                    name="Hovered bar countries",
                    text=h["country_name"],
                    hoverinfo="text+x+y",
                    marker=dict(size=14, color="gold", line=dict(color="black", width=1.5), symbol="circle-open"),
                )
            )

    # Phase III link: map click highlights selected country in scatter.
    if selected_country_iso3:
        sel = d[d["country_iso3"] == selected_country_iso3]
        if not sel.empty:
            fig.add_trace(
                go.Scatter(
                    x=sel["social_spending_pct_gdp"],
                    y=sel["happiness_score"],
                    mode="markers+text",
                    name="Selected country",
                    text=sel["country_name"],
                    textposition="top center",
                    marker=dict(size=18, color="red", line=dict(color="black", width=1.5), symbol="diamond"),
                )
            )
    fig.update_xaxes(tickformat=".1f")
    fig.update_yaxes(tickformat=".1f")
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
    grp["dimension_code"] = grp["dimension"]
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
        custom_data=["dimension_code", "spending_tier"],
        barmode="group",
        orientation="h",
        title="VIZ 4 · GROUPED BAR CHART — Wellbeing Dimensions: High vs. Low Spending Countries",
        labels={"score": "OECD Better Life Index Score", "spending_tier": "Spending Tier", "dimension": "Dimension"},
    )
    fig.update_xaxes(title="OECD Better Life Index Score (0-10)", tickformat=".1f")
    fig.update_yaxes(title="Wellbeing Dimension")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
    return fig


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing cleaned dataset at {DATA_PATH}. Run scripts/clean_datasets.py first."
    )

BASE_DF = add_regions(add_spending_tiers(pd.read_csv(DATA_PATH)))
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
                        html.Label("Year Slider (updates map)"),
                        dcc.Slider(
                            id="year-slider",
                            min=min(YEARS),
                            max=max(YEARS),
                            step=1,
                            marks={y: str(y) for y in YEARS},
                            value=2022 if 2022 in YEARS else YEARS[-1],
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style={"width": "460px"},
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
    Input("year-slider", "value"),
    Input("tier-dropdown", "value"),
    Input("map-fig", "clickData"),
    Input("bar-fig", "hoverData"),
)
def update_figures(selected_year: int, selected_tier: str, map_click_data: dict | None, bar_hover_data: dict | None):
    d = BASE_DF.copy()
    if selected_tier != "all":
        d = d[d["spending_tier"] == selected_tier]

    selected_country_iso3 = None
    if map_click_data and map_click_data.get("points"):
        selected_country_iso3 = map_click_data["points"][0].get("location")

    hovered_bar_countries: list[str] = []
    if bar_hover_data and bar_hover_data.get("points"):
        pt = bar_hover_data["points"][0]
        custom = pt.get("customdata") or []
        if len(custom) >= 2:
            dimension_code = custom[0]
            spending_tier = custom[1]
            if dimension_code in d.columns:
                cdf = build_country_level(d)
                cdf = cdf[(cdf["spending_tier"] == spending_tier) & cdf[dimension_code].notna()]
                hovered_bar_countries = cdf["country_iso3"].dropna().astype(str).tolist()
    return (
        choropleth(d, selected_year),
        timeseries(d, selected_country_iso3=selected_country_iso3),
        scatter(
            d,
            selected_country_iso3=selected_country_iso3,
            hovered_bar_countries=hovered_bar_countries,
        ),
        grouped_bars(d),
    )


if __name__ == "__main__":
    app.run(debug=True)

