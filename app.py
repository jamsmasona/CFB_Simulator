import os
import math
import time
import functools
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ------------------------------------------------------------
# CONFIG & SECRETS
# ------------------------------------------------------------

CFBD_API_KEY = os.getenv("CFBD_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")

st.set_page_config(
    page_title="CFB Institutional Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

if CFBD_API_KEY is None or APP_PASSWORD is None:
    st.error(
        "Environment variables CFBD_API_KEY and APP_PASSWORD are not set.\n\n"
        "Set them in GitHub Secrets / your environment before running this app."
    )
    st.stop()

# ------------------------------------------------------------
# CFBD CLIENT
# ------------------------------------------------------------

BASE_URL = "https://api.collegefootballdata.com"


def cfbd_get(path: str, params: dict | None = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=headers, params=params or {})
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------
# DATA & ENGINE UTILITIES (SIMPLIFIED BUT STRUCTURED)
# ------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def get_teams(season: int = 2026) -> pd.DataFrame:
    data = cfbd_get("/teams/fbs", {"year": season})
    df = pd.DataFrame(data)
    return df[["school", "conference"]].rename(columns={"school": "team"})


@functools.lru_cache(maxsize=32)
def get_team_stats(season: int = 2026) -> pd.DataFrame:
    # Basic offensive/defensive stats as a placeholder
    data = cfbd_get("/stats/team", {"year": season, "seasonType": "regular"})
    df = pd.DataFrame(data)

    # Flatten categories into simple metrics
    rows = []
    for row in df.to_dict("records"):
        team = row.get("team")
        conference = row.get("conference")
        stats = row.get("stats", [])
        stat_map = {s["category"] + "_" + s["stat"]: float(s["value"]) for s in stats}
        rows.append({"team": team, "conference": conference, **stat_map})

    out = pd.DataFrame(rows)
    return out


def build_power_ratings(season: int = 2026) -> pd.DataFrame:
    """
    Very simplified 'institutional-style' power rating:
    - Offensive efficiency proxy
    - Defensive efficiency proxy
    - Tempo proxy
    - Combine into a single rating
    """
    stats = get_team_stats(season)

    # Fallback if stats are sparse
    if stats.empty:
        return pd.DataFrame(columns=["team", "conference", "power_rating"])

    # Simple proxies (you can replace with your full engine logic)
    off_ypp = stats.filter(like="offense_").mean(axis=1)
    def_ypp = stats.filter(like="defense_").mean(axis=1)
    tempo = stats.filter(like="offense_plays").mean(axis=1)

    # Normalize
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    off_z = zscore(off_ypp.fillna(off_ypp.mean()))
    def_z = -zscore(def_ypp.fillna(def_ypp.mean()))  # lower is better
    tempo_z = zscore(tempo.fillna(tempo.mean()))

    rating = 0.55 * off_z + 0.35 * def_z + 0.10 * tempo_z

    pr = pd.DataFrame(
        {
            "team": stats["team"],
            "conference": stats["conference"],
            "power_rating": rating,
        }
    ).dropna()

    pr = pr.sort_values("power_rating", ascending=False).reset_index(drop=True)
    return pr


@functools.lru_cache(maxsize=32)
def get_top25(season: int = 2026) -> pd.DataFrame:
    pr = build_power_ratings(season)
    top25 = pr.head(25).copy()
    top25["rank"] = np.arange(1, len(top25) + 1)
    return top25[["rank", "team", "conference"]]


def simulate_game(
    team_home: str,
    team_away: str,
    season: int = 2026,
    neutral_site: bool = False,
    n_sims: int = 5000,
) -> dict:
    """
    Simplified 'hybrid institutional' simulation:
    - Use power ratings to derive expected margin
    - Add random noise
    - Monte Carlo to get win probabilities, score distribution
    """
    pr = build_power_ratings(season)

    home_row = pr.loc[pr["team"] == team_home]
    away_row = pr.loc[pr["team"] == team_away]

    if home_row.empty or away_row.empty:
        raise ValueError("Missing power rating for one or both teams.")

    home_pr = float(home_row["power_rating"].iloc[0])
    away_pr = float(away_row["power_rating"].iloc[0])

    base_margin = home_pr - away_pr

    # Home field advantage (simple placeholder)
    hfa = 2.5 if not neutral_site else 0.0
    expected_margin = base_margin + hfa

    # Monte Carlo
    margins = np.random.normal(loc=expected_margin, scale=13.0, size=n_sims)

    home_wins = (margins > 0).mean()
    away_wins = 1.0 - home_wins

    # Convert margin to score (very rough)
    total_points = np.random.normal(loc=55.0, scale=10.0, size=n_sims)
    home_scores = (total_points + margins) / 2.0
    away_scores = (total_points - margins) / 2.0

    home_scores = np.clip(home_scores, 3, 70)
    away_scores = np.clip(away_scores, 3, 70)

    return {
        "team_home": team_home,
        "team_away": team_away,
        "expected_margin": expected_margin,
        "home_win_prob": home_wins,
        "away_win_prob": away_wins,
        "home_score_mean": float(home_scores.mean()),
        "away_score_mean": float(away_scores.mean()),
        "margins": margins,
        "home_scores": home_scores,
        "away_scores": away_scores,
    }


# ------------------------------------------------------------
# UI HELPERS
# ------------------------------------------------------------

def format_prob(p: float) -> str:
    return f"{p * 100:0.1f}%"


def margin_density_chart(margins: np.ndarray, team_home: str, team_away: str) -> alt.Chart:
    df = pd.DataFrame({"margin": margins})
    chart = (
        alt.Chart(df)
        .transform_density(
            "margin",
            as_=["margin", "density"],
            extent=[-40, 40],
            steps=200,
        )
        .mark_area(color="#4c78a8", opacity=0.6)
        .encode(
            x=alt.X("margin:Q", title=f"{team_home} margin over {team_away}"),
            y=alt.Y("density:Q", title="Density"),
        )
    )
    return chart


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------

def main():
    # Top bar: title
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0.5rem;'>CFB Institutional Simulator</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#888;'>2026 Season • Hybrid Institutional Engine • Top‑25 Sidebar</p>",
        unsafe_allow_html=True,
    )

    # Sidebar: controls + Top‑25
    with st.sidebar:
        st.markdown("### Controls")

        season = 2026  # locked as requested

        teams_df = get_teams(season)
        team_list = sorted(teams_df["team"].unique())

        team_home = st.selectbox("Home Team", team_list, index=0)
        team_away = st.selectbox("Away Team", team_list, index=1)

        neutral_site = st.checkbox("Neutral Site", value=False)

        n_sims = st.slider("Number of Simulations", 1000, 10000, 5000, step=1000)

        st.markdown("---")
        st.markdown("### Top‑25 (Power Rating)")

        with st.spinner("Computing Top‑25 (cached)..."):
            top25 = get_top25(season)

        for _, row in top25.iterrows():
            st.markdown(
                f"**{int(row['rank']):2d}. {row['team']}**  "
                f"<span style='color:#999; font-size:0.8rem;'>({row['conference']})</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption("Power ratings & rankings are computed once at startup and cached.")

        run_button = st.button("Run Simulation", type="primary")

    # Main layout: hybrid (wide header, centered content)
    if not run_button:
        st.info("Select teams in the sidebar and click **Run Simulation**.")
        return

    # Password gate (from environment, no UI exposure)
    # If APP_PASSWORD is set, we assume the environment is trusted.
    # You can add additional gating here if you want.
    if APP_PASSWORD is None:
        st.error("APP_PASSWORD environment variable is missing.")
        st.stop()

    # Run simulation
    with st.spinner("Running hybrid institutional simulation..."):
        try:
            result = simulate_game(
                team_home=team_home,
                team_away=team_away,
                season=season,
                neutral_site=neutral_site,
                n_sims=n_sims,
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return

    # --------------------------------------------------------
    # EXECUTIVE SUMMARY (FULL‑WIDTH)
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("## Executive Summary")

    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.2])

    with col1:
        st.metric(
            label="Projected Margin",
            value=f"{result['expected_margin']:0.1f} pts",
            delta=None,
        )

    with col2:
        st.metric(
            label=f"{result['team_home']} Win Probability",
            value=format_prob(result["home_win_prob"]),
        )

    with col3:
        st.metric(
            label=f"{result['team_away']} Win Probability",
            value=format_prob(result["away_win_prob"]),
        )

    with col4:
        total_mean = result["home_score_mean"] + result["away_score_mean"]
        st.metric(
            label="Projected Total",
            value=f"{total_mean:0.1f} pts",
        )

    # --------------------------------------------------------
    # CENTERED CONTENT: ANALYTICS + CHARTS
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "<h3 style='text-align:center;'>Score & Margin Distribution</h3>",
        unsafe_allow_html=True,
    )

    center_col = st.columns([0.15, 0.7, 0.15])[1]

    with center_col:
        chart = margin_density_chart(
            result["margins"], result["team_home"], result["team_away"]
        )
        st.altair_chart(chart, use_container_width=True)

    # --------------------------------------------------------
    # SIMULATION OUTPUT TABLE
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("## Simulation Summary")

    sim_df = pd.DataFrame(
        {
            "Team": [result["team_home"], result["team_away"]],
            "Mean Score": [result["home_score_mean"], result["away_score_mean"]],
            "Win Probability": [
                format_prob(result["home_win_prob"]),
                format_prob(result["away_win_prob"]),
            ],
        }
    )

    st.table(sim_df.style.format({"Mean Score": "{:0.1f}"}))

    st.caption(
        "This is a simplified hybrid institutional engine using power ratings, Monte Carlo margin "
        "simulation, and a cached Top‑25. You can replace the rating and simulation logic with your "
        "full internal engine while keeping this UI structure."
    )


if __name__ == "__main__":
    main()
