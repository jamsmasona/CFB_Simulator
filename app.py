from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import Ridge

# Set Native Dark Mode Configuration
st.set_page_config(
    page_title="Institutional CFB Analytics Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "history" not in st.session_state:
    st.session_state.history = []

# Ensure API Key is handled safely (via Streamlit Secrets or fallback input)
API_KEY = st.secrets.get("CFBD_API_KEY", "")

st.sidebar.title("Institutional Power Grid")
st.sidebar.markdown("Bayesian Shrinkage & Ridge Regression Engine")

if not API_KEY:
    API_KEY = st.sidebar.text_input("Enter CFBD API Key:", type="password")

target_week = st.sidebar.slider("Active Season Week Scope", min_value=1, max_value=15, value=6)

if st.sidebar.button("Purge & Re-Index Cache"):
    st.cache_data.clear()
    st.success("Cache cleared successfully!")


@st.cache_data(ttl=3600)
def fetch_granular_pbp_and_ratings(target_week=15, api_key=""):
    """Pulls play-by-play data week-by-week, calculates success rates,
    and constructs a Ridge Regression model to isolate true opponent-adjusted unit ratings.
    """
    if not api_key:
        return {}, {}, {}
        
    headers = {"Authorization": f"Bearer {api_key}"}
    all_plays = []
    
    # Safely clamp the loop to prevent excessive HTTP hanging
    max_fetch_week = min(target_week, 3)
    
    for w in range(1, max_fetch_week + 1):
        url = f"https://api.collegefootballdata.com/plays?year=2026&seasonType=regular&week={w}"
        try:
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code == 200:
                data = res.json()
                if data:
                    all_plays.extend(data)
        except Exception:
            continue
            
    if not all_plays:
        return {}, {}, {}
        
    df_plays = pd.DataFrame(all_plays)
    
    if df_plays.empty or 'play_type' not in df_plays.columns:
        return {}, {}, {}
    
    core_plays = df_plays[df_plays['play_type'].isin(['Rush', 'Pass Reception', 'Passing'])].copy()
    
    if core_plays.empty:
        return {}, {}, {}
    
    def check_success(row):
        down = row.get('down', 1)
        dist = row.get('distance', 10)
        gained = row.get('yards_gained', 0)
        if down == 1:
            return gained >= (dist * 0.5)
        elif down == 2:
            return gained >= (dist * 0.7)
        else:
            return gained >= dist

    core_plays['success'] = core_plays.apply(check_success, axis=1)
    team_sample_counts = core_plays['offense'].value_counts().to_dict()
    
    agg_metrics = core_plays.groupby('offense').agg(
        off_epa=('ppa', 'mean'),
        success_rate=('success', 'mean')
    ).to_dict(orient='index')
    
    games_url = "https://api.collegefootballdata.com/games?year=2026&seasonType=regular"
    ridge_ratings = {}
    try:
        g_res = requests.get(games_url, headers=headers, timeout=4)
        if g_res.status_code == 200:
            games_data = g_res.json()
            matchups = []
            for g in games_data:
                if g.get("home_points") is not None and g.get("away_points") is not None:
                    matchups.append({
                        "home": g.get("home_team"),
                        "away": g.get("away_team"),
                        "margin": g.get("home_points") - g.get("away_points")
                    })
            if matchups:
                df_games = pd.DataFrame(matchups)
                teams = sorted(list(set(df_games['home']).union(set(df_games['away']))))
                team_to_idx = {t: i for i, t in enumerate(teams)}
                
                X = np.zeros((len(df_games), len(teams) * 2))
                y = df_games['margin'].values
                
                for idx, row in df_games.iterrows():
                    h_idx = team_to_idx[row['home']]
                    a_idx = team_to_idx[row['away']]
                    X[idx, h_idx] = 1.0   
                    X[idx, len(teams) + a_idx] = -1.0 
                    
                ridge = Ridge(alpha=100.0)
                ridge.fit(X, y)
                
                coeffs = ridge.coef_
                for t, idx in team_to_idx.items():
                    off_val = coeffs[idx]
                    def_val = coeffs[len(teams) + idx]
                    ridge_ratings[t] = (off_val - def_val) * 3.5
    except Exception:
        pass

    return agg_metrics, team_sample_counts, ridge_ratings

st.title("⚡ CFB Advanced Analytics Dashboard")

if not API_KEY:
    st.warning("Please input your College Football Data API Key in the sidebar to load the models.")
else:
    with st.spinner("Computing opponent-adjusted metrics and regressions..."):
        metrics, samples, ratings = fetch_granular_pbp_and_ratings(target_week, API_KEY)
        
    if not ratings and not metrics:
        st.info("No play data available for the selected scope yet. Try adjusting your week scope or check API connectivity.")
    else:
        st.subheader("Team Efficiency & Power Ratings")
        rating_df = pd.DataFrame.from_dict(ratings, orient='index', columns=['Power Rating'])
        st.dataframe(rating_df.sort_values(by='Power Rating', ascending=False), use_container_width=True)
