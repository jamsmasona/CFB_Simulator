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

# Advanced Sleek Dark Theme Styling
st.markdown(
    """
    <style>
    .stApp, div[data-testid="stAppViewContainer"] { 
        background-color: #090d16 !important; 
    }
    label, p, span, h1, h2, h3, h4, h5, h6, li { 
        color: #f0f6fc !important; 
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px string #30363d !important;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #1f6feb 0%, #238636 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        padding: 14px 28px !important;
        border-radius: 10px !important;
        box-shadow: 0px 6px 20px rgba(31, 111, 235, 0.4) !important;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 14px !important;
        padding: 22px !important;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.6);
    }
    div[data-testid="stMetricLabel"] p {
        color: #8b949e !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #79c0ff !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
    }
    .rival-header {
        color: #79c0ff !important;
        border-bottom: 2px solid #30363d;
        padding-bottom: 8px;
        margin-top: 15px;
        font-size: 1.4rem;
        font-weight: 800;
    }
    .rival-line {
        color: #e6edf3 !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin-bottom: 12px;
    }
    .rival-val {
        color: #79c0ff !important;
        font-family: monospace;
        font-size: 1.2rem !important;
        font-weight: 800 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

CFB_TEAMS = sorted([
    "Alabama", "Arizona", "Arizona State", "Arkansas", "Auburn", "Baylor",
    "Boise State", "BYU", "Cal", "Clemson", "Colorado", "Duke", "East Carolina",
    "Florida", "Florida State", "Georgia", "Georgia Tech", "Houston", "Illinois",
    "Indiana", "Iowa", "Iowa State", "Kansas", "Kansas State", "Kentucky",
    "Louisville", "LSU", "Memphis", "Miami", "Michigan", "Michigan State",
    "Minnesota", "Missouri", "NC State", "Nebraska", "North Carolina",
    "Notre Dame", "Ohio State", "Oklahoma", "Oklahoma State", "Ole Miss",
    "Oregon", "Oregon State", "Penn State", "Pittsburgh", "Purdue", "Rutgers",
    "San José State", "SMU", "South Carolina", "Stanford", "TCU", "Tennessee",
    "Texas", "Texas A&M", "Texas Tech", "UCF", "UCLA", "UNLV", "USC", "Utah",
    "Vanderbilt", "Virginia", "Virginia Tech", "Washington", "Washington State",
    "West Virginia", "Wisconsin"
])
CFB_TEAMS = sorted(list(set(CFB_TEAMS)))

API_KEY = st.secrets["CFBD_API_KEY"]


@st.cache_data(ttl=86400)
def fetch_all_sp_ratings():
    url = "https://api.collegefootballdata.com/ratings/sp?year=2026"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    ratings_dict = {}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            for item in res.json():
                team_name = item.get("team")
                rating_val = item.get("rating")
                if team_name and rating_val is not None:
                    ratings_dict[team_name] = float(rating_val)
    except Exception:
        pass
    return ratings_dict


@st.cache_data(ttl=3600)
def fetch_granular_pbp_and_ratings(target_week=15):
    """Pulls play-by-play data week-by-week, calculates success rates,
    and constructs a Ridge Regression model to isolate true opponent-adjusted unit ratings.
    """
    headers = {"Authorization": f"Bearer {API_KEY}"}
    all_plays = []
    
    # Loop through weeks to assemble complete play data safely
    for w in range(1, target_week + 1):
        url = f"https://api.collegefootballdata.com/plays?year=2026&seasonType=regular&week={w}"
        try:
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                if data:
                    all_plays.extend(data)
        except Exception:
            continue
            
    if not all_plays:
        return {}, {}, {}
        
    df_plays = pd.DataFrame(all_plays)
    
    # Filter out kickoffs, punts, and penalties for core efficiency metrics
    core_plays = df_plays[df_plays['play_type'].isin(['Rush', 'Pass Reception', 'Passing'])].copy()
    
    # Calculate Down-and-Distance Success Rate
    # 1st down: >= 50% yards needed; 2nd down: >= 70%; 3rd/4th: 100%
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
    
    # Aggregate offensive EPA and Success Rates
    agg_metrics = core_plays.groupby('offense').agg(
        off_epa=('ppa', 'mean'),
        success_rate=('success', 'mean')
    ).to_dict(orient='index')
    
    # Ridge Regression for Opponent Adjustments (True Talent Model)
    # Mapping game margins to solve simultaneous offensive/defensive coefficients
    games_url = "https://api.collegefootballdata.com/games?year=2026&seasonType=regular"
    try:
        g_res = requests.get(games_url, headers=headers)
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
                    X[idx, h_idx] = 1.0   # Home offense
                    X[idx, len(teams) + a_idx] = -1.0 # Away defense
                    
                ridge = Ridge(alpha=100.0)
                ridge.fit(X, y)
                
                coeffs = ridge.coef_
                ridge_ratings = {}
                for t, idx in team_to_idx.items():
                    off_val = coeffs[idx]
                    def_val = coeffs[len(teams) + idx]
                    ridge_ratings[t] = (off_val - def_val) * 3.5
            else:
                ridge_ratings = {}
        else:
            ridge_ratings = {}
    except Exception:
        ridge_ratings = {}

    return agg_metrics, team_sample_counts, ridge_ratings


def fetch_bayesian_adjusted_profile(team_name, current_week):
    all_ratings = fetch_all_sp_ratings()
    agg_metrics, sample_counts, ridge_ratings = fetch_granular_pbp_and_ratings(target_week=current_week)
    
    preseason_prior = all_ratings.get(team_name, 15.0)
    total_plays = sample_counts.get(team_name, 50)
    
    # Asymptotic Bayesian Shrinkage Curve (Approaches 1.0 as sample size increases)
    shrinkage_weight = total_plays / (total_plays + 120.0)
    
    live_off_epa = agg_metrics.get(team_name, {}).get('off_epa', 0.0) * 30.0
    ridge_val = ridge_ratings.get(team_name, preseason_prior)
    
    # Blended Institutional Rating combining Bayesian Shrinkage + Ridge Adjusted Performance
    blended_rating = (shrinkage_weight * ridge_val) + ((1.0 - shrinkage_weight) * preseason_prior)
    
    return {
        "sp": blended_rating,
        "stat_metric": live_off_epa,
        "sample_weight": shrinkage_weight,
        "plays": total_plays
    }


# SIDEBAR: INSTITUTIONAL POWER RANKINGS
with st.sidebar:
    st.markdown("## ⚡ Institutional Power Grid")
    st.caption("Bayesian Shrinkage & Ridge Regression Engine")

    if st.button("Purge & Re-Index Cache"):
        st.cache_data.clear()
        st.success("Cache cleared! Live API re-indexed.")

    st.markdown("---")
    current_week_input = st.slider("Active Season Week Scope", min_value=1, max_value=15, value=6)
    
    with st.spinner("Computing Ridge Regression & PBP Metrics..."):
        all_ratings = fetch_all_sp_ratings()
        _, _, ridge_ratings = fetch_granular_pbp_and_ratings(target_week=current_week_input)

    if ridge_ratings:
        sorted_ratings = sorted(
            ridge_ratings.items(), key=lambda x: x[1], reverse=True
        )[:25]
        
        for rank, (team, rating) in enumerate(sorted_ratings, start=1):
            st.markdown(f"**#{rank}** {team} *({rating:.1f})*")
    elif all_ratings:
        sorted_ratings = sorted(all_ratings.items(), key=lambda x: x[1], reverse=True)[:25]
        for rank, (team, rating) in enumerate(sorted_ratings, start=1):
            st.markdown(f"**#{rank}** {team} *({rating:.1f})*")
    else:
        st.warning("API connection error for rankings.")

# Auth & Main Layout
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("⚡ Institutional CFB Analytics Suite")
    pwd = st.text_input("Enter Engine Access Key:", type="password")
    if pwd == st.secrets["APP_PASSWORD"]:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd != "":
        st.error("Invalid Access Key")
else:
    st.title("⚡ Institutional CFB Analytics Suite")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        team_a = st.selectbox("Home Team", CFB_TEAMS, index=CFB_TEAMS.index("Georgia") if "Georgia" in CFB_TEAMS else 0)
    with col2:
        team_b = st.selectbox("Away Team", CFB_TEAMS, index=CFB_TEAMS.index("Auburn") if "Auburn" in CFB_TEAMS else 1)
    with col3:
        st.write("")
        is_neutral = st.checkbox("Neutral Venue", value=False)

    if team_a == team_b:
        st.error("⚠️ Select two distinct programs to execute simulation matrices.")
        st.button("🚀 Execute 25,000 Iteration Simulation", use_container_width=True, disabled=True)
    else:
        st.divider()
        st.subheader("⚙️ Calibration Parameters & Context Matrix")
        
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            venue_tier = st.selectbox(
                "Venue Advantage Profile",
                [
                    "Standard Home Field (2.5 pts)", 
                    "Elite Atmosphere / Hostile (3.6 pts)", 
                    "Cross-Country Travel Penalty (3.1 pts)", 
                    "Neutral Site Grid (0.0 pts)"
                ]
            )
        with c_v2:
            st.info(f"Active Evaluation Horizon: **Week {current_week_input}** Play-by-Play Parsing Enabled.")

        if "Elite" in venue_tier:
            hfa_value = 3.6
        elif "Cross-Country" in venue_tier:
            hfa_value = 3.1
        elif "Neutral" in venue_tier or is_neutral:
            hfa_value = 0.0
        else:
            hfa_value = 2.5

        if st.button("🚀 Execute 25,000 Iteration Simulation", use_container_width=True):
            with st.spinner("Crunching 25,000 Monte Carlo vectors with Bayesian Ridge outputs..."):
                p_a = fetch_bayesian_adjusted_profile(team_a, current_week_input)
                p_b = fetch_bayesian_adjusted_profile(team_b, current_week_input)

            hfa = 0.0 if is_neutral else hfa_value

            sp_diff = p_a["sp"] - p_b["sp"]
            stat_diff = p_a["stat_metric"] - p_b["stat_metric"]
            
            blended_diff = (sp_diff * 0.7) + (stat_diff * 0.3)
            raw_diff = blended_diff + hfa
            
            base_spread = 18.0 * float(np.tanh(raw_diff / 19.5))

            NUM_SIMS = 25000
            simulated_margins = np.random.normal(loc=base_spread, scale=11.8, size=NUM_SIMS)

            wins_a = np.sum(simulated_margins > 0)
            win_prob_a = wins_a / NUM_SIMS
            win_prob_b = 1.0 - win_prob_a

            mean_margin = np.mean(simulated_margins)
            display_spread_a = -mean_margin
            display_spread_b = mean_margin

            total_baseline = 51.0 + ((p_a["sp"] + p_b["sp"]) * 0.12)
            simulated_totals = np.random.normal(loc=total_baseline, scale=8.0, size=NUM_SIMS)

            sim_scores_a = np.maximum(3, np.round((simulated_totals / 2) + (simulated_margins / 2)))
            sim_scores_b = np.maximum(3, np.round((simulated_totals / 2) - (simulated_margins / 2)))

            mean_score_a = int(np.mean(sim_scores_a))
            mean_score_b = int(np.mean(sim_scores_b))

            favored_team = team_a if mean_margin >= 0 else team_b

            st.divider()
            st.subheader("📊 25,000-Run Probability Matrix")

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.metric(label=f"🏠 {team_a} Win Probability", value=f"{win_prob_a*100:.1f}%", delta=f"Line: {display_spread_a:+.1f}")
            with rc2:
                st.metric(label="Projected Score Line", value=f"{mean_score_a} - {mean_score_b}")
            with rc3:
                st.metric(label=f"✈️ {team_b} Win Probability", value=f"{win_prob_b*100:.1f}%", delta=f"Line: {display_spread_b:+.1f}")

            st.progress(win_prob_a, text=f"Model Confidence Ratio: {team_a} ({win_prob_a*100:.1f}%) vs {team_b} ({win_prob_b*100:.1f}%)")

            st.divider()
            st.subheader("📈 Margin Density Waveform")
            hist_vals, bin_edges = np.histogram(simulated_margins, bins=35, density=True)
            chart_df = pd.DataFrame({
                f"Spread Range (← {team_b} | {team_a} →)": bin_edges[:-1],
                "Density Vector": hist_vals
            })
            st.line_chart(chart_df, x=f"Spread Range (← {team_b} | {team_a} →)", y="Density Vector", use_container_width=True)

            st.divider()
            st.subheader("🎯 Executive Game Architecture")
            st.markdown(f"**Core Model Edge:** `{favored_team}` is projected to control the line of scrimmage by **{abs(mean_margin):.1f} points**.")
            st.markdown(f"**Total Projection Index:** Over/Under market line baseline sits at **{total_baseline:.1f} total points**.")

            st.divider()
            st.subheader("📋 Advanced Statistical Box-Score Forecast")
            
            box_1, box_2 = st.columns(2)
            with box_1:
                st.markdown(f"<div class='rival-header'>{team_a} Profile Output</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rival-line'>Bayesian Ridge Rating: <span class='rival-val'>{p_a['sp']:.1f}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rival-line'>Sample Snaps Analyzed: <span class='rival-val'>{p_a['plays']} plays</span></div>", unsafe_allow_html=True)
            with box_2:
                st.markdown(f"<div class='rival-header'>{team_b} Profile Output</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rival-line'>Bayesian Ridge Rating: <span class='rival-val'>{p_b['sp']:.1f}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rival-line'>Sample Snaps Analyzed: <span class='rival-val'>{p_b['plays']} plays</span></div>", unsafe_allow_html=True)
