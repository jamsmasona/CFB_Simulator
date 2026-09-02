from datetime import datetimefrom datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Set Native Dark Mode Configuration
st.set_page_config(
    page_title="2026 CFB Monte Carlo Engine",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session states if they don't exist
if "history" not in st.session_state:
    st.session_state.history = []

# Global Dark CSS Styling
st.markdown(
    """
    <style>
    .stApp, div[data-testid="stAppViewContainer"] { 
        background-color: #0d1117 !important; 
    }
    label, p, span, h1, h2, h3, h4, h5, h6, li { 
        color: #f0f6fc !important; 
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f0f6fc !important;
    }
    div.stButton > button {
        background-color: #238636 !important;
        color: #ffffff !important;
        border: 1px solid #2ea043 !important;
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        box-shadow: 0px 4px 12px rgba(35, 134, 54, 0.4) !important;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 2px solid #30363d !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    div[data-testid="stMetricLabel"] p {
        color: #8b949e !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #58a6ff !important;
        font-size: 2.6rem !important;
        font-weight: 900 !important;
    }
    .stat-header {
        color: #58a6ff !important;
        border-bottom: 2px solid #30363d;
        padding-bottom: 6px;
        margin-top: 15px;
        font-size: 1.4rem;
        font-weight: 800;
    }
    .stat-line {
        color: #e6edf3 !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin-bottom: 10px;
    }
    .stat-val {
        color: #58a6ff !important;
        font-family: monospace;
        font-size: 1.2rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stVegaLiteChart"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# FBS Teams Database
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


@st.cache_data(ttl=86400)
def fetch_advanced_season_stats():
    url = "https://api.collegefootballdata.com/stats/season/advanced?year=2026"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    stats_dict = {}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            for item in res.json():
                team_name = item.get("team")
                offense = item.get("offense", {})
                defense = item.get("defense", {})
                off_epa = offense.get("ppa", 0.0)
                def_epa = defense.get("ppa", 0.0)
                net_epa = float(off_epa) - float(def_epa)
                stats_dict[team_name] = net_epa * 25.0
    except Exception:
        pass
    return stats_dict


@st.cache_data(ttl=3600)
def fetch_completed_games_adjustment():
    """Automatically parses completed games from the CFBD API to build 
    a real-time performance modifier based on actual game margins.
    """
    url = "https://api.collegefootballdata.com/games?year=2026&seasonType=regular"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    adjustments = {}
    team_game_counts = {}
    
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            games = res.json()
            for game in games:
                home_team = game.get("home_team")
                away_team = game.get("away_team")
                home_pts = game.get("home_points")
                away_pts = game.get("away_points")
                
                if home_pts is not None and away_pts is not None:
                    margin = home_pts - away_pts
                    adjustments[home_team] = adjustments.get(home_team, 0.0) + (margin * 0.25)
                    adjustments[away_team] = adjustments.get(away_team, 0.0) - (margin * 0.25)
                    team_game_counts[home_team] = team_game_counts.get(home_team, 0) + 1
                    team_game_counts[away_team] = team_game_counts.get(away_team, 0) + 1
            
            for team in adjustments:
                if team_game_counts.get(team, 0) > 0:
                    adjustments[team] = adjustments[team] / team_game_counts[team]
    except Exception:
        pass
        
    return adjustments


def fetch_advanced_profile(team_name):
    all_ratings = fetch_all_sp_ratings()
    all_stats = fetch_advanced_season_stats()
    auto_adjustments = fetch_completed_games_adjustment()
    
    sp_val = all_ratings.get(team_name, 15.0)
    stat_val = all_stats.get(team_name, sp_val)
    auto_adj = auto_adjustments.get(team_name, 0.0)
    
    return {
        "sp": sp_val + auto_adj, 
        "stat_metric": stat_val + auto_adj,
        "auto_adjustment": auto_adj
    }


# SIDEBAR: DYNAMIC TOP 25 MODEL RANKINGS
with st.sidebar:
    st.markdown("## 🏆 Model Top 25 Rankings")
    st.caption("Auto-calibrated with live game results & SP+")

    if st.button("Clear Cache & Refresh Data"):
        st.cache_data.clear()
        st.success("Cache cleared! Latest data fetched from API.")

    st.markdown("---")
    all_ratings = fetch_all_sp_ratings()
    auto_adjustments = fetch_completed_games_adjustment()

    if all_ratings:
        composite_ratings = {
            team: rating + auto_adjustments.get(team, 0.0) 
            for team, rating in all_ratings.items()
        }
        sorted_ratings = sorted(
            composite_ratings.items(), key=lambda x: x[1], reverse=True
        )[:25]
        
        for rank, (team, rating) in enumerate(sorted_ratings, start=1):
            st.markdown(f"**#{rank}** {team} *({rating:.1f})*")
    else:
        st.warning("Could not load live rankings from API.")

# Authentication Flow
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🏈 2026 College Football Game Predictor")
    pwd = st.text_input("Enter Access Password:", type="password")
    if pwd == st.secrets["APP_PASSWORD"]:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd != "":
        st.error("Incorrect Password")
else:
    st.title("🏈 2026 College Football Game Predictor")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        team_a = st.selectbox(
            "Home Team", CFB_TEAMS, index=CFB_TEAMS.index("North Carolina") if "North Carolina" in CFB_TEAMS else CFB_TEAMS.index("Oregon")
        )
    with col2:
        team_b = st.selectbox(
            "Away Team", CFB_TEAMS, index=CFB_TEAMS.index("TCU") if "TCU" in CFB_TEAMS else CFB_TEAMS.index("Boise State")
        )
    with col3:
        st.write("")
        is_neutral = st.checkbox("Neutral Field Game", value=False)

    same_team_selected = team_a == team_b

    if same_team_selected:
        st.error("⚠️ Invalid Matchup: Please select two different teams to run a valid simulation.")
        st.button("🎲 Run 10,000 Monte Carlo Simulations", use_container_width=True, disabled=True)
    else:
        st.divider()
        st.subheader("🏟️ Venue & Season Progression Settings")
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            venue_type = st.radio(
                "Select Home-Field Advantage Tier",
                [
                    "Standard HFA (2.5 pts)", 
                    "Hostile Environment / Elite Atmos (3.5 pts)", 
                    "Difficult Travel / Long Distance (3.0 pts)", 
                    "Quiet / Low Advantage (1.5 pts)"
                ],
                index=0,
            )
        with col_v2:
            current_week = st.slider(
                "Current Season Week", min_value=1, max_value=15, value=6,
                help="Shifts weight automatically from preseason baseline to live on-field efficiency stats."
            )

        if "Elite Atmos" in venue_type:
            hfa_value = 3.5
        elif "Long Distance" in venue_type:
            hfa_value = 3.0
        elif "Low Advantage" in venue_type:
            hfa_value = 1.5
        else:
            hfa_value = 2.5

        stat_weight = min(0.85, 0.05 * current_week)
        sp_weight = 1.0 - stat_weight

        if st.button("🎲 Run 10,000 Monte Carlo Simulations", use_container_width=True):
            with st.spinner("Processing 10,000 Monte Carlo game iterations with live data..."):
                p_a = fetch_advanced_profile(team_a)
                p_b = fetch_advanced_profile(team_b)

            hfa = 0.0 if is_neutral else hfa_value

            sp_diff = p_a["sp"] - p_b["sp"]
            stat_diff = p_a["stat_metric"] - p_b["stat_metric"]
            
            blended_diff = (sp_diff * sp_weight) + (stat_diff * stat_weight)
            raw_diff = blended_diff + hfa
            
            max_cap = 17.0
            base_spread = max_cap * float(np.tanh(raw_diff / 18.0))

            NUM_SIMS = 10000
            simulated_margins = np.random.normal(
                loc=base_spread, scale=12.5, size=NUM_SIMS
            )

            wins_a = np.sum(simulated_margins > 0)
            wins_b = NUM_SIMS - wins_a
            win_prob_a = wins_a / NUM_SIMS
            win_prob_b = wins_b / NUM_SIMS

            mean_margin = np.mean(simulated_margins)

            display_spread_a = -mean_margin
            display_spread_b = mean_margin

            total_baseline = 48.5 + ((p_a["sp"] + p_b["sp"]) * 0.15)
            simulated_totals = np.random.normal(
                loc=total_baseline, scale=8.5, size=NUM_SIMS
            )

            base_yds_a = 220 + (p_a["sp"] * 3.5) + (simulated_totals * 0.4)
            base_yds_b = 220 + (p_b["sp"] * 3.5) + (simulated_totals * 0.4)

            sim_total_yds_a = np.random.normal(loc=base_yds_a, scale=35.0, size=NUM_SIMS)
            sim_total_yds_b = np.random.normal(loc=base_yds_b, scale=35.0, size=NUM_SIMS)

            sim_pass_yds_a = sim_total_yds_a * 0.58
            sim_rush_yds_a = sim_total_yds_a * 0.42
            sim_pass_yds_b = sim_total_yds_b * 0.58
            sim_rush_yds_b = sim_total_yds_b * 0.42

            sim_scores_a = np.maximum(
                3, np.round((simulated_totals / 2) + (simulated_margins / 2))
            )
            sim_scores_b = np.maximum(
                3, np.round((simulated_totals / 2) - (simulated_margins / 2))
            )

            mean_score_a = int(np.mean(sim_scores_a))
            mean_score_b = int(np.mean(sim_scores_b))

            blowouts_a = (np.sum(simulated_margins >= 14) / NUM_SIMS) * 100
            blowouts_b = (np.sum(simulated_margins <= -14) / NUM_SIMS) * 100
            one_possession = (np.sum(np.abs(simulated_margins) <= 8) / NUM_SIMS) * 100

            favored_team = team_a if mean_margin >= 0 else team_b

            history_entry = {
                "Matchup": f"{team_a} vs {team_b}",
                "Projected Margin": f"{abs(mean_margin):.1f} pts ({favored_team})",
                "Win Probability": f"{win_prob_a*100:.1f}% ({team_a})",
                "Venue": "Neutral" if is_neutral else f"Home ({team_a}, {hfa} pts)",
                "Week": current_week,
            }
            st.session_state.history.insert(0, history_entry)
            if len(st.session_state.history) > 4:
                st.session_state.history.pop()

            st.divider()
            st.subheader("📊 Monte Carlo Simulation Results (10,000 Runs)")

            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.metric(
                    label=f"🏠 {team_a} Win Chance",
                    value=f"{win_prob_a*100:.1f}%",
                    delta=f"Spread: {display_spread_a:+.1f}",
                )
            with res_c2:
                st.metric(
                    label="Projected Score",
                    value=f"{mean_score_a} - {mean_score_b}",
                )
            with res_c3:
                st.metric(
                    label=f"✈️ {team_b} Win Chance",
                    value=f"{win_prob_b*100:.1f}%",
                    delta=f"Spread: {display_spread_b:+.1f}",
                )

            st.progress(
                win_prob_a,
                text=(
                    f"{team_a} ({win_prob_a*100:.1f}%) vs {team_b}"
                    f" ({win_prob_b*100:.1f}%)"
                ),
            )

            st.divider()
            st.subheader("📈 Projected Point Margin Distribution")
            st.caption(
                f"Shows how often each team wins by various margins across 10,000"
                f" simulations. Left side = {team_b} wins | Right side = {team_a} wins"
                " | Center line (0) = Overtime / Toss-up."
            )

            hist_values, bin_edges = np.histogram(
                simulated_margins, bins=30, density=True
            )
            chart_data = pd.DataFrame({
                f"Margin (← {team_b} Wins | {team_a} Wins →)": bin_edges[:-1],
                "Simulation Frequency Density": hist_values,
            })
            st.line_chart(
                chart_data,
                x=f"Margin (← {team_b} Wins | {team_a} Wins →)",
                y="Simulation Frequency Density",
                use_container_width=True,
            )

            st.divider()
            st.subheader("📈 Betting Market Analytics")

            venue_desc = f"Neutral site" if is_neutral else f"Home Field Advantage ({hfa} pts)"
            st.markdown(
                f"**Model Spread Edge:** `{favored_team}` is favored by **{abs(mean_margin):.1f} points** based on Week {current_week} blended metrics and {venue_desc}."
            )
            st.markdown(
                f"**Model Total Baseline:** Projected combined scoring line is **{total_baseline:.1f} points**."
            )
            st.markdown(
                f"**Game Script Distribution:** One-possession game probability (≤ 8 pts) is **{one_possession:.1f}%**. "
                f"Decisive margin probability (14+ pts) is **{blowouts_a:.1f}%** for `{team_a}` and **{blowouts_b:.1f}%** for `{team_b}`."
            )

            st.divider()
            st.subheader("🏈 Simulated Average Game Box Score")

            def calculate_scoring_breakdown(score):
                tds = int(score // 7)
                remainder = int(score % 7)
                fgs = 0
                if remainder == 3:
                    fgs = 1
                elif remainder == 6:
                    fgs = 2
                elif remainder in [1, 2, 4, 5] and tds > 0:
                    tds -= 1
                    fgs = int((score - (tds * 7)) // 3)

                pass_tds = max(0, min(tds, int(round(tds * 0.55))))
                rush_tds = tds - pass_tds
                return pass_tds, rush_tds, fgs

            pass_tds_a, rush_tds_a, fgs_a = calculate_scoring_breakdown(mean_score_a)
            pass_tds_b, rush_tds_b, fgs_b = calculate_scoring_breakdown(mean_score_b)

            total_yds_a = int(np.mean(sim_total_yds_a))
            total_yds_b = int(np.mean(sim_total_yds_b))

            pass_yds_a = int(np.mean(sim_pass_yds_a))
            rush_yds_a = int(np.mean(sim_rush_yds_a))

            pass_yds_b = int(np.mean(sim_pass_yds_b))
            rush_yds_b = int(np.mean(sim_rush_yds_b))

            first_downs_a = round(total_yds_a / 18.0)
            first_downs_b = round(total_yds_b / 18.0)

            box_col1, box_col2 = st.columns(2)
            with box_col1:
                st.markdown(
                    f"<div class='stat-header'>{team_a} (Home) - Blended Rating: { (p_a['sp'] * sp_weight) + (p_a['stat_metric'] * stat_weight):.1f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>First Downs: <span class='stat-val'>{first_downs_a}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Total Offense: <span class='stat-val'>{total_yds_a} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Passing Yards: <span class='stat-val'>{pass_yds_a} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Rushing Yards: <span class='stat-val'>{rush_yds_a} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Passing Touchdowns: <span class='stat-val'>{pass_tds_a}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Rushing Touchdowns: <span class='stat-val'>{rush_tds_a}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Field Goals: <span class='stat-val'>{fgs_a}</span></div>",
                    unsafe_allow_html=True,
                )

            with box_col2:
                st.markdown(
                    f"<div class='stat-header'>{team_b} (Away) - Blended Rating: { (p_b['sp'] * sp_weight) + (p_b['stat_metric'] * stat_weight):.1f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>First Downs: <span class='stat-val'>{first_downs_b}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Total Offense: <span class='stat-val'>{total_yds_b} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Passing Yards: <span class='stat-val'>{pass_yds_b} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Rushing Yards: <span class='stat-val'>{rush_yds_b} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Passing Touchdowns: <span class='stat-val'>{pass_tds_b}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Rushing Touchdowns: <span class='stat-val'>{rush_yds_b // 55}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Field Goals: <span class='stat-val'>{fgs_b}</span></div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("🕒 Recent Simulation History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run a simulation above to populate recent history logs.")
