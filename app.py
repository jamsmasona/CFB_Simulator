from datetime import datetime
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
    .card-title {
        color: #58a6ff !important;
        font-size: 1.3rem !important;
        font-weight: 800 !important;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
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
    "Alabama",
    "Arizona",
    "Arizona State",
    "Arkansas",
    "Auburn",
    "Baylor",
    "Boise State",
    "BYU",
    "Cal",
    "Clemson",
    "Colorado",
    "Duke",
    "East Carolina",
    "Florida",
    "Florida State",
    "Georgia",
    "Georgia Tech",
    "Houston",
    "Illinois",
    "Indiana",
    "Iowa",
    "Iowa State",
    "Kansas",
    "Kansas State",
    "Kentucky",
    "Louisville",
    "LSU",
    "Memphis",
    "Miami",
    "Michigan",
    "Michigan State",
    "Minnesota",
    "Missouri",
    "NC State",
    "Nebraska",
    "North Carolina",
    "Notre Dame",
    "Ohio State",
    "Oklahoma",
    "Oklahoma State",
    "Ole Miss",
    "Oregon",
    "Oregon State",
    "Penn State",
    "Pittsburgh",
    "Purdue",
    "Rutgers",
    "San José State",
    "SMU",
    "South Carolina",
    "Stanford",
    "TCU",
    "Tennessee",
    "Texas",
    "Texas A&M",
    "Texas Tech",
    "UCF",
    "UCLA",
    "UNLV",
    "USC",
    "Utah",
    "Vanderbilt",
    "Virginia",
    "Virginia Tech",
    "Washington",
    "Washington State",
    "West Virginia",
    "Wisconsin",
])
CFB_TEAMS = sorted(list(set(CFB_TEAMS)))

API_KEY = st.secrets.get("CFBD_API_KEY", "")


# Cached API Fetch Function for ALL SP+ Ratings
@st.cache_data(ttl=86400)
def fetch_all_sp_ratings():
    url = "https://api.collegefootballdata.com/ratings/sp?year=2026"
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    
    ratings_dict = {}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            data = res.json()
            for item in data:
                team_name = item.get("team")
                rating_val = item.get("rating")
                if team_name and rating_val is not None:
                    ratings_dict[team_name] = float(rating_val)
        else:
            st.warning(f"API returned non-200 status code: {res.status_code}")
    except Exception as e:
        st.error(f"Failed to connect to CFBD API: {e}")

    return ratings_dict


def fetch_advanced_profile(team_name):
    all_ratings = fetch_all_sp_ratings()
    sp_val = all_ratings.get(team_name, 15.0)
    
    # Dynamic volatility and pace factors derived from SP+ profile
    volatility = 8.5 + (abs(sp_val) * 0.09)
    pace_factor = 1.0 + (0.06 if sp_val > 18 else (-0.06 if sp_val < 2 else 0.0))
    
    return {
        "sp": sp_val,
        "volatility": volatility,
        "pace": pace_factor
    }


# SIDEBAR: DYNAMIC TOP 25 MODEL RANKINGS
with st.sidebar:
    st.markdown("## 🏆 Model Top 25 Rankings")
    st.caption("Updated dynamically via live SP+ metrics")

    if st.button("Clear Cache & Refresh Data"):
        st.cache_data.clear()
        st.success("Cache cleared! Latest data will be fetched.")

    all_ratings = fetch_all_sp_ratings()

    if all_ratings:
        sorted_ratings = sorted(
            all_ratings.items(), key=lambda x: x[1], reverse=True
        )[:25]
        for rank, (team, rating) in enumerate(sorted_ratings, start=1):
            st.markdown(f"**#{rank}** {team} *({rating:.1f})*")
    else:
        st.warning("Could not load live rankings from API.")

# Authentication Flow
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

app_password = st.secrets.get("APP_PASSWORD", "")
if app_password and not st.session_state["authenticated"]:
    st.title("🏈 2026 College Football Game Predictor")
    pwd = st.text_input("Enter Access Password:", type="password")
    if pwd == app_password:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd != "":
        st.error("Incorrect Password")
else:
    st.title("🏈 2026 College Football Game Predictor")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        team_a = st.selectbox(
            "Home Team", CFB_TEAMS, index=CFB_TEAMS.index("Oregon") if "Oregon" in CFB_TEAMS else 0
        )
    with col2:
        team_b = st.selectbox(
            "Away Team", CFB_TEAMS, index=CFB_TEAMS.index("Boise State") if "Boise State" in CFB_TEAMS else 1
        )
    with col3:
        st.write("")
        is_neutral = st.checkbox("Neutral Field Game", value=False)

    same_team_selected = team_a == team_b

    if same_team_selected:
        st.error(
            "⚠️ Invalid Matchup: Please select two different teams to run a valid simulation."
        )
        st.button(
            "🎲 Run 10,000 Monte Carlo Simulations",
            use_container_width=True,
            disabled=True,
        )
    else:
        if st.button("🎲 Run 10,000 Monte Carlo Simulations", use_container_width=True):
            with st.spinner("Processing 10,000 Monte Carlo game iterations with dynamic pace & volatility..."):
                p_a = fetch_advanced_profile(team_a)
                p_b = fetch_advanced_profile(team_b)

                hfa = 0.0 if is_neutral else 2.5
                n_sims = 10000

                # Enhanced Monte Carlo Engine with Pace and Volatility
                expected_diff = (p_a["sp"] - p_b["sp"]) + hfa
                combined_pace = (p_a["pace"] + p_b["pace"]) / 2.0
                base_scoring_mean = 27.5 * combined_pace
                
                score_a_mean = base_scoring_mean + (expected_diff / 2)
                score_b_mean = base_scoring_mean - (expected_diff / 2)

                # Generate 10,000 independent game score outcomes using team-specific volatility and pace modifiers
                sim_scores_a = np.clip(np.random.normal(loc=max(3, score_a_mean), scale=p_a["volatility"], size=n_sims) * np.random.normal(p_a["pace"], 0.02, size=n_sims), 3, 75)
                sim_scores_b = np.clip(np.random.normal(loc=max(3, score_b_mean), scale=p_b["volatility"], size=n_sims) * np.random.normal(p_b["pace"], 0.02, size=n_sims), 3, 75)

                mean_score_a = np.mean(sim_scores_a)
                mean_score_b = np.mean(sim_scores_b)

                simulated_margins = sim_scores_a - sim_scores_b

                wins_a = np.sum(simulated_margins > 0)
                wins_b = np.sum(simulated_margins < 0)
                ties = n_sims - wins_a - wins_b

                win_prob_a = (wins_a + (ties / 2)) / n_sims
                win_prob_b = (wins_b + (ties / 2)) / n_sims

                mean_margin = np.mean(simulated_margins)
                favored_team = team_a if mean_margin > 0 else team_b

                total_scores = sim_scores_a + sim_scores_b
                total_baseline = np.mean(total_scores)

                one_possession = np.mean(np.abs(simulated_margins) <= 8) * 100
                blowouts_a = np.mean(simulated_margins >= 14) * 100
                blowouts_b = np.mean(simulated_margins <= -14) * 100

                # Fully simulated box score metrics derived across the distribution of all 10,000 runs
                sim_total_yds_a = np.clip(np.random.normal(loc=(380 * p_a["pace"]) + (sim_scores_a * 3.5), scale=45, size=n_sims), 150, 750)
                sim_total_yds_b = np.clip(np.random.normal(loc=(380 * p_b["pace"]) + (sim_scores_b * 3.5), scale=45, size=n_sims), 150, 750)
                
                sim_pass_yds_a = sim_total_yds_a * np.random.normal(0.64, 0.05, size=n_sims)
                sim_rush_yds_a = sim_total_yds_a - sim_pass_yds_a
                
                sim_pass_yds_b = sim_total_yds_b * np.random.normal(0.64, 0.05, size=n_sims)
                sim_rush_yds_b = sim_total_yds_b - sim_pass_yds_b

                sim_pass_tds_a = np.clip(np.round(sim_scores_a * np.random.uniform(0.45, 0.65, size=n_sims) / 7), 0, 6)
                sim_rush_tds_a = np.clip(np.round((sim_scores_a / 7) - sim_pass_tds_a), 0, 5)
                
                sim_pass_tds_b = np.clip(np.round(sim_scores_b * np.random.uniform(0.45, 0.65, size=n_sims) / 7), 0, 6)
                sim_rush_tds_b = np.clip(np.round((sim_scores_b / 7) - sim_pass_tds_b), 0, 5)

                sim_fgs_a = np.clip(np.round((sim_scores_a - ((sim_pass_tds_a + sim_rush_tds_a) * 7)) / 3), 0, 5)
                sim_fgs_b = np.clip(np.round((sim_scores_b - ((sim_pass_tds_b + sim_rush_tds_b) * 7)) / 3), 0, 5)

                sim_first_downs_a = np.round(sim_total_yds_a / 17.5)
                sim_first_downs_b = np.round(sim_total_yds_b / 17.5)

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label=f"{team_a} Win Probability", value=f"{win_prob_a*100:.1f}%")
                with res_col2:
                    st.metric(label=f"{team_b} Win Probability", value=f"{win_prob_b*100:.1f}%")

                # Scoreboard Banner Display
                spread_text = f"{favored_team} -{abs(mean_margin):.1f}" if abs(mean_margin) > 0.5 else "Pick 'em"
                st.markdown(
                    f"""
                    <div style="background-color: #161b22; border: 2px solid #30363d; border-radius: 12px; padding: 20px; text-align: center; margin-top: 15px; margin-bottom: 15px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #8b949e; margin-bottom: 5px;">SCOREBOARD PREDICTION</div>
                        <div style="font-size: 2.4rem; font-weight: 900; color: #58a6ff; letter-spacing: 1px;">{team_a} {mean_score_a:.1f} — {team_b} {mean_score_b:.1f}</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #f0f6fc; margin-top: 8px;">Spread: <span style="color: #58a6ff;">{spread_text}</span> &nbsp;|&nbsp; Total Line: <span style="color: #58a6ff;">{total_baseline:.1f}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
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

                st.markdown(
                    f"**Model Spread Edge:** `{favored_team}` is favored by **{abs(mean_margin):.1f} points** based on SP+ differentials, venue adjustments, and pace factors."
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

                avg_first_downs_a = int(np.mean(sim_first_downs_a))
                avg_total_yds_a = int(np.mean(sim_total_yds_a))
                avg_pass_yds_a = int(np.mean(sim_pass_yds_a))
                avg_rush_yds_a = int(np.mean(sim_rush_yds_a))
                avg_pass_tds_a = round(np.mean(sim_pass_tds_a), 1)
                avg_rush_tds_a = round(np.mean(sim_rush_tds_a), 1)
                avg_fgs_a = round(np.mean(sim_fgs_a), 1)

                avg_first_downs_b = int(np.mean(sim_first_downs_b))
                avg_total_yds_b = int(np.mean(sim_total_yds_b))
                avg_pass_yds_b = int(np.mean(sim_pass_yds_b))
                avg_rush_yds_b = int(np.mean(sim_rush_yds_b))
                avg_pass_tds_b = round(np.mean(sim_pass_tds_b), 1)
                avg_rush_tds_b = round(np.mean(sim_rush_tds_b), 1)
                avg_fgs_b = round(np.mean(sim_fgs_b), 1)

                box_col1, box_col2 = st.columns(2)
                with box_col1:
                    st.markdown(
                        f"<div class='stat-header'>{team_a} (Home) - SP+: {p_a['sp']:.1f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>First Downs: <span class='stat-val'>{avg_first_downs_a}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Total Offense: <span class='stat-val'>{avg_total_yds_a} yds</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Passing Yards: <span class='stat-val'>{avg_pass_yds_a} yds</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Rushing Yards: <span class='stat-val'>{avg_rush_yds_a} yds</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Passing Touchdowns: <span class='stat-val'>{avg_pass_tds_a}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Rushing Touchdowns: <span class='stat-val'>{avg_rush_tds_a}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Field Goals: <span class='stat-val'>{avg_fgs_a}</span></div>",
                        unsafe_allow_html=True,
                    )

                with box_col2:
                    st.markdown(
                        f"<div class='stat-header'>{team_b} (Away) - SP+: {p_b['sp']:.1f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>First Downs: <span class='stat-val'>{avg_first_downs_b}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Total Offense: <span class='stat-val'>{avg_total_yds_b} yds</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Passing Yards: <span class='stat-val'>{avg_pass_yds_b} yds</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Rushing Yards: <span class='stat-val'>{avg_rush_yds_b} yds</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Passing Touchdowns: <span class='stat-val'>{avg_pass_tds_b}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Rushing Touchdowns: <span class='stat-val'>{avg_rush_tds_b}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='stat-line'>Field Goals: <span class='stat-val'>{avg_fgs_b}</span></div>",
                        unsafe_allow_html=True,
                    )

                # Log to history
                st.session_state.history.insert(0, {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Home Team": team_a,
                    "Away Team": team_b,
                    "Favored": favored_team,
                    "Spread": f"{abs(mean_margin):.1f}",
                    f"{team_a} Win %": f"{win_prob_a*100:.1f}%",
                    f"{team_b} Win %": f"{win_prob_b*100:.1f}%"
                })

    st.divider()
    st.subheader("🕒 Recent Simulation History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run a simulation above to populate recent history logs.")
