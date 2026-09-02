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

API_KEY = st.secrets["CFBD_API_KEY"]


# Cached API Fetch Function for ALL SP+ Ratings
@st.cache_data(ttl=86400)
def fetch_all_sp_ratings():
    url = "https://api.collegefootballdata.com/ratings/sp?year=2026"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
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
    return {"sp": sp_val}


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
            "Home Team", CFB_TEAMS, index=CFB_TEAMS.index("Oregon")
        )
    with col2:
        team_b = st.selectbox(
            "Away Team", CFB_TEAMS, index=CFB_TEAMS.index("Boise State")
        )
    with col3:
        st.write("")
        is_neutral = st.checkbox("Neutral Field Game", value=False)

    same_team_selected = team_a == team_b

    if same_team_selected:
        st.error(
            "⚠️ Invalid Matchup: Please select two different teams to run a valid"
            " simulation."
        )
        st.button(
            "🎲 Run 10,000 Monte Carlo Simulations",
            use_container_width=True,
            disabled=True,
        )
    else:
        if st.button("🎲 Run 10,000 Monte Carlo Simulations", use_container_width=True):
            with st.spinner("Processing 10,000 Monte Carlo game iterations..."):
                p_a = fetch_advanced_profile(team_a)
                p_b = fetch_advanced_profile(team_b)

            hfa = 0.0 if is_neutral else 2.5

            # Blending SP+ differential with simulated/actual stat differentials based on season week
            sp_diff = p_a["sp"] - p_b["sp"]
            
            # Simulated stat differential placeholder (can be swapped with real box score averages once games are played)
            stat_diff = (p_a["sp"] * 1.1) - (p_b["sp"] * 0.9) 
            
            blended_diff = (sp_diff * sp_weight) + (stat_diff * stat_weight)
            raw_diff = blended_diff + hfa
            
            raw_diff = (p_a["sp"] - p_b["sp"]) + hfa
            max_cap = 17.0
            base_spread = max_cap * float(np.tanh(raw_diff / 18.0))

@@ -114,3 +363,166 @@
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

            st.markdown(
                f"**Model Spread Edge:** `{favored_team}` is favored by **{abs(mean_margin):.1f} points** based on SP+ differentials and venue adjustments."
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
                    f"<div class='stat-header'>{team_a} (Home) - SP+: {p_a['sp']:.1f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>First Downs: <span"
                    f" class='stat-val'>{first_downs_a}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Total Offense: <span"
                    f" class='stat-val'>{total_yds_a} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Passing Yards: <span"
                    f" class='stat-val'>{pass_yds_a} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Rushing Yards: <span"
                    f" class='stat-val'>{rush_yds_a} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Passing Touchdowns: <span"
                    f" class='stat-val'>{pass_tds_a}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Rushing Touchdowns: <span"
                    f" class='stat-val'>{rush_tds_a}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Field Goals: <span"
                    f" class='stat-val'>{fgs_a}</span></div>",
                    unsafe_allow_html=True,
                )

            with box_col2:
                st.markdown(
                    f"<div class='stat-header'>{team_b} (Away) - SP+: {p_b['sp']:.1f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>First Downs: <span"
                    f" class='stat-val'>{first_downs_b}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Total Offense: <span"
                    f" class='stat-val'>{total_yds_b} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Passing Yards: <span"
                    f" class='stat-val'>{pass_yds_b} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Rushing Yards: <span"
                    f" class='stat-val'>{rush_yds_b} yds</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Passing Touchdowns: <span"
                    f" class='stat-val'>{pass_tds_b}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Rushing Touchdowns: <span"
                    f" class='stat-val'>{rush_tds_b}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='stat-line'>Field Goals: <span"
                    f" class='stat-val'>{fgs_b}</span></div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("🕒 Recent Simulation History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run a simulation above to populate recent history logs.")
