from datetime import datetime

import numpy as np

import pandas as pd

import requests

import streamlit as st



# Set Native Dark Mode Configuration

st.set_page_config(

    page_title="2026 CFB Matchup Engine",

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

    "Boise State", "BYU", "Cal", "Clemson", "Colorado", "Duke",

    "East Carolina", "Florida", "Florida State", "Georgia", "Georgia Tech",

    "Houston", "Illinois", "Indiana", "Iowa", "Iowa State", "Kansas",

    "Kansas State", "Kentucky", "Louisville", "LSU", "Memphis", "Miami",

    "Michigan", "Michigan State", "Minnesota", "Missouri", "NC State",

    "Nebraska", "North Carolina", "Notre Dame", "Ohio State", "Oklahoma",

    "Oklahoma State", "Ole Miss", "Oregon", "Oregon State", "Penn State",

    "Pittsburgh", "Purdue", "Rutgers", "San José State", "SMU",

    "South Carolina", "Stanford", "TCU", "Tennessee", "Texas", "Texas A&M",

    "Texas Tech", "UCF", "UCLA", "UNLV", "USC", "Utah", "Vanderbilt",

    "Virginia", "Virginia Tech", "Washington", "Washington State",

    "West Virginia", "Wisconsin",

])



API_KEY = st.secrets.get("CFBD_API_KEY", "")





# Cached API Fetch Function for Offense and Defense SP+ Ratings

@st.cache_data(ttl=86400)

def fetch_all_sp_ratings():

    url = "https://api.collegefootballdata.com/ratings/sp?year=2026"

    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

    

    profiles = {}

    try:

        res = requests.get(url, headers=headers)

        if res.status_code == 200:

            data = res.json()

            for item in data:

                team_name = item.get("team")

                if team_name:

                    net_val = float(item.get("rating", 15.0))

                    

                    off_raw = item.get("offense", net_val)

                    if isinstance(off_raw, dict):

                        off_val = float(off_raw.get("rating", net_val))

                    else:

                        off_val = float(off_raw) if off_raw is not None else net_val



                    def_raw = item.get("defense", net_val)

                    if isinstance(def_raw, dict):

                        def_val = float(def_raw.get("rating", net_val))

                    else:

                        def_val = float(def_raw) if def_raw is not None else net_val

                    

                    profiles[team_name] = {

                        "offense": off_val,

                        "defense": def_val,

                        "net": net_val

                    }

        else:

            st.warning(f"API returned non-200 status code: {res.status_code}")

    except Exception as e:

        st.error(f"Failed to connect to CFBD API: {e}")



    return profiles





def fetch_advanced_profile(team_name):

    all_profiles = fetch_all_sp_ratings()

    if team_name in all_profiles:

        return all_profiles[team_name]

    return {"offense": 15.0, "defense": 15.0, "net": 15.0}





# SIDEBAR: DYNAMIC TOP 25 MODEL RANKINGS

with st.sidebar:

    st.markdown("## 🏆 Model Top 25 Rankings")

    st.caption("Ranked by net SP+ efficiency")



    if st.button("Clear Cache & Refresh Data"):

        st.cache_data.clear()

        st.success("Cache cleared! Latest data will be fetched.")



    all_profiles = fetch_all_sp_ratings()



    if all_profiles:

        sorted_ratings = sorted(

            all_profiles.items(), key=lambda x: x[1]["net"], reverse=True

        )[:25]

        for rank, (team, data) in enumerate(sorted_ratings, start=1):

            st.markdown(f"**#{rank}** {team} *({data['net']:.1f})*")

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

            with st.spinner("Processing anchored Monte Carlo game iterations..."):

                p_a = fetch_advanced_profile(team_a)

                p_b = fetch_advanced_profile(team_b)



                hfa = 2.5 if not is_neutral else 0.0

                n_sims = 10000



                league_avg_scoring = 28.0

                

                net_diff_a = (p_a["net"] - p_b["net"]) + hfa

                net_diff_b = -net_diff_a



                score_a_mean = league_avg_scoring + (net_diff_a * 0.5)

                score_b_mean = league_avg_scoring + (net_diff_b * 0.5)



                sim_scores_a = np.clip(np.random.normal(loc=max(3, score_a_mean), scale=8.5, size=n_sims), 0, 75)

                sim_scores_b = np.clip(np.random.normal(loc=max(3, score_b_mean), scale=8.5, size=n_sims), 0, 75)



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



                sim_total_yds_a = np.clip(np.random.normal(loc=350 + (sim_scores_a * 4.0), scale=40, size=n_sims), 150, 700)

                sim_total_yds_b = np.clip(np.random.normal(loc=350 + (sim_scores_b * 4.0), scale=40, size=n_sims), 150, 700)

                

                sim_pass_yds_a = sim_total_yds_a * 0.62

                sim_rush_yds_a = sim_total_yds_a - sim_pass_yds_a

                sim_pass_yds_b = sim_total_yds_b * 0.62

                sim_rush_yds_b = sim_total_yds_b - sim_pass_yds_b



                sim_pass_tds_a = np.clip(np.round(sim_scores_a * 0.5 / 7), 0, 5)

                sim_rush_tds_a = np.clip(np.round((sim_scores_a / 7) - sim_pass_tds_a), 0, 4)

                sim_pass_tds_b = np.clip(np.round(sim_scores_b * 0.5 / 7), 0, 5)

                sim_rush_tds_b = np.clip(np.round((sim_scores_b / 7) - sim_pass_tds_b), 0, 4)



                res_col1, res_col2 = st.columns(2)

                with res_col1:

                    st.metric(label=f"{team_a} Win Probability", value=f"{win_prob_a*100:.1f}%")

                with res_col2:

                    st.metric(label=f"{team_b} Win Probability", value=f"{win_prob_b*100:.1f}%")



                spread_text = f"{favored_team} -{abs(mean_margin):.1f}" if abs(mean_margin) > 0.5 else "Pick 'em"

                st.markdown(

                    f"""

                    <div style="background-color: #161b22; border: 2px solid #30363d; border-radius: 12px; padding: 20px; text-align: center; margin-top: 15px; margin-bottom: 15px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);">

                        <div style="font-size: 1.1rem; font-weight: 700; color: #8b949e; margin-bottom: 5px;">SCOREBOARD PREDICTION (POWER-ANCHORED MODEL)</div>

                        <div style="font-size: 2.4rem; font-weight: 900; color: #58a6ff; letter-spacing: 1px;">{team_a} {mean_score_a:.1f} — {team_b} {mean_score_b:.1f}</div>

                        <div style="font-size: 1.2rem; font-weight: 600; color: #f0f6fc; margin-top: 8px;">Spread: <span style="color: #58a6ff;">{spread_text}</span> &nbsp;|&nbsp; Total Line: <span style="color: #58a6ff;">{total_baseline:.1f}</span></div>

                    </div>

                    """,

                    unsafe_allow_html=True,

                )



                st.divider()

                st.subheader("📈 Projected Point Margin Distribution")

                st.caption(

                    f"Shows how often each team wins by various margins across 10,000 simulations."

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

                st.subheader("🏈 Simulated Average Game Box Score")



                box_col1, box_col2 = st.columns(2)

                with box_col1:

                    st.markdown(

                        f"<div class='stat-header'>{team_a} (Home)</div>",

                        unsafe_allow_html=True,

                    )

                    st.markdown(f"<div class='stat-line'>Offense Rating: <span class='stat-val'>{p_a['offense']:.1f}</span></div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='stat-line'>Defense Rating: <span class='stat-val'>{p_a['defense']:.1f}</span></div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='stat-line'>Total Offense: <span class='stat-val'>{int(np.mean(sim_total_yds_a))} yds</span></div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='stat-line'>Passing Yards: <span class='stat-val'>{int(np.mean(sim_pass_yds_a))} yds</span></div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='stat-line'>Rushing Yards: <span class='stat-val'>{int(np.mean(sim_rush_yds_a))} yds</span></div>", unsafe_allow_html=True)



                with box_col2:

                    st.markdown(

                        f"<div class='stat-header'>{team_b} (Away)</div>",

                        unsafe_allow_html=True,

                    )

                    st.markdown(f"<div class='stat-line'>Offense Rating: <span class='stat-val'>{p_b['offense']:.1f}</span></div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='stat-line'>Defense Rating: <span class='stat-val'>{p_b['defense']:.1f}</span></div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='stat-line'>Total Offense: <span class='stat-val'>{int(np.mean(sim_total_yds_b))} yds</span></div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='stat-line'>Passing Yards: <span class='stat-val'>{int(np.mean(sim_pass_yds_b))} yds</span></div>", unsafe_allow_html=True)

                    st.markdown(f"<div class='stat-line'>Rushing Yards: <span class='stat-val'>{int(np.mean(sim_rush_yds_b))} yds</span></div>", unsafe_allow_html=True)



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

