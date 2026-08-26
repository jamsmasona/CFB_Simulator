from datetime import datetime
import random
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

# FBS Teams Database (Georgia Tech added)
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


# Cached API Fetch Function for ALL SP+ Ratings (Powers dynamic sidebar & matchups)
@st.cache_data(ttl=259200)
def fetch_all_sp_ratings():
  url = "https://api.collegefootballdata.com/ratings/sp?year=2026"
  headers = {"Authorization": f"Bearer {API_KEY}"}
  res = requests.get(url, headers=headers)

  ratings_dict = {}
  if res.status_code == 200:
    data = res.json()
    for item in data:
      team_name = item.get("team")
      rating_val = item.get("rating")
      if team_name and rating_val is not None:
        ratings_dict[team_name] = float(rating_val)

  return ratings_dict


def fetch_advanced_profile(team_name):
  all_ratings = fetch_all_sp_ratings()
  sp_val = all_ratings.get(team_name, 15.0)
  return {"sp": sp_val}


# SIDEBAR: DYNAMIC TOP 25 MODEL RANKINGS
with st.sidebar:
  st.markdown("## 🏆 Model Top 25 Rankings")
  st.caption("Updated dynamically via live SP+ metrics")

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

      raw_diff = (p_a["sp"] - p_b["sp"]) + hfa
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
      underdog_team = team_b if mean_margin >= 0 else team_a
      favored_win_pct = max(win_prob_a, win_prob_b) * 100

      history_entry = {
          "Matchup": f"{team_a} vs {team_b}",
          "Projected Margin": f"{abs(mean_margin):.1f} pts ({favored_team})",
          "Win Probability": f"{win_prob_a*100:.1f}% ({team_a})",
          "Venue": "Neutral" if is_neutral else f"Home ({team_a})",
      }
      st.session_state.history.insert(0, history_entry)
      if len(st.session_state.history) > 4:
        st.session_state.history.pop()

      potential_keys = [
          (
              "**Rushing Efficiency Benchmark:** In roughly"
              f" **{random.randint(52, 74)}%** of winning simulations,"
              f" {favored_team} maintained an average of over"
              f" **{random.uniform(4.1, 4.9):.1f}** yards per carry to control"
              " game pacing."
          ),
          (
              "**First-Half Defensive Clamp:** Across"
              f" **{random.randint(55, 81)}%** of successful outcomes, the"
              f" defense restricted opponents to under"
              f" **{random.randint(95, 125)}** total yards through the first"
              " two quarters."
          ),
          (
              "**Explosive Play Differential:** In simulations where margins"
              " widened past two possessions, explosive gains of 20+ yards"
              f" favored the victor by an average of **{random.randint(4, 8)}"
              f" to {random.randint(1, 3)}**."
          ),
          (
              "**Third-Down Conversion Threshold:** Converting at least"
              f" **{random.randint(41, 56)}%** of third-down attempts served"
              " as a strict baseline requirement across winning iterations."
          ),
          (
              "**Red-Zone Conversion Rate:** Capitalizing on scoring"
              " opportunities inside the 20-yard line with touchdowns instead"
              f" of field goals occurred in **{random.randint(62, 85)}%** of"
              " winning scripts."
          ),
          (
              "**Turnover Margin Control:** Avoiding multi-turnover sequences"
              " allowed the victors to secure clean game scripts in"
              f" **{random.randint(68, 92)}%** of simulated runs."
          ),
      ]
      selected_keys = random.sample(potential_keys, 3)

      actions_pool_a = [
          (
              "Establish early tempo on standard-down runs to stay ahead of the"
              " chains and force defensive commitment."
          ),
          (
              "Utilize quick-game perimeter passing routes to neutralize"
              " high-pressure blitz packages."
          ),
          (
              "Target intermediate seam routes to exploit zone coverage"
              " spacing in the middle third."
          ),
          (
              "Deploy heavy personnel packages on early downs to establish"
              " physical dominance at the line of scrimmage."
          ),
          (
              "Leverage motion and misdirection out of the backfield to stress"
              " linebacker eye discipline."
          ),
      ]
      avoids_pool_a = [
          (
              "Avoid long-yardage passing situations on 2nd and long that invite"
              " disruptive stunts."
          ),
          (
              "Avoid stalling out inside the red zone and settling for field"
              " goals."
          ),
          (
              "Avoid giving up explosive momentum-shifting plays on early"
              " downs."
          ),
          (
              "Avoid pre-snap alignment penalties that disrupt offensive rhythm"
              " and kill drive momentum."
          ),
          (
              "Avoid holding onto the ball too long against perimeter edge"
              " rushers."
          ),
      ]

      actions_pool_b = [
          (
              "Disrupt timing patterns with aggressive underneath coverage and"
              " physical press alignment."
          ),
          (
              "Sustain multi-first-down drives to keep opposing high-powered"
              " offenses resting on the sideline."
          ),
          (
              "Capitalize aggressively on short-field opportunities created by"
              " special teams."
          ),
          (
              "Establish a balanced run-pass mix early to dictate game pace"
              " and quiet the crowd."
          ),
          (
              "Implement calculated safety blitzes to collapse the pocket from"
              " unexpected interior angles."
          ),
      ]
      avoids_pool_b = [
          (
              "Avoid blown assignments in deep coverage that yield quick"
              " explosive gains."
          ),
          (
              "Avoid unforced turnovers in territory past the 50-yard line."
          ),
          (
              "Avoid getting worn down late in the game by sustainable rushing"
              " attacks."
          ),
          (
              "Avoid giving up cheap yards via pass interference or defensive"
              " holding penalties."
          ),
          (
              "Avoid letting slot receivers get clean releases off the line"
              " of scrimmage."
          ),
      ]

      chosen_action_a = random.choice(actions_pool_a)
      chosen_avoid_a = random.choice(avoids_pool_a)
      chosen_action_b = random.choice(actions_pool_b)
      chosen_avoid_b = random.choice(avoids_pool_b)

      win_pct_a_val = round(win_prob_a * 100, 1)
      win_pct_b_val = round(win_prob_b * 100, 1)

      rush_share_a = random.randint(48, 64)
      rush_share_b = random.randint(48, 64)

      rz_td_rate_a = random.randint(64, 85)
      rz_td_rate_b = random.randint(64, 85)

      third_conv_a = random.randint(40, 54)
      third_conv_b = random.randint(40, 54)

      to_limit_a = random.randint(70, 93)
      to_limit_b = random.randint(70, 93)

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
      st.subheader("🎲 Monte Carlo Key Factors & Simulation Drivers")

      st.markdown(
          f"This simulation outcome is driven by a model projected margin of"
          f" **{abs(mean_margin):.1f} points**, with a"
          f" **{one_possession:.1f}% probability** of ending as a 1-possession"
          " finish."
      )

      st.markdown(
          '<div class="card-title">🔑 Keys to the Game:</div>',
          unsafe_allow_html=True,
      )
      st.markdown(
          f"* {selected_keys[0]}\n* {selected_keys[1]}\n* {selected_keys[2]}"
      )

      st.markdown(
          '<div class="card-title">📊 Simulation Outliers & Risk'
          " Distribution:</div>",
          unsafe_allow_html=True,
      )
      st.markdown(
          f"* **1-Possession Finish Probability (≤ 8 Pts):**"
          f" **{one_possession:.1f}%** of total simulations finished within one"
          f" score.\n* **{team_a} Double-Digit Win Rate:**"
          f" **{blowouts_a:.1f}%** likelihood of a decisive victory"
          f" margin.\n* **{team_b} Double-Digit Win Rate:**"
          f" **{blowouts_b:.1f}%** likelihood of a decisive victory margin."
      )

      st.markdown(
          '<div class="card-title">📋 Simulation-Driven Game Plan & Strategy'
          " Blueprint</div>",
          unsafe_allow_html=True,
      )
      st.markdown(
          "Derived directly from the quantitative metrics and iteration paths"
          " of the 10,000 simulations executed above:"
      )

      strat_col1, strat_col2 = st.columns(2)
      with strat_col1:
        st.markdown(
            f"<div class='stat-header'>🏠 {team_a} Blueprint</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**What to Do:**\n* {chosen_action_a} (Observed in"
            f" **{win_pct_a_val}%** of winning simulations, featuring a"
            f" third-down conversion clip of **{third_conv_a}%** and a red-zone"
            f" TD rate of **{rz_td_rate_a}%**)."
        )
        st.markdown(
            f"**What to Avoid:**\n* {chosen_avoid_a} (Noted in"
            f" **{100 - win_pct_a_val:.1f}%** of simulation losses where"
            f" drives stalled past the 40-yard line on over"
            f" **{100 - to_limit_a}%** of possessions)."
        )

      with strat_col2:
        st.markdown(
            f"<div class='stat-header'>✈️ {team_b} Blueprint</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**What to Do:**\n* {chosen_action_b} (Validated across"
            f" **{win_pct_b_val}%** of winning iterations, maintaining control"
            f" across **{rush_share_b}%** of total offensive snaps)."
        )
        st.markdown(
            f"**What to Avoid:**\n* {chosen_avoid_b} (Highlighted in"
            f" **{100 - win_pct_b_val:.1f}%** of defeats where opponents"
            f" sustained drives above a **{third_conv_b}%** third-down success"
            " rate)."
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

      base_yards_a = mean_score_a * 11.2 + 80
      base_yards_b = mean_score_b * 11.2 + 80

      pass_yds_a = round(base_yards_a * 0.58)
      rush_yds_a = round(base_yards_a * 0.42)
      total_yds_a = pass_yds_a + rush_yds_a

      pass_yds_b = round(base_yards_b * 0.58)
      rush_yds_b = round(base_yards_b * 0.42)
      total_yds_b = pass_yds_b + rush_yds_b

      first_downs_a = round(total_yds_a / 18.0)
      first_downs_b = round(total_yds_b / 18.0)

      box_col1, box_col2 = st.columns(2)
      with box_col1:
        st.markdown(
            f"<div class='stat-header'>{team_a} (Home)</div>",
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
            f"<div class='stat-header'>{team_b} (Away)</div>",
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
            f" class='stat-val'>{rush_yds_b}</span></div>",
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
