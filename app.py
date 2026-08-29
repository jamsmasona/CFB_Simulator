st.divider()
            st.subheader("⚙️ Model Weighting & Season Progression")
            
            # Allow the user to input the current week to adjust stat importance
            current_week = st.slider("Current Season Week", min_value=1, max_value=15, value=6, 
                                     help="As the season progresses, actual game stats gain higher weight over preseason SP+ ratings.")
            
            # Alpha increases as the season goes on (e.g., Week 1 = 10% stats / 90% SP+, Week 15 = 80% stats / 20% SP+)
            stat_weight = min(0.85, 0.05 * current_week)
            sp_weight = 1.0 - stat_weight

            with st.spinner("Processing 10,000 Monte Carlo game iterations with blended metrics..."):
                p_a = fetch_advanced_profile(team_a)
                p_b = fetch_advanced_profile(team_b)

            hfa = 0.0 if is_neutral else 2.5

            # Blending SP+ differential with simulated/actual stat differentials based on season week
            sp_diff = p_a["sp"] - p_b["sp"]
            
            # Simulated stat differential placeholder (can be swapped with real box score averages once games are played)
            stat_diff = (p_a["sp"] * 1.1) - (p_b["sp"] * 0.9) 
            
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
            underdog_team = team_b if mean_margin >= 0 else team_a

            history_entry = {
                "Matchup": f"{team_a} vs {team_b}",
                "Projected Margin": f"{abs(mean_margin):.1f} pts ({favored_team})",
                "Win Probability": f"{win_prob_a*100:.1f}% ({team_a})",
                "Venue": "Neutral" if is_neutral else f"Home ({team_a})",
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
