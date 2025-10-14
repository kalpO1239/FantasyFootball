# predict_player_fixed.py
# Prediction script for fixed WR/TE model
# Usage:
#   python predict_player_fixed.py --artifacts ./artifacts_fixed --player "A.J. Brown" --sim 5000

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def build_latest_features(training_table: pd.DataFrame, player: str, base_metrics, min_games: int = 3):
    tt = training_table[training_table["PLAYER"].str.lower() == player.lower()].copy()
    if tt.empty:
        return None, None, None
    
    # Build comprehensive feature columns for fixed model
    feat_cols = ([f"{c}_lag1" for c in base_metrics] +  # Keep lag1 but will weight less
                 [f"{c}_roll3" for c in base_metrics] + 
                 [f"{c}_roll5" for c in base_metrics] + 
                 [f"{c}_roll8" for c in base_metrics] + 
                 [f"{c}_momentum_short" for c in base_metrics] + 
                 [f"{c}_momentum_medium" for c in base_metrics] + 
                 ["TAR_trend_recent", "REC_trend_recent", "YDS_trend_recent", "TD_trend_recent",
                  "TD_per_TAR_recent", "CTCH_rate_recent", "YPT_recent"] +
                 [f"{c}_consistency_recent" for c in ["TAR", "REC", "YDS", "TD"]] +
                 ["week_number", "is_recent_season", "very_recent"] +
                 [f"{c}_recent_season" for c in ["TAR", "REC", "YDS", "TD"]] +
                 [f"{c}_recent_surge" for c in ["TAR", "REC", "YDS", "TD"]] +
                 [f"{c}_surge_vs_recent" for c in ["TAR", "REC", "YDS", "TD"]] +
                 ["high_volume_tar_recent", "high_volume_rec_recent", "high_volume_yds_recent", "high_scorer_recent", 
                  "elite_targets", "elite_yards", "elite_performer_recent", "consistent_performer_recent",
                  "strong_momentum_tar", "strong_momentum_yds", "strong_momentum_td",
                  "positive_trend_tar", "positive_trend_yds", "positive_trend_td",
                  "is_wr", "is_te"])
    
    tt = tt.sort_values("WEEK")
    tt = tt.dropna(subset=feat_cols)
    if tt.empty:
        return None, None, None
    latest = tt.iloc[-1]
    X = latest[feat_cols].values.reshape(1, -1)
    return X, feat_cols, int(latest["WEEK"])

def soft_participation_factor(table, player, lag_weeks=5, base_weight=0.7):
    """
    Adjust prediction based on how often player appears in the data.
    Players who consistently appear get a boost.
    Players who only occasionally appear get penalized.
    """
    player_df = table[table["PLAYER"].str.lower() == player.lower()].sort_values("WEEK")
    if player_df.empty:
        return base_weight
    
    # Get the latest week in the data
    max_week = table["WEEK"].max()
    
    # Look at the last N weeks chronologically
    recent_weeks_data = player_df[player_df["WEEK"] >= (max_week - lag_weeks + 1)]
    participation = recent_weeks_data.shape[0] / lag_weeks
    
    return base_weight + (1 - base_weight) * participation

def main(args):
    artdir = Path(args.artifacts)
    bundle = joblib.load(artdir / "wrte_fixed_model.joblib")
    resid = joblib.load(artdir / "wrte_fixed_residual_std.joblib")

    model = bundle["model"]
    X_cols = bundle["X_cols"]
    base = bundle["feature_info"]["base_metrics"]

    table = pd.read_parquet(artdir / "wrte_fixed_training_table.parquet")

    X, order_cols, latest_week = build_latest_features(table, args.player, base_metrics=base)

    if order_cols != X_cols:
        row = pd.Series(X.flatten(), index=order_cols)
        X = row.reindex(X_cols).values.reshape(1, -1)

    # Get prediction from fixed model
    point_est = float(model.predict(X)[0])
    
    # Apply soft participation factor
    factor = soft_participation_factor(table, args.player, lag_weeks=5, base_weight=0.7)
    point_est_adj = point_est * factor

    # Monte Carlo simulation
    resid_std = float(resid["residual_std"])
    sims = np.random.normal(loc=point_est_adj, scale=resid_std, size=args.sim)
    p50 = float(np.percentile(sims, 50))
    p10 = float(np.percentile(sims, 10))
    p90 = float(np.percentile(sims, 90))

    # WR/TE-specific thresholds
    thresholds = [8, 12, 15, 20, 25, 30]
    probs = {t: float((sims >= t).mean()) for t in thresholds}

    print(f"\nPlayer: {args.player}")
    print(f"Latest week in data: {latest_week}")
    print(f"Fixed model prediction: {point_est:.2f}")
    print(f"Participation factor: {factor:.3f}")
    print(f"Adjusted prediction: {point_est_adj:.2f}")
    print(f"Monte Carlo percentiles (PPR): P10={p10:.2f}, P50={p50:.2f}, P90={p90:.2f}")
    print("Probabilities of exceeding thresholds:")
    for t in thresholds:
        print(f"  P(PPR ≥ {t}) = {probs[t]:.3f}")
    
    # Show feature breakdown with emphasis on fixed features
    print("\nFeature breakdown (latest values):")
    player_data = table[table["PLAYER"].str.lower() == args.player.lower()].sort_values("WEEK").iloc[-1]
    
    print("Recent season performance (2024 emphasis):")
    for metric in ["TAR", "REC", "YDS", "TD"]:
        recent_season = player_data.get(f"{metric}_recent_season", np.nan)
        recent_surge = player_data.get(f"{metric}_recent_surge", np.nan)
        surge_vs_recent = player_data.get(f"{metric}_surge_vs_recent", np.nan)
        if pd.notna(recent_season):
            print(f"  {metric}_recent_season: {recent_season:.2f}")
        if pd.notna(recent_surge):
            print(f"  {metric}_recent_surge: {recent_surge:.2f}")
        if pd.notna(surge_vs_recent):
            print(f"  {metric}_surge_vs_recent: {surge_vs_recent:.2f}")
    
    print("\nRecent rolling averages (emphasized):")
    for metric in ["TAR", "REC", "YDS", "TD"]:
        roll3 = player_data.get(f"{metric}_roll3", np.nan)
        roll5 = player_data.get(f"{metric}_roll5", np.nan)
        roll8 = player_data.get(f"{metric}_roll8", np.nan)
        if pd.notna(roll3):
            print(f"  {metric}_roll3: {roll3:.2f}")
        if pd.notna(roll5):
            print(f"  {metric}_roll5: {roll5:.2f}")
        if pd.notna(roll8):
            print(f"  {metric}_roll8: {roll8:.2f}")
    
    print("\nRecent momentum indicators:")
    for metric in ["TAR", "REC", "YDS", "TD"]:
        momentum_short = player_data.get(f"{metric}_momentum_short", np.nan)
        momentum_medium = player_data.get(f"{metric}_momentum_medium", np.nan)
        if pd.notna(momentum_short):
            print(f"  {metric}_momentum_short: {momentum_short:.2f}")
        if pd.notna(momentum_medium):
            print(f"  {metric}_momentum_medium: {momentum_medium:.2f}")
    
    print("\nFIXED: Elite performer indicators (more nuanced):")
    elite_features = ["elite_targets", "elite_yards", "elite_performer_recent"]
    for feature in elite_features:
        val = player_data.get(feature, np.nan)
        if pd.notna(val):
            print(f"  {feature}: {val:.0f}")
    
    print("\nNEW: Strong momentum indicators:")
    momentum_features = ["strong_momentum_tar", "strong_momentum_yds", "strong_momentum_td"]
    for feature in momentum_features:
        val = player_data.get(feature, np.nan)
        if pd.notna(val):
            print(f"  {feature}: {val:.0f}")
    
    print("\nNEW: Positive trend indicators:")
    trend_features = ["positive_trend_tar", "positive_trend_yds", "positive_trend_td"]
    for feature in trend_features:
        val = player_data.get(feature, np.nan)
        if pd.notna(val):
            print(f"  {feature}: {val:.0f}")
    
    print("\nRecent volume and performance indicators:")
    volume_features = ["high_volume_tar_recent", "high_volume_rec_recent", "high_volume_yds_recent", "high_scorer_recent", 
                      "consistent_performer_recent"]
    for feature in volume_features:
        val = player_data.get(feature, np.nan)
        if pd.notna(val):
            print(f"  {feature}: {val:.0f}")
    
    print("\nRecent efficiency metrics:")
    efficiency_features = ["TD_per_TAR_recent", "CTCH_rate_recent", "YPT_recent"]
    for feature in efficiency_features:
        val = player_data.get(feature, np.nan)
        if pd.notna(val):
            print(f"  {feature}: {val:.3f}")
    
    print("\nRecent trends:")
    trend_features = ["TAR_trend_recent", "REC_trend_recent", "YDS_trend_recent", "TD_trend_recent"]
    for feature in trend_features:
        val = player_data.get(feature, np.nan)
        if pd.notna(val):
            print(f"  {feature}: {val:.2f}")
    
    # Show lag1 features (reduced influence)
    print("\nLag1 features (reduced influence):")
    for metric in base:
        lag_val = player_data.get(f"{metric}_lag1", np.nan)
        if pd.notna(lag_val):
            print(f"  {metric}_lag1: {lag_val:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="./artifacts_fixed", help="Folder with saved model + table")
    parser.add_argument("--player", required=True, help="Player name (e.g., 'A.J. Brown')")
    parser.add_argument("--sim", type=int, default=5000, help="Monte Carlo simulation count")
    args = parser.parse_args()
    main(args)
