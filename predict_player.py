# predict_player.py
# Usage:
#   python predict_player.py --artifacts ./artifacts --player "A.J. Brown" --sim 5000
#
# Loads the trained WR/TE model and computes a prediction for the next week
# using the most recent lag/rolling features available for that player.
# Also runs a Monte Carlo simulation using the global residual std.

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def build_latest_features(training_table: pd.DataFrame, player: str, base_metrics, min_games: int = 3):
    tt = training_table[training_table["PLAYER"].str.lower() == player.lower()].copy()
    if tt.empty:
        raise ValueError(f"No rows found for player '{player}' in training table.")
    feat_cols = [f"{c}_lag1" for c in base_metrics] + [f"{c}_roll3" for c in base_metrics]
    tt = tt.sort_values("WEEK")
    tt = tt.dropna(subset=feat_cols)
    if tt.empty:
        raise ValueError(f"Not enough history to build features for '{player}'. Need at least {min_games} prior games.")
    latest = tt.iloc[-1]
    X = latest[feat_cols].values.reshape(1, -1)
    return X, feat_cols, int(latest["WEEK"])

def soft_participation_factor(table, player, lag_weeks=3, base_weight=0.7):
    player_df = table[table["PLAYER"].str.lower() == player.lower()].sort_values("WEEK")
    if player_df.empty:
        return base_weight
    recent_weeks = player_df.tail(lag_weeks)
    participation = recent_weeks.shape[0] / lag_weeks
    return base_weight + (1 - base_weight) * participation

def main(args):
    artdir = Path(args.artifacts)
    bundle = joblib.load(artdir / "wrte_model.joblib")
    resid  = joblib.load(artdir / "wrte_residual_std.joblib")

    model = bundle["model"]
    X_cols = bundle["X_cols"]
    base   = bundle["feature_info"]["base_metrics"]

    table = pd.read_parquet(artdir / "wrte_training_table.parquet")

    X, order_cols, latest_week = build_latest_features(table, args.player, base_metrics=base)

    if order_cols != X_cols:
        row = pd.Series(X.flatten(), index=order_cols)
        X = row.reindex(X_cols).values.reshape(1, -1)

    point_est = float(model.predict(X)[0])

    # Soft participation adjustment
    factor = soft_participation_factor(table, args.player, lag_weeks=3, base_weight=0.7)
    point_est_adj = point_est * factor

    # Monte Carlo simulation
    resid_std = float(resid["residual_std"])
    sims = np.random.normal(loc=point_est_adj, scale=resid_std, size=args.sim)
    p50 = float(np.percentile(sims, 50))
    p10 = float(np.percentile(sims, 10))
    p90 = float(np.percentile(sims, 90))

    thresholds = [8, 12, 15, 20]
    probs = {t: float((sims >= t).mean()) for t in thresholds}

    print(f"Player: {args.player}")
    print(f"Latest week in data: {latest_week}")
    print(f"Raw point estimate: {point_est:.2f}")
    print(f"Adjusted point estimate (soft participation): {point_est_adj:.2f}")
    print(f"Monte Carlo percentiles (PPR): P10={p10:.2f}, P50={p50:.2f}, P90={p90:.2f}")
    print("Probabilities of exceeding thresholds:")
    for t in thresholds:
        print(f"  P(PPR ≥ {t}) = {probs[t]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="./artifacts", help="Folder with saved model + table")
    parser.add_argument("--player", required=True, help="Player name (e.g., 'A.J. Brown')")
    parser.add_argument("--sim", type=int, default=5000, help="Monte Carlo simulation count")
    args = parser.parse_args()
    main(args)
