#!/usr/bin/env python3
"""
Enhanced WR/TE Predictor with Minimax Theory and Markov Chains
This script applies advanced mathematical models as a post-processing layer
on top of the base WR/TE predictions from predict_player_fixed.py
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

def load_wr_te_data(artifacts_dir: Path) -> pd.DataFrame:
    """Load the training table from the fixed WR/TE model"""
    training_table_path = artifacts_dir / "wrte_fixed_training_table.parquet"
    if not training_table_path.exists():
        raise FileNotFoundError(f"Training table not found at {training_table_path}")
    
    return pd.read_parquet(training_table_path)

def get_base_prediction(player_name: str, artifacts_dir: Path) -> Tuple[float, Dict]:
    """
    Get the base prediction from predict_player_fixed.py
    Returns the point estimate and additional info
    """
    # Import the predict_player_fixed module
    import sys
    sys.path.append('.')
    from predict_player_fixed import build_latest_features, soft_participation_factor
    
    # Load model artifacts
    bundle = joblib.load(artifacts_dir / "wrte_fixed_model.joblib")
    resid = joblib.load(artifacts_dir / "wrte_fixed_residual_std.joblib")
    
    model = bundle["model"]
    X_cols = bundle["X_cols"]
    base_metrics = bundle["feature_info"]["base_metrics"]
    
    # Load training table
    table = pd.read_parquet(artifacts_dir / "wrte_fixed_training_table.parquet")
    
    # Build features for the player
    X, order_cols, latest_week = build_latest_features(table, player_name, base_metrics)
    
    if X is None:
        return None, {"error": "Player not found or insufficient data"}
    
    # Align feature columns
    if order_cols != X_cols:
        row = pd.Series(X.flatten(), index=order_cols)
        X = row.reindex(X_cols).values.reshape(1, -1)
    
    # Get base prediction
    point_est = float(model.predict(X)[0])
    
    # Apply participation factor
    factor = soft_participation_factor(table, player_name, lag_weeks=5, base_weight=0.7)
    point_est_adj = point_est * factor
    
    return point_est_adj, {
        "base_prediction": point_est,
        "participation_factor": factor,
        "latest_week": latest_week,
        "residual_std": float(resid["residual_std"])
    }

def calculate_minimax_adjustment(player_data: pd.DataFrame, base_prediction: float) -> float:
    """
    Apply Minimax theory to minimize maximum regret
    Based on historical PPR performance scenarios
    """
    if len(player_data) < 3:
        return 0.0
    
    # Define performance scenarios based on historical data
    historical_ppr = player_data['PPR'].dropna()
    if len(historical_ppr) < 2:
        return 0.0
    
    # Calculate scenario-based regret
    worst_case = historical_ppr.quantile(0.1)  # 10th percentile
    median_case = historical_ppr.median()
    mean_case = historical_ppr.mean()
    best_case = historical_ppr.quantile(0.9)   # 90th percentile
    
    # Calculate regret for each scenario
    scenarios = [worst_case, median_case, mean_case, best_case]
    regrets = []
    
    for scenario in scenarios:
        # Regret is the absolute difference between prediction and scenario
        regret = abs(base_prediction - scenario)
        regrets.append(regret)
    
    # Minimax: minimize the maximum regret
    max_regret = max(regrets)
    
    # Adjustment to minimize maximum regret
    # If prediction is too high compared to worst case, reduce it
    # If prediction is too low compared to best case, increase it
    if base_prediction > mean_case:
        # Prediction is optimistic, apply conservative adjustment
        adjustment = -0.1 * (base_prediction - mean_case)
    else:
        # Prediction is conservative, apply optimistic adjustment
        adjustment = 0.05 * (mean_case - base_prediction)
    
    return np.clip(adjustment, -2.0, 2.0)

def markov_chain_prediction(player_data: pd.DataFrame) -> float:
    """
    Use Markov chains to predict next state based on recent performance transitions
    """
    if len(player_data) < 5:
        return 0.0
    
    # Get historical PPR data
    historical_ppr = player_data['PPR'].dropna().values
    if len(historical_ppr) < 5:
        return 0.0
    
    # Discretize PPR into states (quintiles)
    try:
        # Handle cases with insufficient unique values
        unique_values = len(np.unique(historical_ppr))
        if unique_values < 3:
            # Not enough variation, return simple trend (reduced impact)
            recent_avg = np.mean(historical_ppr[-3:])
            overall_avg = np.mean(historical_ppr)
            raw_adjustment = recent_avg - overall_avg
            return raw_adjustment * 0.3  # Reduce impact to 30% of original
        
        # Use fewer bins if we don't have enough unique values
        n_bins = min(5, unique_values)
        states = pd.cut(historical_ppr, bins=n_bins, labels=False, duplicates='drop')
        
        # Fill any NaN values that might occur
        states = states.fillna(0).astype(int)
        
    except Exception:
        # Fallback if discretization fails (reduced impact)
        recent_avg = np.mean(historical_ppr[-3:])
        overall_avg = np.mean(historical_ppr)
        raw_adjustment = recent_avg - overall_avg
        return raw_adjustment * 0.3  # Reduce impact to 30% of original
    
    # Build transition matrix
    n_states = len(np.unique(states))
    if n_states < 2:
        # Fallback if we still don't have enough states (reduced impact)
        recent_avg = np.mean(historical_ppr[-3:])
        overall_avg = np.mean(historical_ppr)
        raw_adjustment = recent_avg - overall_avg
        return raw_adjustment * 0.3  # Reduce impact to 30% of original
        
    transition_matrix = np.zeros((n_states, n_states))
    
    for i in range(len(states) - 1):
        current_state = int(states[i])
        next_state = int(states[i + 1])
        # Ensure indices are within bounds
        if current_state < n_states and next_state < n_states:
            transition_matrix[current_state, next_state] += 1
    
    # Normalize transition matrix
    for i in range(n_states):
        row_sum = transition_matrix[i].sum()
        if row_sum > 0:
            transition_matrix[i] = transition_matrix[i] / row_sum
    
    # Get current state (most recent performance)
    current_state = int(states[-1])
    
    # Predict next state probabilities
    if current_state < n_states and current_state >= 0:
        next_state_probs = transition_matrix[current_state]
        
        # Calculate expected PPR based on state probabilities
        state_centers = []
        for state in range(n_states):
            state_data = historical_ppr[states == state]
            if len(state_data) > 0:
                state_centers.append(np.mean(state_data))
            else:
                state_centers.append(np.mean(historical_ppr))
        
        expected_ppr = np.dot(next_state_probs, state_centers)
        current_ppr = historical_ppr[-1]
        
        # Return adjustment based on Markov prediction (reduced impact)
        raw_adjustment = expected_ppr - current_ppr
        # Scale down Markov chain impact to be more subtle
        return raw_adjustment * 0.3  # Reduce impact to 30% of original
    
    # Fallback (reduced impact)
    recent_avg = np.mean(historical_ppr[-3:])
    overall_avg = np.mean(historical_ppr)
    raw_adjustment = recent_avg - overall_avg
    return raw_adjustment * 0.3  # Reduce impact to 30% of original

def calculate_performance_penalty(player_data: pd.DataFrame) -> float:
    """
    Calculate penalties based on recent performance patterns
    """
    if len(player_data) < 3:
        return 0.0
    
    # Recent performance analysis
    recent_ppr = player_data['PPR'].dropna().tail(5)
    if len(recent_ppr) < 3:
        return 0.0
    
    penalty = 0.0
    
    # Penalty for consistently poor performance
    poor_performance_threshold = 8.0
    poor_performance_rate = (recent_ppr < poor_performance_threshold).mean()
    if poor_performance_rate > 0.6:  # More than 60% of recent games under 8 PPR
        penalty -= 1.5 * poor_performance_rate
    
    # Penalty for high variance (inconsistency)
    if len(recent_ppr) >= 4:
        ppr_std = recent_ppr.std()
        if ppr_std > 8.0:  # High variance
            penalty -= 0.5
    
    # Penalty for declining trend
    if len(recent_ppr) >= 4:
        recent_trend = recent_ppr.tail(3).mean() - recent_ppr.head(2).mean()
        if recent_trend < -2.0:  # Declining by more than 2 PPR
            penalty -= 1.0
    
    # Reward for consistently good performance
    excellent_performance_threshold = 15.0
    excellent_performance_rate = (recent_ppr > excellent_performance_threshold).mean()
    if excellent_performance_rate > 0.6:  # More than 60% of recent games above 15 PPR
        penalty += 1.0 * excellent_performance_rate
    
    # Reward for improving trend
    if len(recent_ppr) >= 4:
        recent_trend = recent_ppr.tail(3).mean() - recent_ppr.head(2).mean()
        if recent_trend > 2.0:  # Improving by more than 2 PPR
            penalty += 1.0
    
    return np.clip(penalty, -3.0, 3.0)

def calculate_volume_consistency_factor(player_data: pd.DataFrame) -> float:
    """
    Calculate adjustment based on target volume consistency
    """
    if len(player_data) < 3:
        return 0.0
    
    # Look at recent target volume
    recent_targets = player_data['TAR'].dropna().tail(5)
    if len(recent_targets) < 3:
        return 0.0
    
    # Reward high and consistent target volume
    avg_targets = recent_targets.mean()
    target_consistency = 1.0 - (recent_targets.std() / (avg_targets + 1.0))
    
    if avg_targets >= 8.0 and target_consistency > 0.7:  # High and consistent targets
        return 1.5
    elif avg_targets >= 6.0 and target_consistency > 0.6:  # Medium-high and consistent
        return 1.0
    elif avg_targets < 4.0:  # Low targets
        return -1.0
    
    return 0.0

def enhanced_prediction(player_name: str, base_prediction: float, artifacts_dir: Path) -> Dict:
    """
    Generate enhanced prediction using minimax theory and Markov chains
    """
    try:
        # Load player data
        table = load_wr_te_data(artifacts_dir)
        player_data = table[table["PLAYER"].str.lower() == player_name.lower()].sort_values("WEEK")
        
        if player_data.empty:
            return {
                "player": player_name,
                "base_prediction": base_prediction,
                "enhanced_prediction": base_prediction,
                "adjustments": {
                    "minimax": 0.0,
                    "markov": 0.0,
                    "performance_penalty": 0.0,
                    "volume_consistency": 0.0
                },
                "error": "Player not found in training data"
            }
        
        # Calculate adjustments
        minimax_adj = calculate_minimax_adjustment(player_data, base_prediction)
        markov_adj = markov_chain_prediction(player_data)
        performance_penalty = calculate_performance_penalty(player_data)
        volume_factor = calculate_volume_consistency_factor(player_data)
        
        # Combine adjustments
        total_adjustment = minimax_adj + markov_adj + performance_penalty + volume_factor
        
        # Apply adjustment to base prediction
        enhanced_pred = base_prediction + total_adjustment
        
        # Ensure reasonable bounds
        enhanced_pred = np.clip(enhanced_pred, 0.0, 40.0)
        
        return {
            "player": player_name,
            "base_prediction": base_prediction,
            "enhanced_prediction": enhanced_pred,
            "adjustments": {
                "minimax": minimax_adj,
                "markov": markov_adj,
                "performance_penalty": performance_penalty,
                "volume_consistency": volume_factor
            },
            "total_adjustment": total_adjustment,
            "latest_week": player_data['WEEK'].iloc[-1] if len(player_data) > 0 else None
        }
        
    except Exception as e:
        return {
            "player": player_name,
            "base_prediction": base_prediction,
            "enhanced_prediction": base_prediction,
            "adjustments": {
                "minimax": 0.0,
                "markov": 0.0,
                "performance_penalty": 0.0,
                "volume_consistency": 0.0
            },
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Enhanced WR/TE Predictor with Minimax Theory and Markov Chains")
    parser.add_argument("--artifacts", default="./artifacts_fixed", help="Path to model artifacts directory")
    parser.add_argument("--player", required=True, help="Player name to predict")
    parser.add_argument("--show_details", action="store_true", help="Show detailed breakdown")
    
    args = parser.parse_args()
    
    artifacts_dir = Path(args.artifacts)
    
    # Get base prediction
    base_pred, base_info = get_base_prediction(args.player, artifacts_dir)
    
    if base_pred is None:
        print(f"Error: {base_info.get('error', 'Unknown error')}")
        return
    
    # Get enhanced prediction
    result = enhanced_prediction(args.player, base_pred, artifacts_dir)
    
    # Display results
    print(f"\n=== Enhanced WR/TE Prediction for {args.player} ===")
    print(f"Base Prediction: {result['base_prediction']:.2f} PPR")
    print(f"Enhanced Prediction: {result['enhanced_prediction']:.2f} PPR")
    print(f"Total Adjustment: {result['total_adjustment']:+.2f} PPR")
    
    if args.show_details:
        print(f"\nDetailed Adjustments:")
        for adjustment_type, value in result['adjustments'].items():
            print(f"  {adjustment_type.replace('_', ' ').title()}: {value:+.2f}")
        
        if 'latest_week' in result and result['latest_week']:
            print(f"\nLatest Week in Data: {result['latest_week']}")
        
        if 'error' in result:
            print(f"\nWarning: {result['error']}")

if __name__ == "__main__":
    main()
