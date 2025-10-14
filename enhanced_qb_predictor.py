#!/usr/bin/env python3
"""
Enhanced QB Predictor with Minimax Theory and Markov Chains
This script applies advanced mathematical models as a post-processing layer
on top of the base QB predictions from the trained model
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

def load_qb_data(artifacts_dir: Path) -> pd.DataFrame:
    """Load the training table from the QB model"""
    training_table_path = artifacts_dir / "qb_training_table.parquet"
    if not training_table_path.exists():
        raise FileNotFoundError(f"Training table not found at {training_table_path}")
    
    return pd.read_parquet(training_table_path)

def get_base_prediction(player_name: str, artifacts_dir: Path) -> Tuple[float, Dict]:
    """
    Get the base prediction from the trained QB model
    Returns the point estimate and additional info
    """
    # Load model artifacts
    bundle = joblib.load(artifacts_dir / "qb_model.joblib")
    resid = joblib.load(artifacts_dir / "qb_residual_std.joblib")
    
    model = bundle["model"]
    X_cols = bundle["X_cols"]
    key_metrics = bundle["feature_info"]["key_metrics"]
    
    # Load training table
    table = pd.read_parquet(artifacts_dir / "qb_training_table.parquet")
    
    # Build features for the player
    player_data = table[table["PLAYER"].str.lower() == player_name.lower()].sort_values("GLOBAL_WEEK")
    
    if player_data.empty:
        return None, {"error": "Player not found or insufficient data"}
    
    # Get the latest week's features
    latest = player_data.iloc[-1]
    
    # Extract features in the same order as training
    X = latest[X_cols].values.reshape(1, -1)
    
    # Get base prediction
    point_est = float(model.predict(X)[0])
    
    return point_est, {
        "base_prediction": point_est,
        "latest_week": int(latest["GLOBAL_WEEK"]),
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
    historical_ppr = player_data['TARGET_PPR'].dropna()
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
    
    # Adjustment to minimize maximum regret (reduced impact)
    # If prediction is too high compared to worst case, reduce it
    # If prediction is too low compared to best case, increase it
    if base_prediction > mean_case:
        # Prediction is optimistic, apply conservative adjustment
        adjustment = -0.05 * (base_prediction - mean_case)  # Reduced from 0.15 to 0.05
    else:
        # Prediction is conservative, apply optimistic adjustment
        adjustment = 0.03 * (mean_case - base_prediction)  # Reduced from 0.08 to 0.03
    
    return np.clip(adjustment, -1.0, 1.0)  # Reduced bounds from 3.0 to 1.0

def markov_chain_prediction(player_data: pd.DataFrame) -> float:
    """
    Use Markov chains to predict next state based on recent performance transitions
    """
    if len(player_data) < 5:
        return 0.0
    
    # Get historical PPR data
    historical_ppr = player_data['TARGET_PPR'].dropna().values
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
            return raw_adjustment * 0.15  # Reduce impact to 15% of original for QBs
        
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
        return raw_adjustment * 0.15  # Reduce impact to 15% of original for QBs
    
    # Build transition matrix
    n_states = len(np.unique(states))
    if n_states < 2:
        # Fallback if we still don't have enough states (reduced impact)
        recent_avg = np.mean(historical_ppr[-3:])
        overall_avg = np.mean(historical_ppr)
        raw_adjustment = recent_avg - overall_avg
        return raw_adjustment * 0.15  # Reduce impact to 15% of original for QBs
        
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
        
        # Return adjustment based on Markov prediction (reduced impact for QBs)
        raw_adjustment = expected_ppr - current_ppr
        # Scale down Markov chain impact to be more subtle for QBs
        return raw_adjustment * 0.15  # Reduce impact to 15% of original for QBs
    
    # Fallback (reduced impact)
    recent_avg = np.mean(historical_ppr[-3:])
    overall_avg = np.mean(historical_ppr)
    raw_adjustment = recent_avg - overall_avg
    return raw_adjustment * 0.25  # Reduce impact to 25% of original for QBs

def calculate_performance_penalty(player_data: pd.DataFrame) -> float:
    """
    Calculate penalties based on recent performance patterns
    """
    if len(player_data) < 3:
        return 0.0
    
    # Recent performance analysis
    recent_ppr = player_data['TARGET_PPR'].dropna().tail(5)
    if len(recent_ppr) < 3:
        return 0.0
    
    penalty = 0.0
    
    # Penalty for consistently poor performance (reduced impact)
    poor_performance_threshold = 15.0  # QBs typically score higher
    poor_performance_rate = (recent_ppr < poor_performance_threshold).mean()
    if poor_performance_rate > 0.6:  # More than 60% of recent games under 15 PPR
        penalty -= 0.5 * poor_performance_rate  # Reduced from 2.0 to 0.5
    
    # Penalty for high variance (inconsistency) - reduced
    if len(recent_ppr) >= 4:
        ppr_std = recent_ppr.std()
        if ppr_std > 10.0:  # High variance for QBs
            penalty -= 0.2  # Reduced from 0.8 to 0.2
    
    # Penalty for declining trend - reduced
    if len(recent_ppr) >= 4:
        recent_trend = recent_ppr.tail(3).mean() - recent_ppr.head(2).mean()
        if recent_trend < -3.0:  # Declining by more than 3 PPR
            penalty -= 0.4  # Reduced from 1.5 to 0.4
    
    # Reward for consistently good performance - reduced
    excellent_performance_threshold = 25.0  # QBs typically score higher
    excellent_performance_rate = (recent_ppr > excellent_performance_threshold).mean()
    if excellent_performance_rate > 0.6:  # More than 60% of recent games above 25 PPR
        penalty += 0.4 * excellent_performance_rate  # Reduced from 1.5 to 0.4
    
    # Reward for improving trend - reduced
    if len(recent_ppr) >= 4:
        recent_trend = recent_ppr.tail(3).mean() - recent_ppr.head(2).mean()
        if recent_trend > 3.0:  # Improving by more than 3 PPR
            penalty += 0.4  # Reduced from 1.5 to 0.4
    
    return np.clip(penalty, -1.5, 1.5)  # Reduced bounds from 4.0 to 1.5

def calculate_passing_consistency_factor(player_data: pd.DataFrame) -> float:
    """
    Calculate adjustment based on passing volume consistency
    """
    if len(player_data) < 3:
        return 0.0
    
    # Look at recent passing attempts
    recent_attempts = player_data['ATT'].dropna().tail(5)
    if len(recent_attempts) < 3:
        return 0.0
    
    # Reward high and consistent passing volume
    avg_attempts = recent_attempts.mean()
    attempt_consistency = 1.0 - (recent_attempts.std() / (avg_attempts + 1.0))
    
    if avg_attempts >= 35.0 and attempt_consistency > 0.7:  # High and consistent attempts
        return 0.8  # Reduced from 2.0 to 0.8
    elif avg_attempts >= 30.0 and attempt_consistency > 0.6:  # Medium-high and consistent
        return 0.4  # Reduced from 1.0 to 0.4
    elif avg_attempts < 25.0:  # Low attempts
        return -0.6  # Reduced from -1.5 to -0.6
    
    return 0.0

def calculate_rushing_upside_factor(player_data: pd.DataFrame) -> float:
    """
    Calculate adjustment based on rushing upside consistency
    """
    if len(player_data) < 3:
        return 0.0
    
    # Look at recent rushing upside
    recent_rushing = player_data['RUSHING_UPSIDE'].dropna().tail(5)
    if len(recent_rushing) < 3:
        return 0.0
    
    # Reward consistent rushing upside
    avg_rushing = recent_rushing.mean()
    rushing_consistency = 1.0 - (recent_rushing.std() / (abs(avg_rushing) + 1.0))
    
    # Simply return the average rushing upside - this is what should be added to passing PPR
    return avg_rushing

def enhanced_prediction(player_name: str, base_prediction: float, artifacts_dir: Path) -> Dict:
    """
    Generate enhanced prediction using minimax theory and Markov chains
    """
    try:
        # Load player data
        table = load_qb_data(artifacts_dir)
        player_data = table[table["PLAYER"].str.lower() == player_name.lower()].sort_values("GLOBAL_WEEK")
        
        if player_data.empty:
            return {
                "player": player_name,
                "base_prediction": base_prediction,
                "enhanced_prediction": base_prediction,
                "adjustments": {
                    "minimax": 0.0,
                    "markov": 0.0,
                    "performance_penalty": 0.0,
                    "passing_consistency": 0.0,
                    "rushing_upside": 0.0
                },
                "error": "Player not found in training data"
            }
        
        # Calculate adjustments
        minimax_adj = calculate_minimax_adjustment(player_data, base_prediction)
        markov_adj = markov_chain_prediction(player_data)
        performance_penalty = calculate_performance_penalty(player_data)
        passing_factor = calculate_passing_consistency_factor(player_data)
        # Note: Rushing factor removed - base model already includes total PPR (passing + rushing)
        
        # Combine adjustments (no rushing factor since base model already includes it)
        total_adjustment = minimax_adj + markov_adj + performance_penalty + passing_factor
        
        # Apply adjustment to base prediction
        enhanced_pred = base_prediction + total_adjustment
        
        # Apply mobile QB correction factor for severely underpredicted players
        # Based on actual vs predicted performance analysis
        recent_ppr = player_data['PPR'].tail(5).mean() if len(player_data) >= 5 else player_data['PPR'].mean()
        
        # If actual performance is significantly higher than base prediction, apply correction
        if recent_ppr > base_prediction * 1.5:  # Actual is 50%+ higher than predicted
            correction_factor = min(1.8, recent_ppr / base_prediction)  # Cap at 1.8x
            enhanced_pred = enhanced_pred * correction_factor
            print(f"  Applied mobile QB correction: {correction_factor:.2f}x")
        
        # Ensure reasonable bounds for QBs
        enhanced_pred = np.clip(enhanced_pred, 0.0, 50.0)
        
        return {
            "player": player_name,
            "base_prediction": base_prediction,
            "enhanced_prediction": enhanced_pred,
            "adjustments": {
                "minimax": minimax_adj,
                "markov": markov_adj,
                "performance_penalty": performance_penalty,
                "passing_consistency": passing_factor
            },
            "total_adjustment": total_adjustment,
            "latest_week": player_data['GLOBAL_WEEK'].iloc[-1] if len(player_data) > 0 else None
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
                "passing_consistency": 0.0,
                "rushing_upside": 0.0
            },
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Enhanced QB Predictor with Minimax Theory and Markov Chains")
    parser.add_argument("--artifacts", default="./artifacts_qb", help="Path to model artifacts directory")
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
    print(f"\n=== Enhanced QB Prediction for {args.player} ===")
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
