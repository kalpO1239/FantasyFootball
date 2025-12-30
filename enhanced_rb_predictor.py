#!/usr/bin/env python3
"""
Enhanced RB PPR Predictor with Minimax Theory and Markov Chains
Post-processes base model predictions for better accuracy
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_rushing_stats(rush_dir: str) -> pd.DataFrame:
    """Load all rushing stats files
    Automatically maps first 18 weeks to 2024 season, remaining weeks to 2025 season
    """
    rush_files = sorted(Path(rush_dir).glob("*.csv"))
    first_season_weeks = 18  # First 18 weeks belong to earliest season (2024)
    
    frames = []
    for global_week_idx, file in enumerate(rush_files, 1):
        df = pd.read_csv(file)
        df.columns = [c.strip() for c in df.columns]
        
        # Map to season and week based on position in file sequence
        if global_week_idx <= first_season_weeks:
            # First 18 weeks → 2024 season
            df["SEASON"] = 2024
            df["WEEK"] = global_week_idx  # Week 1-18 of 2024 season
            df["GLOBAL_WEEK"] = global_week_idx  # Global week 1-18
        else:
            # Remaining weeks → 2025 season
            df["SEASON"] = 2025
            df["WEEK"] = global_week_idx - first_season_weeks  # Week 1-N of 2025 season
            df["GLOBAL_WEEK"] = global_week_idx  # Global week 19+
        
        frames.append(df)
    
    rush = pd.concat(frames, ignore_index=True)
    rush["PLAYER"] = rush["PLAYER"].astype(str).str.strip()
    return rush

def load_ppr_season(ppr_path: str, season: int) -> pd.DataFrame:
    """Load PPR season data with name normalization"""
    df = pd.read_csv(ppr_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Find player column
    player_col = None
    for cand in ["PLAYER", "Player", "Name", "NAME"]:
        if cand in df.columns:
            player_col = cand
            break
    if player_col is None:
        raise ValueError("PPR file must contain a PLAYER/Name column")
    
    # Find week columns
    week_cols = []
    for c in df.columns:
        cu = c.upper()
        if c == player_col:
            continue
        if cu.startswith("W") and cu[1:].isdigit():
            week_cols.append(c)
        elif c.isdigit():
            week_cols.append(c)
    
    if not week_cols:
        raise ValueError("No week columns detected in PPR file")

    long_df = df.melt(id_vars=[player_col], value_vars=week_cols, var_name="WEEK_COL", value_name="PPR")
    
    def parse_week(col: str) -> int:
        cu = str(col).upper()
        if cu.startswith("W") and cu[1:].isdigit():
            return int(cu[1:])
        if col.isdigit():
            return int(col)
        return np.nan

    long_df["WEEK"] = long_df["WEEK_COL"].apply(parse_week)
    long_df["PPR"] = long_df["PPR"].replace({"-": np.nan}).apply(lambda x: pd.to_numeric(x, errors='coerce'))
    long_df["SEASON"] = season
    long_df = long_df.drop(columns=["WEEK_COL"]).dropna(subset=["WEEK"]).copy()
    
    # Normalize player names
    def normalize_player_name(name):
        if pd.isna(name):
            return name
        normalized = str(name).strip().replace("'", "").replace("  ", " ")
        name_mappings = {
            "Devon Achane": "De'Von Achane",
            "De'Von Achane": "De'Von Achane",
        }
        return name_mappings.get(normalized, normalized)
    
    long_df["PLAYER"] = long_df[player_col].apply(normalize_player_name)
    long_df = long_df.drop(columns=[player_col])

    # Set GLOBAL_WEEK based on season
    # First 18 weeks belong to 2024, remaining weeks belong to 2025
    first_season_weeks = 18
    if season == 2024:
        long_df["GLOBAL_WEEK"] = long_df["WEEK"]  # Weeks 1-18 map to GLOBAL_WEEK 1-18
    else:  # season == 2025
        long_df["GLOBAL_WEEK"] = long_df["WEEK"] + first_season_weeks  # Weeks 1-N map to GLOBAL_WEEK 19+

    return long_df

def build_markov_transition_matrix(player_ppr_data, states=5):
    """Build Markov transition matrix for player performance states"""
    if len(player_ppr_data) < 2:
        return np.eye(states), np.linspace(0, 30, states + 1)
    
    # Use GLOBAL_WEEK for chronological ordering if available
    week_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in player_ppr_data.columns else 'WEEK'
    player_ppr_sorted = player_ppr_data.sort_values(week_col)
    
    ppr_values = player_ppr_sorted['PPR'].dropna()
    if len(ppr_values) < 2:
        return np.eye(states), np.linspace(0, 30, states + 1)
    
    # Create state bins based on PPR percentiles
    state_bins = np.percentile(ppr_values, [0, 20, 40, 60, 80, 100])
    player_ppr_sorted_copy = player_ppr_sorted.copy()
    player_ppr_sorted_copy['state'] = pd.cut(player_ppr_sorted_copy['PPR'], bins=state_bins, labels=False, include_lowest=True)
    player_ppr_sorted_copy['state'] = player_ppr_sorted_copy['state'].fillna(0).astype(int)
    
    # Build transition matrix
    transition_matrix = np.zeros((states, states))
    state_counts = np.zeros(states)
    
    for i in range(len(player_ppr_sorted_copy) - 1):
        current_state = player_ppr_sorted_copy.iloc[i]['state']
        next_state = player_ppr_sorted_copy.iloc[i + 1]['state']
        if not pd.isna(current_state) and not pd.isna(next_state):
            transition_matrix[current_state, next_state] += 1
            state_counts[current_state] += 1
    
    # Normalize rows
    for i in range(states):
        if state_counts[i] > 0:
            transition_matrix[i] /= state_counts[i]
        else:
            transition_matrix[i] = np.ones(states) / states
    
    return transition_matrix, state_bins

def minimax_projection(base_prediction, player_ppr_data, risk_aversion=0.4):
    """Apply minimax theory to find robust projection that minimizes maximum regret"""
    # Use GLOBAL_WEEK for chronological ordering if available
    week_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in player_ppr_data.columns else 'WEEK'
    player_ppr_sorted = player_ppr_data.sort_values(week_col)
    recent_ppr = player_ppr_sorted['PPR'].dropna().tail(5)  # Last 5 weeks (matching WR/TE)
    if len(recent_ppr) == 0:
        return base_prediction
    
    # Define scenarios
    worst_case = recent_ppr.min()
    best_case = recent_ppr.max()
    median_case = recent_ppr.median()
    mean_case = recent_ppr.mean()
    
    # Calculate regrets for each scenario
    scenarios = {
        'worst': worst_case,
        'median': median_case,
        'mean': mean_case,
        'best': best_case
    }
    
    regrets = {}
    for name, value in scenarios.items():
        regrets[name] = abs(value - base_prediction)
    
    # Find the scenario with maximum regret
    max_regret_scenario = max(regrets, key=regrets.get)
    max_regret_value = scenarios[max_regret_scenario]
    
    # Minimax adjustment: move prediction toward the scenario that would cause maximum regret
    # This minimizes the worst-case regret
    adjustment = (max_regret_value - base_prediction) * risk_aversion
    minimax_pred = base_prediction + adjustment
    
    return minimax_pred, {
        'scenarios': scenarios,
        'regrets': regrets,
        'max_regret_scenario': max_regret_scenario,
        'adjustment': adjustment
    }

def markov_projection(player_ppr_data, transition_matrix, state_bins, steps_ahead=1):
    """Use Markov chain to project future performance based on state transitions"""
    if len(player_ppr_data) == 0:
        return 0, {}
    
    # Use GLOBAL_WEEK for chronological ordering if available
    week_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in player_ppr_data.columns else 'WEEK'
    player_ppr_sorted = player_ppr_data.sort_values(week_col)
    
    # Determine current state from most recent PPR
    recent_ppr = player_ppr_sorted['PPR'].dropna().tail(1)
    if len(recent_ppr) == 0:
        current_state = 2  # Default to middle state
    else:
        current_state = np.digitize(recent_ppr.iloc[0], state_bins) - 1
        current_state = max(0, min(current_state, len(transition_matrix) - 1))
    
    # Get state probabilities after steps_ahead
    state_probs = np.zeros(len(transition_matrix))
    state_probs[current_state] = 1.0
    
    for _ in range(steps_ahead):
        state_probs = state_probs @ transition_matrix
    
    # Convert state probabilities to PPR prediction
    state_midpoints = [(state_bins[i] + state_bins[i+1]) / 2 for i in range(len(state_bins)-1)]
    markov_pred = np.sum(state_probs * state_midpoints)
    
    return markov_pred, {
        'current_state': current_state,
        'state_probs': state_probs,
        'state_midpoints': state_midpoints
    }

def calculate_performance_penalty(player_ppr_data, base_prediction):
    """
    Calculate penalty based on recent performance patterns
    Now heavily weights last 5 weeks - if recent avg is much different from base, apply strong adjustment
    """
    # Use GLOBAL_WEEK for chronological ordering if available
    week_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in player_ppr_data.columns else 'WEEK'
    player_ppr_sorted = player_ppr_data.sort_values(week_col)
    
    # Recent performance analysis - get last 5 weeks chronologically
    recent_ppr = player_ppr_sorted['PPR'].dropna().tail(5)
    if len(recent_ppr) < 3:
        return 0
    
    recent_avg = recent_ppr.mean()
    
    # HEAVY WEIGHT: Apply strong adjustment based on recent avg vs base
    # Calculate how much recent performance differs from base
    difference = recent_avg - base_prediction
    
    # Apply strong adjustment: 70% of the difference (heavily weight recent performance)
    # If recent avg is much lower than base, this will create a large negative adjustment
    adjustment = 0.70 * difference
    return np.clip(adjustment, -8.0, 8.0)  # Allow larger adjustments

def calculate_recent_absence_penalty(player_ppr_data, max_week: int = None) -> float:
    """
    Calculate penalty for players who haven't been showing up recently
    Focus on the most recent 2 weeks - if they're missing from those, treat as playing badly
    
    Args:
        player_ppr_data: DataFrame with player's weekly data (should have GLOBAL_WEEK column for chronological ordering)
        max_week: Maximum week number in the dataset (if None, uses max from player_data)
    """
    if len(player_ppr_data) == 0:
        return -6.0  # No data at all - severe penalty
    
    # Use GLOBAL_WEEK for chronological ordering if available, otherwise fall back to WEEK
    week_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in player_ppr_data.columns else 'WEEK'
    
    if max_week is None:
        max_week = player_ppr_data[week_col].max()
    
    player_weeks = set(player_ppr_data[week_col].unique())
    
    # Focus on the most recent 2 weeks (most important indicator)
    most_recent_week = max_week
    second_most_recent = max_week - 1
    
    penalty = 0.0
    
    # Check most recent week (most important)
    if most_recent_week not in player_weeks:
        penalty -= 5.0  # Missing the most recent week is a strong negative signal
    else:
        # Player appeared in most recent week - check their performance
        recent_data = player_ppr_data[player_ppr_data[week_col] == most_recent_week]
        if len(recent_data) > 0 and 'PPR' in recent_data.columns:
            recent_ppr = recent_data['PPR'].dropna()
            if len(recent_ppr) > 0 and recent_ppr.iloc[0] < 2.0:
                penalty -= 1.0  # Played but scored very poorly (< 2 PPR)
    
    # Check second most recent week
    if second_most_recent not in player_weeks:
        penalty -= 3.0  # Missing second most recent week
    else:
        # Check their performance in that week
        recent_data = player_ppr_data[player_ppr_data[week_col] == second_most_recent]
        if len(recent_data) > 0 and 'PPR' in recent_data.columns:
            recent_ppr = recent_data['PPR'].dropna()
            if len(recent_ppr) > 0 and recent_ppr.iloc[0] < 2.0:
                penalty -= 0.5  # Played but scored poorly
    
    # If missing BOTH of the last 2 weeks, apply additional severe penalty
    if most_recent_week not in player_weeks and second_most_recent not in player_weeks:
        penalty -= 2.0  # Additional penalty for missing both recent weeks
    
    # Check last 4 weeks overall for context (but lighter weight)
    last_4_weeks = [max_week - i for i in range(4)]
    appearances_last_4 = sum(1 for week in last_4_weeks if week in player_weeks)
    
    # Only apply additional penalty if they appeared in 0 or 1 of the last 4 weeks
    if appearances_last_4 <= 1:
        penalty -= 1.0  # Very limited recent activity
    elif appearances_last_4 == 2:
        penalty -= 0.5  # Limited recent activity
    
    return penalty

def enhanced_prediction(player_name: str, base_prediction: float, artifacts_dir: str):
    """Generate enhanced prediction using minimax theory and Markov chains"""
    
    # Load data
    rush = load_rushing_stats("RushingStats")
    ppr24 = load_ppr_season("RBPPR2024.csv", 2024)
    ppr25 = load_ppr_season("RBPPR2025.csv", 2025)
    
    # Get player PPR data
    ppr = pd.concat([ppr24, ppr25], ignore_index=True)
    player_ppr = ppr[ppr["PLAYER"] == player_name].copy()
    
    if len(player_ppr) == 0:
        # Return base prediction with empty info dict
        return base_prediction, {
            'base_prediction': base_prediction,
            'adjusted_base': base_prediction,
            'minimax_prediction': base_prediction,
            'markov_prediction': base_prediction,
            'performance_penalty': 0.0,
            'absence_penalty': -6.0,  # Severe penalty for no data
            'total_adjustment': -6.0,
            'data_quality': 0.0,
            'minimax_info': {'max_regret_scenario': 'none', 'adjustment': 0.0},
            'markov_info': {'current_state': 2, 'state_probs': np.array([0.2, 0.2, 0.2, 0.2, 0.2])}
        }
    
    # Sort by GLOBAL_WEEK for chronological ordering (if available), otherwise by SEASON, WEEK
    if 'GLOBAL_WEEK' in player_ppr.columns:
        player_ppr = player_ppr.sort_values("GLOBAL_WEEK")
    else:
        player_ppr = player_ppr.sort_values(["SEASON", "WEEK"])
    
    # 1. Build Markov transition matrix
    transition_matrix, state_bins = build_markov_transition_matrix(player_ppr)
    
    # 2. Apply minimax theory
    minimax_pred, minimax_info = minimax_projection(base_prediction, player_ppr)
    
    # 3. Apply Markov chain projection
    markov_pred, markov_info = markov_projection(player_ppr, transition_matrix, state_bins)
    
    # 4. Calculate performance penalty (now heavily weights last 5 weeks)
    performance_penalty = calculate_performance_penalty(player_ppr, base_prediction)
    
    # 5. Calculate absence penalty (for missing recent weeks)
    week_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in ppr.columns else 'WEEK'
    max_week = ppr[week_col].max() if week_col in ppr.columns else player_ppr['WEEK'].max() if 'WEEK' in player_ppr.columns else None
    absence_penalty = calculate_recent_absence_penalty(player_ppr, max_week=max_week)
    
    # 6. Combine all adjustments
    total_adjustment = performance_penalty + absence_penalty
    
    # 7. Apply adjustments to base prediction first
    adjusted_base = base_prediction + total_adjustment
    adjusted_base = np.clip(adjusted_base, 0.0, 40.0)
    
    # 8. Combine predictions with weights (using adjusted base)
    # Weight based on data quality and recency
    data_quality = min(1.0, len(player_ppr) / 20)  # More data = higher weight
    
    # Final prediction: weighted combination using adjusted base
    final_pred = (
        0.35 * adjusted_base +           # Adjusted base model (heavily weighted by recent performance)
        0.30 * minimax_pred +            # Minimax robust prediction
        0.25 * markov_pred +             # Markov chain prediction
        0.10 * base_prediction           # Original base (smaller weight)
    )
    
    return final_pred, {
        'base_prediction': base_prediction,
        'adjusted_base': adjusted_base,
        'minimax_prediction': minimax_pred,
        'markov_prediction': markov_pred,
        'performance_penalty': performance_penalty,
        'absence_penalty': absence_penalty,
        'total_adjustment': total_adjustment,
        'data_quality': data_quality,
        'minimax_info': minimax_info,
        'markov_info': markov_info
    }

def main():
    parser = argparse.ArgumentParser(description="Enhanced RB PPR Prediction with Minimax and Markov")
    parser.add_argument("--player", required=True, help="Player name to predict")
    parser.add_argument("--artifacts", default="./artifacts_rb2", help="Base model artifacts directory")
    parser.add_argument("--sim", type=int, default=5000, help="Number of Monte Carlo simulations")
    args = parser.parse_args()
    
    # Load base model
    artifacts = Path(args.artifacts)
    model_data = joblib.load(artifacts / "rb_model.joblib")
    residual_std = joblib.load(artifacts / "rb_residual_std.joblib")
    if isinstance(residual_std, dict):
        residual_std = residual_std.get('residual_std', 7.5)
    training_table = pd.read_parquet(artifacts / "rb_training_table.parquet")
    
    # Get base prediction using existing model
    # Load data for base prediction
    rush = load_rushing_stats("RushingStats")
    ppr24 = load_ppr_season("RBPPR2024.csv", 2024)
    ppr25 = load_ppr_season("RBPPR2025.csv", 2025)
    
    # Get player data for base prediction
    player_rush = rush[rush["PLAYER"] == args.player].copy()
    if len(player_rush) == 0:
        print(f"Player {args.player} not found in rushing stats")
        return
    
    # Simple base prediction using recent rushing stats
    recent_rush = player_rush.tail(3)  # Last 3 weeks
    if len(recent_rush) == 0:
        base_pred = 10.0  # Default
    else:
        # Simple heuristic: recent yards * 0.1 + recent TDs * 6 + receiving estimate
        recent_yards = recent_rush["YDS"].mean() if "YDS" in recent_rush.columns else 0
        recent_tds = recent_rush["TD"].mean() if "TD" in recent_rush.columns else 0
        base_pred = recent_yards * 0.1 + recent_tds * 6 + 5  # Add 5 for receiving upside
    
    # Apply enhanced prediction
    enhanced_pred, info = enhanced_prediction(args.player, base_pred, args.artifacts)
    
    print(f"=== {args.player} Enhanced Projection ===")
    print(f"Base Prediction: {base_pred:.2f}")
    print(f"Minimax Prediction: {info['minimax_prediction']:.2f}")
    print(f"Markov Prediction: {info['markov_prediction']:.2f}")
    print(f"Performance Penalty: {info['performance_penalty']:.2f}")
    print(f"Enhanced Prediction: {enhanced_pred:.2f}")
    
    # Show minimax details
    print(f"\n=== Minimax Analysis ===")
    print(f"Max Regret Scenario: {info['minimax_info']['max_regret_scenario']}")
    print(f"Adjustment Applied: {info['minimax_info']['adjustment']:.2f}")
    print(f"Scenarios: {info['minimax_info']['scenarios']}")
    
    # Show Markov details
    print(f"\n=== Markov Chain Analysis ===")
    print(f"Current State: {info['markov_info']['current_state']}")
    print(f"State Probabilities: {info['markov_info']['state_probs']}")
    
    # Monte Carlo simulation
    np.random.seed(42)
    simulations = np.random.normal(float(enhanced_pred), float(residual_std), args.sim)
    
    print(f"\n=== Monte Carlo Simulation ({args.sim:,} trials) ===")
    print(f"Mean: {np.mean(simulations):.2f}")
    print(f"Std:  {np.std(simulations):.2f}")
    print(f"Min:  {np.min(simulations):.2f}")
    print(f"Max:  {np.max(simulations):.2f}")
    
    # PPR threshold probabilities
    print(f"\n=== PPR Threshold Probabilities ===")
    thresholds = [10, 15, 20, 25, 30]
    for threshold in thresholds:
        prob = np.mean(simulations >= threshold) * 100
        print(f"P(≥{threshold}): {prob:.1f}%")

if __name__ == "__main__":
    main()
