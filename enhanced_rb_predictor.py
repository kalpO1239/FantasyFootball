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
    """Load all rushing stats files"""
    rush_files = sorted(Path(rush_dir).glob("*.csv"))
    frames = []
    
    for i, file in enumerate(rush_files, 1):
        df = pd.read_csv(file)
        df.columns = [c.strip() for c in df.columns]
        df["WEEK"] = i
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

    if season == 2024:
        long_df = long_df[long_df["WEEK"] <= 16]

    return long_df

def build_markov_transition_matrix(player_ppr_data, states=5):
    """Build Markov transition matrix for player performance states"""
    if len(player_ppr_data) < 2:
        return np.eye(states)
    
    ppr_values = player_ppr_data['PPR'].dropna()
    if len(ppr_values) < 2:
        return np.eye(states)
    
    # Create state bins based on PPR percentiles
    state_bins = np.percentile(ppr_values, [0, 20, 40, 60, 80, 100])
    player_ppr_data = player_ppr_data.copy()
    player_ppr_data['state'] = pd.cut(player_ppr_data['PPR'], bins=state_bins, labels=False, include_lowest=True)
    player_ppr_data['state'] = player_ppr_data['state'].fillna(0).astype(int)
    
    # Build transition matrix
    transition_matrix = np.zeros((states, states))
    state_counts = np.zeros(states)
    
    for i in range(len(player_ppr_data) - 1):
        current_state = player_ppr_data.iloc[i]['state']
        next_state = player_ppr_data.iloc[i + 1]['state']
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
    recent_ppr = player_ppr_data['PPR'].dropna().tail(8)  # Last 8 weeks
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
    
    # Determine current state from most recent PPR
    recent_ppr = player_ppr_data['PPR'].dropna().tail(1)
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
    """Calculate penalty based on recent performance consistency and trends"""
    recent_ppr = player_ppr_data['PPR'].dropna().tail(8)
    if len(recent_ppr) < 3:
        return 0
    
    # Performance consistency penalty
    ppr_std = recent_ppr.std()
    ppr_mean = recent_ppr.mean()
    consistency_penalty = 0
    
    # High variance = inconsistent performance = penalty
    if ppr_std > ppr_mean * 0.5:  # High coefficient of variation
        consistency_penalty += 2.0
    
    # Recent performance trend penalty
    if len(recent_ppr) >= 4:
        recent_4 = recent_ppr.tail(4).mean()
        earlier_4 = recent_ppr.head(4).mean()
        trend_penalty = max(0, (earlier_4 - recent_4) * 0.3)  # Declining performance
        consistency_penalty += trend_penalty
    
    # Poor performance penalty
    poor_performance_penalty = 0
    if ppr_mean < 8:  # Very poor average
        poor_performance_penalty += 3.0
    elif ppr_mean < 12:  # Poor average
        poor_performance_penalty += 1.5
    
    # Inconsistent volume penalty (based on rushing appearances)
    # This would need rushing data, but we can approximate from PPR patterns
    zero_weeks = (recent_ppr == 0).sum()
    if zero_weeks > len(recent_ppr) * 0.3:  # More than 30% zero weeks
        poor_performance_penalty += 2.0
    
    total_penalty = consistency_penalty + poor_performance_penalty
    return total_penalty

def enhanced_prediction(player_name: str, base_prediction: float, artifacts_dir: str):
    """Generate enhanced prediction using minimax theory and Markov chains"""
    
    # Load data
    rush = load_rushing_stats("RushingStats")
    ppr23 = load_ppr_season("RBPPR2023.csv", 2023)
    ppr24 = load_ppr_season("RBPPR2024.csv", 2024)
    
    # Get player PPR data
    ppr = pd.concat([ppr23, ppr24], ignore_index=True)
    player_ppr = ppr[ppr["PLAYER"] == player_name].copy()
    
    if len(player_ppr) == 0:
        print(f"Warning: No PPR data found for {player_name}")
        return base_prediction, {}
    
    # Sort by week
    player_ppr = player_ppr.sort_values(["SEASON", "WEEK"])
    
    # 1. Build Markov transition matrix
    transition_matrix, state_bins = build_markov_transition_matrix(player_ppr)
    
    # 2. Apply minimax theory
    minimax_pred, minimax_info = minimax_projection(base_prediction, player_ppr)
    
    # 3. Apply Markov chain projection
    markov_pred, markov_info = markov_projection(player_ppr, transition_matrix, state_bins)
    
    # 4. Calculate performance penalty
    performance_penalty = calculate_performance_penalty(player_ppr, base_prediction)
    
    # 5. Combine predictions with weights
    # Weight based on data quality and recency
    data_quality = min(1.0, len(player_ppr) / 20)  # More data = higher weight
    recent_weight = min(1.0, len(player_ppr[player_ppr["SEASON"] == 2024]) / 10)  # 2024 data weight
    
    # Final prediction: weighted combination
    final_pred = (
        0.4 * base_prediction +           # Base model
        0.3 * minimax_pred +              # Minimax robust prediction
        0.2 * markov_pred +               # Markov chain prediction
        0.1 * (base_prediction - performance_penalty)  # Penalty-adjusted base
    )
    
    # Apply final penalty
    final_pred = max(0, final_pred - performance_penalty * 0.5)
    
    return final_pred, {
        'base_prediction': base_prediction,
        'minimax_prediction': minimax_pred,
        'markov_prediction': markov_pred,
        'performance_penalty': performance_penalty,
        'data_quality': data_quality,
        'recent_weight': recent_weight,
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
    ppr23 = load_ppr_season("RBPPR2023.csv", 2023)
    ppr24 = load_ppr_season("RBPPR2024.csv", 2024)
    
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
