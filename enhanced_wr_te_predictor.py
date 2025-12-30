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

def build_markov_chain_model(player_data: pd.DataFrame, n_states: int = 5) -> Dict:
    """
    Build a Markov chain model for player performance states
    Returns transition matrix, state boundaries, state centers, and current state probabilities
    """
    historical_ppr = player_data['PPR'].dropna().values
    
    if len(historical_ppr) < 5:
        # Fallback: uniform distribution
        return {
            'transition_matrix': np.ones((n_states, n_states)) / n_states,
            'state_bins': np.linspace(0, 30, n_states + 1),
            'state_centers': np.linspace(5, 25, n_states),
            'current_state_probs': np.ones(n_states) / n_states,
            'current_state': n_states // 2,
            'state_labels': [f'State {i}' for i in range(n_states)]
        }
    
    # Discretize PPR into states using quantiles for better distribution
    try:
        # Use quantile-based binning for more balanced states
        quantiles = np.linspace(0, 100, n_states + 1)
        state_bins = np.percentile(historical_ppr, quantiles)
        # Ensure unique bins
        state_bins = np.unique(state_bins)
        if len(state_bins) < 2:
            state_bins = np.linspace(historical_ppr.min(), historical_ppr.max(), n_states + 1)
        
        states = np.digitize(historical_ppr, state_bins) - 1
        states = np.clip(states, 0, len(state_bins) - 2)  # Clip to valid state indices
        n_actual_states = len(np.unique(states))
        
        if n_actual_states < 2:
            # Fallback to uniform bins
            state_bins = np.linspace(0, max(30, historical_ppr.max()), n_states + 1)
            states = np.digitize(historical_ppr, state_bins) - 1
            states = np.clip(states, 0, n_states - 1)
            n_actual_states = n_states
        
    except Exception:
        # Fallback to uniform bins
        state_bins = np.linspace(0, max(30, historical_ppr.max()), n_states + 1)
        states = np.digitize(historical_ppr, state_bins) - 1
        states = np.clip(states, 0, n_states - 1)
        n_actual_states = n_states
    
    # Build transition matrix
    transition_matrix = np.zeros((n_states, n_states))
    
    for i in range(len(states) - 1):
        current_state = int(states[i])
        next_state = int(states[i + 1])
        if 0 <= current_state < n_states and 0 <= next_state < n_states:
            transition_matrix[current_state, next_state] += 1
    
    # Normalize transition matrix rows (add small smoothing to avoid zeros)
    smoothing = 0.01
    for i in range(n_states):
        row_sum = transition_matrix[i].sum()
        if row_sum > 0:
            transition_matrix[i] = transition_matrix[i] / row_sum
        else:
            # Uniform if no transitions observed
            transition_matrix[i] = np.ones(n_states) / n_states
        # Add small smoothing
        transition_matrix[i] = (1 - smoothing) * transition_matrix[i] + smoothing / n_states
    
    # Calculate state centers (mean PPR for each state)
    state_centers = []
    for state in range(n_states):
        state_mask = (states == state)
        if state_mask.sum() > 0:
            state_centers.append(np.mean(historical_ppr[state_mask]))
        else:
            # Use bin midpoint if no data
            state_centers.append((state_bins[state] + state_bins[min(state + 1, len(state_bins) - 1)]) / 2)
    
    state_centers = np.array(state_centers)
    
    # Get current state and predict next state probabilities
    current_state = int(states[-1])
    current_state = np.clip(current_state, 0, n_states - 1)
    
    # Get next state probabilities from transition matrix
    current_state_probs = transition_matrix[current_state].copy()
    
    return {
        'transition_matrix': transition_matrix,
        'state_bins': state_bins,
        'state_centers': state_centers,
        'current_state_probs': current_state_probs,
        'current_state': current_state,
        'state_labels': [f'State {i} ({state_bins[i]:.1f}-{state_bins[i+1]:.1f})' for i in range(min(n_states, len(state_bins) - 1))]
    }

def calculate_performance_penalty(player_data: pd.DataFrame, base_prediction: float = None) -> float:
    """
    Calculate penalties based on recent performance patterns
    Now heavily weights last 5 weeks - if recent avg is much different from base, apply strong adjustment
    """
    if len(player_data) < 3:
        return 0.0
    
    # Use GLOBAL_WEEK for chronological ordering if available
    week_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in player_data.columns else 'WEEK'
    
    # Recent performance analysis - get last 5 weeks chronologically
    player_data_sorted = player_data.sort_values(week_col)
    recent_ppr = player_data_sorted['PPR'].dropna().tail(5)
    if len(recent_ppr) < 3:
        return 0.0
    
    recent_avg = recent_ppr.mean()
    
    # HEAVY WEIGHT: If base_prediction is provided, adjust strongly based on recent avg vs base
    if base_prediction is not None and base_prediction > 0:
        # Calculate how much recent performance differs from base
        difference = recent_avg - base_prediction
        
        # Apply strong adjustment: 70% of the difference (heavily weight recent performance)
        # If recent avg is much lower than base, this will create a large negative adjustment
        adjustment = 0.70 * difference
        return np.clip(adjustment, -8.0, 8.0)  # Allow larger adjustments
    
    # Fallback: old logic if base_prediction not provided
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

def calculate_recent_absence_penalty(player_data: pd.DataFrame, max_week: int = None) -> float:
    """
    Calculate penalty for players who haven't been showing up recently
    Focus on the most recent 2 weeks - if they're missing from those, treat as playing badly
    
    Args:
        player_data: DataFrame with player's weekly data (should have GLOBAL_WEEK column for chronological ordering)
        max_week: Maximum week number in the dataset (if None, uses max from player_data)
    """
    if len(player_data) == 0:
        return -6.0  # No data at all - severe penalty
    
    # Use GLOBAL_WEEK for chronological ordering if available, otherwise fall back to WEEK
    week_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in player_data.columns else 'WEEK'
    
    if max_week is None:
        max_week = player_data[week_col].max()
    
    player_weeks = set(player_data[week_col].unique())
    
    # Focus on the most recent 2 weeks (most important indicator)
    most_recent_week = max_week
    second_most_recent = max_week - 1
    
    penalty = 0.0
    
    # Check most recent week (most important)
    if most_recent_week not in player_weeks:
        penalty -= 5.0  # Missing the most recent week is a strong negative signal
    else:
        # Player appeared in most recent week - check their performance
        recent_data = player_data[player_data[week_col] == most_recent_week]
        if len(recent_data) > 0 and 'PPR' in recent_data.columns:
            recent_ppr = recent_data['PPR'].dropna()
            if len(recent_ppr) > 0 and recent_ppr.iloc[0] < 2.0:
                penalty -= 1.0  # Played but scored very poorly (< 2 PPR)
    
    # Check second most recent week
    if second_most_recent not in player_weeks:
        penalty -= 3.0  # Missing second most recent week
    else:
        # Check their performance in that week
        recent_data = player_data[player_data[week_col] == second_most_recent]
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

def monte_carlo_simulation(
    base_prediction: float,
    residual_std: float,
    markov_info: Dict,
    n_simulations: int = 10000
) -> Dict:
    """
    Perform improved Monte Carlo simulation using mixture model based on Markov chain states
    This creates a more sophisticated stochastic model that accounts for state uncertainty
    
    Args:
        base_prediction: Adjusted base prediction (already includes absence penalties and other adjustments)
        residual_std: Residual standard deviation from model
        markov_info: Markov chain information with state probabilities and centers
        n_simulations: Number of Monte Carlo simulations
    """
    np.random.seed(42)  # For reproducibility
    
    # Get Markov state probabilities and centers
    state_probs = markov_info.get('current_state_probs', np.ones(5) / 5)
    state_centers = markov_info.get('state_centers', np.linspace(5, 25, 5))
    n_states = len(state_probs)
    
    # Calculate Markov expectation
    markov_expectation = np.dot(state_probs, state_centers)
    
    # Use a mixture model: generate simulations from each state weighted by its probability
    # This better captures the uncertainty in player state transitions
    simulations_list = []
    
    for state_idx in range(n_states):
        state_prob = state_probs[state_idx]
        state_center = state_centers[state_idx]
        
        # Number of simulations from this state
        n_state_sims = int(n_simulations * state_prob)
        
        if n_state_sims > 0:
            # For each state, combine:
            # 1. The state center (Markov prediction)
            # 2. The base prediction (linear model)
            # 3. Uncertainty based on residual std and state variance
            
            # Weight: more weight to base prediction if state is uncertain
            # Also: if base_prediction is very low (< 5), it's likely a missing/injured player, so trust it heavily
            state_confidence = state_prob
            if base_prediction < 2.0:
                # Extremely low base prediction (clipped to ~0, missing player) - trust it almost completely
                base_weight = 0.95  # Almost all weight to the low base
                state_weight = 0.05  # Very little to historical Markov states
            elif base_prediction < 5.0:
                # Very low base prediction (likely missing player) - trust it heavily
                base_weight = 0.80  # Much more weight to the low base
                state_weight = 0.20
            else:
                # Normal prediction - use standard weighting
                base_weight = 1.0 - 0.3 * state_confidence
                state_weight = 0.3 * state_confidence
            
            weighted_mean = base_weight * base_prediction + state_weight * state_center
            
            # Adjust std based on state: higher uncertainty for states with lower probability
            state_std_factor = 1.0 + (1.0 - state_confidence) * 0.5  # Up to 50% more std for uncertain states
            state_sim_std = residual_std * state_std_factor
            
            # Generate simulations for this state
            state_sims = np.random.normal(loc=weighted_mean, scale=state_sim_std, size=n_state_sims)
            simulations_list.append(state_sims)
    
    # Combine all simulations
    if len(simulations_list) > 0:
        simulations = np.concatenate(simulations_list)
        # If we have fewer simulations than requested (due to rounding), fill with overall distribution
        if len(simulations) < n_simulations:
            remaining = n_simulations - len(simulations)
            # Overall distribution: weighted combination
            # If base is very low, trust it more (likely missing player)
            if base_prediction < 2.0:
                overall_mean = 0.95 * base_prediction + 0.05 * markov_expectation
            elif base_prediction < 5.0:
                overall_mean = 0.80 * base_prediction + 0.20 * markov_expectation
            else:
                overall_mean = 0.65 * base_prediction + 0.35 * markov_expectation
            overall_std = residual_std * 1.1  # Slightly higher std for mixture
            remaining_sims = np.random.normal(loc=overall_mean, scale=overall_std, size=remaining)
            simulations = np.concatenate([simulations, remaining_sims])
        simulations = simulations[:n_simulations]  # Ensure exactly n_simulations
    else:
        # Fallback: simple normal distribution
        # If base is very low, trust it more (likely missing player)
        if base_prediction < 2.0:
            combined_mean = 0.95 * base_prediction + 0.05 * markov_expectation
        elif base_prediction < 5.0:
            combined_mean = 0.80 * base_prediction + 0.20 * markov_expectation
        else:
            combined_mean = 0.65 * base_prediction + 0.35 * markov_expectation
        adjusted_std = residual_std * 1.1
        simulations = np.random.normal(loc=combined_mean, scale=adjusted_std, size=n_simulations)
    
    # Note: absence_penalty is typically already incorporated in base_prediction
    # Only apply here if it's provided as a separate adjustment (which we're not doing)
    # This keeps the Monte Carlo focused on stochastic modeling, not deterministic adjustments
    
    # Ensure non-negative
    simulations = np.maximum(simulations, 0.0)
    
    # Calculate combined statistics
    combined_mean = float(np.mean(simulations))
    adjusted_std = float(np.std(simulations))
    
    # Calculate probability distributions for different ranges
    ranges = [
        (0, 5, '<5'),
        (5, 10, '5-10'),
        (10, 15, '10-15'),
        (15, 20, '15-20'),
        (20, 25, '20-25'),
        (25, 30, '25-30'),
        (30, float('inf'), '30+')
    ]
    
    probabilities = {}
    for low, high, label in ranges:
        if high == float('inf'):
            prob = np.mean(simulations >= low)
        else:
            prob = np.mean((simulations >= low) & (simulations < high))
        probabilities[label] = prob
    
    # Also calculate threshold probabilities (common fantasy thresholds)
    thresholds = [8, 10, 12, 15, 18, 20, 25, 30]
    threshold_probs = {}
    for threshold in thresholds:
        threshold_probs[f'≥{threshold}'] = np.mean(simulations >= threshold)
    
    # Calculate percentiles
    percentiles = {
        'p10': float(np.percentile(simulations, 10)),
        'p25': float(np.percentile(simulations, 25)),
        'p50': float(np.percentile(simulations, 50)),
        'p75': float(np.percentile(simulations, 75)),
        'p90': float(np.percentile(simulations, 90)),
        'mean': float(np.mean(simulations)),
        'std': float(np.std(simulations))
    }
    
    return {
        'simulations': simulations,
        'probabilities': probabilities,
        'threshold_probs': threshold_probs,
        'percentiles': percentiles,
        'combined_mean': float(combined_mean),
        'adjusted_std': float(adjusted_std),
        'markov_expectation': float(markov_expectation)
    }

def enhanced_prediction(player_name: str, base_prediction: float, artifacts_dir: Path, residual_std: float = None, n_simulations: int = 10000) -> Dict:
    """
    Generate enhanced prediction using linear base model, Markov chains, and Monte Carlo simulation
    
    Args:
        player_name: Name of player to predict
        base_prediction: Base prediction from linear model
        artifacts_dir: Directory containing model artifacts
        residual_std: Residual standard deviation (if None, will load from artifacts)
        n_simulations: Number of Monte Carlo simulations to run
    """
    try:
        # Load residual std if not provided
        if residual_std is None:
            try:
                resid = joblib.load(artifacts_dir / "wrte_fixed_residual_std.joblib")
                residual_std = float(resid.get("residual_std", 5.0))
            except:
                residual_std = 5.0  # Default fallback
        
        # Load player data
        table = load_wr_te_data(artifacts_dir)
        # Sort by GLOBAL_WEEK if available for chronological ordering, otherwise by WEEK
        sort_col = 'GLOBAL_WEEK' if 'GLOBAL_WEEK' in table.columns else 'WEEK'
        player_data = table[table["PLAYER"].str.lower() == player_name.lower()].sort_values(sort_col)
        
        if player_data.empty:
            return {
                "player": player_name,
                "base_prediction": base_prediction,
                "adjusted_base": base_prediction,
                "enhanced_prediction": base_prediction,
                "adjustments": {
                    "minimax": 0.0,
                    "performance_penalty": 0.0,
                    "volume_consistency": 0.0,
                    "absence_penalty": 0.0
                },
                "total_adjustment": 0.0,
                "markov_info": {},
                "monte_carlo": {
                    "probabilities": {},
                    "threshold_probs": {},
                    "percentiles": {"p50": base_prediction, "mean": base_prediction}
                },
                "error": "Player not found in training data"
            }
        
        # Get max week from full table to check for recent absences
        # Use GLOBAL_WEEK for chronological ordering if available
        if 'GLOBAL_WEEK' in table.columns:
            max_week = table['GLOBAL_WEEK'].max()
        elif 'WEEK' in table.columns:
            max_week = table['WEEK'].max()
        else:
            max_week = player_data['WEEK'].max() if 'WEEK' in player_data.columns else None
        
        # Calculate recent absence penalty (CRITICAL: treat missing players as playing badly)
        absence_penalty = calculate_recent_absence_penalty(player_data, max_week=max_week)
        
        # Build Markov chain model
        markov_info = build_markov_chain_model(player_data, n_states=5)
        
        # Calculate other adjustments (for display/reference)
        minimax_adj = calculate_minimax_adjustment(player_data, base_prediction)
        performance_penalty = calculate_performance_penalty(player_data, base_prediction=base_prediction)
        volume_factor = calculate_volume_consistency_factor(player_data)
        
        # Combine all adjustments including absence penalty
        total_adjustment = minimax_adj + performance_penalty + volume_factor + absence_penalty
        
        # Apply adjustments to base prediction
        adjusted_base = base_prediction + total_adjustment
        adjusted_base = np.clip(adjusted_base, 0.0, 40.0)
        
        # Run improved Monte Carlo simulation with Markov chain integration
        # Note: absence_penalty is already incorporated in adjusted_base
        mc_results = monte_carlo_simulation(
            adjusted_base,
            residual_std,
            markov_info,
            n_simulations=n_simulations
        )
        
        # Enhanced prediction is the median (50th percentile) of simulations
        enhanced_pred = mc_results['percentiles']['p50']
        
        return {
            "player": player_name,
            "base_prediction": base_prediction,
            "adjusted_base": adjusted_base,
            "enhanced_prediction": enhanced_pred,
            "adjustments": {
                "minimax": minimax_adj,
                "performance_penalty": performance_penalty,
                "volume_consistency": volume_factor,
                "absence_penalty": absence_penalty
            },
            "total_adjustment": total_adjustment,
            "markov_info": markov_info,
            "monte_carlo": {
                "probabilities": mc_results['probabilities'],
                "threshold_probs": mc_results['threshold_probs'],
                "percentiles": mc_results['percentiles'],
                "combined_mean": mc_results['combined_mean'],
                "adjusted_std": mc_results['adjusted_std'],
                "markov_expectation": mc_results['markov_expectation']
            },
            "latest_week": player_data['WEEK'].iloc[-1] if len(player_data) > 0 else None
        }
        
    except Exception as e:
        return {
            "player": player_name,
            "base_prediction": base_prediction,
                "adjusted_base": base_prediction,
            "enhanced_prediction": base_prediction,
            "adjustments": {
                "minimax": 0.0,
                "performance_penalty": 0.0,
                    "volume_consistency": 0.0,
                    "absence_penalty": 0.0
                },
                "total_adjustment": 0.0,
                "markov_info": {},
                "monte_carlo": {
                    "probabilities": {},
                    "threshold_probs": {},
                    "percentiles": {"p50": base_prediction, "mean": base_prediction}
            },
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Enhanced WR/TE Predictor with Minimax Theory and Markov Chains")
    parser.add_argument("--artifacts", default="./artifacts_fixed", help="Path to model artifacts directory")
    parser.add_argument("--player", required=True, help="Player name to predict")
    parser.add_argument("--show_details", action="store_true", help="Show detailed breakdown")
    parser.add_argument("--sim", type=int, default=10000, help="Number of Monte Carlo simulations")
    
    args = parser.parse_args()
    
    artifacts_dir = Path(args.artifacts)
    
    # Get base prediction
    base_pred, base_info = get_base_prediction(args.player, artifacts_dir)
    
    if base_pred is None:
        print(f"Error: {base_info.get('error', 'Unknown error')}")
        return
    
    # Get enhanced prediction
    residual_std = base_info.get('residual_std', 5.0)
    result = enhanced_prediction(args.player, base_pred, artifacts_dir, residual_std=residual_std, n_simulations=args.sim)
    
    # Display results
    print(f"\n=== Enhanced WR/TE Prediction for {args.player} ===")
    print(f"Base Prediction: {result['base_prediction']:.2f} PPR")
    print(f"Adjusted Base: {result.get('adjusted_base', result['base_prediction']):.2f} PPR")
    print(f"Enhanced Prediction (P50): {result['enhanced_prediction']:.2f} PPR")
    
    # Monte Carlo results
    if 'monte_carlo' in result and result['monte_carlo']:
        mc = result['monte_carlo']
        print(f"\n=== Monte Carlo Simulation ===")
        print(f"Mean: {mc['percentiles'].get('mean', 0):.2f} PPR")
        print(f"Std:  {mc.get('adjusted_std', 0):.2f} PPR")
        print(f"Percentiles: P10={mc['percentiles'].get('p10', 0):.1f}, "
              f"P25={mc['percentiles'].get('p25', 0):.1f}, "
              f"P50={mc['percentiles'].get('p50', 0):.1f}, "
              f"P75={mc['percentiles'].get('p75', 0):.1f}, "
              f"P90={mc['percentiles'].get('p90', 0):.1f}")
        
        print(f"\n=== Probability Distributions (PPR Ranges) ===")
        if 'probabilities' in mc:
            for label, prob in mc['probabilities'].items():
                print(f"  {label:>6}: {prob*100:5.1f}%")
        
        print(f"\n=== Threshold Probabilities ===")
        if 'threshold_probs' in mc:
            for threshold, prob in mc['threshold_probs'].items():
                print(f"  P({threshold:>4}): {prob*100:5.1f}%")
    
    if args.show_details:
        print(f"\n=== Detailed Adjustments ===")
        for adjustment_type, value in result['adjustments'].items():
            label = adjustment_type.replace('_', ' ').title()
            print(f"  {label}: {value:+.2f}")
        
        if 'markov_info' in result and result['markov_info']:
            mi = result['markov_info']
            print(f"\n=== Markov Chain Analysis ===")
            print(f"Current State: {mi.get('current_state', 'N/A')}")
            if 'state_labels' in mi:
                print(f"State Probabilities:")
                for i, (label, prob) in enumerate(zip(mi['state_labels'], mi.get('current_state_probs', []))):
                    print(f"  {label}: {prob*100:.1f}%")
            if 'markov_expectation' in result.get('monte_carlo', {}):
                print(f"Markov Expectation: {result['monte_carlo']['markov_expectation']:.2f} PPR")
        
        if 'latest_week' in result and result['latest_week']:
            print(f"\nLatest Week in Data: {result['latest_week']}")
        
        if 'error' in result:
            print(f"\nWarning: {result['error']}")

if __name__ == "__main__":
    main()
