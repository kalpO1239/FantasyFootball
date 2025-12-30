#!/usr/bin/env python3
"""
Stochastic Model Accuracy Test - Uses probability distributions instead of point predictions
Evaluates how well the model's probability distributions match actual outcomes
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import enhanced predictors
from enhanced_qb_predictor import get_base_prediction as get_qb_base, enhanced_prediction as qb_enhanced
from enhanced_wr_te_predictor import get_base_prediction as get_wr_te_base, enhanced_prediction as wr_te_enhanced
from enhanced_rb_predictor import enhanced_prediction as rb_enhanced, load_rushing_stats, load_ppr_season

def load_ppr_data_for_week(ppr_file: str, week: int) -> Dict[str, float]:
    """Load actual PPR values for a specific week from a PPR file"""
    df = pd.read_csv(ppr_file)
    df.columns = [c.strip() for c in df.columns]
    
    week_col = str(week)
    if week_col not in df.columns:
        return {}
    
    ppr_dict = {}
    for _, row in df.iterrows():
        player = str(row['Player']).strip()
        ppr_val = row[week_col]
        
        if pd.isna(ppr_val) or ppr_val == '-' or ppr_val == 'BYE' or ppr_val == '':
            continue
        
        try:
            ppr_dict[player] = float(ppr_val)
        except (ValueError, TypeError):
            continue
    
    return ppr_dict

def filter_training_table_by_week(table: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """Filter training table to only include data up to max_global_week"""
    return table[table['GLOBAL_WEEK'] <= max_global_week].copy()

def calculate_log_likelihood(actual: float, simulations: np.ndarray) -> float:
    """
    Calculate log-likelihood of actual value given the simulation distribution
    Uses kernel density estimation to get probability density
    """
    if len(simulations) == 0:
        return -np.inf
    
    # Use KDE to estimate probability density at actual value
    try:
        kde = stats.gaussian_kde(simulations)
        density = kde(actual)[0]
        # Avoid log(0)
        density = max(density, 1e-10)
        return np.log(density)
    except:
        # Fallback: use normal approximation
        mean = np.mean(simulations)
        std = np.std(simulations)
        if std < 0.1:
            std = 0.1
        return stats.norm.logpdf(actual, loc=mean, scale=std)

def calculate_range_probability(actual: float, probabilities: Dict[str, float]) -> float:
    """
    Get the probability assigned to the range containing the actual value
    Returns the probability of the range that contains actual
    """
    ranges = [
        (0, 5, '<5'),
        (5, 10, '5-10'),
        (10, 15, '10-15'),
        (15, 20, '15-20'),
        (20, 25, '20-25'),
        (25, 30, '25-30'),
        (30, float('inf'), '30+')
    ]
    
    for low, high, label in ranges:
        if high == float('inf'):
            if actual >= low:
                return probabilities.get(label, 0.0)
        else:
            if low <= actual < high:
                return probabilities.get(label, 0.0)
    
    return 0.0

def calculate_brier_score(actual: float, threshold: float, predicted_prob: float) -> float:
    """
    Calculate Brier score for a threshold prediction
    Brier score = (predicted_prob - actual_outcome)^2
    where actual_outcome is 1 if actual >= threshold, else 0
    """
    actual_outcome = 1.0 if actual >= threshold else 0.0
    return (predicted_prob - actual_outcome) ** 2

def generate_simulations_from_prediction(prediction: float, residual_std: float, n_simulations: int = 10000) -> np.ndarray:
    """Generate Monte Carlo simulations from a point prediction"""
    np.random.seed(42)
    simulations = np.random.normal(loc=prediction, scale=residual_std, size=n_simulations)
    simulations = np.maximum(simulations, 0.0)  # Ensure non-negative
    return simulations

def get_qb_prediction_with_probs(player_name: str, test_global_week: int, artifacts_dir: Path) -> Tuple[float, Dict]:
    """
    Get QB prediction with probability distributions
    Returns (enhanced_prediction, probability_info)
    """
    # Load training table
    table = pd.read_parquet(artifacts_dir / "qb_training_table.parquet")
    filtered_table = filter_training_table_by_week(table, test_global_week - 1)
    
    if filtered_table.empty:
        return None, None
    
    player_data = filtered_table[filtered_table["PLAYER"].str.lower() == player_name.lower()]
    if player_data.empty:
        return None, None
    
    player_data = player_data.sort_values("GLOBAL_WEEK")
    latest = player_data.iloc[-1]
    
    # Load model artifacts
    bundle = joblib.load(artifacts_dir / "qb_model.joblib")
    resid = joblib.load(artifacts_dir / "qb_residual_std.joblib")
    
    model = bundle["model"]
    X_cols = bundle["X_cols"]
    residual_std = resid.get('residual_std', 8.0) if isinstance(resid, dict) else resid
    
    # Get base prediction
    X = latest[X_cols].values.reshape(1, -1)
    base_pred = float(model.predict(X)[0])
    
    # Get enhanced prediction
    try:
        enhanced_result = qb_enhanced(player_name, base_pred, artifacts_dir)
        enhanced_pred = enhanced_result.get('enhanced_prediction', base_pred)
    except:
        enhanced_pred = base_pred
    
    # Generate simulations
    simulations = generate_simulations_from_prediction(enhanced_pred, residual_std)
    
    # Calculate probability distributions
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
    
    thresholds = [8, 10, 12, 15, 18, 20, 25, 30]
    threshold_probs = {}
    for threshold in thresholds:
        threshold_probs[f'≥{threshold}'] = np.mean(simulations >= threshold)
    
    return enhanced_pred, {
        'simulations': simulations,
        'probabilities': probabilities,
        'threshold_probs': threshold_probs,
        'mean': float(np.mean(simulations)),
        'std': float(np.std(simulations))
    }

def get_wr_te_prediction_with_probs(player_name: str, test_global_week: int, artifacts_dir: Path) -> Tuple[float, Dict]:
    """
    Get WR/TE prediction with probability distributions
    Returns (enhanced_prediction, probability_info)
    """
    import sys
    sys.path.append('.')
    from predict_player_fixed import build_latest_features
    
    # Load training table
    table = pd.read_parquet(artifacts_dir / "wrte_fixed_training_table.parquet")
    filtered_table = filter_training_table_by_week(table, test_global_week - 1)
    
    if filtered_table.empty:
        return None, None
    
    # Load model artifacts
    bundle = joblib.load(artifacts_dir / "wrte_fixed_model.joblib")
    resid = joblib.load(artifacts_dir / "wrte_fixed_residual_std.joblib")
    
    model = bundle["model"]
    X_cols = bundle["X_cols"]
    base_metrics = bundle["feature_info"]["base_metrics"]
    residual_std = resid.get('residual_std', 7.5) if isinstance(resid, dict) else resid
    
    # Build features
    X, order_cols, latest_week = build_latest_features(filtered_table, player_name, base_metrics)
    if X is None:
        return None, None
    
    # Align features
    X_df = pd.DataFrame(X, columns=order_cols)
    for col in X_cols:
        if col not in X_df.columns:
            X_df[col] = 0
    
    X_aligned = X_df[X_cols].values
    base_pred = float(model.predict(X_aligned)[0])
    
    # Get enhanced prediction with full Monte Carlo
    try:
        enhanced_result = wr_te_enhanced(player_name, base_pred, artifacts_dir, residual_std=residual_std)
        enhanced_pred = enhanced_result.get('enhanced_prediction', base_pred)
        mc_results = enhanced_result.get('monte_carlo', {})
        
        if mc_results:
            # Check if simulations array exists
            if 'simulations' in mc_results and len(mc_results['simulations']) > 0:
                return enhanced_pred, {
                    'simulations': np.array(mc_results['simulations']),
                    'probabilities': mc_results.get('probabilities', {}),
                    'threshold_probs': mc_results.get('threshold_probs', {}),
                    'mean': mc_results.get('combined_mean', enhanced_pred),
                    'std': mc_results.get('adjusted_std', residual_std)
                }
    except Exception as e:
        pass
    
    # Fallback: generate simulations
    simulations = generate_simulations_from_prediction(enhanced_pred, residual_std)
    
    # Calculate probabilities
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
    
    thresholds = [8, 10, 12, 15, 18, 20, 25, 30]
    threshold_probs = {}
    for threshold in thresholds:
        threshold_probs[f'≥{threshold}'] = np.mean(simulations >= threshold)
    
    return enhanced_pred, {
        'simulations': simulations,
        'probabilities': probabilities,
        'threshold_probs': threshold_probs,
        'mean': float(np.mean(simulations)),
        'std': float(np.std(simulations))
    }

def get_rb_prediction_with_probs(player_name: str, test_global_week: int, artifacts_dir: str) -> Tuple[float, Dict]:
    """
    Get RB prediction with probability distributions
    Returns (enhanced_prediction, probability_info)
    """
    # Load training table
    table = pd.read_parquet(Path(artifacts_dir) / "rb_training_table.parquet")
    filtered_table = filter_training_table_by_week(table, test_global_week - 1)
    
    if filtered_table.empty:
        return None, None
    
    # Load model artifacts
    model_data = joblib.load(Path(artifacts_dir) / "rb_model.joblib")
    residual_std = joblib.load(Path(artifacts_dir) / "rb_residual_std.joblib")
    if isinstance(residual_std, dict):
        residual_std = residual_std.get('residual_std', 7.5)
    
    # Get player data
    player_data = filtered_table[filtered_table["PLAYER"].str.lower() == player_name.lower()]
    if player_data.empty:
        return None, None
    
    player_data = player_data.sort_values("GLOBAL_WEEK")
    latest = player_data.iloc[-1]
    
    # Get base prediction
    X_cols = model_data["X_cols"]
    X = latest[X_cols].values.reshape(1, -1)
    model = model_data["model"]
    base_pred = float(model.predict(X)[0])
    
    # Get enhanced prediction
    try:
        enhanced_result = rb_enhanced(player_name, base_pred, artifacts_dir)
        if isinstance(enhanced_result, tuple):
            enhanced_pred = enhanced_result[0]
        else:
            enhanced_pred = enhanced_result.get('enhanced_prediction', base_pred)
    except:
        enhanced_pred = base_pred
    
    # Generate simulations
    simulations = generate_simulations_from_prediction(enhanced_pred, residual_std)
    
    # Calculate probabilities
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
    
    thresholds = [8, 10, 12, 15, 18, 20, 25, 30]
    threshold_probs = {}
    for threshold in thresholds:
        threshold_probs[f'≥{threshold}'] = np.mean(simulations >= threshold)
    
    return enhanced_pred, {
        'simulations': simulations,
        'probabilities': probabilities,
        'threshold_probs': threshold_probs,
        'mean': float(np.mean(simulations)),
        'std': float(np.std(simulations))
    }

def test_week_accuracy_stochastic(test_week: int, test_global_week: int, output_dir: Path) -> Dict:
    """
    Test model accuracy using stochastic metrics
    """
    output_dir.mkdir(exist_ok=True)
    print(f"\nTesting Week {test_week} (Global Week {test_global_week})...")
    
    results = {
        'week': test_week,
        'global_week': test_global_week,
        'qb': {'n_players': 0, 'avg_log_likelihood': 0, 'avg_range_prob': 0, 'avg_brier_15': 0, 'avg_brier_20': 0},
        'rb': {'n_players': 0, 'avg_log_likelihood': 0, 'avg_range_prob': 0, 'avg_brier_15': 0, 'avg_brier_20': 0},
        'wr_te': {'n_players': 0, 'avg_log_likelihood': 0, 'avg_range_prob': 0, 'avg_brier_15': 0, 'avg_brier_20': 0}
    }
    
    # Load actual PPR values
    qb_actuals = load_ppr_data_for_week("QBPPR2025.csv", test_week)
    rb_actuals = load_ppr_data_for_week("RBPPR2025.csv", test_week)
    wr_actuals = load_ppr_data_for_week("WRPPR2025.csv", test_week)
    
    # Test QB predictions
    print(f"  Testing {len(qb_actuals)} QBs...")
    qb_log_likelihoods = []
    qb_range_probs = []
    qb_brier_15 = []
    qb_brier_20 = []
    
    for player, actual_ppr in qb_actuals.items():
        try:
            enhanced_pred, prob_info = get_qb_prediction_with_probs(
                player, test_global_week, Path("./artifacts_qb")
            )
            if enhanced_pred is not None and prob_info is not None:
                # Log-likelihood
                ll = calculate_log_likelihood(actual_ppr, prob_info['simulations'])
                qb_log_likelihoods.append(ll)
                
                # Range probability
                range_prob = calculate_range_probability(actual_ppr, prob_info['probabilities'])
                qb_range_probs.append(range_prob)
                
                # Brier scores for key thresholds
                brier_15 = calculate_brier_score(actual_ppr, 15, prob_info['threshold_probs'].get('≥15', 0))
                qb_brier_15.append(brier_15)
                
                brier_20 = calculate_brier_score(actual_ppr, 20, prob_info['threshold_probs'].get('≥20', 0))
                qb_brier_20.append(brier_20)
        except Exception as e:
            continue
    
    if qb_log_likelihoods:
        results['qb']['n_players'] = len(qb_log_likelihoods)
        results['qb']['avg_log_likelihood'] = np.mean(qb_log_likelihoods)
        results['qb']['avg_range_prob'] = np.mean(qb_range_probs)
        results['qb']['avg_brier_15'] = np.mean(qb_brier_15)
        results['qb']['avg_brier_20'] = np.mean(qb_brier_20)
    
    # Test RB predictions
    print(f"  Testing {len(rb_actuals)} RBs...")
    rb_log_likelihoods = []
    rb_range_probs = []
    rb_brier_15 = []
    rb_brier_20 = []
    
    for player, actual_ppr in rb_actuals.items():
        try:
            enhanced_pred, prob_info = get_rb_prediction_with_probs(
                player, test_global_week, "./artifacts_rb2"
            )
            if enhanced_pred is not None and prob_info is not None:
                ll = calculate_log_likelihood(actual_ppr, prob_info['simulations'])
                rb_log_likelihoods.append(ll)
                
                range_prob = calculate_range_probability(actual_ppr, prob_info['probabilities'])
                rb_range_probs.append(range_prob)
                
                brier_15 = calculate_brier_score(actual_ppr, 15, prob_info['threshold_probs'].get('≥15', 0))
                rb_brier_15.append(brier_15)
                
                brier_20 = calculate_brier_score(actual_ppr, 20, prob_info['threshold_probs'].get('≥20', 0))
                rb_brier_20.append(brier_20)
        except Exception as e:
            continue
    
    if rb_log_likelihoods:
        results['rb']['n_players'] = len(rb_log_likelihoods)
        results['rb']['avg_log_likelihood'] = np.mean(rb_log_likelihoods)
        results['rb']['avg_range_prob'] = np.mean(rb_range_probs)
        results['rb']['avg_brier_15'] = np.mean(rb_brier_15)
        results['rb']['avg_brier_20'] = np.mean(rb_brier_20)
    
    # Test WR/TE predictions
    print(f"  Testing {len(wr_actuals)} WRs/TEs...")
    wr_log_likelihoods = []
    wr_range_probs = []
    wr_brier_15 = []
    wr_brier_20 = []
    
    for player, actual_ppr in wr_actuals.items():
        try:
            enhanced_pred, prob_info = get_wr_te_prediction_with_probs(
                player, test_global_week, Path("./artifacts_fixed")
            )
            if enhanced_pred is not None and prob_info is not None:
                ll = calculate_log_likelihood(actual_ppr, prob_info['simulations'])
                wr_log_likelihoods.append(ll)
                
                range_prob = calculate_range_probability(actual_ppr, prob_info['probabilities'])
                wr_range_probs.append(range_prob)
                
                brier_15 = calculate_brier_score(actual_ppr, 15, prob_info['threshold_probs'].get('≥15', 0))
                wr_brier_15.append(brier_15)
                
                brier_20 = calculate_brier_score(actual_ppr, 20, prob_info['threshold_probs'].get('≥20', 0))
                wr_brier_20.append(brier_20)
        except Exception as e:
            continue
    
    if wr_log_likelihoods:
        results['wr_te']['n_players'] = len(wr_log_likelihoods)
        results['wr_te']['avg_log_likelihood'] = np.mean(wr_log_likelihoods)
        results['wr_te']['avg_range_prob'] = np.mean(wr_range_probs)
        results['wr_te']['avg_brier_15'] = np.mean(wr_brier_15)
        results['wr_te']['avg_brier_20'] = np.mean(wr_brier_20)
    
    # Save results
    week_df = pd.DataFrame([{
        'week': test_week,
        'global_week': test_global_week,
        'position': 'QB',
        'n_players': results['qb']['n_players'],
        'avg_log_likelihood': results['qb']['avg_log_likelihood'],
        'avg_range_prob': results['qb']['avg_range_prob'],
        'avg_brier_15': results['qb']['avg_brier_15'],
        'avg_brier_20': results['qb']['avg_brier_20']
    }, {
        'week': test_week,
        'global_week': test_global_week,
        'position': 'RB',
        'n_players': results['rb']['n_players'],
        'avg_log_likelihood': results['rb']['avg_log_likelihood'],
        'avg_range_prob': results['rb']['avg_range_prob'],
        'avg_brier_15': results['rb']['avg_brier_15'],
        'avg_brier_20': results['rb']['avg_brier_20']
    }, {
        'week': test_week,
        'global_week': test_global_week,
        'position': 'WR/TE',
        'n_players': results['wr_te']['n_players'],
        'avg_log_likelihood': results['wr_te']['avg_log_likelihood'],
        'avg_range_prob': results['wr_te']['avg_range_prob'],
        'avg_brier_15': results['wr_te']['avg_brier_15'],
        'avg_brier_20': results['wr_te']['avg_brier_20']
    }])
    
    week_file = output_dir / f"week_{test_week}_stochastic_accuracy.csv"
    week_df.to_csv(week_file, index=False)
    print(f"  Saved results to {week_file}")
    
    return results

def main():
    """Run stochastic accuracy tests for weeks 2-16 in 2025"""
    
    output_dir = Path("./accuracy_results_stochastic")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Stochastic Model Accuracy Test - 2025 Season (Weeks 2-16)")
    print("=" * 80)
    print("\nMetrics:")
    print("  - Log-Likelihood: Higher is better (measures how likely actual outcomes were)")
    print("  - Range Probability: Probability assigned to the range containing actual (higher is better)")
    print("  - Brier Score: Lower is better (measures calibration of threshold probabilities)")
    print("=" * 80)
    
    all_results = []
    
    # Test weeks 2-16 in 2025 (global weeks 20-34)
    for week_2025 in range(2, 17):
        global_week = 18 + week_2025
        
        try:
            week_results = test_week_accuracy_stochastic(week_2025, global_week, output_dir)
            all_results.append(week_results)
        except Exception as e:
            print(f"  Error testing week {week_2025}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary
    summary_data = []
    for result in all_results:
        for position in ['qb', 'rb', 'wr_te']:
            summary_data.append({
                'week': result['week'],
                'global_week': result['global_week'],
                'position': position.upper(),
                'n_players': result[position]['n_players'],
                'avg_log_likelihood': result[position]['avg_log_likelihood'],
                'avg_range_prob': result[position]['avg_range_prob'],
                'avg_brier_15': result[position]['avg_brier_15'],
                'avg_brier_20': result[position]['avg_brier_20']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "stochastic_accuracy_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nSaved summary to {summary_file}")
    
    # Print overall statistics
    for position in ['QB', 'RB', 'WR/TE']:
        pos_data = summary_df[summary_df['position'] == position]
        if len(pos_data) > 0:
            avg_ll = pos_data['avg_log_likelihood'].mean()
            avg_range_prob = pos_data['avg_range_prob'].mean()
            avg_brier_15 = pos_data['avg_brier_15'].mean()
            avg_brier_20 = pos_data['avg_brier_20'].mean()
            total_players = pos_data['n_players'].sum()
            
            print(f"\n{position}:")
            print(f"  Average Log-Likelihood: {avg_ll:.3f} (higher is better)")
            print(f"  Average Range Probability: {avg_range_prob:.3f} (higher is better)")
            print(f"  Average Brier Score (≥15): {avg_brier_15:.3f} (lower is better)")
            print(f"  Average Brier Score (≥20): {avg_brier_20:.3f} (lower is better)")
            print(f"  Total Players Tested: {total_players}")

if __name__ == "__main__":
    main()

