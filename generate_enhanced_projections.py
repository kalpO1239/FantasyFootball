#!/usr/bin/env python3
"""
Generate Enhanced RB Projections for All Players with Bar Graph Visualization
Uses Minimax Theory and Markov Chains for improved predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import our enhanced predictor functions
from enhanced_rb_predictor import (
    load_rushing_stats, load_ppr_season, enhanced_prediction
)

def get_all_players():
    """Get list of all players from rushing stats"""
    rush = load_rushing_stats("RushingStats")
    return sorted(rush["PLAYER"].unique())

def generate_all_projections(artifacts_dir="./artifacts_rb2", max_players=None):
    """Generate enhanced projections for all players"""
    
    print("Loading data...")
    rush = load_rushing_stats("RushingStats")
    ppr23 = load_ppr_season("RBPPR2023.csv", 2023)
    ppr24 = load_ppr_season("RBPPR2024.csv", 2024)
    
    # Get all players
    all_players = get_all_players()
    if max_players:
        all_players = all_players[:max_players]
    
    print(f"Generating enhanced projections for {len(all_players)} players...")
    
    results = []
    
    for player in tqdm(all_players, desc="Processing players"):
        try:
            # Get simple base prediction from rushing stats
            player_rush = rush[rush["PLAYER"] == player].copy()
            if len(player_rush) == 0:
                continue
                
            # Simple base prediction
            recent_rush = player_rush.tail(3)
            if len(recent_rush) == 0:
                base_pred = 10.0
            else:
                recent_yards = recent_rush["YDS"].mean() if "YDS" in recent_rush.columns else 0
                recent_tds = recent_rush["TD"].mean() if "TD" in recent_rush.columns else 0
                base_pred = recent_yards * 0.1 + recent_tds * 6 + 5
            
            # Apply enhanced prediction
            enhanced_pred, info = enhanced_prediction(player, base_pred, artifacts_dir)
            
            # Get player stats for context
            ppr = pd.concat([ppr23, ppr24], ignore_index=True)
            player_ppr = ppr[ppr["PLAYER"] == player]
            recent_ppr = player_ppr["PPR"].dropna().tail(8)
            
            results.append({
                'PLAYER': player,
                'BASE_PREDICTION': base_pred,
                'ENHANCED_PREDICTION': enhanced_pred,
                'MINIMAX_PREDICTION': info['minimax_prediction'],
                'MARKOV_PREDICTION': info['markov_prediction'],
                'PERFORMANCE_PENALTY': info['performance_penalty'],
                'RECENT_PPR_AVG': recent_ppr.mean() if len(recent_ppr) > 0 else 0,
                'RECENT_PPR_STD': recent_ppr.std() if len(recent_ppr) > 0 else 0,
                'DATA_QUALITY': info['data_quality'],
                'MAX_REGRET_SCENARIO': info['minimax_info']['max_regret_scenario'],
                'CURRENT_STATE': info['markov_info']['current_state'],
                'RECENT_WEEKS': len(recent_ppr)
            })
            
        except Exception as e:
            print(f"Error processing {player}: {e}")
            continue
    
    return pd.DataFrame(results)

def create_visualizations(df, output_dir="./enhanced_projections"):
    """Create comprehensive visualizations of the enhanced projections"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # Sort by enhanced prediction
    df_sorted = df.sort_values('ENHANCED_PREDICTION', ascending=False)
    
    # 1. Top 30 Enhanced Projections
    plt.figure(figsize=(15, 10))
    top_30 = df_sorted.head(30)
    
    bars = plt.bar(range(len(top_30)), top_30['ENHANCED_PREDICTION'], 
                   color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    
    # Color bars based on performance penalty
    for i, (idx, row) in enumerate(top_30.iterrows()):
        if row['PERFORMANCE_PENALTY'] > 3:
            bars[i].set_color('red')
            bars[i].set_alpha(0.6)
        elif row['PERFORMANCE_PENALTY'] > 1:
            bars[i].set_color('orange')
            bars[i].set_alpha(0.7)
        else:
            bars[i].set_color('green')
            bars[i].set_alpha(0.8)
    
    plt.xlabel('Players (Ranked by Enhanced Projection)', fontsize=12)
    plt.ylabel('Enhanced PPR Projection', fontsize=12)
    plt.title('Top 30 RB Enhanced Projections (Minimax + Markov Chains)\nRed=High Penalty, Orange=Medium Penalty, Green=Low Penalty', 
              fontsize=14, fontweight='bold')
    
    # Add player names (rotated for readability)
    plt.xticks(range(len(top_30)), top_30['PLAYER'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(top_30['ENHANCED_PREDICTION']):
        plt.text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / 'top_30_enhanced_projections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Comparison: Base vs Enhanced Predictions (Top 20)
    plt.figure(figsize=(15, 8))
    top_20 = df_sorted.head(20)
    
    x = np.arange(len(top_20))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, top_20['BASE_PREDICTION'], width, 
                    label='Base Prediction', color='lightcoral', alpha=0.7)
    bars2 = plt.bar(x + width/2, top_20['ENHANCED_PREDICTION'], width, 
                    label='Enhanced Prediction', color='steelblue', alpha=0.7)
    
    plt.xlabel('Players', fontsize=12)
    plt.ylabel('PPR Projection', fontsize=12)
    plt.title('Base vs Enhanced Predictions (Top 20 RBs)\nEnhanced uses Minimax Theory + Markov Chains', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, top_20['PLAYER'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'base_vs_enhanced_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Penalty Analysis
    plt.figure(figsize=(15, 8))
    
    # Scatter plot: Enhanced Prediction vs Performance Penalty
    scatter = plt.scatter(df['ENHANCED_PREDICTION'], df['PERFORMANCE_PENALTY'], 
                         c=df['RECENT_PPR_AVG'], cmap='RdYlGn', alpha=0.7, s=60)
    
    plt.xlabel('Enhanced PPR Projection', fontsize=12)
    plt.ylabel('Performance Penalty', fontsize=12)
    plt.title('Enhanced Projections vs Performance Penalties\nColor = Recent PPR Average', 
              fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Recent PPR Average', fontsize=10)
    
    # Add trend line
    z = np.polyfit(df['ENHANCED_PREDICTION'], df['PERFORMANCE_PENALTY'], 1)
    p = np.poly1d(z)
    plt.plot(df['ENHANCED_PREDICTION'], p(df['ENHANCED_PREDICTION']), "r--", alpha=0.8)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'performance_penalty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Minimax vs Markov Comparison
    plt.figure(figsize=(15, 8))
    
    top_25 = df_sorted.head(25)
    x = np.arange(len(top_25))
    width = 0.25
    
    plt.bar(x - width, top_25['BASE_PREDICTION'], width, 
            label='Base Prediction', color='lightcoral', alpha=0.7)
    plt.bar(x, top_25['MINIMAX_PREDICTION'], width, 
            label='Minimax Prediction', color='orange', alpha=0.7)
    plt.bar(x + width, top_25['MARKOV_PREDICTION'], width, 
            label='Markov Prediction', color='purple', alpha=0.7)
    
    plt.xlabel('Players', fontsize=12)
    plt.ylabel('PPR Projection', fontsize=12)
    plt.title('Base vs Minimax vs Markov Predictions (Top 25 RBs)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, top_25['PLAYER'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'minimax_markov_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. State Distribution (Markov Analysis)
    plt.figure(figsize=(12, 6))
    
    state_counts = df['CURRENT_STATE'].value_counts().sort_index()
    states = ['Very Low (0)', 'Low (1)', 'Medium (2)', 'High (3)', 'Very High (4)']
    
    bars = plt.bar(range(len(state_counts)), state_counts.values, 
                   color=['red', 'orange', 'yellow', 'lightgreen', 'green'], alpha=0.7)
    
    plt.xlabel('Performance State', fontsize=12)
    plt.ylabel('Number of Players', fontsize=12)
    plt.title('Distribution of Players by Markov Performance State', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(states)), states)
    
    # Add value labels
    for i, v in enumerate(state_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'markov_state_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_path}/")

def main():
    parser = argparse.ArgumentParser(description="Generate Enhanced RB Projections with Visualizations")
    parser.add_argument("--artifacts", default="./artifacts_rb2", help="Model artifacts directory")
    parser.add_argument("--output", default="./enhanced_projections", help="Output directory")
    parser.add_argument("--max-players", type=int, help="Maximum number of players to process")
    parser.add_argument("--csv-output", default="enhanced_rb_projections.csv", help="CSV output filename")
    args = parser.parse_args()
    
    # Generate projections
    df = generate_all_projections(args.artifacts, args.max_players)
    
    if len(df) == 0:
        print("No projections generated!")
        return
    
    # Sort by enhanced prediction
    df = df.sort_values('ENHANCED_PREDICTION', ascending=False).reset_index(drop=True)
    df['RANK'] = range(1, len(df) + 1)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / args.csv_output
    df.to_csv(csv_path, index=False)
    
    print(f"\nEnhanced projections saved to {csv_path}")
    print(f"Generated projections for {len(df)} players")
    
    # Print summary statistics
    print(f"\n=== Enhanced Projections Summary ===")
    print(f"Average Enhanced Projection: {df['ENHANCED_PREDICTION'].mean():.2f}")
    print(f"Median Enhanced Projection: {df['ENHANCED_PREDICTION'].median():.2f}")
    print(f"Max Enhanced Projection: {df['ENHANCED_PREDICTION'].max():.2f}")
    print(f"Min Enhanced Projection: {df['ENHANCED_PREDICTION'].min():.2f}")
    
    print(f"\n=== Top 15 Enhanced Projections ===")
    top_15 = df.head(15)[['RANK', 'PLAYER', 'ENHANCED_PREDICTION', 'PERFORMANCE_PENALTY', 'RECENT_PPR_AVG']]
    print(top_15.to_string(index=False, float_format='%.2f'))
    
    print(f"\n=== Players with Highest Performance Penalties ===")
    high_penalty = df.nlargest(10, 'PERFORMANCE_PENALTY')[['PLAYER', 'ENHANCED_PREDICTION', 'PERFORMANCE_PENALTY', 'RECENT_PPR_AVG']]
    print(high_penalty.to_string(index=False, float_format='%.2f'))
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    create_visualizations(df, args.output)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_path}/")
    print(f"CSV file: {csv_path}")
    print(f"Visualizations: {output_path}/*.png")

if __name__ == "__main__":
    main()
