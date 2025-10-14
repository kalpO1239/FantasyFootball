#!/usr/bin/env python3
"""
Generate Enhanced WR/TE Projections with Minimax Theory and Markov Chains
This script runs the enhanced predictor for all WR/TE players and creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from enhanced_wr_te_predictor import enhanced_prediction, get_base_prediction

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def get_all_players(artifacts_dir: Path) -> list:
    """Get list of all players from the training table"""
    training_table = pd.read_parquet(artifacts_dir / "wrte_fixed_training_table.parquet")
    players = training_table['PLAYER'].unique().tolist()
    return sorted(players)

def generate_all_projections(artifacts_dir: Path = Path("./artifacts_fixed")) -> pd.DataFrame:
    """
    Generate enhanced projections for all WR/TE players
    """
    print("Loading player data...")
    players = get_all_players(artifacts_dir)
    print(f"Found {len(players)} players")
    
    results = []
    
    for i, player in enumerate(players, 1):
        print(f"Processing {player} ({i}/{len(players)})...")
        
        try:
            # Get base prediction
            base_pred, base_info = get_base_prediction(player, artifacts_dir)
            
            if base_pred is None:
                print(f"  Skipping {player}: {base_info.get('error', 'Unknown error')}")
                continue
            
            # Get enhanced prediction
            result = enhanced_prediction(player, base_pred, artifacts_dir)
            
            if 'error' in result:
                print(f"  Warning for {player}: {result['error']}")
            
            # Add to results
            results.append({
                'player': result['player'],
                'base_prediction': result['base_prediction'],
                'enhanced_prediction': result['enhanced_prediction'],
                'minimax_adjustment': result['adjustments']['minimax'],
                'markov_adjustment': result['adjustments']['markov'],
                'performance_penalty': result['adjustments']['performance_penalty'],
                'volume_consistency': result['adjustments']['volume_consistency'],
                'total_adjustment': result['total_adjustment'],
                'latest_week': result.get('latest_week', None)
            })
            
        except Exception as e:
            print(f"  Error processing {player}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('enhanced_prediction', ascending=False)
    
    return df

def create_visualizations(df: pd.DataFrame, output_dir: Path = Path("./enhanced_wr_te_projections")):
    """
    Create bar graph visualizations of the enhanced projections
    """
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # 1. Top 30 Enhanced Projections
    plt.figure(figsize=(15, 10))
    top_30 = df.head(30)
    
    bars = plt.bar(range(len(top_30)), top_30['enhanced_prediction'], 
                   color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    
    # Color bars by adjustment type
    for i, (_, row) in enumerate(top_30.iterrows()):
        if row['total_adjustment'] > 0:
            bars[i].set_color('lightgreen')
        elif row['total_adjustment'] < 0:
            bars[i].set_color('lightcoral')
    
    plt.title('Top 30 Enhanced WR/TE Projections (Minimax + Markov Chains)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Players', fontsize=12)
    plt.ylabel('Enhanced PPR Prediction', fontsize=12)
    plt.xticks(range(len(top_30)), top_30['player'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_30_enhanced_wr_te_projections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Base vs Enhanced Comparison (Top 25)
    plt.figure(figsize=(15, 10))
    top_25 = df.head(25)
    
    x = np.arange(len(top_25))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, top_25['base_prediction'], width, 
                    label='Base Prediction', color='lightblue', alpha=0.7)
    bars2 = plt.bar(x + width/2, top_25['enhanced_prediction'], width,
                    label='Enhanced Prediction', color='orange', alpha=0.7)
    
    plt.title('Base vs Enhanced WR/TE Projections (Top 25)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Players', fontsize=12)
    plt.ylabel('PPR Prediction', fontsize=12)
    plt.xticks(x, top_25['player'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'base_vs_enhanced_wr_te_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Adjustment Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Total adjustments
    top_30_adj = df.head(30)
    colors = ['lightgreen' if adj > 0 else 'lightcoral' for adj in top_30_adj['total_adjustment']]
    
    ax1.bar(range(len(top_30_adj)), top_30_adj['total_adjustment'], color=colors, alpha=0.7)
    ax1.set_title('Total Adjustments by Player (Top 30)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Adjustment (PPR)', fontsize=12)
    ax1.set_xticks(range(len(top_30_adj)))
    ax1.set_xticklabels(top_30_adj['player'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Component breakdown
    components = ['minimax_adjustment', 'markov_adjustment', 'performance_penalty', 'volume_consistency']
    component_labels = ['Minimax', 'Markov', 'Performance', 'Volume']
    
    x = np.arange(len(top_30_adj))
    width = 0.2
    
    for i, (comp, label) in enumerate(zip(components, component_labels)):
        offset = (i - 1.5) * width
        ax2.bar(x + offset, top_30_adj[comp], width, label=label, alpha=0.7)
    
    ax2.set_title('Adjustment Components (Top 30)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Players', fontsize=12)
    ax2.set_ylabel('Adjustment (PPR)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_30_adj['player'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wr_te_adjustment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribution Analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['base_prediction'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title('Base Prediction Distribution')
    plt.xlabel('PPR Points')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    plt.hist(df['enhanced_prediction'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Enhanced Prediction Distribution')
    plt.xlabel('PPR Points')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 3)
    plt.hist(df['total_adjustment'], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.title('Total Adjustment Distribution')
    plt.xlabel('Adjustment (PPR Points)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['base_prediction'], df['enhanced_prediction'], alpha=0.6)
    plt.plot([0, df['base_prediction'].max()], [0, df['base_prediction'].max()], 
             'r--', alpha=0.7, label='No Change')
    plt.title('Base vs Enhanced Predictions')
    plt.xlabel('Base Prediction (PPR)')
    plt.ylabel('Enhanced Prediction (PPR)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wr_te_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate enhanced WR/TE projections"""
    artifacts_dir = Path("./artifacts_fixed")
    output_dir = Path("./enhanced_wr_te_projections")
    
    print("=== Enhanced WR/TE Projection Generator ===")
    print(f"Using artifacts from: {artifacts_dir}")
    print(f"Output directory: {output_dir}")
    
    # Generate projections
    print("\nGenerating enhanced projections...")
    df = generate_all_projections(artifacts_dir)
    
    if df.empty:
        print("No projections generated. Check your artifacts directory.")
        return
    
    # Save to CSV
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "enhanced_wr_te_projections.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} projections to {csv_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, output_dir)
    print(f"Visualizations saved to {output_dir}")
    
    # Display summary
    print(f"\n=== Summary ===")
    print(f"Total players processed: {len(df)}")
    print(f"Average base prediction: {df['base_prediction'].mean():.2f} PPR")
    print(f"Average enhanced prediction: {df['enhanced_prediction'].mean():.2f} PPR")
    print(f"Average total adjustment: {df['total_adjustment'].mean():+.2f} PPR")
    
    print(f"\nTop 10 Enhanced Projections:")
    top_10 = df.head(10)
    for _, row in top_10.iterrows():
        print(f"  {row['player']}: {row['enhanced_prediction']:.2f} PPR "
              f"(Base: {row['base_prediction']:.2f}, Adj: {row['total_adjustment']:+.2f})")
    
    print(f"\nBiggest Positive Adjustments:")
    biggest_positive = df.nlargest(5, 'total_adjustment')
    for _, row in biggest_positive.iterrows():
        print(f"  {row['player']}: +{row['total_adjustment']:.2f} PPR")
    
    print(f"\nBiggest Negative Adjustments:")
    biggest_negative = df.nsmallest(5, 'total_adjustment')
    for _, row in biggest_negative.iterrows():
        print(f"  {row['player']}: {row['total_adjustment']:.2f} PPR")

if __name__ == "__main__":
    main()
