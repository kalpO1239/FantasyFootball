#!/usr/bin/env python3
"""
QB Model Training Script
Trains a comprehensive QB prediction model using passing stats and PPR data
Usage: python3 train_qb_model.py --data_dir ./PassingStats --ppr_2023 QBPPR2023.csv --ppr_2024 QBPPR2024.csv --outfile_dir ./artifacts_qb --min_games 3 --test_weeks 4
"""

import argparse
import glob
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def to_num(x):
    """Convert string values to numeric, handling percentages and special cases"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def load_all_weeks(data_dir: str) -> pd.DataFrame:
    """Load all weekly passing stats files"""
    files = sorted(glob.glob(os.path.join(data_dir, "QB*.csv")))
    if not files:
        raise FileNotFoundError(f"No QB CSVs found in {data_dir}")
    
    frames = []
    for week_idx, f in enumerate(files, start=1):
        df = pd.read_csv(f)
        # Normalize column names - handle the quotes that pandas adds
        df.columns = [c.strip().strip("'\"").upper() for c in df.columns]
        
        # Keep only expected columns if present
        wanted = ["PLAYER NAME","TEAM","TT","CAY","IAY","AYD","AGG%","LCAD","AYTS","ATT","YDS","TD","INT","RATE","COMP%","XCOMP%","+/-"]
        for w in wanted:
            if w not in df.columns:
                raise ValueError(f"Missing column '{w}' in file {f}")
        
        # Standardize player name column
        df = df.rename(columns={"PLAYER NAME": "PLAYER"})
        df["WEEK"] = week_idx
        df["GLOBAL_WEEK"] = week_idx  # Add global week for merging
        
        # Update wanted list to use "PLAYER" instead of "PLAYER NAME"
        wanted_renamed = ["PLAYER","TEAM","TT","CAY","IAY","AYD","AGG%","LCAD","AYTS","ATT","YDS","TD","INT","RATE","COMP%","XCOMP%","+/-"]
        frames.append(df[wanted_renamed + ["WEEK", "GLOBAL_WEEK"]])
    
    data = pd.concat(frames, ignore_index=True)
    
    # Convert numerics
    for col in ["TT","CAY","IAY","AYD","AGG%","LCAD","AYTS","ATT","YDS","TD","INT","RATE","COMP%","XCOMP%","+/-"]:
        data[col] = data[col].apply(to_num)
    
    # Basic cleaning
    data["PLAYER"] = data["PLAYER"].astype(str).str.strip()
    data["TEAM"] = data["TEAM"].astype(str).str.strip()
    
    return data

def load_ppr_data(ppr_2023: str, ppr_2024: str) -> pd.DataFrame:
    """Load and combine QB PPR data from 2023 and 2024"""
    ppr_2023_df = pd.read_csv(ppr_2023)
    ppr_2024_df = pd.read_csv(ppr_2024)
    
    # Process 2023 data (weeks 1-18)
    ppr_2023_long = pd.melt(ppr_2023_df, 
                           id_vars=['#', 'Player', 'Pos', 'Team'],
                           value_vars=[str(i) for i in range(1, 19)],
                           var_name='WEEK',
                           value_name='PPR')
    
    ppr_2023_long = ppr_2023_long.rename(columns={'Player': 'PLAYER', 'Team': 'TEAM'})
    ppr_2023_long['WEEK'] = ppr_2023_long['WEEK'].astype(int)
    ppr_2023_long['SEASON'] = 2023
    ppr_2023_long['GLOBAL_WEEK'] = ppr_2023_long['WEEK']
    
    # Process 2024 data (weeks 19-35)
    ppr_2024_long = pd.melt(ppr_2024_df,
                           id_vars=['#', 'Player', 'Pos', 'Team'],
                           value_vars=[str(i) for i in range(1, 19)],
                           var_name='WEEK',
                           value_name='PPR')
    
    ppr_2024_long = ppr_2024_long.rename(columns={'Player': 'PLAYER', 'Team': 'TEAM'})
    ppr_2024_long['WEEK'] = ppr_2024_long['WEEK'].astype(int)
    ppr_2024_long['SEASON'] = 2024
    ppr_2024_long['GLOBAL_WEEK'] = ppr_2024_long['WEEK'] + 18  # Offset for 2024
    
    # Combine data
    combined_ppr = pd.concat([ppr_2023_long, ppr_2024_long], ignore_index=True)
    
    # Clean PPR data
    combined_ppr['PPR'] = combined_ppr['PPR'].apply(to_num)
    combined_ppr['PLAYER'] = combined_ppr['PLAYER'].astype(str).str.strip()
    
    return combined_ppr[['PLAYER', 'TEAM', 'WEEK', 'GLOBAL_WEEK', 'SEASON', 'PPR']]

def compute_passing_ppr(df: pd.DataFrame) -> pd.Series:
    """Compute passing PPR: 0.04*YDS + 4*TD - 2*INT"""
    return 0.04*df["YDS"].fillna(0) + 4.0*df["TD"].fillna(0) - 2.0*df["INT"].fillna(0)

def build_training_table(passing_data: pd.DataFrame, ppr_data: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """Build comprehensive training table with features and targets"""
    
    # Start with passing data as the base
    merged = passing_data.copy()
    
    # Merge PPR data onto passing rows to get rushing upside
    ppr_key = ppr_data[["PLAYER","GLOBAL_WEEK","PPR"]]
    merged = merged.merge(ppr_key, on=["PLAYER","GLOBAL_WEEK"], how="left")
    
    # Compute passing PPR
    merged["PASSING_PPR"] = compute_passing_ppr(merged)
    
    # Compute rushing upside only when player actually played (PPR is not NaN)
    merged["RUSHING_UPSIDE"] = np.where(
        merged["PPR"].notna(), 
        merged["PPR"] - merged["PASSING_PPR"], 
        np.nan
    )
    
    # For players who didn't play (PPR is NaN), set all passing stats to NaN
    # This ensures they don't get penalized for not playing
    for col in ["TT","CAY","IAY","AYD","AGG%","LCAD","AYTS","ATT","YDS","TD","INT","RATE","COMP%","XCOMP%","+/-","PASSING_PPR"]:
        if col in merged.columns:
            merged[col] = np.where(
                merged["PPR"].isna(),
                np.nan,
                merged[col]
            )
    
    # Set target variable: use PPR only when player actually played
    merged["TARGET_PPR"] = merged["PPR"].copy()
    
    # For players who didn't play (PPR is NaN), we can't predict their performance
    # So we'll exclude these rows from training
    merged = merged.dropna(subset=["TARGET_PPR"])
    
    # Sort by player and week for feature engineering
    merged = merged.sort_values(["PLAYER","GLOBAL_WEEK"]).reset_index(drop=True)
    
    # Feature engineering
    group = merged.groupby("PLAYER", group_keys=False)
    
    # Key metrics for QBs (based on user notes)
    key_metrics = ["TT", "AGG%", "ATT", "YDS", "TD", "INT", "RATE", "COMP%", "XCOMP%"]
    
    # 1-week lags
    for c in key_metrics:
        merged[f"{c}_lag1"] = group[c].shift(1)
    
    # Rolling averages with different windows
    for c in key_metrics:
        merged[f"{c}_roll3"] = group[c].shift(1).rolling(3, min_periods=min_games).mean()
        merged[f"{c}_roll5"] = group[c].shift(1).rolling(5, min_periods=3).mean()
        merged[f"{c}_roll8"] = group[c].shift(1).rolling(8, min_periods=5).mean()
    
    # Momentum indicators
    for c in key_metrics:
        merged[f"{c}_momentum_short"] = merged[f"{c}_roll3"] - merged[f"{c}_roll5"]
        merged[f"{c}_momentum_medium"] = merged[f"{c}_roll5"] - merged[f"{c}_roll8"]
    
    # Rushing upside features
    for c in ["RUSHING_UPSIDE"]:
        merged[f"{c}_lag1"] = group[c].shift(1)
        merged[f"{c}_roll3"] = group[c].shift(1).rolling(3, min_periods=min_games).mean()
        merged[f"{c}_roll5"] = group[c].shift(1).rolling(5, min_periods=3).mean()
        merged[f"{c}_roll8"] = group[c].shift(1).rolling(8, min_periods=5).mean()
        merged[f"{c}_momentum_short"] = merged[f"{c}_roll3"] - merged[f"{c}_roll5"]
        merged[f"{c}_momentum_medium"] = merged[f"{c}_roll5"] - merged[f"{c}_roll8"]
    
    # Recent season performance (last 17 weeks = 2024 season)
    for c in ["ATT", "YDS", "TD", "INT", "RATE", "COMP%"]:
        recent_season = group[c].shift(1).rolling(17, min_periods=8).mean()
        merged[f"{c}_recent_season"] = recent_season
        
        # Recent vs longer-term performance
        recent_3w = group[c].shift(1).rolling(3, min_periods=2).mean()
        merged[f"{c}_recent_surge"] = recent_3w - recent_season.values
    
    # Week number features with recent emphasis
    merged["week_number"] = merged["GLOBAL_WEEK"]
    merged["is_recent_season"] = (merged["GLOBAL_WEEK"] >= 19).astype(int)  # Last 17 weeks (2024)
    merged["very_recent"] = (merged["GLOBAL_WEEK"] >= 31).astype(int)  # Last 5 weeks
    
    # High-volume indicators
    merged["high_volume_att_recent"] = (merged["ATT_roll3"] >= 35).astype(int)  # 35+ attempts recent avg
    merged["high_volume_yds_recent"] = (merged["YDS_roll3"] >= 280).astype(int)  # 280+ yards recent avg
    merged["high_scorer_recent"] = (merged["TD_roll3"] >= 2.0).astype(int)     # 2+ TDs recent avg
    
    # Elite performer indicators
    merged["elite_volume"] = (merged["ATT_roll3"] >= 40).astype(int)  # 40+ attempts recent avg
    merged["elite_performer_recent"] = ((merged["ATT_roll3"] >= 40) | (merged["YDS_roll3"] >= 320)).astype(int)
    
    # Strong momentum indicators
    merged["strong_momentum_att"] = (merged["ATT_momentum_short"] >= 3.0).astype(int)
    merged["strong_momentum_yds"] = (merged["YDS_momentum_short"] >= 25.0).astype(int)
    merged["strong_momentum_td"] = (merged["TD_momentum_short"] >= 0.3).astype(int)
    
    # Positive trend indicators
    merged["positive_trend_att"] = (merged["ATT_momentum_short"] > 0).astype(int)
    merged["positive_trend_yds"] = (merged["YDS_momentum_short"] > 0).astype(int)
    merged["positive_trend_td"] = (merged["TD_momentum_short"] > 0).astype(int)
    
    # Rushing upside indicators
    merged["high_rushing_upside"] = (merged["RUSHING_UPSIDE_roll3"] >= 3.0).astype(int)
    merged["rushing_trend"] = (merged["RUSHING_UPSIDE_momentum_short"] > 0.5).astype(int)
    
    # Efficiency indicators
    merged["high_efficiency_recent"] = (merged["COMP%_roll3"] >= 70.0).astype(int)
    merged["efficiency_trend"] = (merged["COMP%_momentum_short"] > 2.0).astype(int)
    
    # Aggressiveness indicators
    merged["high_aggressiveness"] = (merged["AGG%_roll3"] >= 20.0).astype(int)
    merged["aggressiveness_trend"] = (merged["AGG%_momentum_short"] > 1.0).astype(int)
    
    # Time to throw indicators
    merged["fast_release"] = (merged["TT_roll3"] <= 2.5).astype(int)
    merged["slow_release"] = (merged["TT_roll3"] >= 3.0).astype(int)
    
    # PPR performance adjustment factors (moderate penalties/rewards)
    merged["ppr_performance_roll3"] = group["PPR"].shift(1).rolling(3, min_periods=1).mean()
    merged["ppr_performance_roll5"] = group["PPR"].shift(1).rolling(5, min_periods=2).mean()
    merged["ppr_performance_roll8"] = group["PPR"].shift(1).rolling(8, min_periods=3).mean()
    
    # Moderate penalties for consistently poor PPR performance
    merged["poor_ppr_performance"] = (merged["ppr_performance_roll5"] < 15).astype(int)  # Under 15 points
    merged["very_poor_ppr_performance"] = (merged["ppr_performance_roll5"] < 12).astype(int)  # Under 12 points
    merged["terrible_ppr_performance"] = (merged["ppr_performance_roll5"] < 8).astype(int)  # Under 8 points
    
    # Moderate rewards for consistently good PPR performance
    merged["good_ppr_performance"] = (merged["ppr_performance_roll5"] > 22).astype(int)  # Above 22 points
    merged["excellent_ppr_performance"] = (merged["ppr_performance_roll5"] > 26).astype(int)  # Above 26 points
    merged["elite_ppr_performance"] = (merged["ppr_performance_roll5"] > 30).astype(int)  # Above 30 points
    
    # Recent PPR performance trend
    merged["ppr_performance_trend"] = merged["ppr_performance_roll3"] - merged["ppr_performance_roll5"]
    merged["improving_ppr"] = (merged["ppr_performance_trend"] > 2.0).astype(int)  # Improving by 2+ points
    merged["declining_ppr"] = (merged["ppr_performance_trend"] < -2.0).astype(int)  # Declining by 2+ points
    
    # Passing consistency factor - moderate penalty for players who rarely get 25+ attempts
    merged["passing_appearances"] = group["ATT"].apply(lambda x: x.notna().rolling(8, min_periods=1).sum())
    merged["passing_consistency_rate"] = merged["passing_appearances"] / 8.0
    
    # Moderate penalties for inconsistent passing volume
    merged["inconsistent_passing"] = (merged["passing_consistency_rate"] < 0.5).astype(int)
    merged["very_inconsistent_passing"] = (merged["passing_consistency_rate"] < 0.3).astype(int)
    merged["rarely_passing"] = (merged["passing_consistency_rate"] < 0.2).astype(int)
    
    # Reward consistent passing volume
    merged["consistent_passing"] = (merged["passing_consistency_rate"] >= 0.7).astype(int)
    merged["very_consistent_passing"] = (merged["passing_consistency_rate"] >= 0.8).astype(int)
    
    return merged

def time_based_split(df: pd.DataFrame, test_weeks: int = 4):
    """Split by time: last N weeks serve as test set"""
    max_week = df["GLOBAL_WEEK"].max()
    test_mask = df["GLOBAL_WEEK"] > (max_week - test_weeks)
    train = df[~test_mask].copy()
    test = df[test_mask].copy()
    return train, test

def main(args):
    outdir = Path(args.outfile_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Loading passing stats...")
    passing_data = load_all_weeks(args.data_dir)
    print(f"Loaded {len(passing_data)} passing records")
    
    print("Loading PPR data...")
    ppr_data = load_ppr_data(args.ppr_2023, args.ppr_2024)
    print(f"Loaded {len(ppr_data)} PPR records")
    
    print("Building training table...")
    dfm = build_training_table(passing_data, ppr_data, min_games=args.min_games)
    print(f"Built training table with {len(dfm)} records")
    
    # Define feature columns
    key_metrics = ["TT", "AGG%", "ATT", "YDS", "TD", "INT", "RATE", "COMP%", "xCOMP%"]
    
    feature_cols = []
    
    # Core passing features (high priority) - focus on volume and efficiency
    for c in key_metrics:
        if c in dfm.columns:
            feature_cols += [f"{c}_lag1", f"{c}_roll3", f"{c}_roll5", f"{c}_roll8"]
            feature_cols += [f"{c}_momentum_short", f"{c}_momentum_medium"]
            feature_cols += [f"{c}_recent_season", f"{c}_recent_surge"]
    
    # Rushing upside features (supportive)
    for c in ["RUSHING_UPSIDE"]:
        if c in dfm.columns:
            feature_cols += [f"{c}_lag1", f"{c}_roll3", f"{c}_roll5", f"{c}_roll8"]
            feature_cols += [f"{c}_momentum_short", f"{c}_momentum_medium"]
    
    # Derived features - focus on what actually matters for QBs
    derived_features = [
        "week_number", "is_recent_season", "very_recent",
        "high_volume_att_recent", "high_volume_yds_recent", "high_scorer_recent",
        "elite_volume", "elite_performer_recent",
        "strong_momentum_att", "strong_momentum_yds", "strong_momentum_td",
        "positive_trend_att", "positive_trend_yds", "positive_trend_td",
        "high_rushing_upside", "rushing_trend",
        "high_efficiency_recent", "efficiency_trend",
        "high_aggressiveness", "aggressiveness_trend",
        "fast_release", "slow_release",
        # PPR performance adjustment features (moderate penalties/rewards)
        "ppr_performance_roll3", "ppr_performance_roll5", "ppr_performance_roll8", "ppr_performance_trend",
        "poor_ppr_performance", "very_poor_ppr_performance", "terrible_ppr_performance",
        "good_ppr_performance", "excellent_ppr_performance", "elite_ppr_performance",
        "improving_ppr", "declining_ppr",
        # Passing consistency features
        "passing_appearances", "passing_consistency_rate",
        "inconsistent_passing", "very_inconsistent_passing", "rarely_passing",
        "consistent_passing", "very_consistent_passing"
    ]
    
    # Add derived features that exist in the dataframe
    for feat in derived_features:
        if feat in dfm.columns:
            feature_cols.append(feat)
    
    # Remove any features that don't exist
    feature_cols = [col for col in feature_cols if col in dfm.columns]
    
    y_col = "TARGET_PPR"
    
    print(f"Using {len(feature_cols)} features")
    
    # Time-based split
    train, test = time_based_split(dfm, test_weeks=args.test_weeks)
    X_train, y_train = train[feature_cols].values, train[y_col].values
    X_test, y_test = test[feature_cols].values, test[y_col].values
    
    print(f"Training set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")
    
    # Train Ridge model with cross-validation and imputation
    model = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3,3,13), cv=5))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    mae_tr = mean_absolute_error(y_train, pred_train)
    rmse_tr = np.sqrt(mean_squared_error(y_train, pred_train))
    r2_tr = r2_score(y_train, pred_train)
    
    mae_te = mean_absolute_error(y_test, pred_test)
    rmse_te = np.sqrt(mean_squared_error(y_test, pred_test))
    r2_te = r2_score(y_test, pred_test)
    
    print(f"\n== QB Model Performance ==")
    print("== Training ==")
    print(f"MAE:  {mae_tr:.3f}  RMSE: {rmse_tr:.3f}  R^2: {r2_tr:.3f}")
    print("== Test ==")
    print(f"MAE:  {mae_te:.3f}  RMSE: {rmse_te:.3f}  R^2: {r2_te:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.named_steps['ridge'].coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save artifacts
    joblib.dump({
        "model": model,
        "X_cols": feature_cols,
        "y_col": y_col,
        "feature_info": {
            "key_metrics": key_metrics,
            "uses_lag1": True,
            "uses_roll3": True,
            "uses_roll5": True,
            "uses_roll8": True,
            "uses_momentum": True,
            "uses_recent_season": True,
            "uses_recent_surge": True,
            "uses_strong_momentum": True,
            "uses_positive_trends": True,
            "uses_rushing_upside": True,
            "uses_efficiency_indicators": True,
            "uses_aggressiveness_indicators": True,
            "uses_time_to_throw": True,
            "uses_ppr_performance_adjustment": True,
            "uses_passing_consistency": True,
            "notes": "QB model with comprehensive passing stats and PPR data integration."
        }
    }, outdir / "qb_model.joblib")
    
    # Residual std for Monte Carlo noise
    residuals = y_train - pred_train
    resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 8.0
    joblib.dump({"residual_std": resid_std}, outdir / "qb_residual_std.joblib")
    
    # Save training table
    dfm.to_parquet(outdir / "qb_training_table.parquet", index=False)
    print(f"\nSaved QB model + artifacts to {outdir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./PassingStats", help="Directory with weekly passing CSVs")
    parser.add_argument("--ppr_2023", default="QBPPR2023.csv", help="2023 QB PPR CSV file")
    parser.add_argument("--ppr_2024", default="QBPPR2024.csv", help="2024 QB PPR CSV file")
    parser.add_argument("--outfile_dir", default="./artifacts_qb", help="Where to save model & metadata")
    parser.add_argument("--min_games", type=int, default=3, help="Min prior games for rolling features")
    parser.add_argument("--test_weeks", type=int, default=4, help="Use last N weeks as test set")
    args = parser.parse_args()
    main(args)
