# train_model_fixed.py
# Fixed WR/TE model that properly rewards recent surges and momentum
# Usage:
#   python train_model_fixed.py --data_dir ./receiving_csvs --outfile_dir ./artifacts_fixed --min_games 3 --test_weeks 4

import argparse
import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

KEEP_POS = {"WR", "TE"}

# ---- Helpers ----
def to_num(x):
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
    """Load all weekly receiving stats files
    Automatically maps first 18 weeks to 2024 season, remaining weeks to 2025 season
    Files are sorted by numeric order (Re19 = week 1 of 2024, Re20 = week 2, etc.)
    """
    import re
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")
    
    # Sort files numerically by extracting the number from filename (e.g., Re19 -> 19)
    def get_file_number(f):
        match = re.search(r'Re(\d+)', os.path.basename(f))
        return int(match.group(1)) if match else 0
    
    files = sorted(all_files, key=get_file_number)
    
    first_season_weeks = 18  # First 18 weeks belong to earliest season (2024)
    
    frames = []
    for global_week_idx, f in enumerate(files, start=1):
        df = pd.read_csv(f)
        # Normalize column names
        df.columns = [c.strip().upper() for c in df.columns]
        # Keep only expected columns if present
        wanted = ["PLAYER","TEAM","POS","SEP","TAY","TAY%","REC","TAR","YDS","TD","YAC/R"]
        for w in wanted:
            if w not in df.columns:
                raise ValueError(f"Missing column '{w}' in file {f}")
        # Filter positions
        df = df[df["POS"].isin(KEEP_POS)].copy()
        
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
        
        frames.append(df[wanted + ["WEEK", "GLOBAL_WEEK", "SEASON"]])
    data = pd.concat(frames, ignore_index=True)

    # Convert numerics
    for col in ["SEP","TAY","TAY%","REC","TAR","YDS","TD","YAC/R"]:
        data[col] = data[col].apply(to_num)

    # Basic cleaning
    data["PLAYER"] = data["PLAYER"].astype(str).str.strip()
    data["TEAM"]   = data["TEAM"].astype(str).str.strip()
    data["POS"]    = data["POS"].astype(str).str.strip()

    return data

def compute_target_ppr(df: pd.DataFrame) -> pd.Series:
    # PPR scoring for receiving (no rushing here):
    # 1 * REC + 0.1 * YDS + 6 * TD
    return 1.0*df["REC"].fillna(0) + 0.1*df["YDS"].fillna(0) + 6.0*df["TD"].fillna(0)

def add_fixed_features(df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """
    Build features that properly reward recent surges and momentum.
    Fixed elite performer detection and surge weighting.
    """
    df = df.sort_values(["PLAYER","WEEK"]).copy()

    feat_cols = ["SEP","TAY","TAY%","REC","TAR","YDS","TD","YAC/R"]
    group = df.groupby("PLAYER", group_keys=False)

    # REDUCED: 1-week lags (immediate past performance) - keep but will weight less
    for c in feat_cols:
        df[f"{c}_lag1"] = group[c].shift(1)

    # ENHANCED: Multiple rolling windows with recent emphasis
    for c in feat_cols:
        # 3-week rolling averages (recent trend) - HIGH WEIGHT
        df[f"{c}_roll3"] = group[c].shift(1).rolling(3, min_periods=min_games).mean()
        # 5-week rolling averages (medium trend) - MEDIUM WEIGHT
        df[f"{c}_roll5"] = group[c].shift(1).rolling(5, min_periods=3).mean()
        # 8-week rolling averages (longer trend) - LOWER WEIGHT
        df[f"{c}_roll8"] = group[c].shift(1).rolling(8, min_periods=5).mean()

    # NEW: Recent season performance (last 17 weeks = 2024 season)
    for c in ["TAR", "REC", "YDS", "TD"]:
        # Recent season average (last 17 weeks)
        recent_season = group[c].shift(1).rolling(17, min_periods=8).mean()
        df[f"{c}_recent_season"] = recent_season
        
        # Recent vs longer-term performance
        recent_3w = group[c].shift(1).rolling(3, min_periods=2).mean()
        df[f"{c}_recent_surge"] = recent_3w - recent_season.values

    # ENHANCED: Recent momentum indicators (emphasize recent trends)
    for c in feat_cols:
        # Short-term momentum (3w vs 5w) - HIGH WEIGHT
        df[f"{c}_momentum_short"] = df[f"{c}_roll3"] - df[f"{c}_roll5"]
        # Medium-term momentum (5w vs 8w) - MEDIUM WEIGHT
        df[f"{c}_momentum_medium"] = df[f"{c}_roll5"] - df[f"{c}_roll8"]

    # ENHANCED: Recent volume and efficiency trends
    df["TAR_trend_recent"] = df["TAR_roll3"] - df["TAR_roll5"]  # Recent trend
    df["REC_trend_recent"] = df["REC_roll3"] - df["REC_roll5"]
    df["YDS_trend_recent"] = df["YDS_roll3"] - df["YDS_roll5"]
    df["TD_trend_recent"] = df["TD_roll3"] - df["TD_roll5"]

    # ENHANCED: Recent efficiency metrics
    df["TD_per_TAR_recent"] = df["TD_roll3"] / (df["TAR_roll3"] + 0.1)
    df["CTCH_rate_recent"] = df["REC_roll3"] / (df["TAR_roll3"] + 0.1)
    df["YPT_recent"] = df["YDS_roll3"] / (df["TAR_roll3"] + 0.1)

    # NEW: Recent consistency metrics
    for c in ["TAR", "REC", "YDS", "TD"]:
        # Recent consistency (3-week std)
        df[f"{c}_consistency_recent"] = group[c].shift(1).rolling(3, min_periods=2).std()

    # ENHANCED: Week number features with recent emphasis
    df["week_number"] = df["WEEK"]
    df["is_recent_season"] = (df["GLOBAL_WEEK"] >= 19).astype(int)  # Last N weeks (2025 season)
    df["very_recent"] = (df["GLOBAL_WEEK"] >= (df["GLOBAL_WEEK"].max() - 4)).astype(int)  # Last 5 weeks

    # FIXED: Recent high-volume indicators (based on recent performance)
    df["high_volume_tar_recent"] = (df["TAR_roll3"] >= 8).astype(int)  # 8+ targets recent avg
    df["high_volume_rec_recent"] = (df["REC_roll3"] >= 5).astype(int)  # 5+ receptions recent avg
    df["high_volume_yds_recent"] = (df["YDS_roll3"] >= 60).astype(int)  # 60+ yards recent avg
    df["high_scorer_recent"] = (df["TD_roll3"] >= 0.3).astype(int)     # 0.3+ TDs recent avg

    # FIXED: Elite performer indicators (more nuanced)
    # Elite by targets OR yards (not both)
    df["elite_targets"] = (df["TAR_roll3"] >= 10).astype(int)  # 10+ targets recent avg
    df["elite_yards"] = (df["YDS_roll3"] >= 80).astype(int)    # 80+ yards recent avg
    df["elite_performer_recent"] = ((df["TAR_roll3"] >= 10) | (df["YDS_roll3"] >= 80)).astype(int)
    
    # NEW: Surge indicators (recent improvement)
    for c in ["TAR", "REC", "YDS", "TD"]:
        # Recent surge vs recent season average
        recent_3w = group[c].shift(1).rolling(3, min_periods=2).mean()
        recent_season_avg = group[c].shift(1).rolling(17, min_periods=8).mean()
        df[f"{c}_surge_vs_recent"] = recent_3w - recent_season_avg.values

    # NEW: Momentum strength indicators
    df["strong_momentum_tar"] = (df["TAR_momentum_short"] >= 1.0).astype(int)  # Strong positive momentum
    df["strong_momentum_yds"] = (df["YDS_momentum_short"] >= 10.0).astype(int)  # Strong yardage momentum
    df["strong_momentum_td"] = (df["TD_momentum_short"] >= 0.2).astype(int)     # Strong TD momentum

    # NEW: Trend strength indicators
    df["positive_trend_tar"] = (df["TAR_trend_recent"] > 0).astype(int)  # Positive target trend
    df["positive_trend_yds"] = (df["YDS_trend_recent"] > 0).astype(int)  # Positive yardage trend
    df["positive_trend_td"] = (df["TD_trend_recent"] > 0).astype(int)    # Positive TD trend

    # ENHANCED: Position-specific features
    df["is_wr"] = (df["POS"] == "WR").astype(int)
    df["is_te"] = (df["POS"] == "TE").astype(int)

    # NEW: Recent elite performer indicators
    df["consistent_performer_recent"] = (df["YDS_consistency_recent"] < 30).astype(int)  # Low recent variance

    # Target is *this week's* PPR
    df["PPR"] = compute_target_ppr(df)

    # We will predict PPR using fixed features
    # Drop rows without enough history
    need_cols = ([f"{c}_lag1" for c in feat_cols] +  # Keep lag1 but will weight less
                 [f"{c}_roll3" for c in feat_cols] + 
                 [f"{c}_roll5" for c in feat_cols] + 
                 [f"{c}_roll8" for c in feat_cols] + 
                 [f"{c}_momentum_short" for c in feat_cols] + 
                 [f"{c}_momentum_medium" for c in feat_cols] + 
                 ["TAR_trend_recent", "REC_trend_recent", "YDS_trend_recent", "TD_trend_recent",
                  "TD_per_TAR_recent", "CTCH_rate_recent", "YPT_recent"] +
                 [f"{c}_consistency_recent" for c in ["TAR", "REC", "YDS", "TD"]] +
                 ["week_number", "is_recent_season", "very_recent"] +
                 [f"{c}_recent_season" for c in ["TAR", "REC", "YDS", "TD"]] +
                 [f"{c}_recent_surge" for c in ["TAR", "REC", "YDS", "TD"]] +
                 [f"{c}_surge_vs_recent" for c in ["TAR", "REC", "YDS", "TD"]] +
                 ["high_volume_tar_recent", "high_volume_rec_recent", "high_volume_yds_recent", "high_scorer_recent", 
                  "elite_targets", "elite_yards", "elite_performer_recent", "consistent_performer_recent",
                  "strong_momentum_tar", "strong_momentum_yds", "strong_momentum_td",
                  "positive_trend_tar", "positive_trend_yds", "positive_trend_td",
                  "is_wr", "is_te"])
    
    df_model = df.dropna(subset=need_cols).copy()

    return df_model

def time_based_split(df: pd.DataFrame, test_weeks: int = 4):
    """
    Split by time: last N weeks serve as test set.
    """
    max_week = df["WEEK"].max()
    test_mask = df["WEEK"] > (max_week - test_weeks)
    train = df[~test_mask].copy()
    test  = df[test_mask].copy()
    return train, test

def main(args):
    outdir = Path(args.outfile_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_all_weeks(args.data_dir)
    dfm = add_fixed_features(data, min_games=args.min_games)

    # Fixed feature set
    base = ["SEP","TAY","TAY%","REC","TAR","YDS","TD","YAC/R"]
    X_cols = ([f"{c}_lag1" for c in base] +  # Keep lag1 but will weight less
              [f"{c}_roll3" for c in base] + 
              [f"{c}_roll5" for c in base] + 
              [f"{c}_roll8" for c in base] + 
              [f"{c}_momentum_short" for c in base] + 
              [f"{c}_momentum_medium" for c in base] + 
              ["TAR_trend_recent", "REC_trend_recent", "YDS_trend_recent", "TD_trend_recent",
               "TD_per_TAR_recent", "CTCH_rate_recent", "YPT_recent"] +
              [f"{c}_consistency_recent" for c in ["TAR", "REC", "YDS", "TD"]] +
              ["week_number", "is_recent_season", "very_recent"] +
              [f"{c}_recent_season" for c in ["TAR", "REC", "YDS", "TD"]] +
              [f"{c}_recent_surge" for c in ["TAR", "REC", "YDS", "TD"]] +
              [f"{c}_surge_vs_recent" for c in ["TAR", "REC", "YDS", "TD"]] +
              ["high_volume_tar_recent", "high_volume_rec_recent", "high_volume_yds_recent", "high_scorer_recent", 
               "elite_targets", "elite_yards", "elite_performer_recent", "consistent_performer_recent",
               "strong_momentum_tar", "strong_momentum_yds", "strong_momentum_td",
               "positive_trend_tar", "positive_trend_yds", "positive_trend_td",
               "is_wr", "is_te"])
    
    y_col = "PPR"

    train, test = time_based_split(dfm, test_weeks=args.test_weeks)
    X_train, y_train = train[X_cols].values, train[y_col].values
    X_test,  y_test  = test[X_cols].values,  test[y_col].values

    print(f"Training set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")
    print(f"Features: {len(X_cols)}")

    # Create multiple models for comparison
    models = []
    
    # 1. Ridge Regression with custom weights to emphasize recent data
    from sklearn.linear_model import Ridge
    
    # Create custom weights to emphasize recent data and fix issues
    feature_weights = np.ones(len(X_cols))
    
    # Reduce weight of lag1 features (immediate past)
    for i, col in enumerate(X_cols):
        if '_lag1' in col:
            feature_weights[i] = 0.1  # Reduce lag1 weight to 10%
        elif '_recent_surge' in col or '_surge_vs_recent' in col:
            feature_weights[i] = 4.0  # Increase surge weight to 400%
        elif '_momentum_short' in col:
            feature_weights[i] = 3.0  # Increase short-term momentum to 300%
        elif '_recent_season' in col:
            feature_weights[i] = 2.5  # Increase recent season weight to 250%
        elif '_roll3' in col:
            feature_weights[i] = 2.0  # Increase recent rolling features to 200%
        elif '_roll5' in col:
            feature_weights[i] = 1.5  # Medium weight for 5-week averages
        elif '_roll8' in col or '_momentum_medium' in col:
            feature_weights[i] = 0.5  # Reduce weight of longer-term features
        elif 'elite_targets' in col or 'elite_yards' in col:
            feature_weights[i] = 2.5  # Increase elite indicators
        elif 'strong_momentum' in col:
            feature_weights[i] = 3.5  # Increase strong momentum indicators
        elif 'positive_trend' in col:
            feature_weights[i] = 2.0  # Increase positive trend indicators
        elif 'high_volume' in col and 'recent' in col:
            feature_weights[i] = 2.0  # Increase recent volume indicators
        elif 'is_recent_season' in col or 'very_recent' in col:
            feature_weights[i] = 1.5  # Increase recent season indicators
    
    # Apply weights to features
    X_train_weighted = X_train * feature_weights
    X_test_weighted = X_test * feature_weights
    
    # Train Ridge model with custom weights
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_weighted, y_train)
    
    # Create a custom model that applies weights during prediction
    class WeightedRidgeModel:
        def __init__(self, ridge_model, weights):
            self.ridge = ridge_model
            self.weights = weights
            
        def predict(self, X):
            X_weighted = X * self.weights
            return self.ridge.predict(X_weighted)
    
    weighted_ridge = WeightedRidgeModel(ridge, feature_weights)
    models.append(weighted_ridge)
    
    # 2. Standard Ridge Regression
    ridge_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3,3,13), cv=5))
    ])
    models.append(ridge_model)
    
    # 3. Elastic Net (handles feature selection)
    elastic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("elastic", ElasticNetCV(cv=5, random_state=42))
    ])
    models.append(elastic_model)
    
    # 4. Random Forest (captures non-linear patterns)
    rf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
    ])
    models.append(rf_model)

    # Train individual models and evaluate
    print("\nIndividual Model Performance:")
    individual_scores = []
    model_names = ["Fixed-Weighted Ridge", "Standard Ridge", "Elastic Net", "Random Forest"]
    
    for i, model in enumerate(models):
        if i == 0:  # Fixed-Weighted Ridge already trained
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
        
        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)
        mae_test = mean_absolute_error(y_test, pred_test)
        
        individual_scores.append(r2_test)
        print(f"{model_names[i]}: Train R²={r2_train:.3f}, Test R²={r2_test:.3f}, Test MAE={mae_test:.3f}")

    # Use the best model
    best_model_idx = np.argmax(individual_scores)
    best_model = models[best_model_idx]
    
    # Evaluate best model
    if best_model_idx == 0:  # Fixed-Weighted Ridge
        pred_train_best = best_model.predict(X_train)
        pred_test_best = best_model.predict(X_test)
    else:
        pred_train_best = best_model.predict(X_train)
        pred_test_best = best_model.predict(X_test)

    mae_tr = mean_absolute_error(y_train, pred_train_best)
    rmse_tr = np.sqrt(mean_squared_error(y_train, pred_train_best))
    r2_tr = r2_score(y_train, pred_train_best)

    mae_te = mean_absolute_error(y_test, pred_test_best)
    rmse_te = np.sqrt(mean_squared_error(y_test, pred_test_best))
    r2_te = r2_score(y_test, pred_test_best)

    print(f"\n== Best Model Performance ({model_names[best_model_idx]}) ==")
    print("== Training ==")
    print(f"MAE:  {mae_tr:.3f}  RMSE: {rmse_tr:.3f}  R^2: {r2_tr:.3f}")
    print("== Test ==")
    print(f"MAE:  {mae_te:.3f}  RMSE: {rmse_te:.3f}  R^2: {r2_te:.3f}")

    # Feature importance from best model
    if best_model_idx == 0:  # Fixed-Weighted Ridge
        feature_importance = pd.DataFrame({
            'feature': X_cols,
            'coefficient': ridge.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
    elif hasattr(best_model.named_steps[list(best_model.named_steps.keys())[-1]], 'coef_'):
        # Linear model
        coef = best_model.named_steps[list(best_model.named_steps.keys())[-1]].coef_
        feature_importance = pd.DataFrame({
            'feature': X_cols,
            'coefficient': coef
        }).sort_values('coefficient', key=abs, ascending=False)
    else:
        # Tree-based model
        importances = best_model.named_steps[list(best_model.named_steps.keys())[-1]].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_cols,
            'coefficient': importances
        }).sort_values('coefficient', key=abs, ascending=False)

    print(f"\nTop 10 most important features (from best model):")
    print(feature_importance.head(10))

    # Save artifacts (simplified to avoid pickling issues)
    joblib.dump({
        "model": best_model,
        "X_cols": X_cols,
        "y_col": y_col,
        "feature_info": {
            "base_metrics": base,
            "uses_lag1": True,
            "uses_roll3": True,
            "uses_roll5": True,
            "uses_roll8": True,
            "uses_momentum": True,
            "uses_recent_season": True,
            "uses_recent_surge": True,
            "uses_strong_momentum": True,
            "uses_positive_trends": True,
            "uses_fixed_elite_detection": True,
            "emphasizes_recent_data": True,
            "reduces_old_data_influence": True,
            "notes": "Fixed WR/TE model that properly rewards recent surges and momentum."
        }
    }, outdir / "wrte_fixed_model.joblib")

    # Residual std (global) for Monte Carlo noise
    residuals = y_train - pred_train_best
    resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 5.0
    joblib.dump({"residual_std": resid_std}, outdir / "wrte_fixed_residual_std.joblib")

    # Keep a clean dataset with features so the predictor can recompute for a player
    dfm.to_parquet(outdir / "wrte_fixed_training_table.parquet", index=False)
    print(f"\nSaved fixed model + artifacts to {outdir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with weekly receiving CSVs")
    parser.add_argument("--outfile_dir", default="./artifacts_fixed", help="Where to save model & metadata")
    parser.add_argument("--min_games", type=int, default=3, help="Min prior games for rolling features")
    parser.add_argument("--test_weeks", type=int, default=4, help="Use last N weeks as test set")
    args = parser.parse_args()
    main(args)
