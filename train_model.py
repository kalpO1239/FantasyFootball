# train_model.py
# Usage:
#   python train_model.py --data_dir ./receiving_csvs --outfile_dir ./artifacts --min_games 3 --test_weeks 4
#
# Expects many CSVs (one per week) in --data_dir with headers like:
# PLAYER,TEAM,POS,CUSH,SEP,TAY,TAY%,REC,TAR,CTCH%,YDS,TD,YAC/R,xYAC/R,+/-
#
# Trains for WR/TE only, on weekly rows. Builds lagged/rolling features so you can predict *next* week.

import argparse
import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")
    frames = []
    for week_idx, f in enumerate(files, start=1):
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
        df["WEEK"] = week_idx
        frames.append(df[wanted + ["WEEK"]])
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

def add_lagged_features(df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """
    Build rolling features per player so we predict *next* week using past k games.
    We’ll compute 1-game lag and 3-game rolling means for core signals.
    """
    df = df.sort_values(["PLAYER","WEEK"]).copy()

    feat_cols = ["SEP","TAY","TAY%","REC","TAR","YDS","TD","YAC/R"]
    group = df.groupby("PLAYER", group_keys=False)

    # 1-week lags
    for c in feat_cols:
        df[f"{c}_lag1"] = group[c].shift(1)

    # 3-game rolling means (only when at least 3 prior games exist)
    for c in feat_cols:
        df[f"{c}_roll3"] = group[c].shift(1).rolling(3, min_periods=min_games).mean()

    # Target is *this week’s* PPR
    df["PPR"] = compute_target_ppr(df)

    # We will predict PPR using ONLY lagged/rolling features (no leakage)
    # Drop rows without enough history
    need_cols = [f"{c}_lag1" for c in feat_cols] + [f"{c}_roll3" for c in feat_cols]
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
    dfm  = add_lagged_features(data, min_games=args.min_games)

    # Feature set (easy to expand later with matchup/QB/weather)
    base = ["SEP","TAY","TAY%","REC","TAR","YDS","TD","YAC/R"]
    X_cols = [f"{c}_lag1" for c in base] + [f"{c}_roll3" for c in base]
    y_col  = "PPR"

    train, test = time_based_split(dfm, test_weeks=args.test_weeks)
    X_train, y_train = train[X_cols].values, train[y_col].values
    X_test,  y_test  = test[X_cols].values,  test[y_col].values

    # Model: RidgeCV (stable, interpretable, avoids overfit)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3,3,13), cv=5))
    ])
    model.fit(X_train, y_train)

    # Simple evaluation
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    mae_tr = mean_absolute_error(y_train, pred_train)
    rmse_tr = np.sqrt(mean_squared_error(y_train, pred_train))
    r2_tr = r2_score(y_train, pred_train)

    mae_te = mean_absolute_error(y_test, pred_test)
    rmse_te = np.sqrt(mean_squared_error(y_test, pred_test))
    rmse_te = np.sqrt(mean_squared_error(y_train, pred_train))
    r2_te = r2_score(y_test, pred_test)

    print("== Training ==")
    print(f"MAE:  {mae_tr:.3f}  RMSE: {rmse_tr:.3f}  R^2: {r2_tr:.3f}")
    print("== Test ==")
    print(f"MAE:  {mae_te:.3f}  RMSE: {rmse_te:.3f}  R^2: {r2_te:.3f}")

    # Save artifacts
    joblib.dump({
        "model": model,
        "X_cols": X_cols,
        "y_col": y_col,
        "feature_info": {
            "base_metrics": base,
            "uses_lag1": True,
            "uses_roll3": True,
            "notes": "WR/TE only; future-ready for matchup/QB/weather."
        }
    }, outdir / "wrte_model.joblib")

    # Residual std (global) for Monte Carlo noise
    residuals = y_train - pred_train
    resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 5.0
    joblib.dump({"residual_std": resid_std}, outdir / "wrte_residual_std.joblib")

    # Keep a clean dataset with features so the predictor can recompute for a player
    dfm.to_parquet(outdir / "wrte_training_table.parquet", index=False)
    print(f"Saved model + artifacts to {outdir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with weekly receiving CSVs")
    parser.add_argument("--outfile_dir", default="./artifacts", help="Where to save model & metadata")
    parser.add_argument("--min_games", type=int, default=3, help="Min prior games for rolling features")
    parser.add_argument("--test_weeks", type=int, default=4, help="Use last N weeks as test set")
    args = parser.parse_args()
    main(args)
