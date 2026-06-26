"""
Phase 6: Train per-position regression models to predict current-season PPG.

Features (known at draft time):
  prev_*          — prior-season stats and advanced features
  adp_overall     — consensus draft position
  adp_position_rank
  career_season   — how many NFL seasons the player has played

Target: ppg (fantasy points per game, current season)

Split: train 2016-2023, test 2024-2025 (time-based, no leakage)

Models per position:
  Ridge   — interpretable baseline
  XGBoost — main model

Artifacts saved to models/artifacts/:
  {pos}_ridge.joblib
  {pos}_xgb.joblib
  {pos}_features.json
"""

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Skipping features without any observed values")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

DATA_DIR      = Path(__file__).parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

TRAIN_SEASONS = list(range(2016, 2024))
TEST_SEASONS  = [2024, 2025]

CONTEXT_COLS = ["adp_overall", "adp_position_rank", "career_season"]
ADP_COLS     = ["adp_overall", "adp_position_rank"]


def load_position(pos: str) -> pd.DataFrame:
    path = DATA_DIR / f"training_data_{pos.lower()}.csv"
    df = pd.read_csv(path)
    return df.dropna(subset=["ppg"])


def get_features(df: pd.DataFrame) -> list[str]:
    prev_cols = [c for c in df.columns if c.startswith("prev_")]
    return CONTEXT_COLS + prev_cols


def split(df: pd.DataFrame, features: list[str]):
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test  = df[df["season"].isin(TEST_SEASONS)]
    X_train = train[features]
    y_train = train["ppg"]
    X_test  = test[features]
    y_test  = test["ppg"]
    return X_train, y_train, X_test, y_test, test


def eval_metrics(y_true, y_pred, label: str):
    mae  = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"    {label:<14s}  MAE={mae:.2f}  r={corr:.3f}")
    return {"mae": round(mae, 3), "r": round(float(corr), 3)}


def train_position(pos: str) -> dict:
    print(f"\n{'='*56}")
    print(f"  {pos}")
    print(f"{'='*56}")

    df       = load_position(pos)
    features = get_features(df)
    X_train, y_train, X_test, y_test, _ = split(df, features)

    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {len(features)}")

    # --- ADP-only baseline (how well does just drafting by ADP do?) ---
    adp_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler",  StandardScaler().set_output(transform="pandas")),
        ("model",   Ridge(alpha=1.0)),
    ])
    adp_pipe.fit(X_train[ADP_COLS], y_train)
    adp_pred = adp_pipe.predict(X_test[ADP_COLS])
    print("  Test metrics:")
    adp_metrics = eval_metrics(y_test, adp_pred, "ADP only")

    # --- Ridge (full features) ---
    ridge_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler",  StandardScaler().set_output(transform="pandas")),
        ("model",   Ridge(alpha=10.0)),
    ])
    ridge_pipe.fit(X_train, y_train)
    ridge_pred = ridge_pipe.predict(X_test)
    ridge_metrics = eval_metrics(y_test, ridge_pred, "Ridge (full)")

    # --- XGBoost ---
    xgb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("model",   XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    xgb_pred = xgb_pipe.predict(X_test)
    xgb_metrics = eval_metrics(y_test, xgb_pred, "XGBoost")

    # --- Feature importance (XGBoost built-in) ---
    xgb_model = xgb_pipe.named_steps["model"]
    feat_names = list(xgb_model.feature_names_in_)
    top_features = (
        pd.Series(xgb_model.feature_importances_, index=feat_names)
        .sort_values(ascending=False)
        .head(10)
    )
    print(f"\n  Top 10 features by XGBoost importance:")
    for feat, val in top_features.items():
        print(f"    {feat:<45s} {val:.3f}")

    # --- Save artifacts ---
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_pipe, ARTIFACTS_DIR / f"{pos.lower()}_ridge.joblib")
    joblib.dump(xgb_pipe,   ARTIFACTS_DIR / f"{pos.lower()}_xgb.joblib")
    with open(ARTIFACTS_DIR / f"{pos.lower()}_features.json", "w") as f:
        json.dump(features, f, indent=2)

    return {
        "pos": pos,
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "n_features": len(features),
        "adp":   adp_metrics,
        "ridge": ridge_metrics,
        "xgb":   xgb_metrics,
        "top_features": top_features.to_dict(),
    }


def main():
    results = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        results[pos] = train_position(pos)

    print(f"\n{'='*56}")
    print("  Summary")
    print(f"{'='*56}")
    print(f"  {'Pos':<4}  {'ADP MAE':>8}  {'ADP r':>6}  {'Ridge MAE':>10}  {'Ridge r':>8}  {'XGB MAE':>8}  {'XGB r':>6}")
    for pos, r in results.items():
        print(f"  {pos:<4}  {r['adp']['mae']:>8.2f}  {r['adp']['r']:>6.3f}"
              f"  {r['ridge']['mae']:>10.2f}  {r['ridge']['r']:>8.3f}"
              f"  {r['xgb']['mae']:>8.2f}  {r['xgb']['r']:>6.3f}")

    with open(ARTIFACTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
