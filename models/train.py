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

Evaluation metrics:
  MAE, Pearson r     — standard regression metrics across all players
  Precision@K        — % of top-K predictions that were actual top-K finishers
  Spearman ρ@K       — rank correlation within the actual top-K (did we order the studs right?)
  MAE@K              — PPG error restricted to actual top-K finishers

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

# K = "top-K finishers" threshold per position.
# Matches standard fantasy roster construction (1-QB, 2-RB, 2-WR, 1-TE + flex).
TOP_K = {"QB": 8, "RB": 12, "WR": 12, "TE": 8}


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
    test_seasons = test["season"].reset_index(drop=True)
    return X_train, y_train, X_test, y_test, test_seasons


def _spearman(a, b) -> float:
    a_r = pd.Series(a).rank()
    b_r = pd.Series(b).rank()
    return float(np.corrcoef(a_r, b_r)[0, 1])


def rank_metrics(y_true: pd.Series, y_pred: np.ndarray, seasons: pd.Series, k: int) -> dict:
    """
    Compute fantasy-relevant ranking metrics averaged across test seasons.

    Precision@K  — fraction of predicted top-K who actually finished top-K
                   (did we identify the right elite players?)
    Spearman ρ@K — rank correlation among actual top-K finishers
                   (did we correctly order the studs we found?)
    MAE@K        — PPG error restricted to actual top-K finishers
                   (how wrong are we about the players that matter most?)

    Each metric is computed per season then averaged, so a single fluky season
    doesn't dominate.
    """
    df = pd.DataFrame({
        "y_true":  y_true.values,
        "y_pred":  y_pred,
        "season":  seasons.values,
    })

    p_list, rho_list, mae_list = [], [], []

    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s].copy().reset_index(drop=True)
        if len(sub) < k:
            continue

        sub["true_rank"] = sub["y_true"].rank(ascending=False, method="min")
        sub["pred_rank"] = sub["y_pred"].rank(ascending=False, method="min")

        predicted_top = set(sub[sub["pred_rank"] <= k].index)
        actual_top    = set(sub[sub["true_rank"] <= k].index)

        p_list.append(len(predicted_top & actual_top) / k)

        top_actual = sub.loc[list(actual_top)]
        rho_list.append(_spearman(top_actual["pred_rank"], top_actual["true_rank"]))
        mae_list.append(mean_absolute_error(top_actual["y_true"], top_actual["y_pred"]))

    return {
        f"precision_at_{k}": round(float(np.mean(p_list)), 3),
        f"spearman_at_{k}":  round(float(np.mean(rho_list)), 3),
        f"mae_at_{k}":       round(float(np.mean(mae_list)), 3),
    }


def eval_model(y_true, y_pred, seasons, label: str, k: int) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rm   = rank_metrics(y_true, y_pred, seasons, k)
    prec = rm[f"precision_at_{k}"]
    rho  = rm[f"spearman_at_{k}"]
    maek = rm[f"mae_at_{k}"]
    print(f"    {label:<14s}  MAE={mae:.2f}  r={corr:.3f}  |  "
          f"P@{k}={prec:.2f}  ρ@{k}={rho:.2f}  MAE@{k}={maek:.2f}")
    return {"mae": round(mae, 3), "r": round(float(corr), 3), **rm}


def train_position(pos: str) -> dict:
    print(f"\n{'='*72}")
    print(f"  {pos}")
    print(f"{'='*72}")

    k  = TOP_K[pos]
    df = load_position(pos)
    features = get_features(df)
    X_train, y_train, X_test, y_test, test_seasons = split(df, features)
    y_test = y_test.reset_index(drop=True)

    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
          f"Features: {len(features)} | K={k}")

    print("  Test metrics:")

    # --- ADP-only baseline ---
    adp_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler",  StandardScaler().set_output(transform="pandas")),
        ("model",   Ridge(alpha=1.0)),
    ])
    adp_pipe.fit(X_train[ADP_COLS], y_train)
    adp_pred    = adp_pipe.predict(X_test[ADP_COLS])
    adp_metrics = eval_model(y_test, adp_pred, test_seasons, "ADP only", k)

    # --- Ridge (full features) ---
    ridge_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler",  StandardScaler().set_output(transform="pandas")),
        ("model",   Ridge(alpha=10.0)),
    ])
    ridge_pipe.fit(X_train, y_train)
    ridge_pred    = ridge_pipe.predict(X_test)
    ridge_metrics = eval_model(y_test, ridge_pred, test_seasons, "Ridge (full)", k)

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
    xgb_pred    = xgb_pipe.predict(X_test)
    xgb_metrics = eval_model(y_test, xgb_pred, test_seasons, "XGBoost", k)

    # --- Feature importance ---
    xgb_model  = xgb_pipe.named_steps["model"]
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
        "k": k,
        "adp":   adp_metrics,
        "ridge": ridge_metrics,
        "xgb":   xgb_metrics,
        "top_features": top_features.to_dict(),
    }


def main():
    results = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        results[pos] = train_position(pos)

    print(f"\n{'='*72}")
    print("  Summary — all models, all positions (averaged over 2024-2025 test seasons)")
    print(f"{'='*72}")
    hdr = f"  {'Pos':<4}  {'K':>2}  {'Model':<14}  {'MAE':>5}  {'r':>5}  {'P@K':>5}  {'ρ@K':>5}  {'MAE@K':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pos, r in results.items():
        k = r["k"]
        for mname, m in [("ADP only", r["adp"]), ("Ridge", r["ridge"]), ("XGBoost", r["xgb"])]:
            print(f"  {pos:<4}  {k:>2}  {mname:<14}  "
                  f"{m['mae']:>5.2f}  {m['r']:>5.3f}  "
                  f"{m[f'precision_at_{k}']:>5.2f}  "
                  f"{m[f'spearman_at_{k}']:>5.2f}  "
                  f"{m[f'mae_at_{k}']:>6.2f}")

    with open(ARTIFACTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
