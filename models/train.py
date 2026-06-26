"""
Phase 6: Predict ADP residuals — how much does a player over/underperform their draft slot?

Core idea:
  ADP already prices in most public information (prior stats, injury history, team
  context). A model that predicts raw PPG is mostly re-learning what ADP knows.

  Instead we:
    1. Fit a per-position ADP baseline:  expected_ppg = f(adp_position_rank, adp_overall)
    2. Compute training residuals:       adp_residual = actual_ppg - expected_ppg
    3. Train Ridge + XGBoost to predict adp_residual using prev_* advanced stats
       (ADP columns are excluded — already factored out by step 1)
    4. Final prediction:                 expected_ppg + predicted_residual
    5. Rank players by final prediction and evaluate with fantasy-relevant metrics

  Feature importances now answer "what did ADP miss?" rather than "what predicts PPG?"
  which is the only edge worth having.

Split:     train 2016-2023, test 2024-2025 (time-based, no leakage)
Target:    adp_residual = ppg - adp_expected_ppg
Features:  prev_* columns + career_season  (ADP excluded — in the baseline)

Evaluation metrics (averaged across 2024 and 2025 separately):
  MAE, Pearson r     — overall accuracy of final PPG prediction
  Precision@K        — % of top-K predictions that were actual top-K finishers
  Spearman ρ@K       — rank correlation within actual top-K (did we order the studs right?)
  MAE@K              — PPG error restricted to actual top-K finishers

Artifacts saved to models/artifacts/:
  {pos}_adp.joblib       — ADP baseline (used to compute expected PPG)
  {pos}_ridge.joblib     — Ridge residual model
  {pos}_xgb.joblib       — XGBoost residual model
  {pos}_features.json    — feature list for residual models
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

ADP_COLS  = ["adp_overall", "adp_position_rank"]
META_COLS = ["career_season"]   # known at draft time but not in ADP

# K = elite tier threshold per position (standard 1-QB/2-RB/2-WR/1-TE roster)
TOP_K = {"QB": 8, "RB": 12, "WR": 12, "TE": 8}


def load_position(pos: str) -> pd.DataFrame:
    path = DATA_DIR / f"training_data_{pos.lower()}.csv"
    df = pd.read_csv(path)
    return df.dropna(subset=["ppg"])


def get_resid_features(df: pd.DataFrame) -> list[str]:
    """Features for the residual model: prev_* stats + career_season, no ADP."""
    prev_cols = [c for c in df.columns if c.startswith("prev_")]
    return META_COLS + prev_cols


def split(df: pd.DataFrame, adp_features: list[str], resid_features: list[str]):
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test  = df[df["season"].isin(TEST_SEASONS)]

    X_adp_train  = train[adp_features]
    X_adp_test   = test[adp_features]
    X_res_train  = train[resid_features]
    X_res_test   = test[resid_features]
    y_train      = train["ppg"].reset_index(drop=True)
    y_test       = test["ppg"].reset_index(drop=True)
    seasons_test = test["season"].reset_index(drop=True)

    return (X_adp_train, X_adp_test,
            X_res_train, X_res_test,
            y_train, y_test, seasons_test)


def _spearman(a, b) -> float:
    a_r = pd.Series(a).rank()
    b_r = pd.Series(b).rank()
    return float(np.corrcoef(a_r, b_r)[0, 1])


def rank_metrics(y_true: pd.Series, y_pred: np.ndarray,
                 seasons: pd.Series, k: int) -> dict:
    """
    Precision@K  — fraction of predicted top-K who actually finished top-K
    Spearman ρ@K — rank correlation among actual top-K finishers
    MAE@K        — PPG error restricted to actual top-K finishers

    Each metric computed per season then averaged.
    """
    df = pd.DataFrame({
        "y_true": y_true.values,
        "y_pred": y_pred,
        "season": seasons.values,
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
    print(f"    {label:<22s}  MAE={mae:.2f}  r={corr:.3f}  |  "
          f"P@{k}={prec:.2f}  ρ@{k}={rho:.2f}  MAE@{k}={maek:.2f}")
    return {"mae": round(mae, 3), "r": round(float(corr), 3), **rm}


def train_position(pos: str) -> dict:
    print(f"\n{'='*76}")
    print(f"  {pos}")
    print(f"{'='*76}")

    k            = TOP_K[pos]
    df           = load_position(pos)
    resid_feats  = get_resid_features(df)

    (X_adp_train, X_adp_test,
     X_res_train, X_res_test,
     y_train, y_test, seasons_test) = split(df, ADP_COLS, resid_feats)

    print(f"  Train: {len(y_train)} rows | Test: {len(y_test)} rows | "
          f"Residual features: {len(resid_feats)} | K={k}")

    # -----------------------------------------------------------------------
    # Step 1: ADP baseline — expected PPG given draft position
    # -----------------------------------------------------------------------
    adp_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler",  StandardScaler().set_output(transform="pandas")),
        ("model",   Ridge(alpha=1.0)),
    ])
    adp_pipe.fit(X_adp_train, y_train)
    adp_expected_train = adp_pipe.predict(X_adp_train)
    adp_expected_test  = adp_pipe.predict(X_adp_test)

    # -----------------------------------------------------------------------
    # Step 2: Residuals — how much did each player over/underperform ADP?
    # -----------------------------------------------------------------------
    y_resid_train = pd.Series(y_train.values - adp_expected_train)

    print(f"\n  ADP residual stats (train): "
          f"mean={y_resid_train.mean():.2f}  "
          f"std={y_resid_train.std():.2f}  "
          f"range=[{y_resid_train.min():.1f}, {y_resid_train.max():.1f}]")

    print("\n  Test metrics (final PPG = adp_expected + residual_pred):")

    # ADP-only baseline (no residual correction) — the floor to beat
    adp_metrics = eval_model(y_test, adp_expected_test, seasons_test, "ADP baseline", k)

    # -----------------------------------------------------------------------
    # Step 3a: Ridge residual model
    # -----------------------------------------------------------------------
    ridge_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler",  StandardScaler().set_output(transform="pandas")),
        ("model",   Ridge(alpha=10.0)),
    ])
    ridge_pipe.fit(X_res_train, y_resid_train)
    ridge_final   = adp_expected_test + ridge_pipe.predict(X_res_test)
    ridge_metrics = eval_model(y_test, ridge_final, seasons_test, "ADP + Ridge resid", k)

    # -----------------------------------------------------------------------
    # Step 3b: XGBoost residual model
    # -----------------------------------------------------------------------
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
    xgb_pipe.fit(X_res_train, y_resid_train)
    xgb_final   = adp_expected_test + xgb_pipe.predict(X_res_test)
    xgb_metrics = eval_model(y_test, xgb_final, seasons_test, "ADP + XGB resid", k)

    # -----------------------------------------------------------------------
    # Feature importance — "what did ADP miss?"
    # -----------------------------------------------------------------------
    xgb_model  = xgb_pipe.named_steps["model"]
    feat_names = list(xgb_model.feature_names_in_)
    top_features = (
        pd.Series(xgb_model.feature_importances_, index=feat_names)
        .sort_values(ascending=False)
        .head(10)
    )
    print(f"\n  Top 10 features XGBoost used to correct ADP:")
    for feat, val in top_features.items():
        print(f"    {feat:<45s} {val:.3f}")

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(adp_pipe,   ARTIFACTS_DIR / f"{pos.lower()}_adp.joblib")
    joblib.dump(ridge_pipe, ARTIFACTS_DIR / f"{pos.lower()}_ridge.joblib")
    joblib.dump(xgb_pipe,   ARTIFACTS_DIR / f"{pos.lower()}_xgb.joblib")
    with open(ARTIFACTS_DIR / f"{pos.lower()}_features.json", "w") as f:
        json.dump({"adp": ADP_COLS, "residual": resid_feats}, f, indent=2)

    return {
        "pos": pos,
        "n_train": len(y_train),
        "n_test":  len(y_test),
        "n_resid_features": len(resid_feats),
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

    print(f"\n{'='*76}")
    print("  Summary — test seasons 2024-2025")
    print(f"{'='*76}")
    hdr = f"  {'Pos':<4}  {'K':>2}  {'Model':<22}  {'MAE':>5}  {'r':>5}  {'P@K':>5}  {'ρ@K':>5}  {'MAE@K':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pos, r in results.items():
        k = r["k"]
        for mname, m in [("ADP baseline", r["adp"]),
                         ("ADP + Ridge resid", r["ridge"]),
                         ("ADP + XGB resid",   r["xgb"])]:
            print(f"  {pos:<4}  {k:>2}  {mname:<22}  "
                  f"{m['mae']:>5.2f}  {m['r']:>5.3f}  "
                  f"{m[f'precision_at_{k}']:>5.2f}  "
                  f"{m[f'spearman_at_{k}']:>5.2f}  "
                  f"{m[f'mae_at_{k}']:>6.2f}")

    with open(ARTIFACTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
