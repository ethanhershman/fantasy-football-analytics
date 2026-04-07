"""
Build the training dataset by joining stats + ADP.

Output columns per player-season:
  name, position, team, season,
  ecr_overall, ecr_position,
  actual_finish_position, actual_ppg, total_points,
  prev_total_points, prev_games, prev_<counting stats>...

Training rows: 2021-2024 (need prior year for lag features).
Prediction rows: 2026 (DraftSharks ECR + 2024 stats as "previous year").

Run: python ingestion/build_training_data.py
"""

import pandas as pd
from sqlalchemy import text
from db import get_engine

COUNTING_STATS = [
    "completions", "pass_attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
]


def load_tables(engine):
    with engine.connect() as conn:
        players = pd.read_sql("SELECT * FROM players", conn)
        stats = pd.read_sql("SELECT * FROM season_stats", conn)
        adp = pd.read_sql("SELECT * FROM adp", conn)
    return players, stats, adp


def build_training(players, stats, adp):
    """Build historical training rows (2021-2024)."""
    # Merge stats with player info
    df = stats.merge(players[["player_id", "full_name", "position", "team"]], on="player_id")

    # Compute actual position finish by season (rank by total fantasy points within position)
    df["actual_finish_position"] = (
        df.groupby(["season", "position"])["fantasy_points_ppr"]
        .rank(ascending=False, method="min").astype("Int64")
    )
    df["actual_ppg"] = (df["fantasy_points_ppr"] / df["games_played"]).round(2)

    # Add ECR from FFC ADP
    ffc = adp[adp["source"] == "ffc"][["player_id", "season", "adp_overall", "adp_position_rank"]]
    df = df.merge(
        ffc.rename(columns={"adp_overall": "ecr_overall", "adp_position_rank": "ecr_position"}),
        on=["player_id", "season"],
        how="left",
    )

    # Build previous-year features via self-join on season-1
    prev = stats.copy()
    prev["season"] = prev["season"] + 1  # shift so it aligns with "current" season
    prev_cols = {c: f"prev_{c}" for c in ["fantasy_points_ppr", "games_played"] + COUNTING_STATS}
    prev = prev.rename(columns=prev_cols)
    prev = prev[["player_id", "season"] + list(prev_cols.values())]

    df = df.merge(prev, on=["player_id", "season"], how="left")

    # Only keep seasons where we have a prior year (2021-2024)
    df = df[df["season"].between(2021, 2024)]

    return df


def build_prediction(players, stats, adp):
    """Build 2026 prediction rows using DraftSharks ECR + 2024 stats."""
    ds = adp[adp["source"] == "draftsharks"][["player_id", "adp_overall", "adp_position_rank"]]
    ds = ds.rename(columns={"adp_overall": "ecr_overall", "adp_position_rank": "ecr_position"})

    df = ds.merge(players[["player_id", "full_name", "position", "team"]], on="player_id")
    df["season"] = 2026

    # Previous year = 2024 stats
    prev = stats[stats["season"] == 2024].copy()
    prev_cols = {c: f"prev_{c}" for c in ["fantasy_points_ppr", "games_played"] + COUNTING_STATS}
    prev = prev.rename(columns=prev_cols)
    prev = prev[["player_id"] + list(prev_cols.values())]
    df = df.merge(prev, on="player_id", how="left")

    return df


def main():
    engine = get_engine()
    players, stats, adp = load_tables(engine)

    train = build_training(players, stats, adp)
    pred = build_prediction(players, stats, adp)

    # Standardize column order
    common = [
        "full_name", "position", "team", "season",
        "ecr_overall", "ecr_position",
        "actual_finish_position", "actual_ppg", "fantasy_points_ppr",
        "prev_fantasy_points_ppr", "prev_games_played",
    ] + [f"prev_{c}" for c in COUNTING_STATS]

    train_out = train[[c for c in common if c in train.columns]]
    pred_out = pred[[c for c in common if c in pred.columns]]

    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_out.to_csv(os.path.join(out_dir, "training_data.csv"), index=False)
    pred_out.to_csv(os.path.join(out_dir, "prediction_2026.csv"), index=False)

    print(f"Training data: {len(train_out)} rows → data/training_data.csv")
    print(f"  Seasons: {sorted(train_out['season'].unique())}")
    print(f"  Columns: {list(train_out.columns)}")
    print(f"\nPrediction data: {len(pred_out)} rows → data/prediction_2026.csv")


if __name__ == "__main__":
    main()
