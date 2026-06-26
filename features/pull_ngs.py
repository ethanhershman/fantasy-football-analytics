"""
Bucket A: pull Next Gen Stats from nfl-data-py (2016+).

Source: nfl.import_ngs_data(stat_type=...) for passing, rushing, receiving.
Output: data/features/ngs.parquet, keyed on (player_id, season).

Features extracted per (player_id, season):
  Passing:   avg_time_to_throw, avg_intended_air_yards, aggressiveness,
             completion_percentage_above_expectation (cpoe)
  Rushing:   efficiency, percent_attempts_gte_eight_defenders
  Receiving: avg_separation, avg_cushion, avg_intended_air_yards
"""

from pathlib import Path

import pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2016, 2026))
OUT_PATH = Path(__file__).parent.parent / "data" / "features" / "ngs.parquet"


def _weighted_season_avg(df: pd.DataFrame, value_cols: list[str], weight_col: str) -> pd.DataFrame:
    """
    Aggregate weekly NGS rows to (player_id, season) using a weighted average.
    Weight is the volume column (attempts, targets, rush_attempts) so that a
    heavy-workload week contributes more than a 1-snap appearance.
    """
    df = df.copy()
    for col in value_cols:
        if col in df.columns:
            df[f"_w_{col}"] = df[col] * df[weight_col]

    weighted_cols = [f"_w_{col}" for col in value_cols if col in df.columns]
    agg = df.groupby(["player_id", "season"]).agg(
        {weight_col: "sum", **{c: "sum" for c in weighted_cols}}
    ).reset_index()

    for col in value_cols:
        wcol = f"_w_{col}"
        if wcol in agg.columns:
            agg[col] = (agg[wcol] / agg[weight_col]).where(agg[weight_col] > 0)
            agg = agg.drop(columns=[wcol])

    return agg.drop(columns=[weight_col])


def build_ngs_features(years: list[int]) -> pd.DataFrame:
    """
    Fetch NGS data for all three stat types, aggregate weekly rows to season
    level via weighted averages, and outer-join into a single DataFrame keyed
    on (player_id, season).
    """
    frames = {}

    # --- Passing ---
    passing = nfl.import_ngs_data(stat_type="passing", years=years)
    passing = passing[passing["season_type"] == "REG"].rename(
        columns={"player_gsis_id": "player_id"}
    )
    frames["passing"] = _weighted_season_avg(
        passing,
        value_cols=[
            "avg_time_to_throw",
            "avg_intended_air_yards",
            "aggressiveness",
            "completion_percentage_above_expectation",
        ],
        weight_col="attempts",
    )

    # --- Rushing ---
    rushing = nfl.import_ngs_data(stat_type="rushing", years=years)
    rushing = rushing[rushing["season_type"] == "REG"].rename(
        columns={"player_gsis_id": "player_id"}
    )
    frames["rushing"] = _weighted_season_avg(
        rushing,
        value_cols=[
            "efficiency",
            "percent_attempts_gte_eight_defenders",
        ],
        weight_col="rush_attempts",
    )

    # --- Receiving ---
    receiving = nfl.import_ngs_data(stat_type="receiving", years=years)
    receiving = receiving[receiving["season_type"] == "REG"].rename(
        columns={"player_gsis_id": "player_id"}
    )
    frames["receiving"] = _weighted_season_avg(
        receiving,
        value_cols=[
            "avg_separation",
            "avg_cushion",
            "avg_intended_air_yards",
        ],
        weight_col="targets",
    )

    # Outer-join so a player who only appears in one stat type still gets a row.
    result = frames["passing"]
    for key in ("rushing", "receiving"):
        result = result.merge(frames[key], on=["player_id", "season"], how="outer", suffixes=("", f"_{key}"))

    return result.dropna(subset=["player_id", "season"]).reset_index(drop=True)


def main():
    df = build_ngs_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
