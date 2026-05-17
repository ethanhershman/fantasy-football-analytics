"""
Bucket A: pull pre-aggregated seasonal advanced stats from nfl-data-py.

Source: nfl.import_seasonal_data(years) — no pbp aggregation required.
Output: data/features/seasonal.parquet, keyed on (player_id, season).

Features extracted per (player_id, season):
  target_share, air_yards_share, wopr, racr, ppr_sh
  targets, air_yards, yards_after_catch
  adot  (computed: air_yards / targets)
"""

from pathlib import Path

import pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2016, 2026))
OUT_PATH = Path(__file__).parent.parent / "data" / "features" / "seasonal.parquet"

KEEP_COLS = [
    "player_id",
    "season",
    "target_share",
    "air_yards_share",
    "wopr",
    "racr",
    "ppr_sh",
    "targets",
    "air_yards",
    "yards_after_catch",
]


def build_seasonal_features(years: list[int]) -> pd.DataFrame:
    """
    Fetch seasonal advanced stats and return a DataFrame keyed on
    (player_id, season) with the columns listed in KEEP_COLS plus adot.

    adot is derived as air_yards / targets; rows with targets == 0 get NaN.
    """
    raise NotImplementedError


def main():
    df = build_seasonal_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
