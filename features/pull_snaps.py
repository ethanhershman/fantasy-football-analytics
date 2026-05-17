"""
Bucket A: pull weekly snap counts from nfl-data-py and aggregate to season level.

Source: nfl.import_snap_counts(years) — available 2012+.
Output: data/features/snaps.parquet, keyed on (player_id, season).

Features extracted per (player_id, season):
  snap_pct  (offense_snaps / team_offense_snaps, season average)
  offense_snaps  (season total)
"""

from pathlib import Path

import pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2016, 2026))
OUT_PATH = Path(__file__).parent.parent / "data" / "features" / "snaps.parquet"


def build_snap_features(years: list[int]) -> pd.DataFrame:
    """
    Fetch weekly snap counts, aggregate to (player_id, season) by summing
    offense_snaps and team_offense_snaps, then compute season snap_pct.

    Returns a DataFrame with columns: player_id, season, offense_snaps, snap_pct.
    """
    raise NotImplementedError


def main():
    df = build_snap_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
