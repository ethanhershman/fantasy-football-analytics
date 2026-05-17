"""
Bucket A: pull Next Gen Stats from nfl-data-py (2016+).

Source: nfl.import_ngs_data(stat_type=...) for passing, rushing, receiving.
Output: data/features/ngs.parquet, keyed on (player_id, season).

Features extracted per (player_id, season):
  Passing:   avg_time_to_throw, avg_completed_air_yards (aDOT proxy), aggressiveness
  Rushing:   efficiency, percent_attempts_gte_eight_defenders
  Receiving: avg_separation, avg_cushion, avg_intended_air_yards (aDOT proxy)
"""

from pathlib import Path

import pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2016, 2026))
OUT_PATH = Path(__file__).parent.parent / "data" / "features" / "ngs.parquet"


def build_ngs_features(years: list[int]) -> pd.DataFrame:
    """
    Fetch NGS data for all three stat types, select the relevant columns,
    aggregate to season level (NGS data is weekly), and join into a single
    DataFrame keyed on (player_id, season).

    Season-level values are weighted averages where appropriate (e.g. weight
    avg_separation by targets, avg_time_to_throw by attempts).
    """
    raise NotImplementedError


def main():
    df = build_ngs_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
