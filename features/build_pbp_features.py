"""
Bucket B: aggregate play-by-play data to per-(player_id, season) features.

Source: nfl.import_pbp_data(years, columns=PBP_COLS) — column-filtered load
        keeps memory well under 1 GB for 11 seasons.
Output: data/features/pbp.parquet, keyed on (player_id, season).

Features emitted per (player_id, season):
  Receiver:  rz_targets       (yardline_100 <= 20, play_type == 'pass')
  Rusher:    rz_carries        (yardline_100 <= 20, rush_attempt == 1)
             goalline_carries  (yardline_100 <= 5)
  QB:        designed_rushes   (rush_attempt == 1 and qb_scramble == 0)
             scrambles         (qb_scramble == 1)
  Team-season (joined back to player):
             team_pass_rate    (pass plays / total plays, season level)
             carry_share       (player carries / team rush attempts)
  Pressure:  pressure_to_sack_rate
             # TODO: use pbp pressure column if available; fall back to NGS
             #       sack / pressure numbers for seasons where pbp lacks it.
"""

from pathlib import Path

import pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2016, 2026))
OUT_PATH = Path(__file__).parent.parent / "data" / "features" / "pbp.parquet"

# Minimal column allowlist — keeps the import fast and memory-efficient.
PBP_COLS = [
    "season",
    "season_type",
    "play_type",
    "down",
    "yardline_100",
    "posteam",
    "rush_attempt",
    "qb_scramble",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
    "sack",
    "pass_attempt",
    # pressure columns (present in recent seasons only):
    "was_pressure",
]


def build_team_denominators(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team-season totals used as denominators for share metrics.

    Returns a DataFrame with columns:
      posteam, season, team_pass_plays, team_rush_attempts
    """
    raise NotImplementedError


def build_pbp_features(years: list[int]) -> pd.DataFrame:
    """
    Load pbp data for the given years (REG season only), group by
    (player_id, season), and return the feature set described in the module
    docstring.

    Steps:
      1. import_pbp_data with PBP_COLS allowlist, filter season_type == 'REG'.
      2. build_team_denominators for team_pass_rate and carry_share.
      3. Aggregate receiver features (rz_targets) on receiver_player_id.
      4. Aggregate rusher features (rz_carries, goalline_carries, designed_rushes,
         scrambles) on rusher_player_id.
      5. Join team denominators back and compute share metrics.
      6. Outer-join receiver and rusher frames on (player_id, season).
      7. Compute pressure_to_sack_rate where was_pressure is available;
         leave as NaN otherwise (TODO: NGS fallback).
    """
    raise NotImplementedError


def main():
    df = build_pbp_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
