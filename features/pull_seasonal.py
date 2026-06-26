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
POSITIONS = {"QB", "RB", "WR", "TE"}


def build_seasonal_features(years: list[int]) -> pd.DataFrame:
    """
    Fetch seasonal advanced stats and return a DataFrame keyed on
    (player_id, season) with the columns listed above plus adot.

    adot is derived as air_yards / targets; rows with targets == 0 get NaN.
    """
    df = nfl.import_seasonal_data(years, s_type="REG")

    # nflverse uses receiving_air_yards / receiving_yards_after_catch in some
    # releases; normalise to the shorter names we use everywhere else.
    df = df.rename(columns={
        "receiving_air_yards":        "air_yards",
        "receiving_yards_after_catch": "yards_after_catch",
    })

    df = df[df["position"].isin(POSITIONS)].copy()

    df["adot"] = (df["air_yards"] / df["targets"]).where(df["targets"] > 0)

    keep = [
        "player_id", "season",
        "target_share", "air_yards_share", "wopr", "racr", "ppr_sh",
        "targets", "air_yards", "yards_after_catch", "adot",
    ]
    present = [c for c in keep if c in df.columns]
    return df[present].dropna(subset=["player_id", "season"]).reset_index(drop=True)


def main():
    df = build_seasonal_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(df[["player_id", "season", "target_share", "wopr", "adot"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
