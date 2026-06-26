"""
Bucket A: pull pre-aggregated seasonal advanced stats from nfl-data-py.

Source: nfl.import_seasonal_data(years) for 2016-2024.
        Direct parquet fetch for 2025+ (nflverse moved to a new URL schema).
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

# 2025+ nflverse moved seasonal stats to a new per-season release.
LEGACY_MAX_SEASON = 2024
NEW_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{year}.parquet"


def _fetch_new_season(year: int) -> pd.DataFrame:
    """Fetch a single season from the new nflverse stats_player release."""
    url = NEW_STATS_URL.format(year=year)
    df = pd.read_parquet(url, engine="auto")
    df = df[df["season_type"] == "REG"].copy()
    return df.rename(columns={
        "passing_interceptions": "interceptions",
    })


def build_seasonal_features(years: list[int]) -> pd.DataFrame:
    """
    Fetch seasonal advanced stats and return a DataFrame keyed on
    (player_id, season) with the columns listed above plus adot.
    """
    legacy_years = [y for y in years if y <= LEGACY_MAX_SEASON]
    new_years    = [y for y in years if y > LEGACY_MAX_SEASON]

    frames = []

    if legacy_years:
        df = nfl.import_seasonal_data(legacy_years, s_type="REG")
        df = df.rename(columns={
            "receiving_air_yards":        "air_yards",
            "receiving_yards_after_catch": "yards_after_catch",
        })
        frames.append(df)

    for year in new_years:
        print(f"  Fetching {year} from new nflverse endpoint...")
        df = _fetch_new_season(year)
        df = df.rename(columns={
            "receiving_air_yards":        "air_yards",
            "receiving_yards_after_catch": "yards_after_catch",
        })
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["adot"] = (combined["air_yards"] / combined["targets"]).where(combined["targets"] > 0)

    keep = [
        "player_id", "season",
        "target_share", "air_yards_share", "wopr", "racr", "ppr_sh",
        "targets", "air_yards", "yards_after_catch", "adot",
    ]
    present = [c for c in keep if c in combined.columns]
    return combined[present].dropna(subset=["player_id", "season"]).reset_index(drop=True)


def main():
    df = build_seasonal_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(df[["player_id", "season", "target_share", "wopr", "adot"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
