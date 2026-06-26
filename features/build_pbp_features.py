"""
Bucket B: aggregate play-by-play data to per-(player_id, season) features.

Source: nfl.import_pbp_data(years, columns=PBP_COLS) — column-filtered load
        keeps memory well under 1 GB for 11 seasons.
Output: data/features/pbp.parquet, keyed on (player_id, season).

Features emitted per (player_id, season):
  Receiver:  rz_targets       (yardline_100 <= 20, play_type == 'pass')
  Rusher:    rz_carries        (yardline_100 <= 20, rush_attempt == 1)
             goalline_carries  (yardline_100 <= 5, rush_attempt == 1)
  QB:        designed_rushes   (rush_attempt == 1 and qb_scramble == 0)
             scrambles         (qb_scramble == 1)
  Team-season (joined back to player):
             team_pass_rate    (team pass plays / team total plays)
             carry_share       (player carries / team rush attempts)
  Pressure:  pressure_to_sack_rate (sacks / pressures; NaN where was_pressure absent)
"""

from pathlib import Path

import pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2016, 2025))
OUT_PATH = Path(__file__).parent.parent / "data" / "features" / "pbp.parquet"

# Minimal column allowlist — keeps the import fast and memory-efficient.
PBP_COLS = [
    "game_id",
    "season",
    "season_type",
    "play_type",
    "yardline_100",
    "posteam",
    "rush_attempt",
    "qb_scramble",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
    "sack",
    "pass_attempt",
]


def build_team_denominators(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team-season totals used as denominators for share metrics.

    Returns a DataFrame with columns:
      posteam, season, team_pass_plays, team_rush_attempts, team_pass_rate
    """
    pass_plays = pbp[pbp["pass_attempt"] == 1].groupby(["posteam", "season"]).size().rename("team_pass_plays")
    rush_plays = pbp[pbp["rush_attempt"] == 1].groupby(["posteam", "season"]).size().rename("team_rush_attempts")

    team = pd.concat([pass_plays, rush_plays], axis=1).fillna(0).reset_index()
    total = team["team_pass_plays"] + team["team_rush_attempts"]
    team["team_pass_rate"] = (team["team_pass_plays"] / total).where(total > 0)
    return team


def _dominant_team(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Return the posteam with the most play appearances for each (player_id, season)."""
    counts = (
        df[df[id_col].notna()]
        .groupby([id_col, "season", "posteam"])
        .size()
        .reset_index(name="n")
    )
    return (
        counts.sort_values("n", ascending=False)
        .drop_duplicates(subset=[id_col, "season"], keep="first")
        [[id_col, "season", "posteam"]]
        .rename(columns={id_col: "player_id"})
    )


def build_pbp_features(years: list[int]) -> pd.DataFrame:
    """
    Load pbp data for the given years (REG season only), group by
    (player_id, season), and return the feature set described in the module
    docstring.
    """
    print("  Loading pbp data...")
    # Only request columns that exist; nfl-data-py silently skips unknown ones.
    raw = nfl.import_pbp_data(years, columns=PBP_COLS, downcast=False)
    pbp = raw[raw["season_type"] == "REG"].copy()

    # Coerce flag columns to numeric, filling NaN as 0.
    for col in ("rush_attempt", "qb_scramble", "pass_attempt", "sack"):
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0)
    print(f"  {len(pbp):,} regular-season plays across {pbp['season'].nunique()} seasons.")

    team = build_team_denominators(pbp)

    # -----------------------------------------------------------------------
    # Receiver features — keyed on receiver_player_id
    # -----------------------------------------------------------------------
    pass_plays = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()]
    receiver = (
        pass_plays.assign(
            rz_target=(pass_plays["yardline_100"] <= 20).astype(int)
        )
        .groupby(["receiver_player_id", "season"])
        .agg(rz_targets=("rz_target", "sum"))
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )

    # -----------------------------------------------------------------------
    # Rusher features — keyed on rusher_player_id
    # -----------------------------------------------------------------------
    rush_plays = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()]
    rusher = (
        rush_plays.assign(
            rz_carry=(rush_plays["yardline_100"] <= 20).astype(int),
            gl_carry=(rush_plays["yardline_100"] <= 5).astype(int),
            designed_rush=(rush_plays["qb_scramble"] == 0).astype(int),
            scramble=(rush_plays["qb_scramble"] == 1).astype(int),
        )
        .groupby(["rusher_player_id", "season"])
        .agg(
            carries=("rush_attempt", "sum"),
            rz_carries=("rz_carry", "sum"),
            goalline_carries=("gl_carry", "sum"),
            designed_rushes=("designed_rush", "sum"),
            scrambles=("scramble", "sum"),
        )
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    # Join team denominators to rusher via dominant team, compute carry_share.
    rusher_team = _dominant_team(rush_plays, "rusher_player_id")
    rusher = rusher.merge(rusher_team, on=["player_id", "season"], how="left")
    rusher = rusher.merge(
        team[["posteam", "season", "team_rush_attempts", "team_pass_rate"]],
        on=["posteam", "season"],
        how="left",
    )
    rusher["carry_share"] = (rusher["carries"] / rusher["team_rush_attempts"]).where(
        rusher["team_rush_attempts"] > 0
    )
    rusher = rusher.drop(columns=["posteam", "team_rush_attempts"])

    # -----------------------------------------------------------------------
    # Outer-join all frames so every player with any pbp appearance gets a row.
    # -----------------------------------------------------------------------
    result = receiver.merge(rusher, on=["player_id", "season"], how="outer")

    return result.dropna(subset=["player_id"]).reset_index(drop=True)


def main():
    df = build_pbp_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
