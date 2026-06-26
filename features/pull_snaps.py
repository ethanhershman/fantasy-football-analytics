"""
Bucket A: pull weekly snap counts from nfl-data-py and aggregate to season level.

Source: nfl.import_snap_counts(years) — available 2012+.
Output: data/features/snaps.parquet, keyed on (player_id, season).

Features extracted per (player_id, season):
  offense_snaps  (season total)
  snap_pct       (mean weekly offense_pct across weeks the player appeared)

Note: snap count data uses PFR player IDs, not GSIS IDs. We resolve the
crosswalk via nfl.import_seasonal_rosters(), which carries both pfr_id and
player_id. Players without a pfr_id match are dropped.
"""

from pathlib import Path

import pandas as pd
import nfl_data_py as nfl

YEARS = list(range(2016, 2026))
OUT_PATH = Path(__file__).parent.parent / "data" / "features" / "snaps.parquet"
POSITIONS = {"QB", "RB", "WR", "TE"}


def _build_pfr_crosswalk(years: list[int]) -> pd.DataFrame:
    """Return a DataFrame mapping pfr_id → player_id (GSIS) using seasonal rosters."""
    rosters = nfl.import_seasonal_rosters(years)
    rosters = rosters[["player_id", "pfr_id", "position"]].dropna(subset=["player_id", "pfr_id"])
    rosters = rosters[rosters["position"].isin(POSITIONS)]
    # A PFR ID can appear on multiple seasons; keep one mapping per pfr_id.
    return rosters[["pfr_id", "player_id"]].drop_duplicates(subset=["pfr_id"])


def build_snap_features(years: list[int]) -> pd.DataFrame:
    """
    Fetch weekly snap counts, join GSIS player_id via PFR crosswalk, aggregate
    to (player_id, season), and return offense_snaps + snap_pct.
    """
    snaps = nfl.import_snap_counts(years)

    # Filter to regular-season offensive skill positions.
    snaps = snaps[snaps["game_type"] == "REG"].copy()
    snaps = snaps[snaps["position"].isin(POSITIONS)]

    crosswalk = _build_pfr_crosswalk(years)
    snaps = snaps.merge(crosswalk, on="pfr_player_id", how="left")

    unmatched = snaps["player_id"].isna().sum()
    if unmatched:
        print(f"  {unmatched} snap-count rows had no pfr_id match — dropped.")
    snaps = snaps.dropna(subset=["player_id"])

    agg = (
        snaps.groupby(["player_id", "season"])
        .agg(
            offense_snaps=("offense_snaps", "sum"),
            snap_pct=("offense_pct", "mean"),
        )
        .reset_index()
    )

    return agg


def main():
    df = build_snap_features(YEARS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(df.sort_values("offense_snaps", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
