"""
Download raw play-by-play data for 2010–2025 via nfl_data_py and save one
parquet file per season to data/pbp/.  Seasons already on disk are skipped,
so the script is safe to re-run incrementally after adding new seasons.

The raw files are kept lean — only the columns needed for PBP feature
engineering are retained (raw PBP has 300+ columns per row).

Run: python ingestion/ingest_pbp.py
"""

import os
import nfl_data_py as nfl

SEASONS = list(range(2010, 2026))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pbp")

# Only keep columns relevant to feature engineering.
# Older seasons may be missing some columns (e.g. cpoe before 2006, qb_scramble
# before ~2016); ingest_pbp saves whatever is present and etl/pbp_features.py
# handles missing columns via mergeSchema + coalesce.
KEEP_COLS = [
    "season",
    "week",
    "season_type",
    "play_type",
    "posteam",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
    "pass_attempt",       # 1 on any dropback (complete, incomplete, INT, sack)
    "rush_attempt",       # 1 on designed run plays
    "qb_scramble",        # 1 when QB runs after a designed pass play
    "sack",
    "complete_pass",
    "air_yards",          # distance from LOS to target point (negative = behind LOS)
    "yards_after_catch",
    "yardline_100",       # distance to opponent end zone (1–99)
    "cpoe",               # completion % over expected (~2006+)
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for season in SEASONS:
        path = os.path.join(OUT_DIR, f"pbp_{season}.parquet")
        if os.path.exists(path):
            print(f"  {season}: already cached, skipping.")
            continue
        print(f"  {season}: downloading...", flush=True)
        df = nfl.import_pbp_data([season], downcast=False)
        df = df[df["season_type"] == "REG"]
        # Keep only columns present in this season's data — older seasons may
        # lack newer tracking columns.
        cols = [c for c in KEEP_COLS if c in df.columns]
        df[cols].to_parquet(path, index=False)
        print(f"    {len(df):,} plays → {path}")
    print("Done.")


if __name__ == "__main__":
    main()
