"""
Fetch NFL rosters + seasonal counting stats via nfl_data_py.
Populates: players, season_stats.

This script does two things:
  1. Loads the player roster from nfl_data_py for every season in the SEASONS
     list, deduplicating by player_id, to populate the `players` table.
  2. Fetches regular-season aggregated stats for the same seasons and upserts
     them into `season_stats`, filtering to only players already in the DB.

Run: python ingestion/ingest_nfl_stats.py
"""

import pandas as pd
import nfl_data_py as nfl
from sqlalchemy import text
from db import get_engine

# Seasons to pull roster and stats data for.
# We need rosters from all years (not just the latest) so that players who
# retired or were cut still appear in the players table for historical matching.
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]

# Only include fantasy-relevant skill positions.
POSITIONS = ["QB", "RB", "WR", "TE"]

# Mapping from nfl_data_py column names → our schema's column names.
# Keys are the source columns; values are what we store in season_stats.
STAT_COLS = {
    "games":              "games_played",
    "fantasy_points_ppr": "fantasy_points_ppr",
    "completions":        "completions",
    "attempts":           "pass_attempts",
    "passing_yards":      "passing_yards",
    "passing_tds":        "passing_tds",
    "interceptions":      "interceptions",
    "carries":            "carries",
    "rushing_yards":      "rushing_yards",
    "rushing_tds":        "rushing_tds",
    "receptions":         "receptions",
    "targets":            "targets",
    "receiving_yards":    "receiving_yards",
    "receiving_tds":      "receiving_tds",
}


def _clean(rows):
    """Convert NaN → None so psycopg2 sends SQL NULL instead of 'nan'."""
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]


def load_players(engine):
    """
    Import rosters for all SEASONS and upsert into the players table.

    We pull rosters from every season so that players who were active in
    past years (but may no longer be on a roster) are still in our DB.
    Deduplication by player_id keeps only the most recent roster entry
    (nfl_data_py returns them chronologically).
    """
    print(f"Fetching rosters for {SEASONS}...")
    roster = nfl.import_seasonal_rosters(SEASONS)
    roster = (
        roster[roster["position"].isin(POSITIONS)]
        .dropna(subset=["player_id"])
        # Keep the last occurrence per player_id (most recent season's info).
        .drop_duplicates("player_id", keep="last")
    )
    rows = _clean(
        roster[["player_id", "player_name", "position", "team"]]
        .rename(columns={"player_name": "full_name"})
        .to_dict("records")
    )

    # Upsert: update name/position/team if the player already exists.
    upsert = text("""
        INSERT INTO players (player_id, full_name, position, team)
        VALUES (:player_id, :full_name, :position, :team)
        ON CONFLICT (player_id) DO UPDATE SET
            full_name = EXCLUDED.full_name, position = EXCLUDED.position, team = EXCLUDED.team
    """)
    with engine.begin() as conn:
        conn.execute(upsert, rows)
    print(f"  {len(rows)} players loaded.")


def load_stats(engine):
    """
    Import regular-season aggregated stats for all SEASONS and upsert into
    the season_stats table.

    Only players that already exist in the players table are included
    (prevents foreign-key violations from kickers, DST, etc.).
    """
    print(f"Fetching seasonal stats for {SEASONS}...")
    stats = nfl.import_seasonal_data(SEASONS, s_type="REG").rename(columns=STAT_COLS)

    # Filter to players already in our DB (populated by load_players above).
    with engine.connect() as conn:
        known = set(pd.read_sql("SELECT player_id FROM players", conn)["player_id"])
    stats = stats[stats["player_id"].isin(known)]

    # Build the upsert statement dynamically from STAT_COLS.
    keep = ["player_id", "season"] + list(STAT_COLS.values())
    rows = _clean(stats[keep].to_dict("records"))

    cols = list(STAT_COLS.values())
    val_placeholders = ", ".join(f":{c}" for c in ["player_id", "season"] + cols)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
    upsert = text(f"""
        INSERT INTO season_stats (player_id, season, {', '.join(cols)})
        VALUES ({val_placeholders})
        ON CONFLICT (player_id, season) DO UPDATE SET {update_set}
    """)
    with engine.begin() as conn:
        conn.execute(upsert, rows)
    print(f"  {len(rows)} player-season rows loaded.")


if __name__ == "__main__":
    engine = get_engine()
    load_players(engine)
    load_stats(engine)

    # Print a summary of stats loaded, grouped by position.
    with engine.connect() as conn:
        for row in conn.execute(text("""
            SELECT p.position, COUNT(*) AS n
            FROM season_stats ss JOIN players p USING (player_id)
            GROUP BY p.position ORDER BY p.position
        """)):
            print(f"  {row.position}: {row.n} player-seasons")
