"""
Fetch NFL rosters + seasonal counting stats via nfl_data_py.
Populates: players, season_stats.

Run: python ingestion/ingest_nfl_stats.py
"""

import pandas as pd
import nfl_data_py as nfl
from sqlalchemy import text
from db import get_engine

SEASONS = [2020, 2021, 2022, 2023, 2024]
POSITIONS = ["QB", "RB", "WR", "TE"]

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
    """Convert NaN → None so psycopg2 sends NULL."""
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]


def load_players(engine):
    print("Fetching 2025 rosters...")
    roster = nfl.import_seasonal_rosters([2025])
    roster = (
        roster[roster["position"].isin(POSITIONS)]
        .dropna(subset=["player_id"])
        .drop_duplicates("player_id")
    )
    rows = _clean(
        roster[["player_id", "player_name", "position", "team"]]
        .rename(columns={"player_name": "full_name"})
        .to_dict("records")
    )
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
    print(f"Fetching seasonal stats for {SEASONS}...")
    stats = nfl.import_seasonal_data(SEASONS, s_type="REG").rename(columns=STAT_COLS)

    with engine.connect() as conn:
        known = set(pd.read_sql("SELECT player_id FROM players", conn)["player_id"])
    stats = stats[stats["player_id"].isin(known)]

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

    with engine.connect() as conn:
        for row in conn.execute(text("""
            SELECT p.position, COUNT(*) AS n
            FROM season_stats ss JOIN players p USING (player_id)
            GROUP BY p.position ORDER BY p.position
        """)):
            print(f"  {row.position}: {row.n} player-seasons")
