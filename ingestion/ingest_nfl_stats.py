"""
Phase 1 — Ingestion Script 1: NFL seasonal stats + rosters via nfl-data-py

Pulls:
  - Seasonal player stats (2020–2025) → season_stats table
  - 2026 rosters → players table

Run:
    python ingestion/ingest_nfl_stats.py
"""

import pandas as pd
import nfl_data_py as nfl
from sqlalchemy import text
from db import get_engine

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
POSITIONS = ["QB", "RB", "WR", "TE"]


# ------------------------------------------------------------------
# Rosters → players table
# ------------------------------------------------------------------

def load_players(engine):
    print("Fetching 2026 rosters...")
    rosters = nfl.import_rosters([2026], columns=[
        "player_id", "player_name", "position", "team",
        "age", "years_exp",
    ])

    rosters = rosters[rosters["position"].isin(POSITIONS)].copy()
    rosters = rosters.dropna(subset=["player_id"]).drop_duplicates("player_id")

    rosters = rosters.rename(columns={
        "player_name": "full_name",
        "years_exp":   "years_experience",
    })

    rosters["age"] = rosters["age"].where(rosters["age"].notna(), None)
    rosters["years_experience"] = rosters["years_experience"].where(
        rosters["years_experience"].notna(), None
    )

    rows = rosters[["player_id", "full_name", "position", "team",
                     "age", "years_experience"]].to_dict("records")

    upsert_sql = text("""
        INSERT INTO players (player_id, full_name, position, team, age, years_experience)
        VALUES (:player_id, :full_name, :position, :team, :age, :years_experience)
        ON CONFLICT (player_id) DO UPDATE SET
            full_name        = EXCLUDED.full_name,
            position         = EXCLUDED.position,
            team             = EXCLUDED.team,
            age              = EXCLUDED.age,
            years_experience = EXCLUDED.years_experience
    """)

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    print(f"  Upserted {len(rows)} players.")


# ------------------------------------------------------------------
# Seasonal stats → season_stats table
# ------------------------------------------------------------------

def load_season_stats(engine):
    print(f"Fetching seasonal stats for {SEASONS}...")
    stats = nfl.import_seasonal_data(SEASONS, s_type="REG")

    stats = stats[stats["position"].isin(POSITIONS)].copy()

    # Rename nfl-data-py columns to our schema names
    stats = stats.rename(columns={
        "fantasy_points_ppr": "fantasy_points_ppr",
        "games":              "games_played",
        "targets":            "targets",
        "carries":            "carries",
    })

    # Only keep players we have in the players table
    with engine.connect() as conn:
        known_ids = pd.read_sql("SELECT player_id FROM players", conn)["player_id"].tolist()

    stats = stats[stats["player_id"].isin(known_ids)].copy()

    rows = stats[[
        "player_id", "season", "games_played",
        "fantasy_points_ppr", "targets", "carries",
    ]].to_dict("records")

    # Null out any NaN values so psycopg2 handles them as NULL
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]

    upsert_sql = text("""
        INSERT INTO season_stats (
            player_id, season, games_played,
            fantasy_points_ppr, targets, carries
        )
        VALUES (
            :player_id, :season, :games_played,
            :fantasy_points_ppr, :targets, :carries
        )
        ON CONFLICT (player_id, season) DO UPDATE SET
            games_played       = EXCLUDED.games_played,
            fantasy_points_ppr = EXCLUDED.fantasy_points_ppr,
            targets            = EXCLUDED.targets,
            carries            = EXCLUDED.carries
    """)

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    print(f"  Upserted {len(rows)} player-season rows.")


# ------------------------------------------------------------------
# Play-by-play aggregates → season_stats (target_share, air_yards_share,
#                                          snap_pct, rz_targets, yards_after_catch)
# ------------------------------------------------------------------

def load_pbp_features(engine):
    print(f"Fetching play-by-play data for {SEASONS} (this may take a minute)...")
    pbp = nfl.import_pbp_data(SEASONS, columns=[
        "season", "posteam",
        "receiver_player_id", "passer_player_id",
        "air_yards", "yards_after_catch",
        "pass_attempt", "complete_pass",
        "yardline_100",
    ])

    # --- Target share & air yards share ---
    # Team-level totals per season
    team_totals = (
        pbp[pbp["pass_attempt"] == 1]
        .groupby(["season", "posteam"])
        .agg(
            team_targets=("pass_attempt", "sum"),
            team_air_yards=("air_yards", "sum"),
        )
        .reset_index()
    )

    # Player-level receiving totals
    player_receiving = (
        pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()]
        .groupby(["season", "receiver_player_id"])
        .agg(
            targets=("pass_attempt", "sum"),
            air_yards=("air_yards", "sum"),
            yards_after_catch=("yards_after_catch", "mean"),
        )
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )

    # Red zone targets (yardline_100 <= 20 = inside opponent 20)
    rz = (
        pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()
            & (pbp["yardline_100"] <= 20)]
        .groupby(["season", "receiver_player_id"])
        .agg(rz_targets=("pass_attempt", "sum"))
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )

    player_receiving = player_receiving.merge(rz, on=["season", "player_id"], how="left")

    # We need each player's team to join team totals — pull from players table
    with engine.connect() as conn:
        player_teams = pd.read_sql(
            "SELECT player_id, team FROM players", conn
        )

    player_receiving = player_receiving.merge(player_teams, on="player_id", how="left")
    player_receiving = player_receiving.merge(team_totals, left_on=["season", "team"],
                                               right_on=["season", "posteam"], how="left")

    player_receiving["target_share"] = (
        player_receiving["targets"] / player_receiving["team_targets"]
    ).round(4)
    player_receiving["air_yards_share"] = (
        player_receiving["air_yards"] / player_receiving["team_air_yards"]
    ).round(4)

    rows = player_receiving[[
        "player_id", "season", "target_share", "air_yards_share",
        "rz_targets", "yards_after_catch",
    ]].to_dict("records")
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]

    update_sql = text("""
        UPDATE season_stats SET
            target_share      = :target_share,
            air_yards_share   = :air_yards_share,
            rz_targets        = :rz_targets,
            yards_after_catch = :yards_after_catch
        WHERE player_id = :player_id AND season = :season
    """)

    with engine.begin() as conn:
        conn.execute(update_sql, rows)

    print(f"  Updated PBP features for {len(rows)} player-season rows.")


# ------------------------------------------------------------------
# Snap counts → season_stats (snap_pct)
# ------------------------------------------------------------------

def load_snap_counts(engine):
    print(f"Fetching snap counts for {SEASONS}...")
    snaps = nfl.import_snap_counts(SEASONS)

    snaps = snaps[snaps["position"].isin(POSITIONS)].copy()

    season_snaps = (
        snaps.groupby(["pfr_player_id", "season"])
        .agg(
            total_snaps=("offense_snaps", "sum"),
            total_team_snaps=("offense_pct", "count"),  # will recalculate from pct
            avg_snap_pct=("offense_pct", "mean"),
        )
        .reset_index()
        .rename(columns={"pfr_player_id": "player_id"})
    )

    season_snaps["snap_pct"] = (season_snaps["avg_snap_pct"] / 100).round(4)

    rows = season_snaps[["player_id", "season", "snap_pct"]].to_dict("records")
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]

    update_sql = text("""
        UPDATE season_stats SET snap_pct = :snap_pct
        WHERE player_id = :player_id AND season = :season
    """)

    with engine.begin() as conn:
        conn.execute(update_sql, rows)

    print(f"  Updated snap counts for {len(rows)} player-season rows.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    engine = get_engine()

    load_players(engine)
    load_season_stats(engine)
    load_pbp_features(engine)
    load_snap_counts(engine)

    print("\nDone. Run a quick check:")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT position, COUNT(*) AS players, COUNT(DISTINCT season) AS seasons
            FROM season_stats ss
            JOIN players p USING (player_id)
            GROUP BY position ORDER BY position
        """))
        for row in result:
            print(f"  {row.position}: {row.players} player-seasons across {row.seasons} seasons")
