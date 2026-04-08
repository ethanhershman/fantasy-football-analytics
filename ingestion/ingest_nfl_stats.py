"""
Fetch NFL rosters + seasonal counting stats via nfl_data_py and nflverse.
Populates: season_stats (one row per player per season, including player info).

For 2020–2024 we use the legacy combined parquet from nflverse (weekly data
that we aggregate to season totals). For 2025+ we use the newer per-season
pre-aggregated parquet files from the nflverse stats_player release.

Run: python ingestion/ingest_nfl_stats.py
"""

import pandas as pd
import nfl_data_py as nfl
from sqlalchemy import text
from db import get_engine

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
POSITIONS = ["QB", "RB", "WR", "TE"]

# The legacy combined weekly stats file covers up to 2024.
LEGACY_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet"
# 2025+ uses per-season pre-aggregated files from the stats_player release.
NEW_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{year}.parquet"

# Cutoff: seasons <= this use the legacy weekly file, > this use the new per-season files.
LEGACY_MAX_SEASON = 2024

POSITION_OVERRIDES = {
    "00-0035624": "WR",   # N'Keal Harry — nflverse has TE, but drafted/ranked as WR
    "00-0033357": "TE",   # Taysom Hill — nflverse has QB, but fantasy value is as TE
}

# Column mapping from the new per-season files to our standard names.
NEW_COL_RENAMES = {
    "recent_team": "team",
    "player_display_name": "full_name",
    "games": "games_played",
    "attempts": "pass_attempts",
    "passing_interceptions": "interceptions",
}


def _clean(rows):
    """Convert NaN → None so psycopg2 sends SQL NULL instead of 'nan'."""
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]


def _fetch_legacy_stats(seasons):
    """Fetch weekly stats from the legacy combined parquet and aggregate to season totals."""
    print(f"  Fetching legacy weekly stats for {seasons}...")
    weekly = pd.read_parquet(LEGACY_STATS_URL)
    weekly = weekly[(weekly["season_type"] == "REG") & (weekly["season"].isin(seasons))]

    agg_cols = {
        "week": "count",
        "fantasy_points_ppr": "sum",
        "completions": "sum",
        "attempts": "sum",
        "passing_yards": "sum",
        "passing_tds": "sum",
        "interceptions": "sum",
        "carries": "sum",
        "rushing_yards": "sum",
        "rushing_tds": "sum",
        "receptions": "sum",
        "targets": "sum",
        "receiving_yards": "sum",
        "receiving_tds": "sum",
    }
    stats = weekly.groupby(["player_id", "season"]).agg(agg_cols).reset_index()
    stats = stats.rename(columns={"week": "games_played", "attempts": "pass_attempts"})
    return stats


def _fetch_new_stats(season):
    """Fetch pre-aggregated season stats from the new per-season parquet."""
    url = NEW_STATS_URL.format(year=season)
    print(f"  Fetching new-format stats for {season}...")
    df = pd.read_parquet(url)
    df = df[df["position"].isin(POSITIONS)].copy()
    df = df.rename(columns=NEW_COL_RENAMES)

    keep = [
        "player_id", "season", "full_name", "position", "team",
        "games_played", "fantasy_points_ppr",
        "completions", "pass_attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ]
    return df[keep]


def load_season_stats(engine):
    """
    Fetch rosters and stats for all SEASONS, join them into a single flat
    table, and upsert into season_stats.
    """
    legacy_seasons = [s for s in SEASONS if s <= LEGACY_MAX_SEASON]
    new_seasons = [s for s in SEASONS if s > LEGACY_MAX_SEASON]

    frames = []

    # --- Legacy seasons: fetch weekly data + rosters, aggregate and join ---
    if legacy_seasons:
        print(f"Fetching rosters for {legacy_seasons}...")
        roster = nfl.import_seasonal_rosters(legacy_seasons)
        roster = roster[roster["position"].isin(POSITIONS)].dropna(subset=["player_id"])

        roster["position"] = roster.apply(
            lambda r: POSITION_OVERRIDES.get(r["player_id"], r["position"]), axis=1
        )

        roster = roster.drop_duplicates(["player_id", "season"], keep="last")
        roster = roster[["player_id", "season", "player_name", "position", "team"]].rename(
            columns={"player_name": "full_name"}
        )

        stats = _fetch_legacy_stats(legacy_seasons)
        merged = stats.merge(roster, on=["player_id", "season"], how="inner")
        frames.append(merged)

    # --- New seasons: pre-aggregated files already include player info ---
    for season in new_seasons:
        df = _fetch_new_stats(season)
        # Apply position overrides.
        df["position"] = df.apply(
            lambda r: POSITION_OVERRIDES.get(r["player_id"], r["position"]), axis=1
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # --- Upsert into season_stats ---
    db_cols = [
        "full_name", "position", "team",
        "games_played", "fantasy_points_ppr",
        "completions", "pass_attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ]
    keep = ["player_id", "season"] + db_cols
    rows = _clean(combined[keep].to_dict("records"))

    val_placeholders = ", ".join(f":{c}" for c in keep)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in db_cols)
    upsert = text(f"""
        INSERT INTO season_stats (player_id, season, {', '.join(db_cols)})
        VALUES ({val_placeholders})
        ON CONFLICT (player_id, season) DO UPDATE SET {update_set}
    """)
    with engine.begin() as conn:
        conn.execute(upsert, rows)
    print(f"  {len(rows)} player-season rows loaded.")


if __name__ == "__main__":
    engine = get_engine()
    load_season_stats(engine)

    with engine.connect() as conn:
        for row in conn.execute(text("""
            SELECT position, COUNT(*) AS n
            FROM season_stats
            GROUP BY position ORDER BY position
        """)):
            print(f"  {row.position}: {row.n} player-seasons")
