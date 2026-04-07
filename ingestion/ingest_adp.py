"""
Fetch historical ADP from Fantasy Football Calculator API.
Populates: adp table (source='ffc').

This script pulls Average Draft Position (ADP) data for PPR 12-team leagues
from the Fantasy Football Calculator (FFC) public API for each season in the
SEASONS list. The data is matched to existing player records in the database
by normalized name + position, then upserted into the `adp` table.

Run: python ingestion/ingest_adp.py
"""

import time
import requests
import pandas as pd
from sqlalchemy import text
from db import get_engine

# Historical seasons to pull ADP data for.
# Each year represents a draft year (i.e. pre-season consensus ADP).
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]

# FFC API endpoint — PPR scoring, 12-team league format.
# {year} is replaced at fetch time with each season.
FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year={year}"

# Only include fantasy-relevant skill positions (exclude K, DST, etc.).
POSITIONS = ["QB", "RB", "WR", "TE"]


def _clean(rows):
    """Convert any NaN/NaT values to None so the DB driver sends SQL NULL."""
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]


def fetch_season(season: int) -> pd.DataFrame:
    """
    Hit the FFC API for a single season and return a cleaned DataFrame.

    Returns columns: player_name, position, season, adp_overall, adp_position_rank.
    adp_position_rank is computed here by ranking adp_overall within each position group.
    """
    resp = requests.get(FFC_URL.format(year=season), timeout=15)
    resp.raise_for_status()
    players = resp.json().get("players", [])
    if not players:
        print(f"  {season}: no data.")
        return pd.DataFrame()

    df = pd.DataFrame(players)
    df["season"] = season

    # Filter to skill positions only.
    df = df[df["position"].isin(POSITIONS)].copy()

    # Rename the ADP column and sort so lower ADP = higher draft pick.
    df = df.sort_values("adp").rename(columns={"adp": "adp_overall", "name": "player_name"})

    # Compute within-position rank (e.g. RB1, RB2, ...) based on overall ADP.
    df["adp_position_rank"] = df.groupby("position")["adp_overall"].rank(method="first").astype(int)

    return df[["player_name", "position", "season", "adp_overall", "adp_position_rank"]]


def match_ids(df: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Join ADP rows to the players table by normalized name + position.

    Players that can't be matched (retired players not in the roster, name
    mismatches like "Gabe Davis" vs "Gabriel Davis") are logged and dropped.
    Returns only rows with a valid player_id.
    """
    with engine.connect() as conn:
        players = pd.read_sql("SELECT player_id, full_name, position FROM players", conn)

    # Normalize names to lowercase/stripped for fuzzy-ish matching.
    df["name_norm"] = df["player_name"].str.lower().str.strip()
    players["name_norm"] = players["full_name"].str.lower().str.strip()

    merged = df.merge(players[["player_id", "name_norm", "position"]], on=["name_norm", "position"], how="left")

    n = merged["player_id"].isna().sum()
    if n:
        print(f"    {n} unmatched (retired/name mismatch).")

    return merged.dropna(subset=["player_id"])


def load_adp(engine):
    """
    Main pipeline: fetch ADP for every season, match to player IDs, and upsert
    into the adp table with source='ffc'.
    """
    frames = []
    for season in SEASONS:
        print(f"  Fetching {season}...")
        df = fetch_season(season)
        if df.empty:
            continue
        df = match_ids(df, engine)
        frames.append(df)
        # Polite delay to avoid hammering the FFC API.
        time.sleep(1)

    if not frames:
        print("No ADP data loaded.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["source"] = "ffc"

    # Prepare rows for upsert — drop any with missing player_id (safety check).
    rows = _clean(combined[["player_id", "season", "adp_overall", "adp_position_rank", "source"]].to_dict("records"))
    rows = [r for r in rows if r["player_id"] is not None]

    # Upsert: if a (player_id, season, source) row already exists, update ADP values.
    upsert = text("""
        INSERT INTO adp (player_id, season, adp_overall, adp_position_rank, source)
        VALUES (:player_id, :season, :adp_overall, :adp_position_rank, :source)
        ON CONFLICT (player_id, season, source) DO UPDATE SET
            adp_overall = EXCLUDED.adp_overall, adp_position_rank = EXCLUDED.adp_position_rank
    """)
    with engine.begin() as conn:
        conn.execute(upsert, rows)
    print(f"  Upserted {len(rows)} ADP rows.")


if __name__ == "__main__":
    engine = get_engine()
    load_adp(engine)

    # Print a quick summary of what's in the DB per season.
    with engine.connect() as conn:
        for row in conn.execute(text("""
            SELECT season, COUNT(*) AS n FROM adp WHERE source='ffc'
            GROUP BY season ORDER BY season
        """)):
            print(f"  {row.season}: {row.n} players")
