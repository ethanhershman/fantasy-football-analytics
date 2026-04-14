"""
Fetch historical ADP from Fantasy Football Calculator API.
Populates: adp table (source='ffc').

This script pulls Average Draft Position (ADP) data for PPR 12-team leagues
from the Fantasy Football Calculator (FFC) public API for each season in the
SEASONS list. The data is matched to existing player records in the database
by normalized name + position, then upserted into the `adp` table.

If the FFC API returns no data for a season, the script falls back to a local
CSV at data/ffc_adp_{season}.csv (columns: Name, Position, Overall).

Run: python ingestion/ingest_adp.py
"""

import os
import time
import requests
import pandas as pd
from sqlalchemy import text
from db import get_engine

# Historical seasons to pull ADP data for.
# Each year represents a draft year (i.e. pre-season consensus ADP).
# Covers the full 2016–2025 training window.
SEASONS = list(range(2016, 2026))

# FFC API endpoint — PPR scForing, 12-team league format.
# {year} is replaced at fetch time with each season.
FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year={year}"

# Only include fantasy-relevant skill positions (exclude K, DST, etc.).
POSITIONS = ["QB", "RB", "WR", "TE"]

# Explicit name overrides for cases that can't be solved by normalization alone.
# Maps FFC name → nflverse name.
# Add entries here when the unmatched-player log shows FFC/nflverse name divergence.
NAME_FIXES = {
    # 2020+
    "Chris Herndon":        "Christopher Herndon",
    "Joshua Palmer":        "Josh Palmer",
    # 2016–2019 era
    "Javorius Allen":       "Buck Allen",
    "Robert Kelley":        "Rob Kelley",
}


def _clean(rows):
    """Convert any NaN/NaT values to None so the DB driver sends SQL NULL."""
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]


def _normalize_name(name: str) -> str:
    """
    Normalize a player name for matching by:
      1. Lowercasing and stripping whitespace.
      2. Removing suffixes (Jr., Sr., Jr, Sr, II, III, IV).
      3. Removing dots (so "A.J." becomes "AJ", "D.K." becomes "DK").
      4. Collapsing extra whitespace left behind.
    """
    n = name.lower().strip()
    # Remove common suffixes — order matters (longer patterns first to avoid
    # partial matches, e.g. "iv" before "i").
    import re
    n = re.sub(r'\b(jr\.?|sr\.?|viii|vii|vi|iii|ii|iv|v)\s*$', '', n).strip()
    # Remove dots and apostrophes (A.J. → AJ, D.K. → DK, Le'Veon → LeVeon).
    n = n.replace(".", "").replace("'", "")
    # Collapse any double spaces left behind.
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def fetch_season(season: int) -> pd.DataFrame:
    """
    Hit the FFC API for a single season and return a cleaned DataFrame.

    Returns columns: player_name, position, season, adp_overall, adp_position_rank.
    adp_position_rank is computed here by ranking adp_overall within each position group.
    """
    resp = requests.get(FFC_URL.format(year=season), timeout=15)
    resp.raise_for_status()
    players = resp.json().get("players", [])

    if players:
        df = pd.DataFrame(players)
        df["season"] = season
        # Filter to skill positions only.
        df = df[df["position"].isin(POSITIONS)].copy()
        # Rename the ADP column and sort so lower ADP = higher draft pick.
        df = df.sort_values("adp").rename(columns={"adp": "adp_overall", "name": "player_name"})
    else:
        # Fall back to a local CSV if the API has no data for this season.
        # The CSV is expected at data/ffc_adp_{season}.csv with columns:
        #   Name, Position, Overall (the ADP pick number).
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", f"ffc_adp_{season}.csv")
        if not os.path.exists(csv_path):
            print(f"  {season}: no API data and no local CSV found.")
            return pd.DataFrame()
        print(f"  {season}: no API data — loading from {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"Name": "player_name", "Position": "position", "Overall": "adp_overall"})
        df["season"] = season
        df = df[df["position"].isin(POSITIONS)].copy()
        df = df.sort_values("adp_overall")

    # Compute within-position rank (e.g. RB1, RB2, ...) based on overall ADP.
    df["adp_position_rank"] = df.groupby("position")["adp_overall"].rank(method="first").astype(int)

    return df[["player_name", "position", "season", "adp_overall", "adp_position_rank"]]


def match_ids(df: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Join ADP rows to the players table by normalized name + position.

    1. Applies NAME_FIXES for explicit overrides.
    2. Normalizes both sides (strip suffixes, dots, lowercase) via _normalize_name.
    3. Logs any remaining unmatched players and drops them.

    Returns only rows with a valid player_id.
    """
    # Apply explicit name fixes, then normalize for matching.
    df["match_name"] = df["player_name"].replace(NAME_FIXES)
    df["name_norm"] = df["match_name"].apply(_normalize_name)

    with engine.connect() as conn:
        players = pd.read_sql(
            "SELECT DISTINCT player_id, full_name, position FROM season_stats", conn
        )

    players["name_norm"] = players["full_name"].apply(_normalize_name)

    merged = df.merge(players[["player_id", "name_norm", "position"]], on=["name_norm", "position"], how="left")

    matched = merged.dropna(subset=["player_id"])
    unmatched = merged[merged["player_id"].isna()]

    if not unmatched.empty:
        season = unmatched["season"].iloc[0]
        print(f"    {len(unmatched)} unmatched in {season}:")
        for _, row in unmatched.sort_values("adp_overall").iterrows():
            print(f"      {row['player_name']:25s} {row['position']:4s}  ADP {row['adp_overall']:.1f}")

    return matched


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
