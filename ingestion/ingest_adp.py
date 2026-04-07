"""
Phase 1 — Ingestion Script 2: Historical ADP via Fantasy Football Calculator API

Pulls PPR ADP for each season in SEASONS from the free FFC REST API.
No API key required.

Run:
    python ingestion/ingest_adp.py
"""

import time
import requests
import pandas as pd
from sqlalchemy import text
from db import get_engine

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year={year}"
POSITIONS = ["QB", "RB", "WR", "TE"]


def fetch_adp_for_season(season: int) -> pd.DataFrame:
    url = FFC_URL.format(year=season)
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    players = data.get("players", [])
    if not players:
        print(f"  No data returned for {season}.")
        return pd.DataFrame()

    df = pd.DataFrame(players)
    df["season"] = season
    df["source"] = "ffc_api"

    # FFC returns: name, position, team, adp, times_drafted, high, low, stdev
    df = df.rename(columns={"adp": "adp_overall", "name": "player_name"})
    df = df[df["position"].isin(POSITIONS)].copy()

    # Compute position rank from adp within position
    df = df.sort_values("adp_overall")
    df["adp_position_rank"] = df.groupby("position")["adp_overall"].rank(method="first").astype(int)

    return df[["player_name", "position", "season", "adp_overall", "adp_position_rank", "source"]]


def match_to_player_ids(df: pd.DataFrame, engine) -> pd.DataFrame:
    """
    FFC returns player names, not IDs. Match to player_id by name + position.
    Unmatched players are dropped (they won't be in our roster table).
    """
    with engine.connect() as conn:
        players = pd.read_sql("SELECT player_id, full_name, position FROM players", conn)

    # Normalize names for fuzzy matching
    df["name_norm"] = df["player_name"].str.lower().str.strip()
    players["name_norm"] = players["full_name"].str.lower().str.strip()

    merged = df.merge(
        players[["player_id", "name_norm", "position"]],
        on=["name_norm", "position"],
        how="left",
    )

    unmatched = merged["player_id"].isna().sum()
    if unmatched > 0:
        print(f"    {unmatched} players could not be matched to player_id (likely retired/injured).")

    return merged.dropna(subset=["player_id"])


def load_adp(engine):
    all_seasons = []

    for season in SEASONS:
        print(f"  Fetching ADP for {season}...")
        df = fetch_adp_for_season(season)
        if df.empty:
            continue
        df = match_to_player_ids(df, engine)
        all_seasons.append(df)
        time.sleep(1)  # be respectful to the free API

    if not all_seasons:
        print("No ADP data loaded.")
        return

    combined = pd.concat(all_seasons, ignore_index=True)

    rows = combined[["player_id", "season", "adp_overall", "adp_position_rank", "source"]].to_dict("records")
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]

    upsert_sql = text("""
        INSERT INTO adp_history (player_id, season, adp_overall, adp_position_rank, source)
        VALUES (:player_id, :season, :adp_overall, :adp_position_rank, :source)
        ON CONFLICT (player_id, season, source) DO UPDATE SET
            adp_overall       = EXCLUDED.adp_overall,
            adp_position_rank = EXCLUDED.adp_position_rank
    """)

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    print(f"  Upserted {len(rows)} ADP rows across {len(SEASONS)} seasons.")


if __name__ == "__main__":
    engine = get_engine()
    load_adp(engine)

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT season, COUNT(*) AS players
            FROM adp_history
            WHERE source = 'ffc_api'
            GROUP BY season ORDER BY season
        """))
        print("\nADP rows by season:")
        for row in result:
            print(f"  {row.season}: {row.players} players")
