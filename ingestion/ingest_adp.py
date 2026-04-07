"""
Fetch historical ADP from Fantasy Football Calculator API.
Populates: adp table (source='ffc').

Run: python ingestion/ingest_adp.py
"""

import time
import requests
import pandas as pd
from sqlalchemy import text
from db import get_engine

SEASONS = [2020, 2021, 2022, 2023, 2024]
FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year={year}"
POSITIONS = ["QB", "RB", "WR", "TE"]


def _clean(rows):
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]


def fetch_season(season: int) -> pd.DataFrame:
    resp = requests.get(FFC_URL.format(year=season), timeout=15)
    resp.raise_for_status()
    players = resp.json().get("players", [])
    if not players:
        print(f"  {season}: no data.")
        return pd.DataFrame()

    df = pd.DataFrame(players)
    df["season"] = season
    df = df[df["position"].isin(POSITIONS)].copy()
    df = df.sort_values("adp").rename(columns={"adp": "adp_overall", "name": "player_name"})
    df["adp_position_rank"] = df.groupby("position")["adp_overall"].rank(method="first").astype(int)
    return df[["player_name", "position", "season", "adp_overall", "adp_position_rank"]]


def match_ids(df: pd.DataFrame, engine) -> pd.DataFrame:
    with engine.connect() as conn:
        players = pd.read_sql("SELECT player_id, full_name, position FROM players", conn)
    df["name_norm"] = df["player_name"].str.lower().str.strip()
    players["name_norm"] = players["full_name"].str.lower().str.strip()
    merged = df.merge(players[["player_id", "name_norm", "position"]], on=["name_norm", "position"], how="left")
    n = merged["player_id"].isna().sum()
    if n:
        print(f"    {n} unmatched (retired/name mismatch).")
    return merged.dropna(subset=["player_id"])


def load_adp(engine):
    frames = []
    for season in SEASONS:
        print(f"  Fetching {season}...")
        df = fetch_season(season)
        if df.empty:
            continue
        df = match_ids(df, engine)
        frames.append(df)
        time.sleep(1)

    if not frames:
        print("No ADP data loaded.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["source"] = "ffc"
    rows = _clean(combined[["player_id", "season", "adp_overall", "adp_position_rank", "source"]].to_dict("records"))
    rows = [r for r in rows if r["player_id"] is not None]

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

    with engine.connect() as conn:
        for row in conn.execute(text("""
            SELECT season, COUNT(*) AS n FROM adp WHERE source='ffc'
            GROUP BY season ORDER BY season
        """)):
            print(f"  {row.season}: {row.n} players")
