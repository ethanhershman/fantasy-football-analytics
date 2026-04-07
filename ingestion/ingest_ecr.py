"""
Scrape 2026 ECR/ADP from DraftSharks (PPR, Sleeper, 12-team).
Populates: adp table (source='draftsharks', season=2026).

The page embeds player data as a JS object (vueAppData) — no table scraping needed.

Run: python ingestion/ingest_ecr.py
"""

import re
import json
import requests
import pandas as pd
from sqlalchemy import text
from db import get_engine

URL = "https://www.draftsharks.com/adp/ppr/sleeper/12"
SEASON = 2026
POSITIONS = {"QB", "RB", "WR", "TE"}


def fetch_draftsharks() -> pd.DataFrame:
    print(f"Fetching DraftSharks ADP from {URL}...")
    resp = requests.get(URL, timeout=20)
    resp.raise_for_status()

    # Extract the vueAppData JSON embedded in a <script> tag
    match = re.search(r"var\s+vueAppData\s*=\s*", resp.text)
    if not match:
        raise ValueError("Could not find vueAppData in DraftSharks page.")

    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(resp.text, match.end())

    teams = {t["id"]: t["abbr"] for t in data.get("teams", [])}
    players = data.get("projections", [])

    rows = []
    for p in players:
        pos = p.get("position", "")
        if pos not in POSITIONS:
            continue

        # Find the ADP entry for 12-team leagues
        adp_entry = None
        for key, val in p.get("adps", {}).items():
            if val.get("league_size") == 12:
                adp_entry = val
                break

        if adp_entry is None:
            continue

        team_id = p.get("team_id") or (p.get("team", {}) or {}).get("id")
        rows.append({
            "player_name": f"{p['first_name']} {p['last_name']}".strip(),
            "position": pos,
            "team": teams.get(team_id, ""),
            "adp_overall": adp_entry.get("overall_pick_number"),
        })

    df = pd.DataFrame(rows).dropna(subset=["adp_overall"])
    df = df.sort_values("adp_overall")
    df["adp_position_rank"] = df.groupby("position")["adp_overall"].rank(method="first").astype(int)
    print(f"  Parsed {len(df)} players.")
    return df


def match_ids(df: pd.DataFrame, engine) -> pd.DataFrame:
    with engine.connect() as conn:
        players = pd.read_sql("SELECT player_id, full_name, position FROM players", conn)
    df["name_norm"] = df["player_name"].str.lower().str.strip()
    players["name_norm"] = players["full_name"].str.lower().str.strip()
    merged = df.merge(players[["player_id", "name_norm", "position"]], on=["name_norm", "position"], how="left")
    n = merged["player_id"].isna().sum()
    if n:
        print(f"  {n} unmatched (rookies/name mismatch).")
    return merged.dropna(subset=["player_id"])


def load_ecr(engine):
    df = fetch_draftsharks()
    df = match_ids(df, engine)
    df["season"] = SEASON
    df["source"] = "draftsharks"

    rows = df[["player_id", "season", "adp_overall", "adp_position_rank", "source"]].to_dict("records")
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]
    rows = [r for r in rows if r["player_id"] is not None]

    upsert = text("""
        INSERT INTO adp (player_id, season, adp_overall, adp_position_rank, source)
        VALUES (:player_id, :season, :adp_overall, :adp_position_rank, :source)
        ON CONFLICT (player_id, season, source) DO UPDATE SET
            adp_overall = EXCLUDED.adp_overall, adp_position_rank = EXCLUDED.adp_position_rank
    """)
    with engine.begin() as conn:
        conn.execute(upsert, rows)
    print(f"  Upserted {len(rows)} ECR rows for {SEASON}.")


if __name__ == "__main__":
    engine = get_engine()
    load_ecr(engine)

    with engine.connect() as conn:
        for row in conn.execute(text("""
            SELECT position, COUNT(*) AS n
            FROM adp JOIN players USING (player_id)
            WHERE season = :season AND source = 'draftsharks'
            GROUP BY position ORDER BY position
        """), {"season": SEASON}):
            print(f"  {row.position}: {row.n} players")
