"""
Scrape 2026 ECR/ADP from DraftSharks (PPR, Sleeper, 12-team).
Populates: adp table (source='draftsharks', season=2026).

DraftSharks embeds all player projection data as a JavaScript object
(vueAppData) in the page HTML, so we parse it with regex + JSON rather
than scraping an HTML table.

Run: python ingestion/ingest_ecr.py
"""

import re
import json
import requests
import pandas as pd
from sqlalchemy import text
from db import get_engine

# DraftSharks ADP page — PPR scoring, Sleeper platform, 12-team leagues.
URL = "https://www.draftsharks.com/adp/ppr/sleeper/12"

# Season this ECR data applies to (the upcoming draft year).
SEASON = 2026

# Only include fantasy-relevant skill positions.
POSITIONS = {"QB", "RB", "WR", "TE"}


def fetch_draftsharks() -> pd.DataFrame:
    """
    Download the DraftSharks ADP page and extract player data from the
    embedded vueAppData JavaScript object.

    Returns a DataFrame with columns:
      player_name, position, team, adp_overall, adp_position_rank
    """
    print(f"Fetching DraftSharks ADP from {URL}...")
    resp = requests.get(URL, timeout=20)
    resp.raise_for_status()

    # Locate the vueAppData variable declaration in the page's <script> tags.
    # This contains all player projections, team mappings, and ADP data.
    match = re.search(r"var\s+vueAppData\s*=\s*", resp.text)
    if not match:
        raise ValueError("Could not find vueAppData in DraftSharks page.")

    # Use JSONDecoder.raw_decode to parse the JSON object starting right
    # after the "var vueAppData = " assignment (ignoring trailing JS).
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(resp.text, match.end())

    # Build a team-ID → abbreviation lookup from the embedded teams list.
    teams = {t["id"]: t["abbr"] for t in data.get("teams", [])}

    # "projections" holds one entry per player with nested ADP data.
    players = data.get("projections", [])

    rows = []
    for p in players:
        pos = p.get("position", "")
        if pos not in POSITIONS:
            continue

        # Each player has an "adps" dict keyed by format; find the 12-team entry.
        adp_entry = None
        for key, val in p.get("adps", {}).items():
            if val.get("league_size") == 12:
                adp_entry = val
                break

        # Skip players without a 12-team ADP value.
        if adp_entry is None:
            continue

        # Resolve the team abbreviation from the player's team_id.
        team_id = p.get("team_id") or (p.get("team", {}) or {}).get("id")
        rows.append({
            "player_name": f"{p['first_name']} {p['last_name']}".strip(),
            "position": pos,
            "team": teams.get(team_id, ""),
            "adp_overall": adp_entry.get("overall_pick_number"),
        })

    # Drop rows missing ADP, sort by overall pick, then compute position ranks.
    df = pd.DataFrame(rows).dropna(subset=["adp_overall"])
    df = df.sort_values("adp_overall")
    df["adp_position_rank"] = df.groupby("position")["adp_overall"].rank(method="first").astype(int)
    print(f"  Parsed {len(df)} players.")
    return df


def match_ids(df: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Join scraped player names to the players table by normalized name + position.

    Unmatched players (rookies not yet in the DB, name mismatches) are logged
    and dropped. Returns only rows with a valid player_id.
    """
    with engine.connect() as conn:
        players = pd.read_sql("SELECT player_id, full_name, position FROM players", conn)

    # Normalize names to lowercase/stripped for matching.
    df["name_norm"] = df["player_name"].str.lower().str.strip()
    players["name_norm"] = players["full_name"].str.lower().str.strip()

    merged = df.merge(players[["player_id", "name_norm", "position"]], on=["name_norm", "position"], how="left")
    n = merged["player_id"].isna().sum()
    if n:
        print(f"  {n} unmatched (rookies/name mismatch).")
    return merged.dropna(subset=["player_id"])


def load_ecr(engine):
    """
    Main pipeline: fetch DraftSharks ECR, match to player IDs, and upsert
    into the adp table with source='draftsharks' and season=2026.
    """
    df = fetch_draftsharks()
    df = match_ids(df, engine)
    df["season"] = SEASON
    df["source"] = "draftsharks"

    # Prepare rows for upsert, converting NaN → None for the DB driver.
    rows = df[["player_id", "season", "adp_overall", "adp_position_rank", "source"]].to_dict("records")
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]
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
    print(f"  Upserted {len(rows)} ECR rows for {SEASON}.")


if __name__ == "__main__":
    engine = get_engine()
    load_ecr(engine)

    # Print a quick breakdown by position for verification.
    with engine.connect() as conn:
        for row in conn.execute(text("""
            SELECT position, COUNT(*) AS n
            FROM adp JOIN players USING (player_id)
            WHERE season = :season AND source = 'draftsharks'
            GROUP BY position ORDER BY position
        """), {"season": SEASON}):
            print(f"  {row.position}: {row.n} players")
