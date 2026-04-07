"""
Phase 1 — Ingestion Script 3: FantasyPros ECR (Expert Consensus Rankings)

Scrapes the PPR cheatsheet from FantasyPros once per season and stores
each player's ECR rank in the rankings table.

Scrape policy:
  - One request total, result saved immediately.
  - Do not run this more than once per day.

Run:
    python ingestion/ingest_ecr.py
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import text
from db import get_engine

ECR_URL = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
CURRENT_SEASON = 2026
POSITIONS = ["QB", "RB", "WR", "TE"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_ecr() -> pd.DataFrame:
    print(f"Scraping FantasyPros ECR from {ECR_URL}...")
    time.sleep(2)  # polite delay before the request

    response = requests.get(ECR_URL, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "rank-data"})

    if table is None:
        raise ValueError(
            "Could not find rankings table on FantasyPros page. "
            "The page structure may have changed."
        )

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        rank_text = tds[0].get_text(strip=True)
        player_cell = tds[2]
        pos_text = tds[3].get_text(strip=True).upper()

        if not rank_text.isdigit():
            continue
        if pos_text not in POSITIONS:
            continue

        # Player name is in a link inside the cell
        name_tag = player_cell.find("a")
        player_name = name_tag.get_text(strip=True) if name_tag else player_cell.get_text(strip=True)
        # Strip team abbreviation that sometimes appears in parens, e.g. "CeeDee Lamb (DAL)"
        if "(" in player_name:
            player_name = player_name[:player_name.index("(")].strip()

        rows.append({
            "player_name": player_name,
            "position": pos_text,
            "ecr_rank": int(rank_text),
        })

    df = pd.DataFrame(rows)
    print(f"  Scraped {len(df)} players.")
    return df


def match_to_player_ids(df: pd.DataFrame, engine) -> pd.DataFrame:
    with engine.connect() as conn:
        players = pd.read_sql("SELECT player_id, full_name, position FROM players", conn)

    df["name_norm"] = df["player_name"].str.lower().str.strip()
    players["name_norm"] = players["full_name"].str.lower().str.strip()

    merged = df.merge(
        players[["player_id", "name_norm", "position"]],
        on=["name_norm", "position"],
        how="left",
    )

    unmatched = merged["player_id"].isna().sum()
    if unmatched > 0:
        print(f"  {unmatched} players unmatched (rookies or name mismatches). Dropping.")

    return merged.dropna(subset=["player_id"])


def load_ecr(engine):
    df = scrape_ecr()
    df = match_to_player_ids(df, engine)

    rows = df[["player_id", "ecr_rank"]].copy()
    rows["season"] = CURRENT_SEASON
    rows = rows.to_dict("records")

    upsert_sql = text("""
        INSERT INTO rankings (player_id, season, ecr_rank)
        VALUES (:player_id, :season, :ecr_rank)
        ON CONFLICT (player_id, season) DO UPDATE SET
            ecr_rank   = EXCLUDED.ecr_rank,
            updated_at = NOW()
    """)

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    print(f"  Upserted {len(rows)} ECR rankings for {CURRENT_SEASON}.")


if __name__ == "__main__":
    engine = get_engine()
    load_ecr(engine)

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT p.position, COUNT(*) AS ranked_players
            FROM rankings r
            JOIN players p USING (player_id)
            WHERE r.season = 2026
            GROUP BY p.position ORDER BY p.position
        """))
        print(f"\nECR rankings loaded for {CURRENT_SEASON}:")
        for row in result:
            print(f"  {row.position}: {row.ranked_players} players ranked")
