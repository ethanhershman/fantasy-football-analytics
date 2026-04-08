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

# The ADP key that corresponds to Redraft / PPR / 1QB / Sleeper / 12-team.
# Derived from the AdpDash.js switch logic:
#   type=redraft + scoring=ppr + non-superflex → format_id=11
#   platform=sleeper → source_id=107
#   league_size=12
# Key format is "format_id::source_id::league_size".
ADP_KEY = "11::107::12"

# Explicit name overrides for cases that normalization alone can't fix
# (different first names, nicknames, etc.).
# Maps DraftSharks name → nflverse name.
NAME_FIXES = {
    "Cameron Skattebo":    "Cam Skattebo",
    "Chigoziem Okonkwo":   "Chig Okonkwo",
    "Nathaniel Dell":      "Tank Dell",
    "Cameron Ward":        "Cam Ward",
}


def _normalize_name(name: str) -> str:
    """
    Normalize a player name for matching by:
      1. Lowercasing and stripping whitespace.
      2. Removing suffixes (Jr., Sr., Jr, Sr, II, III, IV).
      3. Removing dots (so "A.J." becomes "AJ", "D.K." becomes "DK").
      4. Collapsing extra whitespace left behind.
    """
    import re
    n = name.lower().strip()
    n = re.sub(r'\b(jr\.?|sr\.?|iii|ii|iv)\s*$', '', n).strip()
    # Remove dots and apostrophes (A.J. → AJ, D.K. → DK, Le'Veon → LeVeon).
    n = n.replace(".", "").replace("'", "")
    n = re.sub(r'\s+', ' ', n).strip()
    return n


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

        # Look up the exact ADP entry for Redraft/PPR/1QB/Sleeper/12-team.
        adp_entry = p.get("adps", {}).get(ADP_KEY)

        # Skip players without an ADP value for this specific configuration.
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


def match_ids(df: pd.DataFrame, engine):
    """
    Join scraped player names to the players table by normalized name + position.

    1. Applies NAME_FIXES to correct known DraftSharks ↔ nflverse mismatches.
    2. Matches by lowercase name + position.
    3. Inserts any remaining unmatched players (rookies) into the players table
       with a generated player_id so they can still get ECR rows.

    Returns a tuple of (matched_df, originally_unmatched_df).
    The originally_unmatched_df is for reporting — all players end up matched.
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

    matched = merged.dropna(subset=["player_id"]).copy()
    unmatched = merged[merged["player_id"].isna()].copy()

    if not unmatched.empty:
        print(f"  {len(unmatched)} unmatched (rookies/unsigned) — assigning generated IDs...")

        # Generate ds- player_ids for rookies. Since the adp table has no FK,
        # these don't need to exist anywhere else.
        for idx, row in unmatched.iterrows():
            pid = f"ds-{row['player_name'].lower().replace(' ', '-').replace('.', '')}"
            unmatched.loc[idx, "player_id"] = pid

        matched = pd.concat([matched, unmatched], ignore_index=True)

    return matched, unmatched


def load_ecr(engine):
    """
    Main pipeline: fetch DraftSharks ECR, match to player IDs, and upsert
    into the adp table with source='draftsharks' and season=2026.

    Returns (matched_df, unmatched_df) so callers can inspect results.
    """
    df = fetch_draftsharks()
    matched, unmatched = match_ids(df, engine)
    matched["season"] = SEASON
    matched["source"] = "draftsharks"

    # Prepare rows for upsert, converting NaN → None for the DB driver.
    rows = matched[["player_id", "season", "adp_overall", "adp_position_rank", "source"]].to_dict("records")
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

    return matched, unmatched


if __name__ == "__main__":
    engine = get_engine()
    matched, rookies = load_ecr(engine)

    veterans = matched[~matched["player_id"].str.startswith("ds-")]
    display_cols = ["player_name", "position", "team", "adp_overall", "adp_position_rank"]

    print(f"\n{'=' * 70}")
    print(f"  VETERAN ECR PLAYERS ({len(veterans)} rows) — matched to season_stats")
    print(f"{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(veterans[display_cols].sort_values("adp_overall").to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"  ROOKIES / UNSIGNED ({len(rookies)} rows) — assigned generated IDs")
    print(f"{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(rookies[display_cols].sort_values("adp_overall").to_string(index=False))
