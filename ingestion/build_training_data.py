"""
Inspect the ingested data by displaying the contents of the players,
season_stats, and adp tables, and report any players that could not be
matched across tables (e.g. ADP names with no player_id match).

This is a diagnostic script — run it after ingestion to verify data quality
and identify name-matching issues that need manual fixes.

Run: python ingestion/build_training_data.py
"""

import pandas as pd
from db import get_engine


def load_tables(engine):
    """Load all three core tables from the database into DataFrames."""
    with engine.connect() as conn:
        players = pd.read_sql("SELECT * FROM players", conn)
        stats = pd.read_sql("SELECT * FROM season_stats", conn)
        adp = pd.read_sql("SELECT * FROM adp", conn)
    return players, stats, adp


def display_table(name: str, df: pd.DataFrame):
    """Pretty-print a summary and the first rows of a table."""
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  {len(df)} rows")
    print(f"{'=' * 60}")
    print(f"Columns: {list(df.columns)}\n")
    # Show a generous preview so the user can eyeball data quality.
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_rows", 30):
        print(df.head(30).to_string(index=False))
    print()


def find_unmatched_stats(players: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Find season_stats rows whose player_id does not appear in the players table.
    This shouldn't happen due to the FK constraint but is a useful sanity check.
    """
    missing = stats[~stats["player_id"].isin(players["player_id"])]
    return missing[["player_id", "season"]].drop_duplicates()


def find_unmatched_adp(players: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """
    Find adp rows whose player_id does not appear in the players table.
    These represent ADP entries that were matched at ingest time but whose
    player record may have since been removed, or data integrity issues.
    """
    missing = adp[~adp["player_id"].isin(players["player_id"])]
    return missing[["player_id", "season", "source"]].drop_duplicates()


def find_players_without_stats(players: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Find players in the players table who have zero season_stats rows.
    These are typically rookies, practice-squad players, or injured reserves
    who were rostered but never recorded meaningful stats.
    """
    missing = players[~players["player_id"].isin(stats["player_id"])]
    return missing[["player_id", "full_name", "position", "team"]]


def find_players_without_adp(players: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """
    Find players in the players table who have zero adp rows.
    These are typically late-round or undrafted players who never appeared
    in ADP data from FFC or DraftSharks.
    """
    missing = players[~players["player_id"].isin(adp["player_id"])]
    return missing[["player_id", "full_name", "position", "team"]]


def main():
    engine = get_engine()
    players, stats, adp = load_tables(engine)

    # --- Display each table ---
    display_table("PLAYERS", players)
    display_table("SEASON_STATS", stats)
    display_table("ADP", adp)

    # --- Report unmatched / orphaned records ---
    print("\n" + "=" * 60)
    print("  UNMATCHED / ORPHANED RECORDS")
    print("=" * 60)

    # Stats rows with no matching player (FK violation check).
    orphan_stats = find_unmatched_stats(players, stats)
    if not orphan_stats.empty:
        print(f"\n  season_stats rows with unknown player_id ({len(orphan_stats)}):")
        print(orphan_stats.to_string(index=False))
    else:
        print("\n  All season_stats rows have a matching player.  OK")

    # ADP rows with no matching player.
    orphan_adp = find_unmatched_adp(players, adp)
    if not orphan_adp.empty:
        print(f"\n  adp rows with unknown player_id ({len(orphan_adp)}):")
        print(orphan_adp.to_string(index=False))
    else:
        print("\n  All adp rows have a matching player.  OK")

    # Players with no stats at all.
    no_stats = find_players_without_stats(players, stats)
    print(f"\n  Players with NO season_stats ({len(no_stats)} of {len(players)}):")
    if not no_stats.empty:
        with pd.option_context("display.max_rows", 50, "display.width", 200):
            print(no_stats.to_string(index=False))

    # Players with no ADP at all.
    no_adp = find_players_without_adp(players, adp)
    print(f"\n  Players with NO adp entries ({len(no_adp)} of {len(players)}):")
    if not no_adp.empty:
        with pd.option_context("display.max_rows", 50, "display.width", 200):
            print(no_adp.to_string(index=False))

    print()


if __name__ == "__main__":
    main()
