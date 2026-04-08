import pandas as pd
from db import get_engine


# Stat columns relevant to each position (current season).
POSITION_STAT_COLS = {
    "QB": [
        "games_played", "fantasy_points_ppr",
        "completions", "pass_attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
    ],
    "RB": [
        "games_played", "fantasy_points_ppr",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ],
    "WR": [
        "games_played", "fantasy_points_ppr",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ],
    "TE": [
        "games_played", "fantasy_points_ppr",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ],
}


def build_position_table(pos, df):
    stat_cols = POSITION_STAT_COLS[pos]
    prev_cols = [f"prev_{c}" for c in stat_cols]

    base_cols = ["full_name", "season", "team", "adp_overall", "adp_position_rank", "finish"]
    return df[df["position"] == pos][base_cols + stat_cols + prev_cols].sort_values(
        ["season", "finish"]
    )


def main():
    engine = get_engine()
    with engine.connect() as conn:
        stats = pd.read_sql("SELECT * FROM season_stats", conn)
        adp = pd.read_sql("SELECT * FROM adp WHERE source = 'ffc'", conn)

    # Only include players who have ever appeared in ADP (fantasy-relevant at least once).
    relevant_players = set(adp["player_id"])
    stats = stats[stats["player_id"].isin(relevant_players)].copy()

    # Compute finish = position rank by PPR points within each (season, position).
    stats["finish"] = (
        stats.groupby(["season", "position"])["fantasy_points_ppr"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    # Average ADP across sources per player-season (only 'ffc' for now).
    adp_agg = (
        adp.groupby(["player_id", "season"])
        .agg(adp_overall=("adp_overall", "mean"), adp_position_rank=("adp_position_rank", "mean"))
        .reset_index()
    )

    # Build prior-season stats by shifting season forward by 1 so they join on the *next* season.
    all_stat_cols = [
        "fantasy_points_ppr", "games_played",
        "completions", "pass_attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ]
    prev = stats[["player_id", "season"] + all_stat_cols].copy()
    prev = prev.rename(columns={c: f"prev_{c}" for c in all_stat_cols})
    prev["season"] = prev["season"] + 1

    # Join ADP and prior-season stats onto current season rows.
    df = stats.merge(adp_agg, on=["player_id", "season"], how="left")
    df = df.merge(prev, on=["player_id", "season"], how="left")

    # Drop 2019 — it's only in the DB to supply prev_* stats for 2020 rows.
    df = df[df["season"] >= 2020]

    for pos in ["QB", "RB", "WR", "TE"]:
        table = build_position_table(pos, df)
        out_path = f"data/training_data_{pos.lower()}.csv"
        table.to_csv(out_path, index=False)
        print(f"\n=== {pos} ({len(table)} rows) -> {out_path} ===")
        print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
