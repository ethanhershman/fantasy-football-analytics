import pandas as pd
from db import get_engine


# Counting-stat columns relevant to each position (current season).
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

# PBP-derived efficiency features relevant to each position.
# prev_ versions of these are also included automatically (see build_training_data).
POSITION_PBP_COLS = {
    "QB": [
        "cpoe",
        "air_yards_per_attempt",
        "rz_pass_attempts",
    ],
    "RB": [
        "target_share",
        "rz_targets",
        "adot",
        "catch_rate",
        "yac_per_reception",
        "air_yards_share",
        "rz_target_share",
        "rz_carries",
        "goalline_carries",
        "rz_carry_share",
        "opportunity_share",
    ],
    "WR": [
        "target_share",
        "rz_targets",
        "adot",
        "catch_rate",
        "yac_per_reception",
        "air_yards_share",
        "rz_target_share",
    ],
    "TE": [
        "target_share",
        "rz_targets",
        "adot",
        "catch_rate",
        "yac_per_reception",
        "air_yards_share",
        "rz_target_share",
    ],
}


def build_position_table(pos, df):
    stat_cols = POSITION_STAT_COLS[pos]
    pbp_cols  = POSITION_PBP_COLS[pos]
    prev_stat_cols = [f"prev_{c}" for c in stat_cols]
    prev_pbp_cols  = [f"prev_{c}" for c in pbp_cols]

    base_cols = ["full_name", "season", "team", "adp_overall", "adp_position_rank", "finish"]
    all_cols = base_cols + stat_cols + pbp_cols + prev_stat_cols + prev_pbp_cols

    # Keep only columns that actually exist (pbp_features may not be loaded yet).
    present = [c for c in all_cols if c in df.columns]

    return df[df["position"] == pos][present].sort_values(["season", "finish"])


def main():
    engine = get_engine()
    with engine.connect() as conn:
        stats = pd.read_sql("SELECT * FROM season_stats", conn)
        adp   = pd.read_sql("SELECT * FROM adp WHERE source = 'ffc'", conn)
        # pbp_features is optional — skip gracefully if the table is empty or absent.
        try:
            pbp = pd.read_sql("SELECT * FROM pbp_features", conn)
        except Exception:
            pbp = pd.DataFrame()

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

    # ------------------------------------------------------------------
    # Prior-season counting stats (shift season +1 so they join on the
    # *next* season's rows).
    # ------------------------------------------------------------------
    all_stat_cols = [
        "fantasy_points_ppr", "games_played",
        "completions", "pass_attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ]
    prev_stats = stats[["player_id", "season"] + all_stat_cols].copy()
    prev_stats = prev_stats.rename(columns={c: f"prev_{c}" for c in all_stat_cols})
    prev_stats["season"] = prev_stats["season"] + 1

    # ------------------------------------------------------------------
    # Prior-season PBP features (same shift-forward pattern).
    # ------------------------------------------------------------------
    if not pbp.empty:
        all_pbp_cols = [
            c for c in pbp.columns if c not in ("player_id", "season")
        ]
        prev_pbp = pbp[["player_id", "season"] + all_pbp_cols].copy()
        prev_pbp = prev_pbp.rename(columns={c: f"prev_{c}" for c in all_pbp_cols})
        prev_pbp["season"] = prev_pbp["season"] + 1
    else:
        prev_pbp = pd.DataFrame()

    # ------------------------------------------------------------------
    # Join everything onto current-season stats rows.
    # ------------------------------------------------------------------
    df = stats.merge(adp_agg, on=["player_id", "season"], how="left")
    df = df.merge(prev_stats, on=["player_id", "season"], how="left")
    if not pbp.empty:
        df = df.merge(pbp,      on=["player_id", "season"], how="left")
        df = df.merge(prev_pbp, on=["player_id", "season"], how="left")

    # Drop pre-2016 seasons — they exist in the DB only to supply prev_* stats
    # for the 2016 training rows (e.g. 2016 needs 2015 prev_ data).
    df = df[df["season"] >= 2016]

    for pos in ["QB", "RB", "WR", "TE"]:
        table = build_position_table(pos, df)
        out_path = f"data/training_data_{pos.lower()}.csv"
        table.to_csv(out_path, index=False)
        print(f"\n=== {pos} ({len(table)} rows) -> {out_path} ===")
        print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
