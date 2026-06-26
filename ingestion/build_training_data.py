from pathlib import Path

import pandas as pd
from db import get_engine

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"

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

# Advanced feature columns per position (from the features/ parquets).
# Both current-season and prev_* versions are included in the mega table:
# current-season for EDA/correlation work; prev_* as model predictors.
#
# Column sources:
#   seasonal.parquet  — target_share, air_yards_share, wopr, racr, adot,
#                        air_yards, yards_after_catch, ppr_sh
#   ngs.parquet       — avg_time_to_throw, avg_intended_air_yards (passing),
#                        avg_intended_air_yards_receiving (receiving),
#                        aggressiveness, completion_percentage_above_expectation,
#                        efficiency, percent_attempts_gte_eight_defenders,
#                        avg_separation, avg_cushion
#   snaps.parquet     — offense_snaps, snap_pct
#   pbp.parquet       — rz_targets, rz_carries, goalline_carries,
#                        designed_rushes, scrambles, carry_share,
#                        team_pass_rate, pressure_to_sack_rate
POSITION_FEATURE_COLS = {
    "QB": [
        # NGS passing
        "avg_time_to_throw",
        "avg_intended_air_yards",           # passing aDOT proxy (no suffix — from passing frame)
        "aggressiveness",
        "completion_percentage_above_expectation",
        # snaps
        "snap_pct",
        "offense_snaps",
        # pbp
        "designed_rushes",
        "scrambles",
        "team_pass_rate",
        "pressure_to_sack_rate",
    ],
    "RB": [
        # seasonal
        "target_share",
        "air_yards_share",
        "wopr",
        "racr",
        "adot",
        "air_yards",
        "yards_after_catch",
        # NGS rushing + receiving
        "efficiency",
        "percent_attempts_gte_eight_defenders",
        "avg_separation",
        "avg_cushion",
        # snaps
        "snap_pct",
        "offense_snaps",
        # pbp
        "rz_targets",
        "rz_carries",
        "goalline_carries",
        "carry_share",
        "team_pass_rate",
    ],
    "WR": [
        # seasonal
        "target_share",
        "air_yards_share",
        "wopr",
        "racr",
        "adot",
        "air_yards",
        "yards_after_catch",
        # NGS receiving
        "avg_separation",
        "avg_cushion",
        "avg_intended_air_yards_receiving",  # receiving aDOT from NGS (suffixed by merge)
        # snaps
        "snap_pct",
        "offense_snaps",
        # pbp
        "rz_targets",
        "team_pass_rate",
    ],
    "TE": [
        # seasonal
        "target_share",
        "air_yards_share",
        "wopr",
        "racr",
        "adot",
        "air_yards",
        "yards_after_catch",
        # NGS receiving
        "avg_separation",
        "avg_cushion",
        "avg_intended_air_yards_receiving",
        # snaps
        "snap_pct",
        "offense_snaps",
        # pbp
        "rz_targets",
        "team_pass_rate",
    ],
}


def load_features() -> pd.DataFrame:
    """
    Read the four feature parquets from data/features/ and outer-join on
    (player_id, season). Columns already present from an earlier parquet are
    not duplicated. Returns an empty DataFrame if no parquets exist yet.
    """
    parquet_names = ["seasonal", "ngs", "snaps", "pbp"]
    result = None

    for name in parquet_names:
        path = FEATURES_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"  {name} features not found — run features/pull_{name}.py first")
            continue
        df = pd.read_parquet(path)
        print(f"  Loaded {name}: {len(df):,} player-seasons")

        if result is None:
            result = df
        else:
            # Only bring in columns not already present to avoid duplicates.
            new_cols = ["player_id", "season"] + [
                c for c in df.columns if c not in result.columns
            ]
            result = result.merge(df[new_cols], on=["player_id", "season"], how="outer")

    return result if result is not None else pd.DataFrame()


def build_position_table(pos: str, df: pd.DataFrame) -> pd.DataFrame:
    stat_cols    = POSITION_STAT_COLS[pos]
    feature_cols = POSITION_FEATURE_COLS[pos]
    prev_stat_cols    = [f"prev_{c}" for c in stat_cols]
    prev_feature_cols = [f"prev_{c}" for c in feature_cols]

    base_cols = [
        "full_name", "season", "career_season", "team",
        "adp_overall", "adp_position_rank", "finish",
        "ppg", "prev_ppg", "ppg_delta", "pts_delta",
    ]
    all_cols  = base_cols + stat_cols + feature_cols + prev_stat_cols + prev_feature_cols

    # Keep only columns that actually exist (features may not all be loaded yet).
    present = [c for c in all_cols if c in df.columns]
    return df[df["position"] == pos][present].sort_values(["season", "finish"])


def main():
    engine = get_engine()
    with engine.connect() as conn:
        stats = pd.read_sql("SELECT * FROM season_stats", conn)
        adp   = pd.read_sql("SELECT * FROM adp WHERE source = 'ffc'", conn)

    # Only include players who have ever appeared in ADP (fantasy-relevant at least once).
    relevant_players = set(adp["player_id"])
    stats = stats[stats["player_id"].isin(relevant_players)].copy()

    # Outcome 1: finish = position rank by PPR points within each (season, position).
    stats["finish"] = (
        stats.groupby(["season", "position"])["fantasy_points_ppr"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    # Career season number (1 = first season this player appears in the dataset).
    stats["career_season"] = (
        stats.groupby("player_id")["season"]
        .rank(method="first")
        .astype(int)
    )

    # PPG — shifted forward so prev_ppg lands as a predictor on the next row.
    stats["ppg"] = (
        stats["fantasy_points_ppr"] / stats["games_played"]
    ).where(stats["games_played"] > 0)

    # Average ADP across sources per player-season (only 'ffc' for now).
    adp_agg = (
        adp.groupby(["player_id", "season"])
        .agg(adp_overall=("adp_overall", "mean"), adp_position_rank=("adp_position_rank", "mean"))
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Prior-season counting stats — shift season +1 so they join on the
    # *next* season's rows.
    # ------------------------------------------------------------------
    all_stat_cols = [
        "ppg", "fantasy_points_ppr", "games_played",
        "completions", "pass_attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ]
    prev_stats = stats[["player_id", "season"] + all_stat_cols].copy()
    prev_stats = prev_stats.rename(columns={c: f"prev_{c}" for c in all_stat_cols})
    prev_stats["season"] = prev_stats["season"] + 1

    # ------------------------------------------------------------------
    # Load feature parquets and build prev_* versions.
    # ------------------------------------------------------------------
    print("\nLoading feature parquets...")
    features = load_features()

    if not features.empty:
        feat_cols = [c for c in features.columns if c not in ("player_id", "season")]
        prev_features = features[["player_id", "season"] + feat_cols].copy()
        prev_features = prev_features.rename(columns={c: f"prev_{c}" for c in feat_cols})
        prev_features["season"] = prev_features["season"] + 1
    else:
        prev_features = pd.DataFrame()

    # ------------------------------------------------------------------
    # Join everything onto current-season stats rows.
    # ------------------------------------------------------------------
    df = stats.merge(adp_agg,    on=["player_id", "season"], how="left")
    df = df.merge(prev_stats,    on=["player_id", "season"], how="left")

    if not features.empty:
        # Current-season features (for EDA; not used as model inputs at draft time).
        feat_new_cols = ["player_id", "season"] + [
            c for c in features.columns if c not in df.columns
        ]
        df = df.merge(features[feat_new_cols], on=["player_id", "season"], how="left")
        # Prior-season features (used as model predictors).
        df = df.merge(prev_features, on=["player_id", "season"], how="left")

    # Fill undrafted players with end-of-draft proxy values so ADP is never null.
    df["adp_overall"]       = df["adp_overall"].fillna(300.0)
    df["adp_position_rank"] = df["adp_position_rank"].fillna(75.0)

    # Delta features: how much did this player improve/decline vs last season?
    df["ppg_delta"] = df["ppg"] - df["prev_ppg"]
    df["pts_delta"] = df["fantasy_points_ppr"] - df["prev_fantasy_points_ppr"]

    # Drop pre-2016 seasons — they exist only to supply prev_* stats for 2016 rows.
    df = df[df["season"] >= 2016]

    for pos in ["QB", "RB", "WR", "TE"]:
        table = build_position_table(pos, df)
        out_path = f"data/training_data_{pos.lower()}.csv"
        table.to_csv(out_path, index=False)
        print(f"\n=== {pos} ({len(table)} rows) -> {out_path} ===")
        print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
