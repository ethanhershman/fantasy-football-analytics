"""
PySpark ETL: aggregate play-by-play data into per-player-season features
and upsert into the pbp_features table.

Reads:  data/pbp/pbp_{year}.parquet  (produced by ingestion/ingest_pbp.py)
Writes: pbp_features table in Postgres (via .toPandas() + db.py — no JDBC JAR needed)

Schema differences across seasons are handled via mergeSchema.  Columns absent
from older seasons (e.g. cpoe, qb_scramble) are back-filled with 0 / NULL so
aggregations degrade gracefully rather than raising errors.

Run (from project root): python etl/pbp_features.py
"""

import os
import sys

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from sqlalchemy import text

# Allow importing db.py from the ingestion directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from db import get_engine

PBP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pbp"))


# ---------------------------------------------------------------------------
# Spark helpers
# ---------------------------------------------------------------------------

def _ensure_col(df, col_name, default):
    """Add col_name with a constant default if it doesn't exist; coalesce NULLs otherwise."""
    if col_name not in df.columns:
        return df.withColumn(col_name, F.lit(default))
    return df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(default)))


def _dominant_team(df, id_col):
    """
    For each (player, season) return the posteam with the most play appearances.
    Handles mid-season trades by picking the majority team.
    """
    counts = (
        df.filter(F.col(id_col).isNotNull())
        .groupBy(id_col, "season", "posteam")
        .agg(F.count("*").alias("n"))
    )
    w = Window.partitionBy(id_col, "season").orderBy(F.col("n").desc())
    return (
        counts
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select(F.col(id_col).alias("player_id"), "season", "posteam")
    )


# ---------------------------------------------------------------------------
# Team-level denominators (shared across all share metrics)
# ---------------------------------------------------------------------------

def build_team_stats(pbp):
    """
    Season-level team totals used as denominators for target_share,
    air_yards_share, rz_target_share, rz_carry_share, and opportunity_share.
    """
    pass_plays = pbp.filter(F.col("pass_attempt") == 1)
    rush_plays = pbp.filter(
        (F.col("rush_attempt") == 1) | (F.col("qb_scramble") == 1)
    )

    team_pass = pass_plays.groupBy("posteam", "season").agg(
        F.count("*").alias("team_targets"),
        F.sum(F.coalesce(F.col("air_yards"), F.lit(0.0))).alias("team_air_yards"),
        F.sum(F.when(F.col("yardline_100") <= 20, 1).otherwise(0)).alias("team_rz_targets"),
    )
    team_rush = rush_plays.groupBy("posteam", "season").agg(
        F.count("*").alias("team_rush_attempts"),
        F.sum(F.when(F.col("yardline_100") <= 20, 1).otherwise(0)).alias("team_rz_rushes"),
    )
    return team_pass.join(team_rush, on=["posteam", "season"], how="outer")


# ---------------------------------------------------------------------------
# Per-role aggregations
# ---------------------------------------------------------------------------

def build_passer_features(pbp):
    """
    Features keyed on passer_player_id:
      cpoe, air_yards_per_attempt, rz_pass_attempts
    """
    throws = pbp.filter(
        (F.col("pass_attempt") == 1)
        & (F.col("sack") != 1)
        & F.col("passer_player_id").isNotNull()
    )
    return (
        throws.groupBy("passer_player_id", "season").agg(
            F.count("*").alias("_n_throws"),
            F.sum(F.coalesce(F.col("air_yards"), F.lit(0.0))).alias("_sum_air_yards"),
            F.avg("cpoe").alias("cpoe"),
            F.sum(F.when(F.col("yardline_100") <= 20, 1).otherwise(0)).alias("rz_pass_attempts"),
        )
        .withColumn(
            "air_yards_per_attempt",
            F.when(F.col("_n_throws") > 0,
                   F.col("_sum_air_yards") / F.col("_n_throws")),
        )
        .select(
            F.col("passer_player_id").alias("player_id"),
            "season", "cpoe", "air_yards_per_attempt", "rz_pass_attempts",
        )
    )


def build_receiver_features(pbp, team_stats):
    """
    Features keyed on receiver_player_id:
      targets, rz_targets, adot, catch_rate, yac_per_reception,
      target_share, air_yards_share, rz_target_share
    Also returns receiver_teams for use in opportunity_share.
    """
    targeted = pbp.filter(
        (F.col("play_type") == "pass")
        & F.col("receiver_player_id").isNotNull()
    )
    player_team = _dominant_team(targeted, "receiver_player_id")

    agg = (
        targeted.groupBy("receiver_player_id", "season").agg(
            F.count("*").alias("targets"),
            F.sum(F.coalesce(F.col("complete_pass"), F.lit(0))).alias("_receptions"),
            F.sum(F.coalesce(F.col("air_yards"), F.lit(0.0))).alias("_sum_air_yards"),
            F.sum(F.coalesce(F.col("yards_after_catch"), F.lit(0.0))).alias("_sum_yac"),
            F.sum(F.when(F.col("yardline_100") <= 20, 1).otherwise(0)).alias("rz_targets"),
        )
        .withColumnRenamed("receiver_player_id", "player_id")
    )

    team_cols = team_stats.select(
        "posteam", "season", "team_targets", "team_air_yards", "team_rz_targets"
    )

    features = (
        agg
        .join(player_team, on=["player_id", "season"], how="left")
        .join(team_cols, on=["posteam", "season"], how="left")
        .withColumn("adot",
            F.when(F.col("targets") > 0,
                   F.col("_sum_air_yards") / F.col("targets")))
        .withColumn("catch_rate",
            F.when(F.col("targets") > 0,
                   F.col("_receptions") / F.col("targets")))
        .withColumn("yac_per_reception",
            F.when(F.col("_receptions") > 0,
                   F.col("_sum_yac") / F.col("_receptions")))
        .withColumn("target_share",
            F.when(F.col("team_targets") > 0,
                   F.col("targets") / F.col("team_targets")))
        .withColumn("air_yards_share",
            F.when(F.col("team_air_yards") > 0,
                   F.col("_sum_air_yards") / F.col("team_air_yards")))
        .withColumn("rz_target_share",
            F.when(F.col("team_rz_targets") > 0,
                   F.col("rz_targets") / F.col("team_rz_targets")))
        .select(
            "player_id", "season",
            "targets", "rz_targets",
            "adot", "catch_rate", "yac_per_reception",
            "target_share", "air_yards_share", "rz_target_share",
            "posteam",  # carried through for opportunity_share
        )
    )
    return features


def build_rusher_features(pbp, team_stats):
    """
    Features keyed on rusher_player_id:
      rz_carries, goalline_carries, rz_carry_share
    posteam is carried through for opportunity_share.
    """
    rushes = pbp.filter(
        ((F.col("rush_attempt") == 1) | (F.col("qb_scramble") == 1))
        & F.col("rusher_player_id").isNotNull()
    )
    player_team = _dominant_team(rushes, "rusher_player_id")

    agg = (
        rushes.groupBy("rusher_player_id", "season").agg(
            F.count("*").alias("carries"),
            F.sum(F.when(F.col("yardline_100") <= 20, 1).otherwise(0)).alias("rz_carries"),
            F.sum(F.when(F.col("yardline_100") <= 5, 1).otherwise(0)).alias("goalline_carries"),
        )
        .withColumnRenamed("rusher_player_id", "player_id")
    )

    team_cols = team_stats.select(
        "posteam", "season", "team_rush_attempts", "team_rz_rushes"
    )

    return (
        agg
        .join(player_team, on=["player_id", "season"], how="left")
        .join(team_cols, on=["posteam", "season"], how="left")
        .withColumn("rz_carry_share",
            F.when(F.col("team_rz_rushes") > 0,
                   F.col("rz_carries") / F.col("team_rz_rushes")))
        .select(
            "player_id", "season",
            "carries", "rz_carries", "goalline_carries", "rz_carry_share",
            "posteam",  # carried through for opportunity_share
        )
    )


def build_opportunity_share(receiver_df, rusher_df, team_stats):
    """
    opportunity_share = (targets + carries) / (team_targets + team_rush_attempts)

    For players with carries (RBs, QBs), posteam comes from rusher_df.
    For pure receivers without carries, posteam comes from receiver_df.
    """
    rec = receiver_df.select(
        "player_id", "season", "targets",
        F.col("posteam").alias("rec_team"),
    )
    rush = rusher_df.select(
        "player_id", "season", "carries",
        F.col("posteam").alias("rush_team"),
    )

    combined = (
        rec.join(rush, on=["player_id", "season"], how="outer")
        # Prefer rusher's team (more plays for a runner); fall back to receiver's team.
        .withColumn("posteam", F.coalesce(F.col("rush_team"), F.col("rec_team")))
        .withColumn("opportunities",
            F.coalesce(F.col("targets"), F.lit(0))
            + F.coalesce(F.col("carries"), F.lit(0)))
    )

    team_denom = team_stats.select(
        "posteam", "season", "team_targets", "team_rush_attempts"
    )

    return (
        combined
        .join(team_denom, on=["posteam", "season"], how="left")
        .withColumn("opportunity_share",
            F.when(
                (F.col("team_targets") + F.col("team_rush_attempts")) > 0,
                F.col("opportunities") / (F.col("team_targets") + F.col("team_rush_attempts"))
            ))
        .select("player_id", "season", "opportunity_share")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _clean(rows):
    """Convert NaN/NaT → None so psycopg2 sends SQL NULL."""
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows]


def main():
    if not os.path.isdir(PBP_DIR) or not os.listdir(PBP_DIR):
        print(f"No PBP parquet files found at {PBP_DIR}.")
        print("Run ingestion/ingest_pbp.py first.")
        return

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("pbp_features")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("Reading PBP parquets...")
    pbp = (
        spark.read
        .option("mergeSchema", "true")
        .parquet(PBP_DIR)
        .filter(F.col("season_type") == "REG")
    )

    # Back-fill integer columns absent from older seasons; leave float tracking
    # columns (cpoe, air_yards) as NULL — aggregations skip NULLs by default.
    pbp = _ensure_col(pbp, "pass_attempt", 0)
    pbp = _ensure_col(pbp, "rush_attempt", 0)
    pbp = _ensure_col(pbp, "qb_scramble", 0)
    pbp = _ensure_col(pbp, "sack", 0)
    pbp = _ensure_col(pbp, "complete_pass", 0)

    print(f"  Total play rows: {pbp.count():,}")

    print("Building team stats...")
    team_stats = build_team_stats(pbp).cache()

    print("Building passer features...")
    passer = build_passer_features(pbp)

    print("Building receiver features...")
    receiver = build_receiver_features(pbp, team_stats)

    print("Building rusher features...")
    rusher = build_rusher_features(pbp, team_stats)

    print("Building opportunity share...")
    opp = build_opportunity_share(receiver, rusher, team_stats)

    # -----------------------------------------------------------------------
    # Combine: full outer join so every player with any PBP appearance gets a row.
    # Drop the internal posteam columns used only for team joins.
    # -----------------------------------------------------------------------
    print("Combining features...")
    combined = (
        passer
        .join(receiver.drop("posteam"), on=["player_id", "season"], how="outer")
        .join(rusher.drop("posteam"), on=["player_id", "season"], how="outer")
        .join(opp, on=["player_id", "season"], how="outer")
    )

    # -----------------------------------------------------------------------
    # Write to Postgres via pandas (no JDBC JAR required).
    # -----------------------------------------------------------------------
    print("Converting to pandas and writing to pbp_features...")
    result = combined.toPandas()
    print(f"  {len(result):,} player-season rows.")

    db_cols = [c for c in result.columns if c not in ("player_id", "season")]
    rows = _clean(result[["player_id", "season"] + db_cols].to_dict("records"))
    rows = [r for r in rows if r.get("player_id") is not None]

    val_placeholders = ", ".join(f":{c}" for c in ["player_id", "season"] + db_cols)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in db_cols)
    upsert = text(f"""
        INSERT INTO pbp_features (player_id, season, {', '.join(db_cols)})
        VALUES ({val_placeholders})
        ON CONFLICT (player_id, season) DO UPDATE SET {update_set}
    """)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(upsert, rows)
    print(f"  Upserted {len(rows)} rows into pbp_features.")

    with engine.connect() as conn:
        for row in conn.execute(text(
            "SELECT season, COUNT(*) AS n FROM pbp_features GROUP BY season ORDER BY season"
        )):
            print(f"  {row.season}: {row.n} players")

    spark.stop()


if __name__ == "__main__":
    main()
