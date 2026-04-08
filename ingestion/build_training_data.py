import pandas as pd
from db import get_engine


def main():
    engine = get_engine()
    with engine.connect() as conn:
        stats = pd.read_sql("SELECT * FROM season_stats WHERE season = 2025", conn)
        adp = pd.read_sql("SELECT * FROM adp WHERE season = 2026 AND adp_overall < 50", conn)

    rb_adp = adp[adp["player_id"].isin(
        stats[stats["position"] == "RB"]["player_id"]
    )]

    result = rb_adp.merge(
        stats[stats["position"] == "RB"][["player_id", "full_name", "carries", "rushing_yards", "rushing_tds"]],
        on="player_id"
    ).sort_values("rushing_yards", ascending=False)

    print(result[["full_name", "adp_overall", "carries", "rushing_yards", "rushing_tds"]].to_string(index=False))


if __name__ == "__main__":
    main()
