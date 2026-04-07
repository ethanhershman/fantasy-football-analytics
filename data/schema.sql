-- Fantasy Football Analytics Platform
-- PostgreSQL Schema
-- Run this once to initialize the database: psql -d ffdb -f data/schema.sql

-- ============================================================
-- players
-- One row per player. Referenced by all other tables.
-- ============================================================
CREATE TABLE IF NOT EXISTS players (
    player_id         TEXT PRIMARY KEY,
    full_name         TEXT NOT NULL,
    position          TEXT NOT NULL CHECK (position IN ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')),
    team              TEXT,
    age               INTEGER,
    years_experience  INTEGER
);

-- ============================================================
-- season_stats
-- One row per player per season. Main feature table.
-- ============================================================
CREATE TABLE IF NOT EXISTS season_stats (
    player_id           TEXT NOT NULL REFERENCES players(player_id),
    season              INTEGER NOT NULL,
    games_played        INTEGER,
    fantasy_points_ppr  NUMERIC,
    points_per_game     NUMERIC GENERATED ALWAYS AS (
                            CASE WHEN games_played > 0
                                 THEN fantasy_points_ppr / games_played
                                 ELSE NULL END
                        ) STORED,
    targets             INTEGER,
    target_share        NUMERIC,
    air_yards_share     NUMERIC,
    snap_pct            NUMERIC,
    carries             INTEGER,
    rz_targets          INTEGER,
    yards_after_catch   NUMERIC,
    PRIMARY KEY (player_id, season)
);

-- ============================================================
-- adp_history
-- One row per player per season per source.
-- ============================================================
CREATE TABLE IF NOT EXISTS adp_history (
    player_id         TEXT NOT NULL REFERENCES players(player_id),
    season            INTEGER NOT NULL,
    adp_overall       NUMERIC,
    adp_position_rank INTEGER,
    source            TEXT NOT NULL,  -- 'ffc_api', 'sleeper', 'fantasypros'
    PRIMARY KEY (player_id, season, source)
);

-- ============================================================
-- rankings
-- Model output. Overwritten each time the model is retrained.
-- ============================================================
CREATE TABLE IF NOT EXISTS rankings (
    player_id            TEXT NOT NULL REFERENCES players(player_id),
    season               INTEGER NOT NULL,
    projected_points_pg  NUMERIC,
    model_rank_overall   INTEGER,
    model_rank_position  INTEGER,
    ecr_rank             INTEGER,
    value_score          NUMERIC,  -- ecr_rank - model_rank_overall; positive = undervalued
    updated_at           TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (player_id, season)
);

-- ============================================================
-- Draft assistant tables (Phase 6)
-- ============================================================
CREATE TABLE IF NOT EXISTS draft_sessions (
    session_id    TEXT PRIMARY KEY,
    draft_slot    INTEGER NOT NULL,
    num_teams     INTEGER NOT NULL DEFAULT 12,
    total_rounds  INTEGER NOT NULL DEFAULT 15,
    scoring       TEXT NOT NULL DEFAULT 'ppr',
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS draft_picks (
    pick_id       SERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES draft_sessions(session_id),
    pick_number   INTEGER NOT NULL,
    round_number  INTEGER NOT NULL,
    team_slot     INTEGER NOT NULL,
    player_id     TEXT REFERENCES players(player_id),
    is_user_pick  BOOLEAN NOT NULL DEFAULT FALSE,
    picked_at     TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_season_stats_season ON season_stats(season);
CREATE INDEX IF NOT EXISTS idx_season_stats_player ON season_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_adp_history_season  ON adp_history(season);
CREATE INDEX IF NOT EXISTS idx_rankings_season     ON rankings(season);
CREATE INDEX IF NOT EXISTS idx_draft_picks_session ON draft_picks(session_id);
