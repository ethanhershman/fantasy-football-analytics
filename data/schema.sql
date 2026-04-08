-- Fantasy Football Analytics — Simplified Schema
-- Run: psql -d ffdb -f data/schema.sql

DROP TABLE IF EXISTS draft_picks CASCADE;
DROP TABLE IF EXISTS draft_sessions CASCADE;
DROP TABLE IF EXISTS rankings CASCADE;
DROP TABLE IF EXISTS adp_history CASCADE;
DROP TABLE IF EXISTS adp CASCADE;
DROP TABLE IF EXISTS season_stats CASCADE;

CREATE TABLE season_stats (
    player_id         TEXT NOT NULL,
    season            INTEGER NOT NULL,
    full_name         TEXT NOT NULL,
    position          TEXT NOT NULL,
    team              TEXT,
    games_played      INTEGER,
    fantasy_points_ppr NUMERIC,
    completions       INTEGER,
    pass_attempts     INTEGER,
    passing_yards     INTEGER,
    passing_tds       INTEGER,
    interceptions     INTEGER,
    carries           INTEGER,
    rushing_yards     INTEGER,
    rushing_tds       INTEGER,
    receptions        INTEGER,
    targets           INTEGER,
    receiving_yards   INTEGER,
    receiving_tds     INTEGER,
    PRIMARY KEY (player_id, season)
);

CREATE TABLE adp (
    player_id          TEXT NOT NULL,
    season             INTEGER NOT NULL,
    adp_overall        NUMERIC,
    adp_position_rank  INTEGER,
    source             TEXT NOT NULL,
    PRIMARY KEY (player_id, season, source)
);

CREATE INDEX idx_season_stats_season ON season_stats(season);
CREATE INDEX idx_adp_season ON adp(season);
