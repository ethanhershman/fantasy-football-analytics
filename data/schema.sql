-- Fantasy Football Analytics — Schema
-- Run: psql -d ffdb -f data/schema.sql

DROP TABLE IF EXISTS training_data_qb CASCADE;
DROP TABLE IF EXISTS training_data_rb CASCADE;
DROP TABLE IF EXISTS training_data_wr CASCADE;
DROP TABLE IF EXISTS training_data_te CASCADE;
DROP TABLE IF EXISTS adp CASCADE;
DROP TABLE IF EXISTS pbp_features CASCADE;
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

CREATE TABLE pbp_features (
    player_id               TEXT    NOT NULL,
    season                  INTEGER NOT NULL,
    -- Passing (QB)
    cpoe                    NUMERIC,
    air_yards_per_attempt   NUMERIC,
    rz_pass_attempts        INTEGER,
    -- Receiving (WR / TE / RB)
    targets                 INTEGER,
    rz_targets              INTEGER,
    adot                    NUMERIC,
    catch_rate              NUMERIC,
    yac_per_reception       NUMERIC,
    target_share            NUMERIC,
    air_yards_share         NUMERIC,
    rz_target_share         NUMERIC,
    -- Rushing (RB / QB scrambles)
    carries                 INTEGER,
    rz_carries              INTEGER,
    goalline_carries        INTEGER,
    rz_carry_share          NUMERIC,
    -- Combined
    opportunity_share       NUMERIC,
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

CREATE TABLE training_data_qb (
    full_name              TEXT NOT NULL,
    season                 INTEGER NOT NULL,
    team                   TEXT,
    adp_overall            NUMERIC,
    adp_position_rank      NUMERIC,
    finish                 INTEGER,
    games_played           INTEGER,
    fantasy_points_ppr     NUMERIC,
    completions            INTEGER,
    pass_attempts          INTEGER,
    passing_yards          INTEGER,
    passing_tds            INTEGER,
    interceptions          INTEGER,
    carries                INTEGER,
    rushing_yards          INTEGER,
    rushing_tds            INTEGER,
    prev_games_played      INTEGER,
    prev_fantasy_points_ppr NUMERIC,
    prev_completions       INTEGER,
    prev_pass_attempts     INTEGER,
    prev_passing_yards     INTEGER,
    prev_passing_tds       INTEGER,
    prev_interceptions     INTEGER,
    prev_carries           INTEGER,
    prev_rushing_yards     INTEGER,
    prev_rushing_tds       INTEGER,
    PRIMARY KEY (full_name, season)
);

CREATE TABLE training_data_rb (
    full_name              TEXT NOT NULL,
    season                 INTEGER NOT NULL,
    team                   TEXT,
    adp_overall            NUMERIC,
    adp_position_rank      NUMERIC,
    finish                 INTEGER,
    games_played           INTEGER,
    fantasy_points_ppr     NUMERIC,
    carries                INTEGER,
    rushing_yards          INTEGER,
    rushing_tds            INTEGER,
    receptions             INTEGER,
    targets                INTEGER,
    receiving_yards        INTEGER,
    receiving_tds          INTEGER,
    prev_games_played      INTEGER,
    prev_fantasy_points_ppr NUMERIC,
    prev_carries           INTEGER,
    prev_rushing_yards     INTEGER,
    prev_rushing_tds       INTEGER,
    prev_receptions        INTEGER,
    prev_targets           INTEGER,
    prev_receiving_yards   INTEGER,
    prev_receiving_tds     INTEGER,
    PRIMARY KEY (full_name, season)
);

CREATE TABLE training_data_wr (
    full_name              TEXT NOT NULL,
    season                 INTEGER NOT NULL,
    team                   TEXT,
    adp_overall            NUMERIC,
    adp_position_rank      NUMERIC,
    finish                 INTEGER,
    games_played           INTEGER,
    fantasy_points_ppr     NUMERIC,
    receptions             INTEGER,
    targets                INTEGER,
    receiving_yards        INTEGER,
    receiving_tds          INTEGER,
    prev_games_played      INTEGER,
    prev_fantasy_points_ppr NUMERIC,
    prev_receptions        INTEGER,
    prev_targets           INTEGER,
    prev_receiving_yards   INTEGER,
    prev_receiving_tds     INTEGER,
    PRIMARY KEY (full_name, season)
);

CREATE TABLE training_data_te (
    full_name              TEXT NOT NULL,
    season                 INTEGER NOT NULL,
    team                   TEXT,
    adp_overall            NUMERIC,
    adp_position_rank      NUMERIC,
    finish                 INTEGER,
    games_played           INTEGER,
    fantasy_points_ppr     NUMERIC,
    receptions             INTEGER,
    targets                INTEGER,
    receiving_yards        INTEGER,
    receiving_tds          INTEGER,
    prev_games_played      INTEGER,
    prev_fantasy_points_ppr NUMERIC,
    prev_receptions        INTEGER,
    prev_targets           INTEGER,
    prev_receiving_yards   INTEGER,
    prev_receiving_tds     INTEGER,
    PRIMARY KEY (full_name, season)
);
