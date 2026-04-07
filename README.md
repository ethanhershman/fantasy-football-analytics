# Fantasy Football Analytics Platform

A two-person data science portfolio project that ranks NFL players for the 2026 season using historical statistics and machine learning, and provides an interactive draft assistant for 12-man PPR leagues.

## What It Does

1. **Player Ranking & Valuation** — An XGBoost model trained on 5 years of NFL data predicts PPR fantasy points per game for each player. Rankings are compared to expert consensus (FantasyPros ECR) to identify over/undervalued players.

2. **Draft Assistant** — An interactive snake draft simulator that recommends picks based on value over replacement, positional need, and roster construction.

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | Python, nfl-data-py, requests, BeautifulSoup |
| File storage | AWS S3 |
| Database | PostgreSQL (local → AWS RDS) |
| ETL | PySpark |
| ML model | XGBoost + Scikit-learn + SHAP |
| API | FastAPI + Docker |
| Frontend | Streamlit + Plotly |

## Project Structure

```
├── data/           # Schema SQL, data dictionary
├── ingestion/      # Scripts to pull stats, ADP, ECR into PostgreSQL
├── etl/            # PySpark play-by-play pipeline
├── models/         # XGBoost training scripts and notebooks
├── api/            # FastAPI backend + Dockerfile
├── frontend/       # Streamlit app
├── notebooks/      # EDA and model experiment notebooks
├── requirements.txt
└── docker-compose.yml
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/ethanhershman/fantasy-football-analytics.git
cd fantasy-football-analytics
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 3. Initialize the database

```bash
psql -d ffdb -f data/schema.sql
```

### 4. Run ingestion

```bash
cd ingestion
python ingest_nfl_stats.py   # seasonal stats + rosters
python ingest_adp.py         # historical ADP
python ingest_ecr.py         # expert consensus rankings
```

## Development Phases

- [x] Phase 1 — Data ingestion & local PostgreSQL
- [ ] Phase 2 — PySpark ETL & AWS S3
- [ ] Phase 3 — XGBoost model & SHAP analysis
- [ ] Phase 4 — FastAPI backend & Docker
- [ ] Phase 5 — Streamlit frontend
- [ ] Phase 6 — Draft assistant
