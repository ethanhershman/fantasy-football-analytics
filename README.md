# Fantasy Football Analytics Platform

A two-person data science portfolio project that builds a machine learning model to predict NFL player performance relative to consensus draft ADP, and packages the output into a website with player rankings and an interactive draft assistant for 12-team PPR leagues.

## Goals

1. **Player valuation model** — Predict player performance relative to consensus ADP. The objective is identifying where the model and the market disagree, so undervalued and overvalued players surface clearly.
2. **Website** — Expose the model output as player rankings against consensus and include a draft tool that tracks which players have already been picked and recommends a strategy that maximizes expected points for the user's roster.

## Current Progress

Data ingestion and EDA are complete. Per-position mega tables (QB, RB, WR, TE) cover 2016–2025 with one row per player-season containing ADP, finish, PPG, career season number, prior-season stats, and advanced features as predictors. Feature engineering scripts for NGS, snap counts, seasonal advanced stats, and play-by-play are implemented and ready to run.

| Source | Coverage | Notes |
| --- | --- | --- |
| nfl-data-py | 2016–2025 seasonal stats | All skill positions |
| FantasyFootballCalculator API | 2016–2025 ADP | PPR / 12-team, top ~180 players per season |
| DraftSharks | 2026 ADP | Top ~260 players |

## Data Sources

- **nfl-data-py** — Open-source Python wrapper around nflverse data. Used for seasonal player statistics.
- **FantasyFootballCalculator API** — Free REST API providing historical PPR ADP for 12-team leagues.
- **DraftSharks** — Source for 2026 ADP (deeper coverage than FFC was offering for the upcoming season).

### Future expansion

The mega tables will be extended back to 2015 (11 seasons total), and `nfl-data-py`'s `import_pbp_data()` will be used to pull the same 11 seasons of play-by-play data. This will be aggregated to engineer position-specific features (red-zone usage, air yards, target shares, carry %, route participation, etc.). At ~500,000 rows the dataset fits comfortably in pandas after column-filtered loads via `nfl.import_pbp_data(years, columns=[...])`.

## Tech Stack

| Layer | Tool | Status |
| --- | --- | --- |
| Language | Python 3 | In use |
| Data ingestion | nfl-data-py, requests | In use |
| Storage (current) | PostgreSQL (local `ffdb`) | In use |
| Storage (next) | AWS RDS (managed PostgreSQL) | Planned |
| File storage (cloud) | AWS S3 | Planned |
| ETL | pandas (+ DuckDB if needed) | Planned |
| Modeling | Scikit-learn (baseline), XGBoost, SHAP | Planned |
| Visualization | Matplotlib, Seaborn, Plotly | Planned |
| Backend | FastAPI + Docker | Planned |
| Frontend | Streamlit (initial), React (possible) | Planned |
| Hosting | AWS EC2 + RDS | Planned |
| Version control | Git + GitHub | In use |

## Project Structure

```
├── data/           # Schema SQL, data dictionary
├── ingestion/      # Scripts to pull stats and ADP into PostgreSQL
├── features/       # Play-by-play and seasonal feature engineering (pandas)
├── models/         # XGBoost training scripts and notebooks (planned)
├── api/            # FastAPI backend + Dockerfile (planned)
├── frontend/       # Streamlit / React app (planned)
├── notebooks/      # EDA and model experiment notebooks
├── requirements.txt
└── setup.sh
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
python ingest_nfl_stats.py    # seasonal stats from nfl-data-py
python ingest_ffc_adp.py      # historical ADP from FantasyFootballCalculator
python ingest_draftsharks.py  # 2026 ADP from DraftSharks
```

## Roadmap

- [x] **Phase 1** — Data ingestion + mega tables (2016–2025)
- [x] **Phase 2** — EDA: ADP vs finish, PPG correlations, player profiles, steal/bust analysis
- [x] **Phase 3** — Bucket A feature scripts (seasonal advanced stats, NGS, snap counts)
- [x] **Phase 4** — Bucket B play-by-play features (red zone, carry share, pressure rate, team pass rate)
- [ ] **Phase 5** — Migrate PostgreSQL to AWS RDS
- [ ] **Phase 6** — Baseline (Ridge) + XGBoost models per position + SHAP
- [ ] **Phase 7** — FastAPI backend + Docker
- [ ] **Phase 8** — Streamlit frontend (rankings dashboard)
- [ ] **Phase 9** — Draft assistant logic + draft board UI
- [ ] **Phase 10** — Cloud deploy (RDS + EC2 or equivalent)

## Collaborators

- Ethan Hershman
- Colin Daugherty
