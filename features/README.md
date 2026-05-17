# features/

Feature engineering for the fantasy football model. All outputs are keyed on
`(player_id, season)` and written as parquet files under `data/features/`.

The mega-table builder (`ingestion/build_training_data.py`) left-joins each
parquet output onto the per-position tables as additional feature columns.

---

## Bucket A — pre-aggregated by nfl-data-py (no pbp aggregation needed)

These pull from nfl-data-py endpoints that already return season-level numbers.
No play-by-play needed; each script is a thin wrapper + column selection.

| Script | Source function | Output |
| --- | --- | --- |
| `pull_seasonal.py` | `nfl.import_seasonal_data()` | `data/features/seasonal.parquet` |
| `pull_ngs.py` | `nfl.import_ngs_data()` | `data/features/ngs.parquet` |
| `pull_snaps.py` | `nfl.import_snap_counts()` | `data/features/snaps.parquet` |

## Bucket B — aggregate from pbp with pandas groupby

| Script | Source function | Output |
| --- | --- | --- |
| `build_pbp_features.py` | `nfl.import_pbp_data(columns=[...])` | `data/features/pbp.parquet` |

## Bucket C — deferred (future work, v1+)

These require data sources with spotty historical coverage; do not implement yet.

- **Slot vs outside rate / in-line vs flexed** — needs FTN/participation data,
  unavailable for most seasons pre-2022.
- **Route participation rate** — same coverage gap as slot rate.
- **Yards before contact** — requires player-tracking data not in nflverse.
