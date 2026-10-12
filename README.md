# SportsEdge

A runnable research starter for **NBA, NFL, and MLB home-win prediction**. It includes CSV ingestion, point-in-time head-to-head and availability features, three baselines, walk-forward evaluation, a local dashboard, and a read-only API.

**Draft status:** the included generator creates fictional games and players. Its results demonstrate functioning software, not predictive skill on real sports. No real historical dataset, injury feed, odds feed, betting execution, or profitability evaluation is included.

## Run locally

Requires Python 3.11 or newer. No third-party runtime dependencies or API keys.

```sh
python -m sportsedge demo --output data/demo --seed 42
python -m sportsedge run --games data/demo/games.csv --availability data/demo/availability.csv --data-kind synthetic --output runs/latest
python -m unittest discover -s tests -v
python -m sportsedge serve --report runs/latest/report.json --port 8765
```

Open http://127.0.0.1:8765. Stop with Ctrl+C. The server binds only to localhost and loads the report at startup; restart it after generating a new report. `GET /api/health` reports health; `GET /api/report` serves the full report, including predictions, model coefficients, fold provenance, and calibration bins. It is a local research server, not a production deployment.

Optional installation: `python -m pip install -e .`, then use `sportsedge` instead of `python -m sportsedge`.

## Bring historical data

Export licensed/provider data to the following normalized CSV schemas, then run the same pipeline. Inputs are read-only, hashed with SHA-256, and identified in every report. Keep raw provider responses, provider/version, retrieval times, transformations and licenses alongside your data. Never manufacture historical observation times from a current feed.

`games.csv` header:

```csv
game_id,league,home,away,start_at,prediction_at,result_known_at,home_score,away_score,neutral
example-1,NBA,TEAM-A,TEAM-B,2024-01-01T20:00:00Z,2024-01-01T19:00:00Z,2024-01-02T01:00:00Z,102,99,0
```

- Unique, stable game IDs; league is NBA, NFL, or MLB. Stable team identifiers must be consistent across files.
- All times must carry timezone offsets. `prediction_at` must precede `start_at`. Use the actual intended decision time, not a retrospective convenient time.
- `result_known_at` is when the final result became available to your system, after the start. Using a scheduled end time can leak delayed games.
- For upcoming games, leave both scores and result time blank. With sufficient past training games, their predictions appear in `upcoming` in the JSON report.
- Scores are nonnegative integers. `neutral` is 0 or 1. Ties are omitted from training/evaluation; the target is **home win conditional on a decisive result**, including for NFL. This is not an unconditional NFL moneyline probability.

`availability.csv` header:

```csv
game_id,team,player_id,published_at,observed_at,status,weight
example-1,TEAM-A,player-7,2024-01-01T15:00:00Z,2024-01-01T15:05:00Z,out,0.5
```

- Snapshots are specific to a game/team/player, never silently carried across fixtures.
- Both publication and observation times must be strictly before the prediction cutoff. Late-ingested backdated reports are excluded. Exact-cutoff timestamps are conservatively excluded.
- Latest eligible publication wins per player; later observation breaks publication-time ties. Duplicate snapshot keys are rejected. Explicit `available` updates supersede older `out` reports.
- Supported status factors: available 0, probable .25, questionable .5, doubtful .75, out 1. These are fixed heuristic severity weights, not calibrated probabilities of missing a game.
- `weight` is a finite number in [0,1] representing prespecified player importance. It must have been known at snapshot time. Do not use end-of-season minutes, retrospectively estimated player value, or eventual participation.
- Missing availability is unknown, not confirmed health. A coverage-difference feature indicates whether any eligible snapshot was present for each team; this is coarse report coverage, not complete roster coverage. Include explicit available reports where possible.

```sh
python -m sportsedge run --games data/historical/games.csv --availability data/historical/availability.csv --data-kind historical --min-train 100 --batch-size 30 --output runs/historical
```

The historical label is supplied by the operator and does not certify provenance. Omit `--availability` if historical snapshots are unavailable. No injury history is fabricated by the pipeline.

## Models and leakage controls

Each league has separate training, ratings, and histories:

1. **Home prior:** smoothed training-only home-win rate.
2. **Elo:** 1500 initial ratings, K=20, fixed 50-point home advantage (zero at neutral venues); ties update ratings as 0.5.
3. **Logistic:** fixed L2-regularized batch gradient descent, using Elo difference, smoothed last-10 form, smoothed last-10 H2H oriented toward the current home team, capped rest difference, weighted absence difference, availability coverage difference, and neutral venue.

Features use only results known strictly before each row's cutoff, ordered by result availability then game ID. Logistic training uses each historical game's original prediction-time feature vector. There is no whole-dataset scaling or random train/test split.

Fold boundaries are chronological blocks of all fixtures, independent of outcomes. At each block's earliest cutoff, training labels must already be known. Model weights stay fixed for that block; history features can update between predictions as new results become known. This represents sequential deployment, not a batch of predictions all made at the fold start. Minimum training count defaults to 20; cold-start blocks are skipped. No hyperparameter search is performed.

Reports include accuracy, Brier score, log loss, five-bin calibration, held-out predictions and training cutoff audits. Do not select a model based on this demo. For real research, use multiple seasons, larger training windows, a final untouched season, and uncertainty estimates before drawing conclusions.

Timestamp gating cannot repair incorrectly sourced timestamps or retrospectively revised input. The initial schema holds one final result per game and does not model result revisions, schedule/roster revisions, or a bitemporal warehouse. Team renames, season resets, starting pitchers/QBs, detailed lineup strength, market odds and travel are future extensions. The simple point-in-time feature builder scans history for each row; it prioritizes auditability over large-dataset speed.

## Reproducibility and tests

The synthetic generator uses a local seeded random generator; the pipeline is deterministic and writes `report.json`, `features.csv`, and (when nonempty) `predictions.csv`. Rerunning with the same inputs/configuration yields identical output on the same Python runtime. Generated data/runs are ignored by Git. Keep Python version and code commit with your experiment records.

Tests cover late result exclusion, publication/observation cutoffs, availability revisions, H2H direction, league isolation, timezone validation, duplicate IDs, metric values, deterministic output, fold embargoes and future-label perturbations. GitHub Actions runs tests and a demo on Python 3.11–3.13.

## Repository layout

- `sportsedge/core.py`: schema checks, features and models.
- `sportsedge/pipeline.py`: reproducible demo and walk-forward evaluation.
- `sportsedge/cli.py`: command line, local API and dashboard server.
- `sportsedge/dashboard.html`: league filter, metrics and prediction review.
- `tests/`: leakage and pipeline regression tests.

No source data or secrets should be committed. Provider ingestion adapters and real-data evaluation are the next steps; this draft already supports their normalized CSV outputs.
