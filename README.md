# SportsEdge

Outcome-prediction research for **NBA, NFL, MLB and NHL**. Includes current pregame forecasts, reproducible public-result ingestion, time-aware H2H and availability features, six model variants, chronological backtesting, saved model artifacts and a responsive local dashboard/API.

**Current status:** prospective forecasts are generated for upcoming games; the expanded dataset spans 2023–October 2026. Historical evaluations explicitly use a result-delay assumption because archives lack historical observation timestamps. Injury data is **not available** in these runs. The dashboard separates upcoming forecasts, historical model performance, and prospective outcome tracking. This is a research prototype, not evidence of betting profitability.

## Predict current and upcoming games

The prospective pipeline retrieves current NBA, NFL, MLB and NHL schedules and final results, refits the models, and saves predictions for scheduled games that have not started. It covers pregame outcomes, **not in-game predictions**. Run after creating the historical archive below:

```sh
python -m sportsedge forecast --history data/historical/games.csv --ml --days 7
python -m sportsedge serve --report runs/historical/report.json --forecast runs/forecast/report.json --scorecard runs/forecast/scorecard.json --port 8767
```

The dashboard's **Upcoming game forecasts** section shows home/away probabilities and local game times. Historical evaluation remains separate. `GET /api/forecast` returns the most recently published forecast snapshot. The server reloads this file on each request; rerun the forecast command to fetch new schedules/results and refit. The refresh button reloads saved predictions; it does not train a model. No background refresh or scheduled service is installed.

- All predictions in a snapshot use the same actual generation cutoff, including games several days away. They cannot use results from games that finish between now and that future fixture. Completed, in-progress, canceled, postponed, exhibition and undated/TBD fixtures are excluded. Games starting during training are removed before publication; the dashboard also hides games once they start.
- Each refresh fetches up to the preceding 730 days of results (`--lookback-days`, 7–730), plus the requested forecast horizon (`--days`, 1–30). Old archive dates are not counted again. Team aliases align ESPN with the existing NFL/MLB archive; Oakland/Athletics share a history, while Arizona/Utah NHL histories remain separate. Unknown teams fail closed. The provider's schedule can still be revised after collection.
- Model training combines the verified archive with newly retrieved older finals. **Training features remain a reconstruction:** new historical labels use a seven-day delay after start, not verified historical publication times. This assumption may be wrong for unusually delayed/suspended games or revised data. Recent finals are included in forecast features only after their actual retrieval time. These inference features use observed results, not the reconstructed training availability times.
- Training windows can contain gaps if the archive ends before the requested lookback. Use a sufficiently long lookback or regenerate the archive. The default covers the gap from the supplied 2023–2024 archive through October 2026. New model fits include previously evaluated data and must not inherit the old backtest scores as their performance estimate. Future forecast outcomes need a separate prospective evaluation.
- `--availability path/to/availability.csv` accepts existing timestamped reports. There is no automatic injury/lineup provider yet; missing coverage is unknown. A model trained without injury examples has not learned an injury effect. NFL probabilities remain conditional on a decisive result; NHL includes overtime/shootouts.
- Every refresh creates an immutable `runs/forecast/snapshots/<timestamp>/` folder with raw payloads and retrieval timestamps, normalized training and inference-history CSVs, hashes, features, model artifacts and `forecast.json`. Only after successful completion does it atomically replace `runs/forecast/report.json`. Failed refreshes leave the previous snapshot intact. The UI flags snapshots older than 24 hours; always check the displayed generation time.

The ESPN scoreboard adapter is unofficial and may change or omit events. Fetching a schedule does not guarantee lineup accuracy, forecast accuracy, or profitability. Raw data and forecasts stay local and are excluded from Git.

Reproduce a saved fit without fetching current data (use the snapshot path printed by the forecast command):

```sh
python -m sportsedge replay --snapshot runs/forecast/snapshots/YOUR_TIMESTAMP --output runs/replay
```

Replay verifies input hashes and reuses the original cutoff and model settings. It is explicitly marked as an old snapshot. For exact ML reproducibility, use the same locked dependencies. Replay includes all fixtures eligible at the original cutoff; the published forecast may have removed games that started during model fitting.

### Track actual forecast outcomes

Every successful new forecast refresh also updates `runs/forecast/scorecard.json`. It compares saved pregame forecasts to later observed finals, choosing the **earliest issued forecast per game/model** to avoid counting a game multiple times or choosing a favorable forecast after the result. Changed schedules/matchups and ties are excluded and counted explicitly. Finals must have been observed after forecast issuance and before scoring time. Earlier observed finals are retained from the immutable snapshot history even when they leave the latest rolling data window.

To score independently using an observed-result CSV:

```sh
python -m sportsedge score --snapshots runs/forecast/snapshots --results runs/forecast/snapshots/YOUR_TIMESTAMP/observed.csv
```

Use **observed** results, not a reconstructed training CSV. `/api/scorecard` and the Model performance view show pending, scored and excluded games. Zero scored games means accuracy is unknown, not zero. Local files are auditable but not externally timestamp-attested; the scorer cannot prove a user has not edited them. Never manufacture or backdate snapshots to claim prospective performance.

## Quick start

Python 3.11+; core demo needs no third-party packages. From the repository directory:

```sh
python -m sportsedge demo --seed 42
python -m sportsedge run --games data/demo/games.csv --availability data/demo/availability.csv --output runs/demo
python -m sportsedge serve --report runs/demo/report.json --port 8765
```

Open http://127.0.0.1:8765. Use the league/model dropdowns to inspect predictions, calibration and injury coverage. The server binds only to loopback, loads its report at startup, and stops with Ctrl+C. Restart it after regenerating a report.

## Reproduce the real-data experiment

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-ml.lock.txt
python -m sportsedge.ingest --years 2023 2024 --historical-lag-days 7 --output data/historical
python -m sportsedge run --games data/historical/games.csv --ml --min-train 100 --test-from 2024-01-01T00:00:00Z --output runs/historical
python -m sportsedge serve --report runs/historical/report.json --port 8766
python -m unittest discover -s tests -v
```

`pip install -e ".[ml]"` is an alternative installation with scikit-learn 1.7.2 and compatible dependency versions. The lock file records the tested environment. To limit CPU use, set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` before running.

The default ingest covers all four leagues. Use `--leagues NHL` to restrict scope. The adapter fetches daily ESPN scoreboards with three concurrent requests and retries; requests are cached by URL with retrieval times and SHA-256 hashes. Cached bytes are verified on reuse. Source manifests record the transformed CSV hash; training rejects mismatches. Keep cached archives for reproducibility because upstream archives can be revised.

### Data sources and scope

- **NBA/NHL:** [ESPN scoreboards](https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates=20240101), one daily query per requested calendar date. Only completed regular-season/playoff events involving recognized clubs are retained. All-Star events and preseason are excluded. Only game identity, venue neutrality, date, teams and final scores are extracted; postgame records, leaders, statistics and market odds are ignored. ESPN's endpoint is unofficial and may change. Club allowlists require maintenance for future expansion/renames; ARI and UTAH currently have separate histories.
- **NFL:** [nflverse games](https://github.com/nflverse/nfldata/blob/master/data/games.csv), selected by season. A selected season includes its January/February postseason in the following calendar year. No odds, eventual starter/QB fields or weather outcomes are consumed.
- **MLB:** [Retrosheet annual game logs](https://www.retrosheet.org/gamelogs/index.html), regular-season archives. Suspended games use the completion date in the [documented completion field](https://www.retrosheet.org/gamelogs/glfields.txt) before applying the result delay. Forfeits and protests are excluded. Game numbers preserve doubleheaders. Designated home/away sides are used; neutral-site classification is not available in this initial adapter.

Retrosheet data is used courtesy of Retrosheet. Consult each provider's terms before redistributing its data. Raw archives and normalized datasets are ignored by Git; only code, provenance summaries and aggregate experiment results are published.

### Timing policy: explicit reconstruction

The default ingest, **without** `--historical-lag-days`, uses the actual archive retrieval time as `result_known_at`. It does not pretend results were observed historically, so it cannot provide historical training labels at old cutoffs.

The explicit reconstruction option represents games at a UTC date boundary, places the decision one hour earlier, and makes results available seven days after the game/completion date in the example above. This deliberately excludes same-day and recent results. Date-only boundaries are **not actual tipoff/puck-drop times**. This conservative research assumption cannot certify point-in-time correctness of revised schedules/results. Schedule revisions and data corrections remain unmodeled. Rest is a proxy measured from the last *known-result* game, not a complete schedule-derived rest measure. Do not present these results as live, executable or fully timestamp-verified performance.

## Feature engineering and models

Thirteen fixed features are calculated separately for each league:

- Elo difference (1500 initial rating, K=20, 50-point home advantage, zero at neutral venues).
- Smoothed last-10 form and scaled/clipped last-10 scoring-margin difference.
- H2H last-10 win rate, sample count, recency-weighted win rate (365-day half-life) and age of latest meeting.
- Capped rest difference, weighted absence difference, report-coverage difference, separate home/away report-presence indicators, and neutral venue.

Baselines are a smoothed training-only home prior, Elo, and fixed L2-regularized logistic regression. `--ml` adds histogram gradient boosting (100 iterations, seven leaves, minimum 30 samples/leaf, L2=2, learning rate=.05, seed=42, early stopping disabled), a logistic ablation with all four direct H2H features zeroed, and chronologically calibrated logistic regression. The original five variants' settings were fixed before the initial historical benchmark. Stronger model capacity does not guarantee stronger results.

The calibrated model uses the first 80% of each pre-test training window for the base logistic model and the later 20% for Platt calibration (`LogisticRegression`, C=1, fixed seed). Base labels must be known strictly before the first calibration prediction; all calibration labels must be known before the outer evaluation/forecast cutoff. It requires at least 100 base and 30 calibration examples with both classes; otherwise it explicitly falls back to plain logistic. It retains its earlier base fit rather than refitting it on the calibration labels. Artifacts record both cutoffs and sample sizes. There is no random split or use of test labels in fitting.

The calibration extension was designed after inspecting the five-model 2026 comparison. Its 2026 comparison is therefore a **development benchmark**, not a new untouched confirmatory test. Calibration improved NBA Brier score in that comparison and worsened it for the other leagues; it is not the default forecast model. Reserve later outcomes for confirmation.

`--test-from` trains each league once using only labels known before the boundary. Weights remain frozen throughout evaluation; features update sequentially as prior game results become available. This represents rolling prediction with frozen weights, not all predictions made on January 1. Without this option, the original expanding-window backtest retrains at each chronological fixture block. The two protocols should not be conflated.

Targets are home wins conditional on a decisive final result. **NHL includes overtime and shootouts**, not regulation-only outcomes. NFL ties are excluded from training/evaluation (and represented as 0.5 for historical form/Elo); the output is not an unconditional three-way moneyline probability.

## Injuries and availability

CSV inputs retain one snapshot per game/team/player/publication/observation. Both timestamps must be strictly before prediction time; the latest eligible publication wins, with observation time breaking ties. Reports older than 72 hours are treated as unknown. Explicit `available` updates supersede older `out` statuses. Coverage indicators distinguish missing reports from known availability; they indicate any report, not full-roster coverage.

Supported heuristic severity factors: available=0, probable=.25, questionable=.5, doubtful=.75, out=1. These are not calibrated probabilities. Player importance is a finite weight in [0,1] that must have been known at report time. Never derive historical weights or availability from eventual starters, participation, season-end statistics or today's injury list.

To begin collecting genuine snapshots, normalize permitted provider reports into a JSON list containing `game_id`, `team`, `player_id`, `published_at`, `status`, and `weight`, then:

```sh
python -m sportsedge.availability --input incoming.json --games data/mydata/games.csv --output data/mydata/availability.csv
```

The command records the actual current `observed_at`, ignores any observation time supplied by the input, validates all rows before replacing the CSV, and preserves input payloads with hashes under `availability_raw/`. It is a single-writer import command, not a scheduled provider integration. The real-data experiment has **0% injury coverage**, so no injury benefit is claimed. A licensed timestamped archive or prospective collection is needed to evaluate that benefit.

## Normalized CSV schemas

```csv
game_id,league,home,away,start_at,prediction_at,result_known_at,home_score,away_score,neutral
example-1,NHL,BOS,NYR,2024-01-01T20:00:00Z,2024-01-01T19:00:00Z,2024-01-02T01:00:00Z,3,2,0
```

IDs must be unique, team IDs stable, timestamps offset-aware, scores nonnegative integers, and neutral 0/1. Prediction must precede the start; result availability must follow it. Upcoming fixtures leave both scores and result time blank. Their predictions appear under `upcoming` when enough past training data exists. Historical adapters currently import completed games only.

```csv
game_id,team,player_id,published_at,observed_at,status,weight
example-1,BOS,player-7,2024-01-01T15:00:00Z,2024-01-01T15:05:00Z,out,0.5
```

Run your own verified data with `--games ... --availability ... --data-kind historical`. The data-kind flag does not certify provenance; ingestion manifest classification takes precedence when present.

## Outputs and API

- `report.json`: metrics, five-bin calibration, prediction rows, model coefficients, fold cutoffs, data hashes, runtime versions, coverage and timing policy.
- `features.csv` and `predictions.csv`: reproducible research artifacts.
- `<LEAGUE>-logistic.json`: feature order, coefficients and training provenance.
- `models/<LEAGUE>-gradient-boosting.joblib`: last trained fold's model and provenance. These are evaluation models, not models refit on the held-out set. Load only your own trusted joblib artifacts.
- `GET /api/health` and `GET /api/report`: local read-only API.

See [VALIDATION.md](VALIDATION.md) and [experiments/historical-summary.json](experiments/historical-summary.json) for the measured run. The first run is a benchmark, not a final untouched test for subsequent model selection. Reserve later data before future tuning. No market odds, transaction costs, uncertainty estimates, or betting returns are evaluated.

## Tests and limitations

Tests exercise future-result exclusion, late and stale reports, H2H direction, league isolation, exact-cutoff embargoes, snapshot observation times, suspended MLB completion, All-Star filtering, NHL support, frozen training, deterministic reports, schema validation and saved model artifacts. GitHub Actions tests core and ML configurations.

Feature construction rescans prior games and prioritizes auditability over scale. There is no season reset, roster revision model, dedicated goalie/pitcher/QB feature, full lineup coverage, in-game model or automatic retraining schedule. Current injury feeds and later untouched evaluation windows remain future work.
