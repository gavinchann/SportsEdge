# Validation — October 16, 2026

## Prospective forecasting and dashboard update

- 31 tests pass with ML dependencies; core-only validation passes with four optional ML tests skipped. New coverage includes current schedule filtering, observation-time embargoes, source identity mapping, replay integrity, chronological probability calibration, earliest-forecast scoring, retained final observations, and HTTP assets/API behavior.
- JavaScript syntax validation passes. Desktop and 390px mobile layouts were exercised in the browser; a mobile navigation overflow was found and corrected. Date filtering, matchup model comparison and historical evaluation navigation work.
- The 0.3.0 wheel builds and includes the HTML, CSS and JavaScript dashboard assets.
- A 730-day current-source refresh retrieved 2,952 daily scoreboards and 9,770 recent completed results beyond the original archive. The resulting fit used 20,504 decisive examples (20,505 completed games including one tie). The seven-day snapshot contains 81 unique upcoming fixtures: NBA 16, NFL 14, MLB 4, NHL 47. These counts are snapshot-specific, not service coverage guarantees.
- The saved prospective tracker initially has 81 pending games and zero scored games. No prospective accuracy or betting return is established.
- The local app separates current forecasts, historical model comparison, prospective tracking and methodology. Injury coverage is visibly unknown; an automated injury feed is not connected.

## Newer-season development benchmark

Training freezes before January 1, 2026 UTC. The reconstructed evaluation covers 4,200 decisive games: NBA 821, NFL 94, MLB 2,453 and NHL 832. Features update sequentially using earlier results under the seven-day training reconstruction policy. The archive and recent data use different source timing representations; this is not a fully timestamp-observed historical test.

Plain logistic Brier scores: NBA 0.20728, NFL 0.22892, MLB 0.24744, NHL 0.24547. Chronological Platt-calibrated logistic scores: NBA 0.20375, NFL 0.23035, MLB 0.24801, NHL 0.24644. Lower is better. Calibration improved NBA and worsened the other three leagues in this run, so the UI's default remains plain logistic. The calibrated model also uses an earlier base fitting window, so these differences are not an isolated causal estimate of calibration's effect.

The calibration extension was designed after the original five-model 2026 results were inspected. This is a **development comparison**, not an untouched confirmatory test. No hyperparameter search was performed against these results; future outcomes remain necessary for confirmation. Full aggregate measurements and fold cutoffs are recorded in `experiments/evaluation-2026-summary.json`.

Reproduce from a forecast snapshot's normalized training CSV:

```sh
python -m sportsedge run --games runs/forecast/snapshots/YOUR_TIMESTAMP/games.csv --ml --min-train 100 --test-from 2026-01-01T00:00:00Z --output runs/evaluation-2026
```

Exact reproduction requires the saved snapshot because new refreshes may include revised upstream data. An old snapshot is not a current prediction. The earlier benchmark below is retained as historical evidence.

## Original October 14 validation

Local environment: Windows, Python 3.12.14, scikit-learn 1.7.2. Exact installed dependencies are in `requirements-ml.lock.txt`.

## Checks performed

- 20 tests pass with ML dependencies installed.
- Core-only run (`python -S -m unittest discover -s tests -v`): 18 pass; two optional ML tests skip.
- Tests cover future-result perturbations, frozen training weights, late/stale injury exclusion, H2H orientation, duplicate/schema handling, cache-manifest validation, prospective observation timestamps, All-Star exclusion, suspended-game completion dates and NHL model artifacts.
- Real-data ingestion and training completed locally. Models and full report are in `runs/historical/` (ignored by Git).
- The dashboard exposes league/model selectors, calibration, timing assumptions, prediction review and explicitly zero injury-report coverage.

## Real-data experiment

Ingested 10,811 games: NBA 2,566; NFL 570; MLB 4,859; NHL 2,816. Sources are ESPN daily scoreboards, nflverse game archives and Retrosheet game logs; see README for links and scope. Queries target 2023–2024. NFL season selection includes postseason in the following calendar year, and UTC game dates can fall the day after a scoreboard query date.

Training weights were frozen before January 1, 2024 UTC. Only results whose assumed availability preceded that boundary entered fitting. Evaluation uses sequentially updated features and covers 5,486 unique decisive games: NBA 1,316; NFL 314; MLB 2,429; NHL 1,427. Five model variants produced 27,430 probability records. Fixed hyperparameters were not tuned on these held-out outcomes.

Lowest Brier score on this particular benchmark:

- NBA: Elo, 0.21863 (64.29% accuracy); home prior 0.24955.
- NFL: Elo, 0.22430 (63.69% accuracy); home prior 0.24750.
- MLB: logistic regression, 0.24537 (55.08% accuracy); home prior 0.24953.
- NHL: Elo, 0.23940 (59.22% accuracy); home prior 0.24883.

These are descriptive, post-evaluation comparisons, not preselected deployment winners or statistically established improvements. Gradient boosting did not beat the best baseline on Brier score in any league. Direct H2H features had small mixed effects: removing them slightly improved logistic Brier score for NBA/NHL and slightly worsened it for MLB/NFL. No confidence intervals or significance tests have been calculated.

## Interpretation limits

Result-publication timestamps were not supplied by the archives. This experiment deliberately uses a **seven-day availability delay** after the recorded game/completion date and date-boundary prediction times. Those are explicit reconstruction assumptions, not actual historical observations. Recent-game results are omitted; revised schedules/results remain a source of uncertainty. Suspended MLB results are delayed from the documented completion date.

Historical injury coverage is **zero**. The new prospective capture command records real observation timestamps and supports later injury research, but it is not connected to a live injury provider. No current-lineup benefit is claimed. No odds, wagering returns, calibration fitting, or live performance was evaluated.

The full aggregate metrics, calibration bins, input hashes, runtime and training cutoffs are in `experiments/historical-summary.json`. The full report hash is included there for local verification. A later untouched evaluation period is required before tuning based on these results. Remote CI is configured but must be checked separately.
