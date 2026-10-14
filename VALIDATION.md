# Validation — October 14, 2026

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
