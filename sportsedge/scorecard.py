"""Score saved pregame forecasts once genuinely later final results are observed."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from .core import LEAGUES, metrics, outcome, read_games, stamp


def score_forecasts(snapshots, results_path, output, as_of=None):
    cutoff = stamp(as_of) if as_of else datetime.now(timezone.utc)
    results = {g["game_id"]: g for g in read_games(results_path)}
    result_sources = []
    # Retain known finals even when a later refresh's rolling lookback no longer
    # includes them. Prefer the most recent eligible observation/correction.
    for observed_path in sorted(Path(snapshots).glob("*/observed.csv")):
        result_sources.append({"snapshot": observed_path.parent.name, "sha256": hashlib.sha256(observed_path.read_bytes()).hexdigest()})
        for g in read_games(observed_path):
            known = g["result_known_at"]
            previous = results.get(g["game_id"])
            if known and known < cutoff and (not previous or not previous["result_known_at"] or previous["result_known_at"] >= cutoff or known > previous["result_known_at"]):
                results[g["game_id"]] = g
    selected, sources = {}, []
    for path in sorted(Path(snapshots).glob("*/forecast.json")):
        payload = path.read_bytes()
        report = json.loads(payload)
        if report.get("data_kind") != "prospective":
            continue
        issued = stamp(report["finished_at"])
        generated = stamp(report["generated_at"])
        if generated > issued:
            raise ValueError("Forecast generation occurs after issuance")
        if issued >= cutoff:
            continue
        digest = hashlib.sha256(payload).hexdigest()
        sources.append({"snapshot": path.parent.name, "sha256": digest})
        seen = set()
        for r in report["upcoming"]:
            probability = r["probability"]
            if r["league"] not in LEAGUES or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid saved forecast")
            start, predicted = stamp(r["start_at"]), stamp(r["prediction_at"])
            if predicted != generated or not predicted <= issued < start or r.get("actual") is not None:
                raise ValueError("Saved forecast was not issued before the scheduled start")
            key = r["game_id"], r["model"]
            if key in seen:
                raise ValueError("Duplicate game/model within a forecast snapshot")
            seen.add(key)
            candidate = dict(r, issued_at=issued.isoformat(), snapshot=path.parent.name, forecast_sha256=digest)
            if key not in selected or issued < stamp(selected[key]["issued_at"]):
                selected[key] = candidate
    scored, pending, excluded = [], [], []
    for r in selected.values():
        g = results.get(r["game_id"])
        if not g or g["result_known_at"] is None or g["result_known_at"] >= cutoff:
            pending.append(r)
            continue
        reason = None
        if (g["league"], g["home"], g["away"]) != (r["league"], r["home"], r["away"]):
            reason = "matchup_changed"
        elif g["start_at"] != stamp(r["start_at"]):
            reason = "schedule_changed"
        elif g["result_known_at"] <= stamp(r["issued_at"]):
            reason = "result_not_observed_after_issuance"
        elif outcome(g) is None:
            reason = "tie_excluded"
        if reason:
            excluded.append(dict(r, exclusion=reason))
        else:
            scored.append(dict(r, actual=outcome(g), result_observed_at=g["result_known_at"].isoformat()))
    models = sorted({r["model"] for r in selected.values()})
    report = {"schema_version": 1, "data_kind": "prospective_scorecard", "scored_at": cutoff.isoformat(),
              "policy": "Earliest issued forecast per game/model. Changed schedules/matchups and ties excluded. Only results observed after issuance and before scoring cutoff count.",
              "warning": "Small prospective samples do not establish stable accuracy or betting profitability. Exclusions are reported; this is not a complete schedule audit.",
              "summary": {l: {m: metrics([r for r in scored if r["league"] == l and r["model"] == m]) for m in models} for l in sorted(LEAGUES)},
              "unique_games": len({r["game_id"] for r in selected.values()}),
              "scored_games": len({r["game_id"] for r in scored}), "pending_games": len({r["game_id"] for r in pending}),
              "excluded_games": len({r["game_id"] for r in excluded}),
              "scored": scored, "pending": pending, "excluded": excluded,
              "source_snapshots": sources, "result_snapshots": result_sources,
              "results_sha256": hashlib.sha256(Path(results_path).read_bytes()).hexdigest()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report
