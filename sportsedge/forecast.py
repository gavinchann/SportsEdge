"""Prospective pregame forecasts; immutable source snapshots and explicit training assumptions."""
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .core import read_games, stamp
from .ingest import CLUBS, cached
from .pipeline import GAME_COLUMNS, run, write_csv

SPORTS = {"NBA": "basketball/nba", "NFL": "football/nfl", "MLB": "baseball/mlb", "NHL": "hockey/nhl"}
ALIASES = {
    "NFL": {"WSH": "WAS", "LAR": "LA"},
    "MLB": {"LAA": "ANA", "CHW": "CHA", "CWS": "CHA", "CHC": "CHN", "KC": "KCA", "LAD": "LAN", "NYY": "NYA", "NYM": "NYN", "SD": "SDN", "SF": "SFN", "STL": "SLN", "TB": "TBA", "WSH": "WAS", "ATH": "OAK"},
    "NBA": {"UTA": "UTAH"}, "NHL": {"UTA": "UTAH"},
}
TEAMS = {**CLUBS,
    "NFL": set("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split()),
    "MLB": set("ANA ARI ATL BAL BOS CHA CHN CIN CLE COL DET HOU KCA LAN MIA MIL MIN NYA NYN OAK PHI PIT SDN SEA SFN SLN TBA TEX TOR WAS".split())}


def parse_live(payload, league, observed):
    """Only final or explicitly scheduled fixtures; ignore odds, records and eventual lineups."""
    rows = []
    for event in payload.get("events", []):
        if event.get("season", {}).get("type") not in (2, 3):
            continue
        competition = event["competitions"][0]
        if competition.get("type", {}).get("abbreviation") == "ALLSTAR":
            continue
        status = competition.get("status", event.get("status", {})).get("type", {})
        final = status.get("completed") and status.get("name") == "STATUS_FINAL"
        scheduled = status.get("name") == "STATUS_SCHEDULED" and status.get("state") == "pre"
        if not (final or scheduled) or competition.get("timeValid") is False:
            continue
        sides = {p["homeAway"]: p for p in competition["competitors"]}
        h, a = sides["home"], sides["away"]
        home, away = [ALIASES[league].get(p["team"]["abbreviation"], p["team"]["abbreviation"]) for p in (h, a)]
        exhibition = {"AL", "NL", "EAST", "WEST", "AFC", "NFC", "USA", "CAN", "SWE", "FIN"}
        if home in exhibition or away in exhibition:
            continue
        if home not in TEAMS[league] or away not in TEAMS[league]:
            raise ValueError(f"Unrecognized {league} team: {home}/{away}; update mappings before forecasting")
        start = stamp(competition.get("date", event["date"]))
        seen = stamp(observed)
        if final and start >= seen:
            raise ValueError("Final result has a future start time")
        if scheduled and start <= seen:
            continue
        rows.append(dict(zip(GAME_COLUMNS, [league + "-" + event["id"], league, home, away,
            start.isoformat(), (start - timedelta(hours=1)).isoformat(), observed if final else "",
            int(h["score"]) if final else "", int(a["score"]) if final else "", int(competition.get("neutralSite", False))])))
    return rows


def serialize(g):
    return {k: (v.isoformat() if isinstance(v, datetime) else "" if v is None else v) for k, v in g.items() if k in GAME_COLUMNS}


def prepare(archive, live, as_of, horizon):
    """Keep actual observed results for inference; reconstruction is limited to training."""
    archive = [g for g in archive if g["result_known_at"] and g["result_known_at"] < as_of]
    last_archive = {l: max((g["start_at"].date() for g in archive if g["league"] == l), default=None) for l in SPORTS}
    actual = list(archive)
    training = list(archive)
    fixtures = []
    for g in live:
        if g["home_score"] is None:
            if as_of < g["start_at"] <= horizon:
                fixtures.append(dict(g, prediction_at=as_of))
        elif g["result_known_at"] < as_of:
            # Archive and ESPN can use different IDs. Exclude overlapping archive dates,
            # preserving same-day MLB doubleheaders within the ESPN snapshot itself.
            if last_archive[g["league"]] and g["start_at"].date() <= last_archive[g["league"]]:
                continue
            actual.append(g)
            reconstructed = g["start_at"] + timedelta(days=7)
            if reconstructed < as_of:
                training.append(dict(g, result_known_at=reconstructed))
    return training, actual, fixtures


def forecast(history_path, output="runs/forecast", days=7, lookback_days=730, ml=False, availability_path=None, min_train=100):
    if not 1 <= days <= 30 or not 7 <= lookback_days <= 730:
        raise ValueError("days must be 1..30 and lookback-days 7..730")
    started = datetime.now(timezone.utc)
    output = Path(output)
    snapshot = output / "snapshots" / started.strftime("%Y%m%dT%H%M%S%fZ")
    snapshot.mkdir(parents=True, exist_ok=False)
    archive = read_games(history_path)
    source_path = Path(history_path).with_name("source.json")
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    digest = hashlib.sha256(Path(history_path).read_bytes()).hexdigest()
    if source.get("games_sha256") != digest or source.get("kind") != "historical_reconstructed":
        raise ValueError("Forecast training requires a hash-verified historical_reconstructed archive")
    tasks = [(league, (started + timedelta(days=offset)).strftime("%Y%m%d"))
             for league in SPORTS for offset in range(-lookback_days, days + 1)]
    def fetch(task):
        league, date = task
        url = f"https://site.api.espn.com/apis/site/v2/sports/{SPORTS[league]}/scoreboard?dates={date}&limit=1000"
        body, meta = cached(url, snapshot / "raw")
        payload = json.loads(body)
        if "events" not in payload or not isinstance(payload["events"], list):
            raise ValueError("Unexpected scoreboard schema")
        return parse_live(payload, league, meta["retrieved_at"]), meta
    unique, sources = {}, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, (rows, meta) in enumerate(pool.map(fetch, tasks)):
            sources.append(meta)
            for g in rows:
                # Later observations supersede prior schedule/status snapshots.
                previous = unique.get(g["game_id"])
                if not previous or meta["retrieved_at"] > previous[0]:
                    unique[g["game_id"]] = (meta["retrieved_at"], g)
            if index % 100 == 0:
                print(f"Current schedules/results: {index + 1}/{len(tasks)} snapshots", flush=True)
    write_csv(snapshot / "observed.csv", GAME_COLUMNS, [g for _, g in unique.values()])
    as_of = datetime.now(timezone.utc)
    training, actual, fixtures = prepare(archive, read_games(snapshot / "observed.csv"), as_of, as_of + timedelta(days=days))
    write_csv(snapshot / "inference-history.csv", GAME_COLUMNS, [serialize(g) for g in actual])
    write_csv(snapshot / "games.csv", GAME_COLUMNS, [serialize(g) for g in training + fixtures])
    if availability_path:
        copied_availability = snapshot / "availability.csv"
        copied_availability.write_bytes(Path(availability_path).read_bytes())
        availability_path = copied_availability
    manifest = {"kind": "historical_reconstructed", "result_lag_days": 7,
        "timing_policy": "Training uses archived timing assumptions plus a seven-day delay for newly retrieved historical finals. Forecast features use actual retrieval times for new results; all forecasts share the generation cutoff.",
        "games_sha256": hashlib.sha256((snapshot / "games.csv").read_bytes()).hexdigest(),
        "history_sha256": digest, "history_source": source, "sources": sources,
        "inference_history_sha256": hashlib.sha256((snapshot / "inference-history.csv").read_bytes()).hexdigest(),
        "availability_sha256": hashlib.sha256(Path(availability_path).read_bytes()).hexdigest() if availability_path else None,
        "ml": ml, "min_train": min_train,
        "generated_at": as_of.isoformat(), "lookback_days": lookback_days}
    (snapshot / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Refitting models on {len(training)} past games; forecasting {len(fixtures)} fixtures", flush=True)
    report = run(snapshot / "games.csv", availability_path, snapshot / "models", min_train=min_train,
                 ml=ml, test_from=as_of.isoformat(), forecast_history=actual)
    # Model fitting can take time: never present a newly generated pregame prediction
    # for a fixture that started while fitting was in progress.
    finished = datetime.now(timezone.utc)
    report["upcoming"] = [r for r in report["upcoming"] if stamp(r["start_at"]) > finished]
    report.update(data_kind="prospective", generated_at=as_of.isoformat(), finished_at=finished.isoformat(),
        warning="Pregame forecasts, not evaluated results. Refitted on reconstructed historical features; current results were observed at retrieval. Injuries/lineups are unknown unless timestamped reports are supplied. No live accuracy or profitability established.",
        forecast={"horizon_days": days, "lookback_days": lookback_days, "snapshot": str(snapshot),
                  "schedule_first_observed_at": min(m["retrieved_at"] for m in sources),
                  "schedule_last_observed_at": max(m["retrieved_at"] for m in sources),
                  "fixtures": len({r["game_id"] for r in report["upcoming"]}),
                  "training_games": len(training), "observed_recent_results": len(actual) - len(archive),
                  "training_reconstruction_delay_days": 7,
                  "missing_leagues": [l for l in SPORTS if not any(r["league"] == l for r in report["upcoming"])]})
    for r in report["upcoming"]:
        r["away_probability"] = 1 - r["probability"]
    upcoming_ids = {r["game_id"] for r in report["upcoming"]}
    with (snapshot / "models" / "features.csv").open(newline="", encoding="utf-8") as handle:
        report["matchup_context"] = {
            r["game_id"]: {"h2h_games": round(float(r["h2h_count"]) * 10),
                           "h2h_smoothed_home_win_rate": float(r["h2h"]) + .5,
                           "elo_home_minus_away": float(r["elo_diff"]) * 400,
                           "home_availability_reported": bool(float(r["home_coverage"])),
                           "away_availability_reported": bool(float(r["away_coverage"]))}
            for r in csv.DictReader(handle) if r["game_id"] in upcoming_ids}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (snapshot / "forecast.json").write_text(text, encoding="utf-8")
    temporary = output / "report.tmp"
    temporary.write_text(text, encoding="utf-8")
    from .scorecard import score_forecasts
    score_forecasts(output / "snapshots", snapshot / "observed.csv", output / "scorecard.json")
    temporary.replace(output / "report.json")
    return report


def replay(snapshot, output):
    """Recompute a saved snapshot without fetching data or claiming it is current."""
    snapshot, output = Path(snapshot), Path(output)
    if output.resolve() == snapshot.resolve() or snapshot.resolve() in output.resolve().parents:
        raise ValueError("Replay output must be outside the immutable snapshot")
    manifest = json.loads((snapshot / "source.json").read_text(encoding="utf-8"))
    for name, key in (("games.csv", "games_sha256"), ("inference-history.csv", "inference_history_sha256"), ("availability.csv", "availability_sha256")):
        if manifest.get(key) and hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"Snapshot integrity check failed: {name}")
    if not manifest.get("inference_history_sha256"):
        raise ValueError("Snapshot has no verified inference history")
    saved_config = json.loads((snapshot / "models" / "report.json").read_text(encoding="utf-8"))["config"]
    report = run(snapshot / "games.csv", snapshot / "availability.csv" if manifest.get("availability_sha256") else None,
                 output, min_train=manifest.get("min_train", saved_config["min_train"]), ml=manifest.get("ml", saved_config["ml"]), test_from=manifest["generated_at"],
                 forecast_history=read_games(snapshot / "inference-history.csv"), calibrate=bool(saved_config.get("probability_calibration")))
    report.update(data_kind="forecast_replay", generated_at=manifest["generated_at"],
                  warning="Offline replay of an old forecast snapshot, not a current prediction or performance evaluation.")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
