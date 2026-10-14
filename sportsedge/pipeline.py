import csv
import hashlib
import json
import random
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .core import FEATURES, LEAGUES, Logistic, features, metrics, outcome, read_availability, read_games, sigmoid, stamp

GAME_COLUMNS = ["game_id", "league", "home", "away", "start_at", "prediction_at", "result_known_at", "home_score", "away_score", "neutral"]
AVAILABILITY_COLUMNS = ["game_id", "team", "player_id", "published_at", "observed_at", "status", "weight"]


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def demo(directory, seed=42, count=120):
    """Fabricated teams, results and snapshots: tests plumbing, not real skill."""
    if count < 30:
        raise ValueError("Demo needs at least 30 games per league")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    games, snapshots = [], []
    origin = datetime(2024, 1, 1, 20, tzinfo=timezone.utc)
    for league in sorted(LEAGUES):
        for i in range(count):
            h, a = rng.sample(range(6), 2)
            start = origin + timedelta(days=i * (7 if league == "NFL" else 2))
            cutoff = start - timedelta(hours=1)
            gid = f"SYNTHETIC-{league}-{i:04}"
            home, away = f"{league}-Demo-{h}", f"{league}-Demo-{a}"
            missing = rng.random() < 0.2
            win = rng.random() < sigmoid(0.25 + (h - a) * 0.25 - missing * 0.5)
            base = {"NBA": 100, "NFL": 20, "MLB": 3, "NHL": 2}[league]
            games.append(dict(zip(GAME_COLUMNS, [gid, league, home, away, start.isoformat(), cutoff.isoformat(), (start + timedelta(hours=4)).isoformat(), base + int(win), base + int(not win), 0])))
            for team in (home, away):
                snapshots.append(dict(zip(AVAILABILITY_COLUMNS, [gid, team, f"{team}-player-1", (cutoff - timedelta(hours=3)).isoformat(), (cutoff - timedelta(hours=2)).isoformat(), "out" if team == home and missing else "available", 0.7])))
            # Deliberately late data, excluded by the point-in-time join.
            snapshots.append(dict(zip(AVAILABILITY_COLUMNS, [gid, home, f"{home}-player-1", (start + timedelta(minutes=1)).isoformat(), (start + timedelta(minutes=2)).isoformat(), "out", 1])))
    write_csv(directory / "games.csv", GAME_COLUMNS, games)
    write_csv(directory / "availability.csv", AVAILABILITY_COLUMNS, snapshots)
    (directory / "source.json").write_text(json.dumps({"kind": "synthetic", "seed": seed, "games_per_league": count}, indent=2), encoding="utf-8")


def run(games_path, availability_path, output, min_train=20, batch_size=20, data_kind="unverified", ml=False, test_from=None):
    if min_train < 2 or batch_size < 1:
        raise ValueError("min_train >= 2 and batch_size >= 1 required")
    games = read_games(games_path)
    availability = read_availability(availability_path, games)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_path = Path(games_path).with_name("source.json")
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    digest = hashlib.sha256(Path(games_path).read_bytes()).hexdigest()
    if source.get("games_sha256") and source["games_sha256"] != digest:
        raise ValueError("Input hash differs from ingestion manifest; regenerate or use a new dataset directory")
    if source.get("kind") in ("historical_reconstructed", "historical_observed", "synthetic"):
        data_kind = source["kind"]
    if ml:
        from sklearn.ensemble import HistGradientBoostingClassifier
        import joblib
        import sklearn
        import numpy
        (output / "models").mkdir(exist_ok=True)
    models = ["home_prior", "elo", "logistic"] + (["gradient_boosting", "logistic_no_h2h"] if ml else [])
    h2h_indices = [FEATURES.index(name) for name in ("h2h", "h2h_count", "h2h_recency", "h2h_age")]
    def without_h2h(x):
        return [0 if j in h2h_indices else v for j, v in enumerate(x)]
    vectors = {g["game_id"]: features(g, games, availability) for g in games}
    predictions, future, folds = [], [], []
    for league in sorted(LEAGUES):
        league_games = [g for g in games if g["league"] == league]
        # Batch all fixtures, including ties/unplayed rows, without using labels to select fold boundaries.
        boundary = stamp(test_from) if test_from else None
        blocks = [(0, [g for g in league_games if g["prediction_at"] >= boundary])] if boundary else [(i, league_games[i:i + batch_size]) for i in range(0, len(league_games), batch_size)]
        for index, block in blocks:
            if not block:
                continue
            cutoff = boundary or block[0]["prediction_at"]
            train = [g for g in league_games if g["prediction_at"] < cutoff and g["result_known_at"] and g["result_known_at"] < cutoff and outcome(g) is not None]
            if len(train) < min_train:
                continue
            model = Logistic().fit([(vectors[g["game_id"]], outcome(g)) for g in train])
            if ml:
                tree = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=2, learning_rate=0.05, early_stopping=False, random_state=42)
                tree.fit([vectors[g["game_id"]] for g in train], [outcome(g) for g in train])
                ablation = Logistic().fit([(without_h2h(vectors[g["game_id"]]), outcome(g)) for g in train])
                tree_probs = tree.predict_proba([vectors[g["game_id"]] for g in block])
                class_list = list(tree.classes_)
                home_index = class_list.index(1) if 1 in class_list else None
                tree_home = [float(p[home_index]) if home_index is not None else 0.0 for p in tree_probs]
                joblib.dump({"model": tree, "features": FEATURES, "trained_before": cutoff.isoformat(), "data_sha256": digest}, output / "models" / f"{league}-gradient-boosting.joblib")
            prior = (sum(outcome(g) for g in train) + 1) / (len(train) + 2)
            fold_id = f"{league}-{index // batch_size}"
            folds.append({"fold": fold_id, "league": league, "trained_before": cutoff.isoformat(), "training_n": len(train), "latest_training_result": max(g["result_known_at"] for g in train).isoformat(), "weights": dict(zip(["intercept"] + FEATURES, model.weights))})
            (output / f"{league}-logistic.json").write_text(json.dumps({"features": FEATURES, "weights": model.weights, "trained_before": cutoff.isoformat(), "training_n": len(train), "data_sha256": digest}, indent=2), encoding="utf-8")
            for block_index, g in enumerate(block):
                x = vectors[g["game_id"]]
                probabilities = {"home_prior": prior, "elo": sigmoid((400 * x[0] + (0 if g["neutral"] else 50)) * 2.302585092994046 / 400), "logistic": model.predict(x)}
                if ml:
                    probabilities.update(gradient_boosting=tree_home[block_index], logistic_no_h2h=ablation.predict(without_h2h(x)))
                for name, p in probabilities.items():
                    record = {"game_id": g["game_id"], "league": league, "home": g["home"], "away": g["away"], "prediction_at": g["prediction_at"].isoformat(), "fold": fold_id, "model": name, "probability": p, "actual": outcome(g)}
                    if outcome(g) is not None:
                        predictions.append(record)
                    elif g["home_score"] is None:
                        future.append(record)
    summary = {league: {name: metrics([r for r in predictions if r["league"] == league and r["model"] == name]) for name in models} for league in sorted(LEAGUES)}
    source_hashes = {"games": hashlib.sha256(Path(games_path).read_bytes()).hexdigest()}
    if availability_path:
        source_hashes["availability"] = hashlib.sha256(Path(availability_path).read_bytes()).hexdigest()
    report = {"schema_version": 1, "data_kind": data_kind, "warning": "Synthetic results validate software only; no real-world performance claim." if data_kind == "synthetic" else "User-supplied data; historical availability and timing provenance require verification.", "config": {"min_train": min_train, "batch_size": batch_size, "features": FEATURES, "seed": 42, "target": "home win conditional on a decisive result"}, "source_sha256": source_hashes, "games": len(games), "availability_rows": len(availability), "ties_excluded": sum(g["home_score"] is not None and outcome(g) is None for g in games), "summary": summary, "folds": folds, "predictions": predictions, "upcoming": future}
    report["runtime"] = {"python": platform.python_version(), "sportsedge": "0.2.0"}
    if ml:
        report["runtime"].update(sklearn=sklearn.__version__, numpy=numpy.__version__)
    report["config"].update(ml=ml, test_from=test_from, evaluation="frozen_weights_sequential_features" if test_from else "expanding_window", gradient_boosting={"max_iter": 100, "max_leaf_nodes": 7, "min_samples_leaf": 30, "l2_regularization": 2, "learning_rate": 0.05, "early_stopping": False, "random_state": 42} if ml else None)
    report["source"] = {key: source[key] for key in ("kind", "years", "counts", "result_lag_days", "timing_policy", "availability") if key in source}
    report["availability_coverage"] = {league: {"games": sum(g["league"] == league for g in games), "home_reported": sum(vectors[g["game_id"]][-2] for g in games if g["league"] == league), "away_reported": sum(vectors[g["game_id"]][-1] for g in games if g["league"] == league)} for league in sorted(LEAGUES)}
    if data_kind.startswith("historical_"):
        report["warning"] = f"Real archived results. {source.get('timing_policy', '')} Injury coverage is reported separately. No betting profitability or live performance established."
    report["date_ranges"] = {l: {"first": min((g["start_at"].isoformat() for g in games if g["league"] == l), default=None), "last": max((g["start_at"].isoformat() for g in games if g["league"] == l), default=None)} for l in sorted(LEAGUES)}
    report["schema_version"] = 2
    report["config"]["availability_max_age_hours"] = 72
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "features.csv", ["game_id", "prediction_at"] + FEATURES, [{"game_id": g["game_id"], "prediction_at": g["prediction_at"].isoformat(), **dict(zip(FEATURES, vectors[g["game_id"]]))} for g in games])
    if predictions:
        write_csv(output / "predictions.csv", list(predictions[0]), predictions)
    elif (output / "predictions.csv").exists():
        (output / "predictions.csv").unlink()
    return report
