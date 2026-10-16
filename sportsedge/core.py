"""Strict schemas, point-in-time features, and dependency-free baseline models."""
import csv
import math
from datetime import datetime

LEAGUES = {"NBA", "NFL", "MLB", "NHL"}
FEATURES = ["elo_diff", "form_diff", "h2h", "rest_diff", "absence_diff", "coverage_diff", "neutral",
            "h2h_count", "h2h_recency", "h2h_age", "margin_diff", "home_coverage", "away_coverage"]
STATUS = {"available": 0.0, "probable": 0.25, "questionable": 0.5, "doubtful": 0.75, "out": 1.0}


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a UTC offset")
    return result


def read_games(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for g in rows:
        if not g["game_id"] or g["game_id"] in seen:
            raise ValueError("Empty or duplicate game_id")
        seen.add(g["game_id"])
        if g["league"] not in LEAGUES or g["home"] == g["away"] or not g["home"] or not g["away"]:
            raise ValueError("Invalid league or teams")
        for key in ("start_at", "prediction_at", "result_known_at"):
            g[key] = stamp(g[key]) if g[key] else None
        if not g["start_at"] or not g["prediction_at"] or g["prediction_at"] >= g["start_at"]:
            raise ValueError("Prediction must precede start")
        if g["neutral"] not in ("0", "1"):
            raise ValueError("neutral must be 0 or 1")
        g["neutral"] = int(g["neutral"])
        for key in ("home_score", "away_score"):
            g[key] = int(g[key]) if g[key] else None
            if g[key] is not None and g[key] < 0:
                raise ValueError("Negative score")
        complete = g["home_score"] is not None and g["away_score"] is not None
        if (g["home_score"] is None) != (g["away_score"] is None) or complete != (g["result_known_at"] is not None):
            raise ValueError("Scores and result_known_at must all be present or all blank")
        if complete and g["result_known_at"] <= g["start_at"]:
            raise ValueError("Result must become known after start")
    return sorted(rows, key=lambda g: (g["prediction_at"], g["game_id"]))


def read_availability(path, games):
    if path is None:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {g["game_id"]: g for g in games}
    seen = set()
    for r in rows:
        g = lookup.get(r["game_id"])
        if not g or r["team"] not in (g["home"], g["away"]) or not r["player_id"] or r["status"] not in STATUS:
            raise ValueError("Invalid availability record")
        r["published_at"], r["observed_at"] = stamp(r["published_at"]), stamp(r["observed_at"])
        r["weight"] = float(r["weight"])
        if not math.isfinite(r["weight"]) or not 0 <= r["weight"] <= 1 or r["observed_at"] < r["published_at"]:
            raise ValueError("Invalid availability weight or chronology")
        key = (r["game_id"], r["team"], r["player_id"], r["published_at"], r["observed_at"])
        if key in seen:
            raise ValueError("Duplicate availability snapshot")
        seen.add(key)
    return rows


def outcome(g):
    if g["home_score"] is None or g["home_score"] == g["away_score"]:
        return None
    return int(g["home_score"] > g["away_score"])


def sigmoid(x):
    return 1 / (1 + math.exp(-max(-35, min(35, x))))


def features(target, games, availability, max_availability_age_hours=72):
    cutoff = target["prediction_at"]
    history = sorted((g for g in games if g["league"] == target["league"] and g["result_known_at"] and g["result_known_at"] < cutoff), key=lambda g: (g["start_at"], g["game_id"]))
    ratings, records, last, meetings, margins = {}, {}, {}, [], {}
    home, away = target["home"], target["away"]
    for g in history:
        h, a = g["home"], g["away"]
        rh, ra = ratings.get(h, 1500), ratings.get(a, 1500)
        y = 0.5 if outcome(g) is None else outcome(g)
        expected = sigmoid((rh - ra + (0 if g["neutral"] else 50)) * math.log(10) / 400)
        ratings[h], ratings[a] = rh + 20 * (y - expected), ra - 20 * (y - expected)
        records.setdefault(h, []).append(y)
        records.setdefault(a, []).append(1 - y)
        scale = {"NBA": 20, "NFL": 14, "MLB": 5, "NHL": 3}[g["league"]]
        margin = max(-2, min(2, (g["home_score"] - g["away_score"]) / scale))
        margins.setdefault(h, []).append(margin)
        margins.setdefault(a, []).append(-margin)
        for team in (h, a):
            last[team] = max(last.get(team, g["start_at"]), g["start_at"])
        if {h, a} == {home, away}:
            meetings.append((y if h == home else 1 - y, g["start_at"]))
    def form(team):
        r = records.get(team, [])[-10:]
        return (sum(r) + 2) / (len(r) + 4)
    def rest(team):
        return min(14, max(0, (target["start_at"] - last[team]).total_seconds() / 86400)) if team in last else 14
    latest = {}
    for r in availability:
        if r["game_id"] == target["game_id"] and r["published_at"] < cutoff and r["observed_at"] < cutoff and (cutoff - r["published_at"]).total_seconds() <= max_availability_age_hours * 3600:
            key = (r["team"], r["player_id"])
            if key not in latest or (r["published_at"], r["observed_at"]) > (latest[key]["published_at"], latest[key]["observed_at"]):
                latest[key] = r
    def absence(team):
        return sum(r["weight"] * STATUS[r["status"]] for (t, _), r in latest.items() if t == team)
    def coverage(team):
        return int(any(t == team for t, _ in latest))
    h2h = meetings[-10:]
    ages = [(cutoff - date).total_seconds() / 86400 for _, date in h2h]
    weights = [0.5 ** (age / 365) for age in ages]
    def margin_form(team):
        values = margins.get(team, [])[-10:]
        return sum(values) / (len(values) + 4)
    return [(ratings.get(home, 1500) - ratings.get(away, 1500)) / 400,
            form(home) - form(away), (sum(y for y, _ in h2h) + 2) / (len(h2h) + 4) - 0.5,
            (rest(home) - rest(away)) / 14, absence(away) - absence(home),
            coverage(home) - coverage(away), target["neutral"], len(h2h) / 10,
            (sum(y * w for (y, _), w in zip(h2h, weights)) + 2) / (sum(weights) + 4) - 0.5,
            min(min(ages), 730) / 730 if ages else 1,
            margin_form(home) - margin_form(away), coverage(home), coverage(away)]


class Logistic:
    def fit(self, rows):
        # Fixed hyperparameters; no test-fold tuning or whole-dataset scaling.
        self.weights = [0.0] * (len(FEATURES) + 1)
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            x = np.array([[1.0] + list(x) for x, _ in rows])
            y = np.array([y for _, y in rows])
            w = np.zeros(x.shape[1])
            for _ in range(250):
                error = 1 / (1 + np.exp(-np.clip(x @ w, -35, 35))) - y
                penalty = 0.01 * w
                penalty[0] = 0
                w -= 0.15 * (x.T @ error / len(y) + penalty)
            self.weights = w.tolist()
            return self
        for _ in range(250):
            grad = [0.0] * len(self.weights)
            for x, y in rows:
                xx = [1.0] + x
                err = sigmoid(sum(a * b for a, b in zip(self.weights, xx))) - y
                for j, v in enumerate(xx):
                    grad[j] += err * v
            for j in range(len(self.weights)):
                self.weights[j] -= 0.15 * (grad[j] / len(rows) + (0.01 * self.weights[j] if j else 0))
        return self

    def predict(self, x):
        return sigmoid(sum(a * b for a, b in zip(self.weights, [1.0] + x)))


def metrics(rows):
    if not rows:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None, "calibration": []}
    pairs = [(max(1e-12, min(1 - 1e-12, r["probability"])), r["actual"]) for r in rows]
    bins = []
    for i in range(5):
        values = [(p, y) for p, y in pairs if i / 5 <= p < (i + 1) / 5]
        if values:
            bins.append({"lower": i / 5, "n": len(values), "predicted": sum(p for p, _ in values) / len(values), "observed": sum(y for _, y in values) / len(values)})
    return {"n": len(pairs), "accuracy": sum((p >= 0.5) == y for p, y in pairs) / len(pairs), "brier": sum((p - y) ** 2 for p, y in pairs) / len(pairs), "log_loss": -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in pairs) / len(pairs), "calibration": bins}
