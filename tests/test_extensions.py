import copy
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from sportsedge.availability import capture
from sportsedge.core import FEATURES, features, read_availability, read_games
from sportsedge.ingest import parse_scoreboard, parse_mlb, row
from sportsedge.pipeline import demo, run


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        demo(self.root, count=40)
        self.games = read_games(self.root / "games.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_nhl_scoreboard_and_all_star_exclusion(self):
        c = {"status": {"type": {"completed": True}}, "competitors": [{"homeAway": "home", "team": {"abbreviation": "BOS"}, "score": "3"}, {"homeAway": "away", "team": {"abbreviation": "NYR"}, "score": "2"}]}
        event = {"id": "1", "date": "2023-01-01T22:00Z", "season": {"type": 2}, "competitions": [c]}
        parsed = parse_scoreboard({"events": [event]}, 7, "2026-01-01T00:00Z", "NHL")
        self.assertEqual(parsed[0]["league"], "NHL")
        self.assertEqual(parsed[0]["result_known_at"], "2023-01-08T00:00:00+00:00")
        c["competitors"][0]["team"]["abbreviation"] = "MET"
        self.assertEqual(parse_scoreboard({"events": [event]}, 7, "2026-01-01T00:00Z", "NHL"), [])

    def test_suspended_mlb_uses_completion_date(self):
        import csv
        import io
        fields = [""] * 161
        for i, v in {0: "20230401", 1: "0", 3: "NYA", 6: "BOS", 9: "2", 10: "3", 13: "20230501,park,1,1,20"}.items():
            fields[i] = v
        handle = io.StringIO()
        csv.writer(handle).writerow(fields)
        parsed = parse_mlb(handle.getvalue(), 7, "2026-01-01T00:00Z")
        self.assertEqual(parsed[0]["result_known_at"], "2023-05-08T00:00:00+00:00")

    def test_strict_observed_policy_does_not_backdate(self):
        g = row("1", "NHL", "BOS", "NYR", "2023-01-01", 3, 2, None, "2026-01-01T00:00:00Z")
        self.assertEqual(g["result_known_at"], "2026-01-01T00:00:00Z")

    def test_stale_availability_and_missing_indicators(self):
        g = self.games[20]
        r = {"game_id": g["game_id"], "team": g["home"], "player_id": "1", "status": "out", "weight": 1, "published_at": g["prediction_at"] - timedelta(days=4), "observed_at": g["prediction_at"] - timedelta(days=3)}
        x = features(g, self.games, [r])
        self.assertEqual(x[-2:], [0, 0])
        r["published_at"] = r["observed_at"] = g["prediction_at"] - timedelta(hours=1)
        x = features(g, self.games, [r])
        self.assertEqual(x[-2:], [1, 0])
        self.assertEqual(x[4], -1)

    def test_manifest_rejects_changed_csv(self):
        import hashlib
        path = self.root / "games.csv"
        (self.root / "source.json").write_text(json.dumps({"kind": "historical_reconstructed", "games_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        with path.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Input hash differs"):
            run(path, None, self.root / "out")

    def test_capture_overrides_claimed_observation_time(self):
        g = self.games[0]
        payload = [{"game_id": g["game_id"], "team": g["home"], "player_id": "p", "status": "out", "weight": 0.5, "published_at": "2023-01-01T00:00Z", "observed_at": "2023-01-01T00:00Z"}]
        path = self.root / "incoming.json"
        path.write_text(json.dumps(payload))
        with patch("sportsedge.availability.utcnow", return_value=datetime(2026, 1, 1, tzinfo=timezone.utc)):
            capture(path, self.root / "games.csv", self.root / "snapshots.csv")
        snapshots = read_availability(self.root / "snapshots.csv", self.games)
        self.assertEqual(snapshots[0]["observed_at"].year, 2026)
        self.assertEqual(features(g, self.games, snapshots)[4], 0)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "ML extra not installed")
    def test_frozen_training_and_saved_model(self):
        report = run(self.root / "games.csv", None, self.root / "trained", min_train=10, ml=True, test_from="2024-02-15T00:00:00Z")
        self.assertTrue(report["folds"])
        self.assertTrue((self.root / "trained/models/NHL-gradient-boosting.joblib").exists())
        self.assertIn("gradient_boosting", report["summary"]["NHL"])
        for f in report["folds"]:
            self.assertLess(f["latest_training_result"], f["trained_before"])
        self.assertEqual(len(FEATURES), 13)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "ML extra not installed")
    def test_holdout_labels_do_not_change_fitted_weights(self):
        import csv
        from sportsedge.pipeline import write_csv
        path = self.root / "games.csv"
        first = run(path, None, self.root / "first", min_train=10, ml=True, test_from="2024-02-15T00:00:00Z")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for r in rows:
            if r["prediction_at"] >= "2024-02-15":
                r["home_score"], r["away_score"] = "0", "9"
        write_csv(path, fields, rows)
        second = run(path, None, self.root / "second", min_train=10, ml=True, test_from="2024-02-15T00:00:00Z")
        self.assertEqual(first["folds"], second["folds"])
        for league in ("NBA", "MLB", "NHL"):
            initial = [r for r in first["predictions"] if r["league"] == league and r["model"] == "gradient_boosting"][0]
            repeated = [r for r in second["predictions"] if r["game_id"] == initial["game_id"] and r["model"] == "gradient_boosting"][0]
            self.assertEqual(initial["probability"], repeated["probability"])


if __name__ == "__main__":
    unittest.main()
