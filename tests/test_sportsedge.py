import copy
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from sportsedge.core import features, metrics, read_games, read_availability, stamp
from sportsedge.pipeline import demo, run


class SportsEdgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        demo(self.root, count=40)
        self.games = read_games(self.root / "games.csv")
        self.availability = read_availability(self.root / "availability.csv", self.games)

    def tearDown(self):
        self.temp.cleanup()

    def test_future_results_cannot_change_features(self):
        target = self.games[30]
        before = features(target, self.games, self.availability)
        changed = copy.deepcopy(self.games)
        for g in changed:
            if g["result_known_at"] >= target["prediction_at"]:
                g["home_score"], g["away_score"] = 999, 0
        self.assertEqual(before, features(target, changed, self.availability))

    def test_late_observation_excluded_even_if_published_early(self):
        target = self.games[30]
        baseline = features(target, self.games, self.availability)
        late = {"game_id": target["game_id"], "team": target["home"], "player_id": "extra", "published_at": target["prediction_at"] - timedelta(days=1), "observed_at": target["prediction_at"], "status": "out", "weight": 1.0}
        self.assertEqual(baseline, features(target, self.games, self.availability + [late]))
        late["observed_at"] -= timedelta(seconds=1)
        self.assertAlmostEqual(baseline[4] - 1, features(target, self.games, self.availability + [late])[4])

    def test_latest_available_snapshot_replaces_out(self):
        target = self.games[30]
        report = {"game_id": target["game_id"], "team": target["home"], "player_id": "test", "published_at": target["prediction_at"] - timedelta(hours=3), "observed_at": target["prediction_at"] - timedelta(hours=2), "status": "out", "weight": 1.0}
        update = dict(report, status="available", published_at=target["prediction_at"] - timedelta(hours=1), observed_at=target["prediction_at"] - timedelta(minutes=30))
        self.assertEqual(features(target, self.games, [report, update])[4], 0)

    def test_h2h_orientation_and_league_isolation(self):
        target = copy.deepcopy(self.games[-1])
        prior = copy.deepcopy(target)
        prior.update(game_id="prior", home=target["away"], away=target["home"], home_score=0, away_score=1, start_at=target["prediction_at"] - timedelta(days=2), result_known_at=target["prediction_at"] - timedelta(days=1))
        x = features(target, [prior], [])
        self.assertGreater(x[2], 0)
        prior["league"] = "NBA" if target["league"] != "NBA" else "NFL"
        self.assertEqual(features(target, [prior], [])[2], 0)

    def test_result_at_cutoff_is_excluded(self):
        target = self.games[30]
        prior = copy.deepcopy(target)
        prior["result_known_at"] = target["prediction_at"]
        self.assertEqual(features(target, [prior], []), features(target, [], []))

    def test_naive_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            stamp("2024-01-01T10:00:00")

    def test_duplicate_games_rejected(self):
        path = self.root / "games.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines + [lines[1]]))
        with self.assertRaises(ValueError):
            read_games(path)

    def test_metrics_known_values(self):
        m = metrics([{"probability": 0.5, "actual": 1}, {"probability": 0.5, "actual": 0}])
        self.assertEqual(m["brier"], 0.25)
        self.assertAlmostEqual(m["log_loss"], 0.69314718056)
        self.assertIsNone(metrics([])["brier"])

    def test_ties_excluded_and_upcoming_predicted(self):
        import csv
        from sportsedge.pipeline import write_csv
        path = self.root / "games.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row["game_id"].endswith("0038"):
                row["home_score"] = row["away_score"]
            if row["game_id"].endswith("0039"):
                row["home_score"] = row["away_score"] = row["result_known_at"] = ""
        write_csv(path, fields, rows)
        report = run(path, None, self.root / "out")
        self.assertEqual(report["ties_excluded"], 3)
        self.assertEqual(len(report["upcoming"]), 9)
        self.assertTrue(all(r["actual"] is None for r in report["upcoming"]))
        self.assertTrue(all(m["n"] == 18 for league in report["summary"].values() for m in league.values()))

    def test_bad_availability_weight_rejected(self):
        path = self.root / "availability.csv"
        path.write_text(path.read_text().replace(",0.7", ",nan"))
        with self.assertRaises(ValueError):
            read_availability(path, self.games)

    def test_reproducible_pipeline_and_fold_embargo(self):
        args = (self.root / "games.csv", self.root / "availability.csv")
        report = run(*args, self.root / "a", data_kind="synthetic")
        run(*args, self.root / "b", data_kind="synthetic")
        self.assertEqual((self.root / "a/report.json").read_bytes(), (self.root / "b/report.json").read_bytes())
        for fold in report["folds"]:
            self.assertLess(stamp(fold["latest_training_result"]), stamp(fold["trained_before"]))
        for league in ("NBA", "NFL", "MLB"):
            self.assertEqual(report["summary"][league]["logistic"]["n"], 20)
        self.assertTrue(all(0 < r["probability"] < 1 for r in report["predictions"]))

    def test_future_label_changes_cannot_change_earlier_predictions(self):
        path = self.root / "games.csv"
        first = run(path, None, self.root / "first")
        import csv
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row["game_id"].endswith("0039"):
                row["home_score"], row["away_score"] = "999", "0"
        from sportsedge.pipeline import write_csv
        write_csv(path, fields, rows)
        second = run(path, None, self.root / "second")
        self.assertEqual([(r["game_id"], r["probability"]) for r in first["predictions"]], [(r["game_id"], r["probability"]) for r in second["predictions"]])


if __name__ == "__main__":
    unittest.main()
