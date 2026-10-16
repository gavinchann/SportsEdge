import json
import tempfile
import unittest
from pathlib import Path
from sportsedge.pipeline import GAME_COLUMNS, write_csv
from sportsedge.scorecard import score_forecasts


class ScorecardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.results = self.root / "results.csv"
        self.game = dict(zip(GAME_COLUMNS, ["NHL-1", "NHL", "BOS", "NYR", "2026-10-16T20:00Z", "2026-10-16T19:00Z", "2026-10-17T01:00Z", 3, 2, 0]))
        write_csv(self.results, GAME_COLUMNS, [self.game])

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self, name, time="2026-10-16T18:00Z", p=.6):
        directory = self.root / "snapshots" / name
        directory.mkdir(parents=True)
        report = {"data_kind": "prospective", "generated_at": time, "finished_at": time,
                  "upcoming": [{"game_id": "NHL-1", "league": "NHL", "home": "BOS", "away": "NYR",
                                "model": "logistic", "probability": p, "actual": None,
                                "start_at": "2026-10-16T20:00Z", "prediction_at": time}]}
        (directory / "forecast.json").write_text(json.dumps(report))
        return directory

    def score(self, cutoff="2026-10-18T00:00Z"):
        return score_forecasts(self.root / "snapshots", self.results, self.root / "score.json", cutoff)

    def test_earliest_forecast_scored_once_and_pending_before_result(self):
        self.snapshot("first", p=.6)
        self.snapshot("second", "2026-10-16T19:00Z", p=.99)
        self.assertEqual(self.score("2026-10-17T01:00Z")["pending_games"], 1)
        scored = self.score()
        self.assertEqual(scored["scored_games"], 1)
        self.assertEqual(len(scored["scored"]), 1)
        self.assertAlmostEqual(scored["summary"]["NHL"]["logistic"]["brier"], .16)

    def test_schedule_change_excluded_and_late_forecast_rejected(self):
        path = self.snapshot("first")
        write_csv(self.results, GAME_COLUMNS, [dict(self.game, start_at="2026-10-16T21:00Z")])
        self.assertEqual(self.score()["excluded"][0]["exclusion"], "schedule_changed")
        payload = json.loads((path / "forecast.json").read_text())
        payload["finished_at"] = "2026-10-16T20:01Z"
        (path / "forecast.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "before the scheduled"):
            self.score()

    def test_retains_old_observed_finals_when_current_feed_is_empty(self):
        path = self.snapshot("first")
        write_csv(path / "observed.csv", GAME_COLUMNS, [self.game])
        write_csv(self.results, GAME_COLUMNS, [])
        self.assertEqual(self.score()["scored_games"], 1)


if __name__ == "__main__":
    unittest.main()
