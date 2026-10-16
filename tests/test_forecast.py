import copy
import hashlib
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from sportsedge.core import features, read_games, stamp
from sportsedge.forecast import parse_live, prepare, serialize, replay
from sportsedge.pipeline import GAME_COLUMNS, demo, run, write_csv


class ForecastTests(unittest.TestCase):
    def event(self, status="STATUS_SCHEDULED", start="2026-10-17T00:00Z", league="NHL"):
        return {"events": [{"id": "123", "date": start, "season": {"type": 2}, "competitions": [{
            "status": {"type": {"name": status, "state": "pre" if status == "STATUS_SCHEDULED" else "post", "completed": status == "STATUS_FINAL"}},
            "competitors": [{"homeAway": "home", "team": {"abbreviation": "BOS"}, "score": "3"},
                            {"homeAway": "away", "team": {"abbreviation": "NYR" if league == "NHL" else "NYY"}, "score": "2"}]}]}]}

    def test_only_future_scheduled_games_and_actual_result_observations(self):
        seen = "2026-10-16T12:00:00Z"
        r = parse_live(self.event(), "NHL", seen)[0]
        self.assertEqual(r["home_score"], "")
        self.assertEqual(r["result_known_at"], "")
        for status in ("STATUS_IN_PROGRESS", "STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_DELAYED"):
            self.assertEqual(parse_live(self.event(status), "NHL", seen), [])
        self.assertEqual(parse_live(self.event(start=seen), "NHL", seen), [])
        r = parse_live(self.event("STATUS_FINAL", "2026-10-15T00:00Z"), "NHL", seen)[0]
        self.assertEqual(r["result_known_at"], seen)
        self.assertEqual(r["home_score"], 3)

    def test_team_identity_and_preseason_exclusion(self):
        payload = self.event(league="MLB")
        self.assertEqual(parse_live(payload, "MLB", "2026-10-16T12:00Z")[0]["away"], "NYA")
        payload["events"][0]["season"]["type"] = 1
        self.assertEqual(parse_live(payload, "MLB", "2026-10-16T12:00Z"), [])

    def test_prepare_embargo_overlap_and_common_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.csv"
            rows = parse_live(self.event("STATUS_FINAL", "2026-10-01T00:00Z"), "NHL", "2026-10-16T10:00Z")
            rows += parse_live(self.event(), "NHL", "2026-10-16T10:00Z")
            rows[1]["game_id"] = "upcoming"
            write_csv(path, GAME_COLUMNS, rows)
            live = read_games(path)
            cutoff = stamp("2026-10-16T12:00Z")
            training, actual, fixtures = prepare([], live, cutoff, cutoff + timedelta(days=7))
            self.assertEqual(fixtures[0]["prediction_at"], cutoff)
            self.assertEqual(training[0]["result_known_at"], stamp("2026-10-08T00:00Z"))
            self.assertEqual(actual[0]["result_known_at"], stamp("2026-10-16T10:00Z"))
            changed = copy.deepcopy(live)
            changed[0]["result_known_at"] = cutoff
            self.assertEqual(prepare([], changed, cutoff, cutoff + timedelta(days=7))[1], [])
            # Different provider IDs must not double-count overlapping historical dates.
            archive = [dict(actual[0], game_id="archive-id")]
            self.assertEqual(len(prepare(archive, live, cutoff, cutoff + timedelta(days=7))[1]), 1)

    def test_forecast_uses_observed_history_not_training_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo(root, count=30)
            games = [g for g in read_games(root / "games.csv") if g["league"] == "NHL"]
            cutoff = games[-1]["start_at"] + timedelta(days=10)
            upcoming = dict(games[-1], game_id="future", prediction_at=cutoff,
                            start_at=cutoff + timedelta(days=1), home_score=None, away_score=None, result_known_at=None)
            write_csv(root / "games.csv", GAME_COLUMNS, [serialize(g) for g in games + [upcoming]])
            first = run(root / "games.csv", None, root / "out", min_train=10,
                        test_from=cutoff.isoformat(), forecast_history=[])
            self.assertEqual(first["predictions"], [])
            self.assertEqual(len(first["upcoming"]), 3)
            elo = next(r for r in first["upcoming"] if r["model"] == "elo")
            from sportsedge.core import sigmoid
            self.assertAlmostEqual(elo["probability"], sigmoid(50 * 2.302585092994046 / 400))
            # Results arriving exactly at/after cutoff never affect a forecast.
            late = [dict(g, result_known_at=cutoff) for g in games]
            self.assertEqual(features(upcoming, late, []), features(upcoming, [], []))

    def test_offline_replay_verifies_inputs_and_reuses_original_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            demo(snapshot, count=30)
            games = [g for g in read_games(snapshot / "games.csv") if g["league"] == "NHL"]
            cutoff = games[-1]["start_at"] + timedelta(days=10)
            upcoming = dict(games[-1], game_id="future", prediction_at=cutoff,
                            start_at=cutoff + timedelta(days=1), home_score=None, away_score=None, result_known_at=None)
            write_csv(snapshot / "games.csv", GAME_COLUMNS, [serialize(g) for g in games + [upcoming]])
            write_csv(snapshot / "inference-history.csv", GAME_COLUMNS, [serialize(g) for g in games])
            manifest = {"kind": "historical_reconstructed", "generated_at": cutoff.isoformat(), "min_train": 10, "ml": False,
                        "games_sha256": hashlib.sha256((snapshot / "games.csv").read_bytes()).hexdigest(),
                        "inference_history_sha256": hashlib.sha256((snapshot / "inference-history.csv").read_bytes()).hexdigest()}
            (snapshot / "source.json").write_text(json.dumps(manifest))
            original = run(snapshot / "games.csv", None, snapshot / "models", min_train=10,
                           test_from=cutoff.isoformat(), forecast_history=games)
            repeated = replay(snapshot, root / "replay")
            self.assertEqual(original["upcoming"], repeated["upcoming"])
            self.assertEqual(repeated["data_kind"], "forecast_replay")
            (snapshot / "inference-history.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"):
                replay(snapshot, root / "another")


if __name__ == "__main__":
    unittest.main()
