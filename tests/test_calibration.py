import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from sportsedge.calibration import CalibratedLogistic
from sportsedge.core import Logistic, features, outcome, read_games, stamp
from sportsedge.pipeline import demo


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "ML extra not installed")
class CalibrationTests(unittest.TestCase):
    def test_later_calibration_labels_never_enter_base_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo(root, count=200)
            train = [g for g in read_games(root / "games.csv") if g["league"] == "NHL"]
            boundary = train[160]["prediction_at"]
            train[159]["result_known_at"] = boundary
            vectors = {g["game_id"]: features(g, train, []) for g in train}
            fallback = Logistic().fit([(vectors[g["game_id"]], outcome(g)) for g in train])
            first = CalibratedLogistic().fit(train, vectors, fallback)
            self.assertTrue(first.metadata["fitted"])
            self.assertLess(stamp(first.metadata["latest_base_result"]), boundary)
            changed = copy.deepcopy(train)
            for g in changed[160:]:
                g["home_score"], g["away_score"] = g["away_score"], g["home_score"]
            second = CalibratedLogistic().fit(changed, vectors, fallback)
            self.assertEqual(first.base.weights, second.base.weights)
            self.assertNotEqual(first.slope, second.slope)
            for x in vectors.values():
                self.assertTrue(0 < first.predict(x) < 1)

    def test_small_sample_explicitly_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            demo(directory, count=30)
            train = [g for g in read_games(Path(directory) / "games.csv") if g["league"] == "NHL"]
            vectors = {g["game_id"]: features(g, train, []) for g in train}
            fallback = Logistic().fit([(vectors[g["game_id"]], outcome(g)) for g in train])
            calibrated = CalibratedLogistic().fit(train, vectors, fallback)
            self.assertFalse(calibrated.metadata["fitted"])
            for x in vectors.values():
                self.assertAlmostEqual(calibrated.predict(x), fallback.predict(x))
