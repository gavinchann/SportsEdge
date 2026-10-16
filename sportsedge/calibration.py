"""Probability calibration on a later, strictly pre-test chronological slice."""
import math
from .core import FEATURES, Logistic, outcome, sigmoid


def logit(p):
    p = max(1e-8, min(1 - 1e-8, p))
    return math.log(p / (1 - p))


class CalibratedLogistic:
    def fit(self, train, vectors, fallback):
        ordered = sorted(train, key=lambda g: (g["prediction_at"], g["game_id"]))
        boundary = ordered[min(len(ordered) - 1, int(.8 * len(ordered)))]["prediction_at"]
        base = [g for g in ordered if g["prediction_at"] < boundary and g["result_known_at"] < boundary]
        calibration = [g for g in ordered if g["prediction_at"] >= boundary]
        self.base, self.slope, self.intercept = fallback, 1.0, 0.0
        self.metadata = {"method": "chronological_platt", "split_fraction": .8,
                         "calibration_from": boundary.isoformat(), "base_training_n": len(base),
                         "calibration_n": len(calibration), "fitted": False}
        if len(base) < 100 or len(calibration) < 30 or len({outcome(g) for g in base}) < 2 or len({outcome(g) for g in calibration}) < 2:
            self.metadata["fallback"] = "Plain logistic: insufficient chronological calibration examples/classes"
            return self
        from sklearn.linear_model import LogisticRegression
        self.base = Logistic().fit([(vectors[g["game_id"]], outcome(g)) for g in base])
        inputs = [[logit(self.base.predict(vectors[g["game_id"]]))] for g in calibration]
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        calibrator.fit(inputs, [outcome(g) for g in calibration])
        self.slope = float(calibrator.coef_[0][0])
        self.intercept = float(calibrator.intercept_[0])
        self.metadata.update(fitted=True, latest_base_result=max(g["result_known_at"] for g in base).isoformat(),
                             latest_calibration_result=max(g["result_known_at"] for g in calibration).isoformat())
        return self

    def predict(self, x):
        return sigmoid(self.intercept + self.slope * logit(self.base.predict(x)))

    def artifact(self):
        return {"features": FEATURES, "base_weights": self.base.weights, "slope": self.slope,
                "intercept": self.intercept, "calibration": self.metadata}
