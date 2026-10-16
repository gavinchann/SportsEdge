import unittest
from unittest.mock import patch
from sportsedge.injuries import normalize, collect


class InjuryTests(unittest.TestCase):
    def test_timestamps_aliases_and_stale_context(self):
        def row(id, date, team="WSH"):
            return {"id": id, "date": date, "status": "Out", "athlete": {
                "displayName": "Example Player", "team": {"abbreviation": team}}}
        payload = {"status": "success", "injuries": [{"injuries": [
            row("1", "2026-10-10T00:00Z"), row("2", "2026-10-17T00:00Z"),
            row("3", "2026-10-16T00:00Z", "UNKNOWN"), row("4", "2026-10-16T00:00Z")]}]}
        report = normalize(payload, "NFL", "2026-10-16T12:00Z")
        self.assertEqual(report["rejected_rows"], 2)
        self.assertEqual(len(report["reports"]), 2)
        self.assertEqual(report["reports"][0]["team"], "WAS")
        self.assertTrue(report["reports"][0]["stale"])
        self.assertFalse(report["reports"][1]["stale"])
        self.assertEqual(report["reports"][0]["observed_at"], "2026-10-16T12:00Z")

    def test_failure_is_not_healthy_coverage(self):
        with patch("sportsedge.injuries.cached", side_effect=OSError("offline")):
            result = collect("unused")
        self.assertEqual(len(result["leagues"]), 4)
        self.assertTrue(all(r["status"] == "unavailable" for r in result["leagues"].values()))
        with self.assertRaises(ValueError):
            normalize({"status": "success"}, "NFL", "2026-10-16T12:00Z")
