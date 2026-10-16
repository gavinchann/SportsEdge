"""Observed team-level injury context, deliberately excluded from model inputs."""
import json
from datetime import timedelta

from .core import stamp
from .ingest import cached


def normalize(payload, league, observed):
    from .forecast import ALIASES, TEAMS
    cutoff = stamp(observed)
    if payload.get("status") != "success" or not isinstance(payload.get("injuries"), list):
        raise ValueError("Unexpected injury feed schema")
    rows, rejected = {}, 0
    for group in payload["injuries"]:
        for item in group.get("injuries", []):
            try:
                athlete = item["athlete"]
                code = athlete["team"]["abbreviation"]
                team = ALIASES[league].get(code, code)
                published = stamp(item["date"])
                if team not in TEAMS[league] or published > cutoff:
                    raise ValueError("Unknown team or future publication")
                key = str(item["id"])
                row = {"report_id": key, "team": team, "player": athlete["displayName"],
                       "status": item["status"], "published_at": published.isoformat(),
                       "observed_at": observed, "stale": cutoff - published > timedelta(hours=72)}
                if not row["player"] or not row["status"]:
                    raise ValueError("Missing name or status")
                if key not in rows or published > stamp(rows[key]["published_at"]):
                    rows[key] = row
            except (KeyError, TypeError, ValueError):
                rejected += 1
    return {"status": "available", "observed_at": observed, "reports": list(rows.values()),
            "rejected_rows": rejected}


def collect(directory):
    from .forecast import SPORTS
    feeds = {}
    for league, sport in SPORTS.items():
        try:
            body, meta = cached(f"https://site.api.espn.com/apis/site/v2/sports/{sport}/injuries?limit=1000", directory)
            feeds[league] = normalize(json.loads(body), league, meta["retrieved_at"])
            feeds[league]["source"] = meta
        except (OSError, ValueError, TypeError, KeyError) as error:
            feeds[league] = {"status": "unavailable", "reports": [], "error_type": type(error).__name__}
    return {"scope": "Team-level reports; not confirmed game availability. Context only, excluded from model inputs.",
            "leagues": feeds}
