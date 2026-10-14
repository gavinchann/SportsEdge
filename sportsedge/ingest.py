"""Cached public-result adapters. Historical availability times are assumptions, not observations."""
import csv
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .pipeline import GAME_COLUMNS, write_csv

CLUBS = {
    "NBA": set("ATL BKN BOS CHA CHI CLE DAL DEN DET GS HOU IND LAC LAL MEM MIA MIL MIN NO NY OKC ORL PHI PHX POR SA SAC TOR UTAH WSH".split()),
    "NHL": set("ANA ARI BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LA MIN MTL NJ NSH NYI NYR OTT PHI PIT SEA SJ STL TB TOR UTAH VAN VGK WPG WSH".split()),
}


def cached(url, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    body_path, meta_path = directory / (key + ".bin"), directory / (key + ".json")
    if body_path.exists() and meta_path.exists():
        body, meta = body_path.read_bytes(), json.loads(meta_path.read_text())
        if hashlib.sha256(body).hexdigest() != meta["sha256"] or meta["url"] != url:
            raise ValueError("Raw cache integrity check failed")
        return body, meta
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SportsEdge-research/0.2"})
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read()
            break
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    meta = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": hashlib.sha256(body).hexdigest()}
    body_path.write_bytes(body)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return body, meta


def row(gid, league, home, away, day, hs, aws, lag_days, observed, neutral=0, completed_day=None):
    # Date-only sources use a day-boundary representation, not an invented tipoff time.
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(completed_day or day).replace(tzinfo=timezone.utc)
    known = observed if lag_days is None else (end + timedelta(days=lag_days)).isoformat()
    return dict(zip(GAME_COLUMNS, [gid, league, home, away, start.isoformat(), (start - timedelta(hours=1)).isoformat(), known, int(hs), int(aws), neutral]))


def parse_scoreboard(payload, lag_days, observed, league="NBA"):
    rows = []
    for e in payload.get("events", []):
        if e.get("season", {}).get("type") not in (2, 3):
            continue  # Exclude preseason and exhibitions.
        c = e["competitions"][0]
        if not c.get("status", e.get("status", {})).get("type", {}).get("completed"):
            continue
        sides = {p["homeAway"]: p for p in c["competitors"]}
        h, a = sides["home"], sides["away"]
        if h["team"]["abbreviation"] not in CLUBS[league] or a["team"]["abbreviation"] not in CLUBS[league]:
            continue  # All-Star events can be tagged as regular-season by ESPN.
        # Deliberately ignore endpoint records, leaders, odds and statistics: they are postgame.
        rows.append(row(league + "-" + e["id"], league, h["team"]["abbreviation"], a["team"]["abbreviation"], e["date"][:10], h["score"], a["score"], lag_days, observed, int(c.get("neutralSite", False))))
    return rows


def parse_nfl(text, years, lag_days, observed):
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if int(r["season"]) not in years or r["game_type"] == "PRE" or not r["home_score"] or not r["away_score"]:
            continue
        rows.append(row("NFL-" + r["game_id"], "NFL", r["home_team"], r["away_team"], r["gameday"], float(r["home_score"]), float(r["away_score"]), lag_days, observed, int(r["location"] == "Neutral")))
    return rows


def parse_mlb(text, lag_days, observed):
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if r[14] or r[15]:
            continue  # Exclude forfeits/protests rather than infer adjudication dates.
        day = datetime.strptime(r[0], "%Y%m%d").date().isoformat()
        completion = datetime.strptime(r[13].split(",")[0], "%Y%m%d").date().isoformat() if r[13] else day
        rows.append(row(f"MLB-{r[0]}-{r[6]}-{r[3]}-{r[1]}", "MLB", r[6], r[3], day, r[10], r[9], lag_days, observed, completed_day=completion))
    return rows


def ingest(output, years, lag_days=None, leagues=("NBA", "NFL", "MLB", "NHL")):
    if lag_days is not None and lag_days < 2:
        raise ValueError("Historical reconstruction lag must be at least two days")
    output = Path(output)
    raw = output / "raw"
    games, sources = [], []
    if "NFL" in leagues:
        body, meta = cached("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", raw)
        games.extend(parse_nfl(body.decode(), years, lag_days, meta["retrieved_at"]))
        sources.append(meta)
    if "MLB" in leagues:
        for year in years:
            body, meta = cached(f"https://www.retrosheet.org/gamelogs/gl{year}.zip", raw)
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                text = archive.read(f"gl{year}.txt").decode("utf-8-sig")
            games.extend(parse_mlb(text, lag_days, meta["retrieved_at"]))
            sources.append(meta)
    for league in (l for l in ("NBA", "NHL") if l in leagues):
        days = []
        for year in years:
            day = datetime(year, 1, 1)
            while day.year == year:
                days.append(day.strftime("%Y%m%d"))
                day += timedelta(days=1)
        def fetch(day):
            sport = "basketball/nba" if league == "NBA" else "hockey/nhl"
            body, meta = cached(f"https://site.api.espn.com/apis/site/v2/sports/{sport}/scoreboard?dates={day}", raw)
            return parse_scoreboard(json.loads(body), lag_days, meta["retrieved_at"], league), meta
        with ThreadPoolExecutor(max_workers=3) as pool:
            for index, (rows, meta) in enumerate(pool.map(fetch, days)):
                games.extend(rows)
                sources.append(meta)
                if index % 60 == 0:
                    print(f"{league}: cached {index + 1}/{len(days)} daily scoreboards", flush=True)
    unique = {}
    for g in games:
        # ESPN date queries may overlap; conflicting normalized records fail closed.
        previous = unique.get(g["game_id"])
        if previous and previous != g:
            if lag_days is None:
                g["result_known_at"] = min(previous["result_known_at"], g["result_known_at"])
                previous = dict(previous, result_known_at=g["result_known_at"])
            if previous != g:
                raise ValueError(f"Conflicting duplicate: {g['game_id']}")
        unique[g["game_id"]] = g
    games = sorted(unique.values(), key=lambda g: (g["prediction_at"], g["game_id"]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "games.csv", GAME_COLUMNS, games)
    manifest = {"kind": "historical_reconstructed" if lag_days is not None else "historical_observed", "years": years, "leagues": list(leagues), "result_lag_days": lag_days, "timing_policy": "Date-boundary predictions; final results delayed from completion date by specified lag. This is an assumption, not historical observation evidence." if lag_days is not None else "Results become known only at actual archive retrieval time; cannot support retrospective training.", "availability": "Not supplied; unknown, never inferred from final rosters or participation.", "sources": sources, "counts": {l: sum(g["league"] == l for g in games) for l in leagues}, "games_sha256": hashlib.sha256((output / "games.csv").read_bytes()).hexdigest()}
    (output / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"kind": manifest["kind"], "counts": manifest["counts"]}), flush=True)
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/historical")
    p.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    p.add_argument("--historical-lag-days", type=int)
    p.add_argument("--leagues", nargs="+", choices=["NBA", "NFL", "MLB", "NHL"], default=["NBA", "NFL", "MLB", "NHL"])
    args = p.parse_args()
    ingest(args.output, args.years, args.historical_lag_days, args.leagues)
