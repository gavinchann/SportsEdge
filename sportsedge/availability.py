"""Append normalized injury reports with the actual local observation time."""
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from .core import read_availability, read_games
from .pipeline import AVAILABILITY_COLUMNS, write_csv


def utcnow():
    return datetime.now(timezone.utc)


def capture(input_path, games_path, output_path):
    """Input JSON list needs every availability field except observed_at.

    Single-writer command. Call only on normalized reports obtained from a permitted source.
    Publication time is retained; observed_at supplied by the input is intentionally ignored.
    """
    payload = Path(input_path).read_bytes()
    incoming = json.loads(payload)
    if not isinstance(incoming, list) or not incoming:
        raise ValueError("Expected a nonempty JSON list of normalized reports")
    observed = utcnow().isoformat()
    rows = [{k: observed if k == "observed_at" else r[k] for k in AVAILABILITY_COLUMNS} for r in incoming]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output.exists():
        with output.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    fd, name = tempfile.mkstemp(dir=output.parent, suffix=".csv")
    os.close(fd)
    temporary = Path(name)
    try:
        write_csv(temporary, AVAILABILITY_COLUMNS, existing + rows)
        read_availability(temporary, read_games(games_path))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    archive = output.parent / "availability_raw"
    archive.mkdir(exist_ok=True)
    key = hashlib.sha256(payload + observed.encode()).hexdigest()
    (archive / f"{key}.json").write_text(json.dumps({"observed_at": observed, "input_sha256": hashlib.sha256(payload).hexdigest(), "input": incoming}, indent=2), encoding="utf-8")
    return {"appended": len(rows), "observed_at": observed}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--games", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(capture(a.input, a.games, a.output)))
