import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .pipeline import demo, run


def serve(report_path, port, forecast_path=None, scorecard_path=None):
    report = Path(report_path).read_bytes()
    json.loads(report)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?")[0]
            if route == "/":
                body, kind = Path(__file__).with_name("dashboard.html").read_bytes(), "text/html; charset=utf-8"
            elif route in ("/assets/dashboard.css", "/assets/dashboard.js"):
                body = Path(__file__).with_name(route.rsplit("/", 1)[1]).read_bytes()
                kind = "text/css; charset=utf-8" if route.endswith(".css") else "text/javascript; charset=utf-8"
            elif route == "/api/report":
                body, kind = report, "application/json"
            elif route == "/api/forecast":
                if not forecast_path or not Path(forecast_path).exists():
                    self.send_error(404, "No forecast snapshot available")
                    return
                body, kind = Path(forecast_path).read_bytes(), "application/json"
            elif route == "/api/scorecard":
                if not scorecard_path or not Path(scorecard_path).exists():
                    self.send_error(404, "No prospective scorecard available")
                    return
                body, kind = Path(scorecard_path).read_bytes(), "application/json"
            elif route == "/api/health":
                body, kind = b'{"status":"ok"}', "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"SportsEdge: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="SportsEdge outcome research")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Generate explicitly synthetic CSV inputs")
    d.add_argument("--output", default="data/demo")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--count", type=int, default=120)
    r = sub.add_parser("run", help="Validate CSV inputs and walk-forward evaluate")
    r.add_argument("--games", required=True)
    r.add_argument("--availability")
    r.add_argument("--output", default="runs/latest")
    r.add_argument("--min-train", type=int, default=20)
    r.add_argument("--batch-size", type=int, default=20)
    r.add_argument("--data-kind", choices=["synthetic", "historical", "unverified"], default="unverified")
    r.add_argument("--ml", action="store_true", help="Include gradient boosting and H2H ablation (install ML extras)")
    r.add_argument("--test-from", help="Freeze training before an offset-aware timestamp; evaluate subsequent games")
    s = sub.add_parser("serve", help="Serve a report and dashboard on loopback only")
    s.add_argument("--report", default="runs/latest/report.json")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--forecast", help="Prospective report; reloads when a new snapshot is published")
    s.add_argument("--scorecard", help="Prospective outcome scorecard")
    f = sub.add_parser("forecast", help="Fetch current schedules/results and refit pregame models")
    f.add_argument("--history", default="data/historical/games.csv")
    f.add_argument("--output", default="runs/forecast")
    f.add_argument("--days", type=int, default=7)
    f.add_argument("--lookback-days", type=int, default=730)
    f.add_argument("--availability")
    f.add_argument("--min-train", type=int, default=100)
    f.add_argument("--ml", action="store_true")
    replay_parser = sub.add_parser("replay", help="Recompute a saved forecast snapshot offline")
    replay_parser.add_argument("--snapshot", required=True)
    replay_parser.add_argument("--output", required=True)
    score = sub.add_parser("score", help="Score saved forecasts against later observed final results")
    score.add_argument("--snapshots", default="runs/forecast/snapshots")
    score.add_argument("--results", required=True)
    score.add_argument("--output", default="runs/forecast/scorecard.json")
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.output, args.seed, args.count)
        print(f"Synthetic inputs written to {args.output}")
    elif args.command == "run":
        report = run(args.games, args.availability, args.output, args.min_train, args.batch_size, args.data_kind, args.ml, args.test_from)
        print(json.dumps({"data_kind": report["data_kind"], "summary": report["summary"]}, indent=2))
    elif args.command == "forecast":
        from .forecast import forecast
        report = forecast(args.history, args.output, args.days, args.lookback_days, args.ml, args.availability, args.min_train)
        print(json.dumps(report["forecast"], indent=2))
    elif args.command == "replay":
        from .forecast import replay
        report = replay(args.snapshot, args.output)
        print(json.dumps({"generated_at": report["generated_at"], "predictions": len(report["upcoming"])}))
    elif args.command == "score":
        from .scorecard import score_forecasts
        report = score_forecasts(args.snapshots, args.results, args.output)
        print(json.dumps({k: report[k] for k in ("scored_games", "pending_games", "excluded_games")}))
    else:
        serve(args.report, args.port, args.forecast, args.scorecard)


if __name__ == "__main__":
    main()
