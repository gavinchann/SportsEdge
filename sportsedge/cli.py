import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .pipeline import demo, run


def serve(report_path, port):
    report = Path(report_path).read_bytes()
    json.loads(report)
    html = Path(__file__).with_name("dashboard.html").read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?")[0]
            if route == "/":
                body, kind = html, "text/html; charset=utf-8"
            elif route == "/api/report":
                body, kind = report, "application/json"
            elif route == "/api/health":
                body, kind = b'{"status":"ok"}', "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
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
    s = sub.add_parser("serve", help="Serve a report and dashboard on loopback only")
    s.add_argument("--report", default="runs/latest/report.json")
    s.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.output, args.seed, args.count)
        print(f"Synthetic inputs written to {args.output}")
    elif args.command == "run":
        report = run(args.games, args.availability, args.output, args.min_train, args.batch_size, args.data_kind)
        print(json.dumps({"data_kind": report["data_kind"], "summary": report["summary"]}, indent=2))
    else:
        serve(args.report, args.port)


if __name__ == "__main__":
    main()
