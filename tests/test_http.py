import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from sportsedge.cli import serve


class HttpTests(unittest.TestCase):
    def test_assets_and_dynamic_forecast_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("report", "forecast", "scorecard"):
                (root / (name + ".json")).write_text(json.dumps({"version": 1}))
            ready = threading.Event()
            servers = []
            def factory(address, handler):
                server = ThreadingHTTPServer(address, handler)
                servers.append(server)
                ready.set()
                return server
            with patch("sportsedge.cli.ThreadingHTTPServer", factory), patch("builtins.print"), patch("sportsedge.cli.BaseHTTPRequestHandler.log_message"):
                worker = threading.Thread(target=serve, args=(root / "report.json", 0, root / "forecast.json", root / "scorecard.json"), daemon=True)
                worker.start()
                self.assertTrue(ready.wait(5))
                base = f"http://127.0.0.1:{servers[0].server_port}"
                try:
                    for route, content_type in (("/", "text/html"), ("/assets/dashboard.css", "text/css"), ("/assets/dashboard.js", "text/javascript"), ("/api/report", "application/json"), ("/api/forecast", "application/json"), ("/api/scorecard", "application/json")):
                        with urllib.request.urlopen(base + route, timeout=5) as response:
                            self.assertEqual(response.status, 200)
                            self.assertIn(content_type, response.headers["Content-Type"])
                            self.assertEqual(response.headers["Cache-Control"], "no-store")
                            self.assertTrue(response.read())
                    (root / "forecast.json").write_text('{"version":2}')
                    with urllib.request.urlopen(base + "/api/forecast", timeout=5) as response:
                        self.assertEqual(json.load(response)["version"], 2)
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(base + "/assets/../README.md", timeout=5)
                    self.assertEqual(error.exception.code, 404)
                finally:
                    servers[0].shutdown()
                    worker.join(5)
                    self.assertFalse(worker.is_alive())
