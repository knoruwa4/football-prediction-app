"""Run the read-only local JSON API."""

from pathlib import Path
import os
import sys
from wsgiref.simple_server import make_server

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_prediction.api import create_app


def main() -> None:
    host = os.environ.get("FOOTBALL_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FOOTBALL_API_PORT", "8000"))
    app = create_app(PROJECT_ROOT / "reports" / "prediction_history.csv")
    with make_server(host, port, app) as server:
        print(f"Football prediction API listening on http://{host}:{port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
