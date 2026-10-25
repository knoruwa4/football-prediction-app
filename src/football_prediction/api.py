"""Read-only JSON API for saved football forecasts and their performance."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = PROJECT_ROOT / "reports" / "prediction_history.csv"
PROBABILITY_FIELDS = (
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
)


def _read_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _json_response(start_response, status: str, payload: object):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
        ],
    )
    return [body]


def _prediction_payload(row: dict[str, str]) -> dict[str, object]:
    payload: dict[str, object] = dict(row)
    for field in PROBABILITY_FIELDS + (
        "home_expected_goals",
        "away_expected_goals",
        "score_1_probability",
        "score_2_probability",
        "score_3_probability",
    ):
        value = payload.get(field)
        if value not in (None, ""):
            payload[field] = float(value)
    for field in ("actual_home_goals", "actual_away_goals"):
        value = payload.get(field)
        if value not in (None, ""):
            payload[field] = int(value)
    for field in ("home_team_history_available", "away_team_history_available"):
        value = payload.get(field)
        if value not in (None, ""):
            payload[field] = value.casefold() == "true"
    return payload


def _performance(rows: list[dict[str, str]]) -> dict[str, object]:
    evaluated = [row for row in rows if row.get("evaluation_status") == "evaluated"]
    pending = sum(row.get("evaluation_status") != "evaluated" for row in rows)
    if not evaluated:
        return {
            "evaluated_count": 0,
            "pending_count": pending,
            "outcome_accuracy": None,
            "multiclass_brier_score": None,
        }

    correct = 0
    brier_total = 0.0
    labels = ("home", "draw", "away")
    for row in evaluated:
        probabilities = {
            "home": float(row["home_win_probability"]),
            "draw": float(row["draw_probability"]),
            "away": float(row["away_win_probability"]),
        }
        actual = row["actual_outcome"]
        correct += max(probabilities, key=probabilities.get) == actual
        brier_total += sum(
            (probabilities[label] - int(label == actual)) ** 2
            for label in labels
        )
    count = len(evaluated)
    return {
        "evaluated_count": count,
        "pending_count": pending,
        "outcome_accuracy": correct / count,
        "multiclass_brier_score": brier_total / count,
    }


def create_app(ledger_path: Path = DEFAULT_LEDGER):
    """Create a WSGI app backed by a prediction ledger path."""

    def app(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        if method != "GET":
            return _json_response(
                start_response,
                "405 Method Not Allowed",
                {"error": "Only GET requests are supported."},
            )

        path = environ.get("PATH_INFO", "/")
        if path == "/":
            return _json_response(
                start_response,
                "200 OK",
                {
                    "service": "Football Prediction API",
                    "endpoints": {
                        "health": "/health",
                        "predictions": "/api/predictions",
                        "pending_predictions": "/api/predictions?status=pending",
                        "performance": "/api/performance",
                    },
                },
            )
        if path == "/favicon.ico":
            start_response("204 No Content", [("Content-Length", "0")])
            return [b""]
        if path == "/health":
            return _json_response(start_response, "200 OK", {"status": "ok"})
        if path not in {"/api/predictions", "/api/performance"}:
            return _json_response(
                start_response, "404 Not Found", {"error": "Not found."}
            )

        try:
            rows = _read_ledger(ledger_path)
        except (OSError, csv.Error, UnicodeError) as exc:
            return _json_response(
                start_response,
                "500 Internal Server Error",
                {"error": f"Could not read prediction ledger: {exc}"},
            )

        if path == "/api/performance":
            return _json_response(start_response, "200 OK", _performance(rows))

        query = parse_qs(environ.get("QUERY_STRING", ""))
        status = query.get("status", [None])[0]
        match_date = query.get("match_date", [None])[0]
        if status not in (None, "pending", "evaluated"):
            return _json_response(
                start_response,
                "400 Bad Request",
                {"error": "status must be 'pending' or 'evaluated'."},
            )
        if match_date:
            try:
                date.fromisoformat(match_date)
            except ValueError:
                return _json_response(
                    start_response,
                    "400 Bad Request",
                    {"error": "match_date must use YYYY-MM-DD format."},
                )

        filtered = [
            row
            for row in rows
            if (status is None or row.get("evaluation_status") == status)
            and (match_date is None or row.get("match_date") == match_date)
        ]
        return _json_response(
            start_response,
            "200 OK",
            {"count": len(filtered), "predictions": [_prediction_payload(r) for r in filtered]},
        )

    return app


application = create_app()
