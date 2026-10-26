import csv
import io
import json

from football_prediction.api import create_app


def _request(app, path, method="GET"):
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path.split("?", 1)[0],
        "QUERY_STRING": path.split("?", 1)[1] if "?" in path else "",
    }
    body = b"".join(app(environ, start_response))
    return captured["status"], json.loads(body)


def _write_ledger(path):
    fields = [
        "prediction_id", "match_date", "home_team", "away_team", "model",
        "competition",
        "model_version", "home_win_probability", "draw_probability",
        "away_win_probability", "home_expected_goals", "away_expected_goals",
        "evaluation_status", "actual_home_goals", "actual_away_goals",
        "actual_outcome",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {
                "prediction_id": "p1", "match_date": "2025-01-10",
                "home_team": "Arsenal", "away_team": "Chelsea",
                "competition": "Premier League",
                "model": "joint_poisson", "model_version": "0.1.0",
                "home_win_probability": "0.5", "draw_probability": "0.3",
                "away_win_probability": "0.2", "home_expected_goals": "1.4",
                "away_expected_goals": "0.9", "evaluation_status": "evaluated",
                "actual_home_goals": "2", "actual_away_goals": "0",
                "actual_outcome": "home",
            },
            {
                "prediction_id": "p2", "match_date": "2025-01-11",
                "home_team": "Liverpool", "away_team": "Everton",
                "competition": "Ligue 1",
                "model": "joint_poisson", "model_version": "0.1.0",
                "home_win_probability": "0.4", "draw_probability": "0.3",
                "away_win_probability": "0.3", "home_expected_goals": "1.2",
                "away_expected_goals": "1.0", "evaluation_status": "pending",
            },
        ])


def test_predictions_endpoint_returns_typed_forecasts_and_filters(tmp_path):
    ledger = tmp_path / "prediction_history.csv"
    _write_ledger(ledger)

    status, result = _request(create_app(ledger), "/api/predictions?status=pending")

    assert status == "200 OK"
    assert result["count"] == 1
    prediction = result["predictions"][0]
    assert prediction["prediction_id"] == "p2"
    assert prediction["home_win_probability"] == 0.4
    assert prediction["home_expected_goals"] == 1.2


def test_performance_endpoint_calculates_metrics_from_evaluated_rows(tmp_path):
    ledger = tmp_path / "prediction_history.csv"
    _write_ledger(ledger)

    status, result = _request(create_app(ledger), "/api/performance")

    assert status == "200 OK"
    assert result["evaluated_count"] == 1
    assert result["pending_count"] == 1
    assert result["outcome_accuracy"] == 1.0
    assert result["multiclass_brier_score"] == 0.38


def test_api_has_health_route_and_rejects_invalid_queries_and_methods(tmp_path):
    app = create_app(tmp_path / "missing.csv")

    root_status, root = _request(app, "/")
    assert root_status == "200 OK"
    assert root["endpoints"]["predictions"] == "/api/predictions"
    assert _request(app, "/health") == ("200 OK", {"status": "ok"})
    assert _request(app, "/api/predictions?status=unknown")[0] == "400 Bad Request"
    assert _request(app, "/api/predictions?match_date=tomorrow")[0] == "400 Bad Request"
    assert _request(app, "/api/predictions?competition=Unknown")[0] == "400 Bad Request"
    assert _request(app, "/api/predictions", method="POST")[0] == "405 Method Not Allowed"
    assert _request(app, "/elsewhere")[0] == "404 Not Found"


def test_predictions_and_performance_can_be_filtered_by_competition(tmp_path):
    ledger = tmp_path / "prediction_history.csv"
    _write_ledger(ledger)
    app = create_app(ledger)

    pending_status, pending = _request(
        app, "/api/predictions?status=pending&competition=Ligue%201"
    )
    performance_status, performance = _request(
        app, "/api/performance?competition=Premier%20League"
    )

    assert pending_status == "200 OK"
    assert pending["count"] == 1
    assert pending["predictions"][0]["competition"] == "Ligue 1"
    assert performance_status == "200 OK"
    assert performance["evaluated_count"] == 1
    assert performance["pending_count"] == 0
