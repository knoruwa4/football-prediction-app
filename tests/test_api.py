import csv
import io
import json
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace

from football_prediction.api import create_app


def _request(app, path, method="GET", payload=None):
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path.split("?", 1)[0],
        "QUERY_STRING": path.split("?", 1)[1] if "?" in path else "",
        "CONTENT_LENGTH": str(len(json.dumps(payload).encode())) if payload is not None else "0",
        "wsgi.input": BytesIO(json.dumps(payload).encode() if payload is not None else b""),
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


def test_fixture_catalog_only_offers_verified_future_fixtures(tmp_path):
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    (fixture_root / "E0_2026-10-26.csv").write_text(
        "match_date,kickoff_utc,home_team,away_team,source_url\n"
        "2026-10-26,11:59,Past FC,Old FC,https://example.com/fixtures\n"
        "2026-10-26,12:01,Home FC,Away FC,https://example.com/fixtures\n"
        "2026-10-26,12:02,No Source FC,Other FC,http://example.com/fixtures\n",
        encoding="utf-8",
    )
    app = create_app(
        tmp_path / "ledger.csv",
        fixture_root=fixture_root,
        now_provider=lambda: datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc),
    )

    status, result = _request(app, "/api/fixtures?competition=Premier%20League")

    assert status == "200 OK"
    assert result["count"] == 1
    assert result["fixtures"][0]["home_team"] == "Home FC"
    assert result["fixtures"][0]["fixture_id"]


def test_site_prediction_uses_history_and_persists_once(tmp_path, monkeypatch):
    from football_prediction import api

    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    (fixture_root / "E0_2026-10-27.csv").write_text(
        "match_date,kickoff_utc,home_team,away_team,source_url\n"
        "2026-10-27,15:00,Home FC,Away FC,https://example.com/fixture\n",
        encoding="utf-8",
    )
    historical = [SimpleNamespace(home_team="Home FC", away_team="Away FC")]
    monkeypatch.setattr(api, "load_league_history", lambda *args: historical)

    def fake_predict(matches, fixtures, match_date, **kwargs):
        assert matches is historical
        assert match_date.isoformat() == "2026-10-27"
        assert kwargs["competition"] == "Premier League"
        assert kwargs["model_type"] == "dixon_coles"
        forecast = SimpleNamespace(
            home_win=0.5, draw=0.25, away_win=0.25, home_xg=1.4, away_xg=0.9,
            model="dixon_coles_joint_poisson",
        )
        return [(fixtures[0][0], fixtures[0][1], forecast)]

    monkeypatch.setattr(api, "predict_fixtures", fake_predict)
    now = lambda: datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    app = create_app(tmp_path / "ledger.csv", fixture_root=fixture_root, now_provider=now)
    _, fixtures = _request(app, "/api/fixtures?competition=Premier%20League")
    fixture_id = fixtures["fixtures"][0]["fixture_id"]

    status, response = _request(app, "/api/predict", method="POST", payload={"fixture_id": fixture_id, "model_type": "dixon_coles"})
    repeat_status, repeat = _request(app, "/api/predict", method="POST", payload={"fixture_id": fixture_id, "model_type": "dixon_coles"})

    assert status == "201 Created"
    assert repeat_status == "200 OK"
    assert repeat["already_saved"] is True
    assert response["prediction"]["home_win_probability"] == 0.5
    assert _request(app, "/api/predictions?status=pending")[1]["count"] == 1


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


def test_competition_catalog_reports_all_data_sources_and_availability(tmp_path):
    from football_prediction.football_data import COMPETITIONS

    external = tmp_path / "external_top_tiers"
    external.mkdir()
    (external / "BOL_2017-2023.csv").write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Season\n"
        "BOL,03/05/2023,Home FC,Away FC,2,1,H,2023\n",
        encoding="utf-8",
    )
    app = create_app(
        tmp_path / "ledger.csv",
        data_root=tmp_path,
        fixture_root=tmp_path / "fixtures",
    )

    status, result = _request(app, "/api/competitions")

    assert status == "200 OK"
    assert result["count"] == len(COMPETITIONS) == 60
    by_code = {item["code"]: item for item in result["competitions"]}
    assert by_code["BOL"]["name"] == "Bolivian División Profesional"
    assert by_code["BOL"]["data_available"] is True
    assert by_code["UCL"]["data_available"] is False


def test_site_prediction_loads_supplemental_competition_history(tmp_path, monkeypatch):
    from football_prediction import api

    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    (fixture_root / "BOL_2026-11-03.csv").write_text(
        "match_date,kickoff_utc,home_team,away_team,source_url\n"
        "2026-11-03,15:00,Home FC,Away FC,https://example.com/fixture\n",
        encoding="utf-8",
    )
    historical = [SimpleNamespace(home_team="Home FC", away_team="Away FC")]
    loaded = []
    def fake_load(data_root, code):
        loaded.append(code)
        return historical
    monkeypatch.setattr(api, "load_external_competition_history", fake_load)

    def fake_predict(matches, fixtures, match_date, **kwargs):
        assert matches is historical
        assert kwargs["competition"] == "Bolivian División Profesional"
        return [(fixtures[0][0], fixtures[0][1], SimpleNamespace(
            home_win=0.5, draw=0.25, away_win=0.25, home_xg=1.4, away_xg=0.9,
            model="joint_attack_defence_poisson",
        ))]
    monkeypatch.setattr(api, "predict_fixtures", fake_predict)
    app = create_app(
        tmp_path / "ledger.csv",
        data_root=tmp_path,
        fixture_root=fixture_root,
        now_provider=lambda: datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc),
    )
    _, catalog = _request(app, "/api/fixtures?competition=Bolivian%20Divisi%C3%B3n%20Profesional")
    fixture_id = catalog["fixtures"][0]["fixture_id"]

    status, result = _request(
        app, "/api/predict", method="POST",
        payload={"fixture_id": fixture_id, "model_type": "joint_attack_defence"},
    )

    assert status == "201 Created"
    assert loaded == ["BOL"]
    assert result["prediction"]["competition"] == "Bolivian División Profesional"
