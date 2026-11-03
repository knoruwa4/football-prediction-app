"""Read-only JSON API for saved football forecasts and their performance."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, time, timezone
from math import exp
from pathlib import Path
from urllib.parse import parse_qs, unquote

from football_prediction.football_data import COMPETITIONS
from football_prediction.fixtures import historical_team_name
from football_prediction.league_data import (
    FIRST_SEASON_YEAR,
    LAST_SEASON_YEAR,
    MAIN_LEAGUE_CODES,
    EXTRA_LEAGUE_CODES,
    EXTERNAL_COMPETITION_CODES,
    extra_league_file,
    external_competition_file,
    load_external_competition_history,
    load_extra_league_history,
    load_league_history,
    season_file,
)
from football_prediction.batch_predictor import predict_fixtures
from football_prediction.prediction_store import append_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = PROJECT_ROOT / "reports" / "prediction_history.csv"
PROBABILITY_FIELDS = (
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
)
MAX_GOALS = 10
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_FIXTURE_ROOT = DEFAULT_DATA_ROOT / "fixtures"
MODEL_VERSION = "0.1.0"
MODEL_NAMES = {
    "joint_attack_defence": "joint_attack_defence_poisson",
    "dixon_coles": "dixon_coles_joint_poisson",
    "elo_poisson": "elo_adjusted_joint_poisson",
}


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


def _fixture_code(path: Path, row: dict[str, str]) -> str | None:
    code = (row.get("competition_code") or "").strip().upper()
    if code:
        return code if code in COMPETITIONS else None
    stem = path.stem
    prefix = stem.split("_", 1)[0].upper()
    if prefix == "EPL":
        return "E0"
    return prefix if prefix in COMPETITIONS else None


def _fixture_id(code: str, row: dict[str, str]) -> str:
    key = "|".join((code, row["match_date"], row["kickoff_utc"], row["home_team"].casefold(), row["away_team"].casefold()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _catalog(fixture_root: Path) -> list[dict[str, str]]:
    fixtures = []
    if not fixture_root.is_dir():
        return fixtures
    for path in sorted(fixture_root.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"match_date", "kickoff_utc", "home_team", "away_team", "source_url"}
            if not required.issubset(reader.fieldnames or []):
                continue
            seen = set()
            for row in reader:
                code = _fixture_code(path, row)
                try:
                    match_day = date.fromisoformat(row["match_date"].strip())
                    kickoff = time.fromisoformat(row["kickoff_utc"].strip())
                except (ValueError, AttributeError):
                    continue
                home, away = row["home_team"].strip(), row["away_team"].strip()
                source = row["source_url"].strip()
                if not code or not home or not away or kickoff.tzinfo or not source.startswith("https://"):
                    continue
                clean = {"match_date": match_day.isoformat(), "kickoff_utc": kickoff.strftime("%H:%M"), "home_team": home, "away_team": away, "source_url": source}
                key = (code, clean["match_date"], clean["kickoff_utc"], home.casefold(), away.casefold())
                if key in seen:
                    continue
                seen.add(key)
                clean["competition_code"] = code
                clean["competition"] = COMPETITIONS[code]
                clean["fixture_id"] = _fixture_id(code, clean)
                fixtures.append(clean)
    return fixtures


def _is_upcoming(fixture: dict[str, str], now: datetime) -> bool:
    kickoff = datetime.combine(date.fromisoformat(fixture["match_date"]), time.fromisoformat(fixture["kickoff_utc"]), tzinfo=timezone.utc)
    return kickoff > now.astimezone(timezone.utc)


def _prediction_is_upcoming(row: dict[str, str], now: datetime) -> bool:
    """Return whether a saved forecast has a valid kickoff still in the future."""
    try:
        kickoff = datetime.combine(
            date.fromisoformat(row["match_date"]),
            time.fromisoformat(row["kickoff_utc"]),
            tzinfo=timezone.utc,
        )
    except (KeyError, ValueError, TypeError):
        return False
    return kickoff > now.astimezone(timezone.utc)


def _competition_has_history(data_root: Path, code: str) -> bool:
    if code in MAIN_LEAGUE_CODES:
        return any(
            season_file(data_root, year, code).is_file()
            for year in range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1)
        )
    if code in EXTRA_LEAGUE_CODES:
        return extra_league_file(data_root, code).is_file()
    if code in EXTERNAL_COMPETITION_CODES:
        return external_competition_file(data_root, code).is_file()
    return False


def _scorelines(home_xg: float, away_xg: float):
    def probs(xg: float):
        result = [exp(-xg)]
        for goals in range(1, MAX_GOALS + 1):
            result.append(result[-1] * xg / goals)
        return result
    home_probs, away_probs = probs(home_xg), probs(away_xg)
    return sorted(((home_probs[h] * away_probs[a], h, a) for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1)), reverse=True)[:3]


def create_app(
    ledger_path: Path = DEFAULT_LEDGER,
    data_root: Path = DEFAULT_DATA_ROOT,
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
    now_provider=lambda: datetime.now(timezone.utc),
):
    """Create the JSON API for saved forecasts and sourced fixture predictions."""

    def app(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")
        if method not in {"GET", "POST"}:
            return _json_response(
                start_response,
                "405 Method Not Allowed",
                {"error": "Only GET requests are supported."},
            )

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
                        "fixtures": "/api/fixtures?competition=Premier%20League",
                        "competitions": "/api/competitions",
                        "predict": "/api/predict",
                        "prediction_models": ["joint_attack_defence", "dixon_coles", "elo_poisson"],
                    },
                },
            )
        if path == "/favicon.ico":
            start_response("204 No Content", [("Content-Length", "0")])
            return [b""]
        if path == "/health":
            return _json_response(start_response, "200 OK", {"status": "ok"})
        if path == "/api/competitions" and method == "GET":
            catalog = [
                {
                    "code": code,
                    "name": name,
                    "data_available": _competition_has_history(data_root, code),
                }
                for code, name in sorted(
                    COMPETITIONS.items(), key=lambda item: item[1].casefold()
                )
            ]
            return _json_response(
                start_response, "200 OK", {"count": len(catalog), "competitions": catalog}
            )
        if path == "/api/fixtures" and method == "GET":
            query = parse_qs(environ.get("QUERY_STRING", ""))
            competition = query.get("competition", [None])[0]
            code = next((key for key, value in COMPETITIONS.items() if value == competition), None)
            if competition and not code:
                return _json_response(start_response, "400 Bad Request", {"error": "competition must be one of the supported leagues."})
            try:
                fixtures = [fixture for fixture in _catalog(fixture_root) if (code is None or fixture["competition_code"] == code) and _is_upcoming(fixture, now_provider())]
            except (OSError, csv.Error, UnicodeError) as exc:
                return _json_response(start_response, "500 Internal Server Error", {"error": f"Could not read fixture catalog: {exc}"})
            return _json_response(start_response, "200 OK", {"count": len(fixtures), "fixtures": fixtures})

        if path == "/api/predict" and method == "POST":
            try:
                length = int(environ.get("CONTENT_LENGTH") or 0)
                body = environ["wsgi.input"].read(length)
                request_data = json.loads(body or b"{}")
                fixture_id = request_data.get("fixture_id") if isinstance(request_data, dict) else None
                if not isinstance(fixture_id, str):
                    raise ValueError("fixture_id is required.")
                model_type = request_data.get("model_type", "joint_attack_defence")
                if model_type not in {"joint_attack_defence", "dixon_coles", "elo_poisson"}:
                    raise ValueError("model_type must be joint_attack_defence, dixon_coles, or elo_poisson.")
                fixture = next((item for item in _catalog(fixture_root) if item["fixture_id"] == fixture_id), None)
                if fixture is None:
                    return _json_response(start_response, "404 Not Found", {"error": "Verified fixture not found."})
                now = now_provider()
                if not _is_upcoming(fixture, now):
                    return _json_response(start_response, "409 Conflict", {"error": "This fixture has already kicked off."})
                rows = _read_ledger(ledger_path)
                existing = next((row for row in rows if row.get("competition") == fixture["competition"] and row.get("match_date") == fixture["match_date"] and row.get("home_team", "").casefold() == fixture["home_team"].casefold() and row.get("away_team", "").casefold() == fixture["away_team"].casefold() and row.get("model") == MODEL_NAMES[model_type]), None)
                if existing:
                    return _json_response(start_response, "200 OK", {"prediction": _prediction_payload(existing), "already_saved": True})
                code = fixture["competition_code"]
                years = range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1)
                if code in MAIN_LEAGUE_CODES:
                    matches = load_league_history(data_root, code, years)
                elif code in EXTRA_LEAGUE_CODES:
                    matches = load_extra_league_history(data_root, code, years)
                else:
                    matches = load_external_competition_history(data_root, code)
                match_date = date.fromisoformat(fixture["match_date"])
                model_home = historical_team_name(fixture["home_team"])
                model_away = historical_team_name(fixture["away_team"])
                result = predict_fixtures(matches, [(model_home, model_away)], match_date, model_type=model_type, allow_unknown_teams=True, competition=fixture["competition"])[0][2]
                scorelines = _scorelines(result.home_xg, result.away_xg)
                known_teams = {team for match in matches for team in (match.home_team, match.away_team)}
                report = {"match_date": fixture["match_date"], "competition": fixture["competition"], "kickoff_utc": fixture["kickoff_utc"], "home_team": fixture["home_team"], "away_team": fixture["away_team"], "home_team_history_available": model_home in known_teams, "away_team_history_available": model_away in known_teams, "fixture_source_url": fixture["source_url"], "model": result.model, "model_version": MODEL_VERSION, "home_win_probability": round(result.home_win, 6), "draw_probability": round(result.draw, 6), "away_win_probability": round(result.away_win, 6), "home_expected_goals": round(result.home_xg, 4), "away_expected_goals": round(result.away_xg, 4)}
                for rank, (probability, hg, ag) in enumerate(scorelines, 1):
                    report[f"score_{rank}"] = f"{hg}-{ag}"
                    report[f"score_{rank}_probability"] = round(probability, 6)
                append_predictions(ledger_path, [report])
                saved = next(row for row in _read_ledger(ledger_path) if row.get("match_date") == fixture["match_date"] and row.get("competition") == fixture["competition"] and row.get("home_team") == fixture["home_team"] and row.get("away_team") == fixture["away_team"] and row.get("model") == MODEL_NAMES[model_type])
                return _json_response(start_response, "201 Created", {"prediction": _prediction_payload(saved), "already_saved": False})
            except (ValueError, TypeError, KeyError, OSError, csv.Error, UnicodeError) as exc:
                return _json_response(start_response, "400 Bad Request", {"error": str(exc)})

        if path in {"/api/predictions", "/api/performance", "/api/fixtures", "/api/competitions", "/api/predict"} and method != "GET":
            return _json_response(start_response, "405 Method Not Allowed", {"error": "Method not allowed for this endpoint."})
        if method != "GET" or path not in {"/api/predictions", "/api/performance"}:
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

        query = parse_qs(environ.get("QUERY_STRING", ""))
        status = query.get("status", [None])[0]
        match_date = query.get("match_date", [None])[0]
        competition = query.get("competition", [None])[0]
        upcoming = query.get("upcoming", ["false"])[0]
        if upcoming not in {"true", "false"}:
            return _json_response(
                start_response,
                "400 Bad Request",
                {"error": "upcoming must be 'true' or 'false'."},
            )
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

        if competition and competition not in COMPETITIONS.values():
            return _json_response(
                start_response,
                "400 Bad Request",
                {"error": "competition must be one of the supported leagues."},
            )

        selected_rows = [
            row
            for row in rows
            if competition is None
            or row.get("competition", "") == competition
            or (
                competition == "Premier League"
                and not row.get("competition", "")
            )
        ]

        if path == "/api/performance":
            return _json_response(
                start_response, "200 OK", _performance(selected_rows)
            )

        filtered = [
            row
            for row in selected_rows
            if (status is None or row.get("evaluation_status") == status)
            and (match_date is None or row.get("match_date") == match_date)
            and (
                upcoming != "true"
                or _prediction_is_upcoming(row, now_provider())
            )
        ]
        return _json_response(
            start_response,
            "200 OK",
            {"count": len(filtered), "predictions": [_prediction_payload(r) for r in filtered]},
        )

    return app


application = create_app()
