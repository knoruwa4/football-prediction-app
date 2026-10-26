import csv
from datetime import date

from football_prediction.data import MatchResult
from football_prediction.prediction_store import (
    append_predictions,
    evaluate_prediction_file,
)


def test_append_predictions_preserves_each_forecast_with_identity(tmp_path):
    path = tmp_path / "prediction_history.csv"
    forecast = {
        "match_date": "2025-01-10",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "model": "joint_poisson",
        "model_version": "0.1.0",
        "home_win_probability": 0.5,
        "draw_probability": 0.25,
        "away_win_probability": 0.25,
    }

    append_predictions(path, [forecast])
    append_predictions(path, [forecast])

    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["prediction_id"] != rows[1]["prediction_id"]
    assert rows[0]["prediction_timestamp"]
    assert rows[0]["evaluation_status"] == "pending"
    assert rows[0]["home_win_probability"] == "0.5"


def test_evaluation_adds_actual_result_without_changing_forecast(tmp_path):
    path = tmp_path / "prediction_history.csv"
    append_predictions(path, [{
        "match_date": "2025-01-10",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "model": "joint_poisson",
        "home_win_probability": 0.5,
        "draw_probability": 0.25,
        "away_win_probability": 0.25,
    }])
    actual = MatchResult(
        home_team="Arsenal",
        away_team="Chelsea",
        home_goals=2,
        away_goals=1,
        match_date=date(2025, 1, 10),
        competition="EPL",
    )

    rows = evaluate_prediction_file(path, [actual])

    assert rows[0]["evaluation_status"] == "evaluated"
    assert rows[0]["actual_home_goals"] == "2"
    assert rows[0]["actual_away_goals"] == "1"
    assert rows[0]["actual_outcome"] == "home"
    assert rows[0]["home_win_probability"] == "0.5"


def test_evaluation_leaves_unmatched_prediction_pending(tmp_path):
    path = tmp_path / "prediction_history.csv"
    append_predictions(path, [{
        "match_date": "2025-01-10",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "model": "joint_poisson",
    }])

    rows = evaluate_prediction_file(path, [])

    assert rows[0]["evaluation_status"] == "pending"
    assert rows[0]["actual_outcome"] == ""


def test_append_migrates_legacy_ledger_header_before_adding_competition(tmp_path):
    path = tmp_path / "prediction_history.csv"
    path.write_text(
        "match_date,home_team,away_team,model,evaluation_status\n"
        "2025-01-10,Arsenal,Chelsea,old_model,pending\n",
        encoding="utf-8",
    )

    append_predictions(path, [{
        "match_date": "2025-01-11",
        "competition": "La Liga",
        "home_team": "Sevilla",
        "away_team": "Valencia",
        "model": "joint_attack_defence_poisson",
    }])

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames
    assert "competition" in fields
    assert rows[0]["home_team"] == "Arsenal"
    assert rows[0]["competition"] == ""
    assert rows[1]["competition"] == "La Liga"
    assert len(rows[1]) == len(fields)


def test_evaluation_matches_same_fixture_name_with_competition_key(tmp_path):
    path = tmp_path / "prediction_history.csv"
    append_predictions(path, [{
        "match_date": "2025-01-10",
        "competition": "Ligue 1",
        "home_team": "United",
        "away_team": "City",
        "model": "joint_poisson",
    }])
    actual = [
        MatchResult(date(2025, 1, 10), "Premier League", "United", "City", 4, 0),
        MatchResult(date(2025, 1, 10), "Ligue 1", "United", "City", 0, 2),
    ]

    rows = evaluate_prediction_file(path, actual)

    assert rows[0]["actual_home_goals"] == "0"
    assert rows[0]["actual_away_goals"] == "2"
    assert rows[0]["actual_outcome"] == "away"
