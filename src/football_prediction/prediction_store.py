"""Append and evaluate durable CSV prediction records."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from football_prediction.data import MatchResult


PREDICTION_FIELDS = (
    "prediction_id",
    "prediction_timestamp",
    "match_date",
    "kickoff_utc",
    "home_team",
    "away_team",
    "home_team_history_available",
    "away_team_history_available",
    "fixture_source_url",
    "model",
    "model_version",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "home_expected_goals",
    "away_expected_goals",
    "score_1",
    "score_1_probability",
    "score_2",
    "score_2_probability",
    "score_3",
    "score_3_probability",
    "evaluation_status",
    "actual_home_goals",
    "actual_away_goals",
    "actual_outcome",
)


def append_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    """Append forecasts without changing prior records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            record = {field: "" for field in PREDICTION_FIELDS}
            record.update(row)
            record["prediction_id"] = str(uuid4())
            record["prediction_timestamp"] = timestamp
            record["evaluation_status"] = "pending"
            writer.writerow(record)


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals == away_goals:
        return "draw"
    return "away"


def evaluate_prediction_file(
    path: Path, actual_matches: list[MatchResult]
) -> list[dict[str, str]]:
    """Match forecasts to results and persist evaluation fields in place."""
    result_by_key = {
        (
            match.match_date.isoformat(),
            match.home_team.strip().casefold(),
            match.away_team.strip().casefold(),
        ): match
        for match in actual_matches
    }
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    for field in PREDICTION_FIELDS:
        if field not in fields:
            fields.append(field)

    for row in rows:
        key = (
            row["match_date"],
            row["home_team"].strip().casefold(),
            row["away_team"].strip().casefold(),
        )
        actual = result_by_key.get(key)
        if actual is None:
            row["evaluation_status"] = "pending"
            continue
        row["actual_home_goals"] = str(actual.home_goals)
        row["actual_away_goals"] = str(actual.away_goals)
        row["actual_outcome"] = _outcome(actual.home_goals, actual.away_goals)
        row["evaluation_status"] = "evaluated"

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return rows
