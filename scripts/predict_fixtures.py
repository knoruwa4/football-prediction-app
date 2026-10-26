
"""Predict fixtures and save probabilities and scorelines to CSV."""

from datetime import date, datetime, timezone
import argparse
from math import exp
from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_prediction.batch_predictor import predict_fixtures
from football_prediction.fixtures import (
    historical_team_name,
    load_verified_fixtures,
)
from football_prediction.football_data import COMPETITIONS
from football_prediction.league_data import (
    FIRST_SEASON_YEAR,
    LAST_SEASON_YEAR,
    MAIN_LEAGUE_CODES,
    load_extra_league_history,
    load_league_history,
)
from football_prediction.prediction_store import append_predictions


DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
PREDICTION_LEDGER = REPORT_DIR / "prediction_history.csv"
MATCH_DATE = date(2026, 10, 25)
FIXTURE_FILE = PROJECT_ROOT / "data" / "fixtures" / "epl_2026-10-25.csv"

MAX_GOALS = 10


def poisson_probabilities(expected_goals: float) -> list[float]:
    """Calculate probabilities of scoring 0 through MAX_GOALS goals."""
    probabilities = [exp(-expected_goals)]

    for goals in range(1, MAX_GOALS + 1):
        probabilities.append(
            probabilities[-1] * expected_goals / goals
        )

    return probabilities


def most_likely_scorelines(home_xg: float, away_xg: float):
    """Return the three most likely exact scorelines."""
    home_probs = poisson_probabilities(home_xg)
    away_probs = poisson_probabilities(away_xg)

    scores = [
        (home_probs[home] * away_probs[away], home, away)
        for home in range(MAX_GOALS + 1)
        for away in range(MAX_GOALS + 1)
    ]

    return sorted(scores, reverse=True)[:3]


def main():
    parser = argparse.ArgumentParser(
        description="Predict sourced fixtures using one league's local history."
    )
    parser.add_argument(
        "--competition-code", choices=tuple(COMPETITIONS), default="E0",
        help="Football-Data division/country code (default: E0 Premier League)",
    )
    parser.add_argument(
        "--match-date", type=date.fromisoformat, default=MATCH_DATE,
        help="Fixture date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--fixture-file", type=Path,
        help="CSV of verified fixtures with a source URL per row (required outside E0)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=DATA_DIR,
        help="Root containing main-season folders and the extra/ country CSVs",
    )
    args = parser.parse_args()

    fixture_file = args.fixture_file
    if fixture_file is None:
        if args.competition_code != "E0":
            parser.error("--fixture-file is required for non-Premier League competitions")
        fixture_file = FIXTURE_FILE

    years = range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1)
    if args.competition_code in MAIN_LEAGUE_CODES:
        matches = load_league_history(
            args.data_root, args.competition_code, years
        )
    else:
        matches = load_extra_league_history(
            args.data_root, args.competition_code, years
        )
    print(
        f"Loaded {len(matches)} {COMPETITIONS[args.competition_code]} "
        f"matches from seasons {FIRST_SEASON_YEAR}/{str(FIRST_SEASON_YEAR + 1)[-2:]} "
        f"through {LAST_SEASON_YEAR}/{str(LAST_SEASON_YEAR + 1)[-2:]}"
    )

    match_date = args.match_date
    print(f"Prediction date: {match_date}\n")

    fixtures = load_verified_fixtures(fixture_file, match_date)
    fixture_pairs = [
        (
            historical_team_name(fixture["home_team"]),
            historical_team_name(fixture["away_team"]),
        )
        for fixture in fixtures
    ]

    results = predict_fixtures(
        matches=matches,
        fixtures=fixture_pairs,
        match_date=match_date,
        model_type="joint_attack_defence",
        allow_unknown_teams=True,
        competition=COMPETITIONS[args.competition_code],
    )

    report_rows = []

    fixture_metadata = {
        (
            historical_team_name(fixture["home_team"]),
            historical_team_name(fixture["away_team"]),
        ): fixture
        for fixture in fixtures
    }
    historical_teams = {
        team
        for match in matches
        for team in (match.home_team, match.away_team)
    }

    for model_home_team, model_away_team, probabilities in results:
        home_team, away_team = model_home_team, model_away_team
        fixture = fixture_metadata[(home_team, away_team)]
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]
        scorelines = most_likely_scorelines(
            probabilities.home_xg,
            probabilities.away_xg,
        )

        print(f"{home_team} vs {away_team}")
        print(
            f"  Home: {probabilities.home_win:.1%} | "
            f"Draw: {probabilities.draw:.1%} | "
            f"Away: {probabilities.away_win:.1%}"
        )
        print(
            f"  Expected goals: {probabilities.home_xg:.2f} - "
            f"{probabilities.away_xg:.2f}"
        )

        for rank, (probability, hg, ag) in enumerate(scorelines, start=1):
            print(f"  Score #{rank}: {hg}-{ag} ({probability:.1%})")

        row = {
            "match_date": match_date.isoformat(),
            "competition": COMPETITIONS[args.competition_code],
            "kickoff_utc": fixture["kickoff_utc"],
            "home_team": home_team,
            "away_team": away_team,
            "home_team_history_available": (
                model_home_team in historical_teams
            ),
            "away_team_history_available": (
                model_away_team in historical_teams
            ),
            "fixture_source_url": fixture["source_url"],
            "model": probabilities.model,
            "model_version": "0.1.0",
            "home_win_probability": round(probabilities.home_win, 6),
            "draw_probability": round(probabilities.draw, 6),
            "away_win_probability": round(probabilities.away_win, 6),
            "home_expected_goals": round(probabilities.home_xg, 4),
            "away_expected_goals": round(probabilities.away_xg, 4),
        }

        for rank, (probability, hg, ag) in enumerate(scorelines, start=1):
            row[f"score_{rank}"] = f"{hg}-{ag}"
            row[f"score_{rank}_probability"] = round(probability, 6)

        report_rows.append(row)
        print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_name = (
        f"fixture_predictions_{match_date.isoformat()}_{run_timestamp}.csv"
        if args.competition_code == "E0"
        else f"fixture_predictions_{args.competition_code}_{match_date.isoformat()}_{run_timestamp}.csv"
    )
    report_path = REPORT_DIR / report_name

    fieldnames = list(report_rows[0].keys()) if report_rows else []

    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    append_predictions(PREDICTION_LEDGER, report_rows)

    print(f"Saved {len(report_rows)} predictions to:")
    print(report_path)
    print(f"Appended forecasts to: {PREDICTION_LEDGER}")


if __name__ == "__main__":
    main()
