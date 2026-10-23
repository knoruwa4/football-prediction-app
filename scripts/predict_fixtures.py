
"""Predict fixtures and save probabilities and scorelines to CSV."""

from datetime import date
from math import exp
from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_prediction.batch_predictor import predict_fixtures
from football_prediction.football_data import load_football_data


DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"
MATCH_DATE = date(2026, 10, 25)

FIXTURES = [
    ("Arsenal", "Chelsea"),
    ("Liverpool", "Manchester City"),
    ("Tottenham", "Manchester United"),
]

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
    csv_files = sorted(DATA_DIR.glob("epl_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No epl_*.csv files found in {DATA_DIR}"
        )

    matches = []
    for csv_file in csv_files:
        loaded = load_football_data(csv_file)
        matches.extend(loaded)
        print(f"Loaded {len(loaded)} matches from {csv_file.name}")

    print(f"\nTotal historical matches loaded: {len(matches)}")
    print(f"Prediction date: {MATCH_DATE}\n")

    results = predict_fixtures(
        matches=matches,
        fixtures=FIXTURES,
        match_date=MATCH_DATE,
        model_type="joint_attack_defence",
        allow_unknown_teams=True,
    )

    report_rows = []

    for home_team, away_team, probabilities in results:
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
            "match_date": MATCH_DATE.isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "model": probabilities.model,
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
    report_path = REPORT_DIR / (
        f"fixture_predictions_{MATCH_DATE.isoformat()}.csv"
    )

    fieldnames = list(report_rows[0].keys()) if report_rows else []

    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Saved {len(report_rows)} predictions to:")
    print(report_path)


if __name__ == "__main__":
    main()
