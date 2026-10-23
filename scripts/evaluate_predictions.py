
"""Evaluate saved football predictions against completed match results."""

from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_prediction.football_data import load_football_data


DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"


def outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals == away_goals:
        return "draw"
    return "away"


def main():
    prediction_files = sorted(
        REPORT_DIR.glob("fixture_predictions_*.csv")
    )

    if not prediction_files:
        raise FileNotFoundError(
            f"No fixture_predictions_*.csv files found in {REPORT_DIR}. "
            "Run scripts/predict_fixtures.py first."
        )

    prediction_file = prediction_files[-1]
    print(f"Reading predictions from: {prediction_file.name}")

    predictions = []
    with prediction_file.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        predictions = list(csv.DictReader(file))

    actual_matches = {}
    for csv_file in sorted(DATA_DIR.glob("epl_*.csv")):
        for match in load_football_data(csv_file):
            key = (
                match.match_date.isoformat(),
                match.home_team.strip().casefold(),
                match.away_team.strip().casefold(),
            )
            actual_matches[key] = match

    evaluated = []
    for prediction in predictions:
        key = (
            prediction["match_date"],
            prediction["home_team"].strip().casefold(),
            prediction["away_team"].strip().casefold(),
        )

        actual = actual_matches.get(key)
        if actual is None:
            print(
                f"Not evaluated (no matching result): "
                f"{prediction['home_team']} vs {prediction['away_team']}"
            )
            continue

        predicted_probs = {
            "home": float(prediction["home_win_probability"]),
            "draw": float(prediction["draw_probability"]),
            "away": float(prediction["away_win_probability"]),
        }
        actual_outcome = outcome(actual.home_goals, actual.away_goals)

        evaluated.append((prediction, actual, predicted_probs, actual_outcome))

    if not evaluated:
        print(
            "\nNo matching completed results were found. This is normal "
            "if your predictions are for future fixtures."
        )
        return

    brier_sum = 0.0
    correct = 0
    for prediction, actual, probs, actual_outcome in evaluated:
        actual_vector = {
            "home": int(actual_outcome == "home"),
            "draw": int(actual_outcome == "draw"),
            "away": int(actual_outcome == "away"),
        }

        brier_sum += sum(
            (probs[label] - actual_vector[label]) ** 2
            for label in ("home", "draw", "away")
        )

        predicted_outcome = max(probs, key=probs.get)
        correct += predicted_outcome == actual_outcome

    count = len(evaluated)
    print(f"\nEvaluated matches: {count}")
    print(f"Outcome accuracy: {correct / count:.1%}")
    print(f"Multiclass Brier score: {brier_sum / count:.4f}")
    print(
        "\nLower Brier scores are better. Accuracy alone does not "
        "measure how well-calibrated the probabilities are."
    )


if __name__ == "__main__":
    main()
