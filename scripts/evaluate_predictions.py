
"""Evaluate saved football predictions against completed match results."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_prediction.league_data import load_all_competition_histories
from football_prediction.prediction_store import evaluate_prediction_file


DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
PREDICTION_LEDGER = REPORT_DIR / "prediction_history.csv"


def main():
    if not PREDICTION_LEDGER.exists():
        raise FileNotFoundError(
            f"No prediction ledger found at {PREDICTION_LEDGER}. "
            "Run scripts/predict_fixtures.py first."
        )
    print(f"Evaluating prediction ledger: {PREDICTION_LEDGER.name}")

    histories = load_all_competition_histories(DATA_DIR)
    actual_matches = [
        match
        for league_matches in histories.values()
        for match in league_matches
    ]

    rows = evaluate_prediction_file(
        PREDICTION_LEDGER, actual_matches
    )
    evaluated = [row for row in rows if row["evaluation_status"] == "evaluated"]
    pending = [row for row in rows if row["evaluation_status"] != "evaluated"]
    for prediction in pending:
        print(
            f"Pending result: {prediction['home_team']} vs "
            f"{prediction['away_team']} ({prediction['match_date']})"
        )

    if not evaluated:
        print(
            f"\nEvaluated matches: 0; pending forecasts: {len(pending)}. "
            "No matching completed results were found yet."
        )
        return

    brier_sum = 0.0
    correct = 0
    for prediction in evaluated:
        predicted_probs = {
            "home": float(prediction["home_win_probability"]),
            "draw": float(prediction["draw_probability"]),
            "away": float(prediction["away_win_probability"]),
        }
        actual_outcome = prediction["actual_outcome"]
        actual_vector = {
            "home": int(actual_outcome == "home"),
            "draw": int(actual_outcome == "draw"),
            "away": int(actual_outcome == "away"),
        }

        brier_sum += sum(
            (predicted_probs[label] - actual_vector[label]) ** 2
            for label in ("home", "draw", "away")
        )

        predicted_outcome = max(predicted_probs, key=predicted_probs.get)
        correct += predicted_outcome == actual_outcome

    count = len(evaluated)
    print(f"\nEvaluated matches: {count}")
    print(f"Pending forecasts: {len(pending)}")
    print(f"Outcome accuracy: {correct / count:.1%}")
    print(f"Multiclass Brier score: {brier_sum / count:.4f}")
    print(
        "\nLower Brier scores are better. Accuracy alone does not "
        "measure how well-calibrated the probabilities are."
    )


if __name__ == "__main__":
    main()
