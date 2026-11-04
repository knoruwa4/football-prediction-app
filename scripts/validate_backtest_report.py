
import csv
import math
from collections import defaultdict
from pathlib import Path


REPORT_PATH = Path("reports/epl_multiseason_predictions.csv")

MODELS = {
    "Team strength": (
        "model_home_win",
        "model_draw",
        "model_away_win",
    ),
    "Historical frequency": (
        "frequency_home_win",
        "frequency_draw",
        "frequency_away_win",
    ),
    "Venue Poisson": (
        "venue_home_win",
        "venue_draw",
        "venue_away_win",
    ),
}

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def calculate_scores(rows, probability_columns):
    """Calculate multiclass Brier score and log loss."""
    if not rows:
        return None

    brier_total = 0.0
    log_loss_total = 0.0

    for row in rows:
        actual = OUTCOME_INDEX[row["actual_outcome"].strip()]

        probabilities = [
            float(row[column]) for column in probability_columns
        ]

        if any(
            not math.isfinite(p) or p < 0.0 or p > 1.0
            for p in probabilities
        ):
            raise ValueError(
                f"Invalid probability values in row: {row}"
            )

        if not math.isclose(sum(probabilities), 1.0, abs_tol=0.001):
            raise ValueError(
                f"Probabilities do not sum to approximately 1: {row}"
            )

        brier_total += sum(
            (p - (1.0 if i == actual else 0.0)) ** 2
            for i, p in enumerate(probabilities)
        )

        log_loss_total -= math.log(max(probabilities[actual], 1e-15))

    count = len(rows)
    return brier_total / count, log_loss_total / count


def print_scores(title, rows):
    print(f"\n{title} ({len(rows)} matches)")
    print(f"{'Model':<24} {'Brier score':>12} {'Log loss':>12}")
    print("-" * 50)

    for model_name, columns in MODELS.items():
        scores = calculate_scores(rows, columns)

        if scores is None:
            print(f"{model_name:<24} {'No data':>12}")
            continue

        brier, log_loss = scores
        print(f"{model_name:<24} {brier:>12.4f} {log_loss:>12.4f}")


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Report not found: {REPORT_PATH.resolve()}"
        )

    with REPORT_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "season_file",
            "actual_outcome",
            *[column for columns in MODELS.values() for column in columns],
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        rows = list(reader)

    invalid_outcomes = sorted({
        row["actual_outcome"].strip()
        for row in rows
        if row["actual_outcome"].strip() not in OUTCOME_INDEX
    })

    if invalid_outcomes:
        raise ValueError(f"Unexpected match outcomes: {invalid_outcomes}")

    if not rows:
        raise ValueError("The CSV contains no prediction rows.")

    print(f"Report: {REPORT_PATH.resolve()}")
    print(f"Rows loaded: {len(rows)}")

    print_scores("OVERALL RESULTS", rows)

    seasons = defaultdict(list)
    for row in rows:
        seasons[row["season_file"]].append(row)

    for season in sorted(seasons):
        print_scores(season, seasons[season])


if __name__ == "__main__":
    main()
