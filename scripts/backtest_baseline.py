
import csv
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


REPORT_PATH = Path("reports/epl_multiseason_predictions.csv")
OUTPUT_PATH = Path("reports/epl_baseline_predictions.csv")

OUTCOMES = ("H", "D", "A")
PRIOR_MATCHES = 50.0

MODEL_COLUMNS = {
    "Team strength": (
        "model_home_win", "model_draw", "model_away_win"
    ),
    "Historical frequency": (
        "frequency_home_win", "frequency_draw", "frequency_away_win"
    ),
    "Venue Poisson": (
        "venue_home_win", "venue_draw", "venue_away_win"
    ),
}


def score_predictions(rows, probability_columns):
    """Return multiclass Brier score and log loss."""
    if not rows:
        return None

    brier_sum = 0.0
    log_loss_sum = 0.0

    for row in rows:
        actual = OUTCOMES.index(row["actual_outcome"])
        probabilities = [
            float(row[column]) for column in probability_columns
        ]

        brier_sum += sum(
            (p - (1.0 if i == actual else 0.0)) ** 2
            for i, p in enumerate(probabilities)
        )
        log_loss_sum -= math.log(max(probabilities[actual], 1e-15))

    n = len(rows)
    return brier_sum / n, log_loss_sum / n


def score_baseline(rows):
    if not rows:
        return None

    brier_sum = 0.0
    log_loss_sum = 0.0

    for row in rows:
        actual = OUTCOMES.index(row["actual_outcome"])
        probabilities = [
            float(row["baseline_home_win"]),
            float(row["baseline_draw"]),
            float(row["baseline_away_win"]),
        ]

        brier_sum += sum(
            (p - (1.0 if i == actual else 0.0)) ** 2
            for i, p in enumerate(probabilities)
        )
        log_loss_sum -= math.log(max(probabilities[actual], 1e-15))

    n = len(rows)
    return brier_sum / n, log_loss_sum / n


def print_scores(title, rows):
    print(f"\n{title} ({len(rows)} matches)")
    print(f"{'Model':<24} {'Brier score':>12} {'Log loss':>12}")
    print("-" * 50)

    results = [("Historical-frequency baseline", score_baseline(rows))]
    results.extend(
        (name, score_predictions(rows, columns))
        for name, columns in MODEL_COLUMNS.items()
    )

    for name, scores in results:
        if scores is None:
            continue
        brier, log_loss = scores
        print(f"{name:<30} {brier:>10.4f} {log_loss:>12.4f}")


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find prediction report: {REPORT_PATH.resolve()}"
        )

    with REPORT_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Prediction report contains no rows.")

    required = {
        "date", "season_file", "actual_outcome",
        *[c for cols in MODEL_COLUMNS.values() for c in cols],
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    for row in rows:
        row["match_date"] = date.fromisoformat(row["date"][:10])
        row["actual_outcome"] = row["actual_outcome"].strip()

        if row["actual_outcome"] not in OUTCOMES:
            raise ValueError(
                f"Unexpected outcome {row['actual_outcome']!r} "
                f"on {row['date']}"
            )

    # Process whole dates together. Matches on the same date cannot
    # use one another's results, avoiding same-day information leakage.
    rows.sort(key=lambda r: (r["match_date"], r["season_file"]))

    historical_counts = Counter()
    baseline_rows = []
    i = 0

    while i < len(rows):
        current_date = rows[i]["match_date"]
        j = i

        while j < len(rows) and rows[j]["match_date"] == current_date:
            j += 1

        previous_matches = sum(historical_counts.values())

        # Use a uniform prior equivalent to 50 matches spread equally
        # across home wins, draws and away wins.
        denominator = previous_matches + PRIOR_MATCHES
        probabilities = {
            outcome: (
                historical_counts[outcome] + PRIOR_MATCHES / 3
            ) / denominator
            for outcome in OUTCOMES
        }

        # Predict every match on this date before adding that date's results.
        for row in rows[i:j]:
            row["baseline_home_win"] = probabilities["H"]
            row["baseline_draw"] = probabilities["D"]
            row["baseline_away_win"] = probabilities["A"]
            baseline_rows.append(row)

        # Only after all predictions for the date are made do we
        # incorporate that date's actual results.
        for row in rows[i:j]:
            historical_counts[row["actual_outcome"]] += 1

        i = j

    output_columns = [
        "date", "season_file", "home_team", "away_team",
        "actual_outcome",
        "baseline_home_win", "baseline_draw", "baseline_away_win",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        for row in baseline_rows:
            writer.writerow({key: row[key] for key in output_columns})

    print(f"Input report: {REPORT_PATH.resolve()}")
    print(f"Matches evaluated: {len(baseline_rows)}")
    print(f"Baseline prior: {PRIOR_MATCHES:.0f} matches, equally split")
    print(f"Baseline predictions saved to: {OUTPUT_PATH.resolve()}")

    print_scores("OVERALL COMPARISON", baseline_rows)

    by_season = defaultdict(list)
    for row in baseline_rows:
        by_season[row["season_file"]].append(row)

    for season in sorted(by_season):
        print_scores(season, by_season[season])


if __name__ == "__main__":
    main()
