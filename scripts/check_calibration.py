
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "epl_multiseason_predictions.csv"

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

OUTCOMES = {"H": 0, "D": 1, "A": 2}
OUTCOME_NAMES = ("Home", "Draw", "Away")
BIN_WIDTH = 0.10


def print_calibration(rows, model_name, columns):
    print(f"\n{model_name}")
    print("Outcome  Probability range  Matches  Predicted  Actual")

    for outcome_index, outcome_name in enumerate(OUTCOME_NAMES):
        for bin_index in range(10):
            lower = bin_index * BIN_WIDTH
            upper = lower + BIN_WIDTH

            selected = [
                row for row in rows
                if lower <= float(row[columns[outcome_index]]) < upper
            ]

            if bin_index == 9:
                selected = [
                    row for row in rows
                    if lower <= float(row[columns[outcome_index]]) <= 1.0
                ]

            if not selected:
                continue

            predicted = sum(
                float(row[columns[outcome_index]])
                for row in selected
            ) / len(selected)

            actual = sum(
                1 for row in selected
                if OUTCOMES[row["actual_outcome"]] == outcome_index
            ) / len(selected)

            print(
                f"{outcome_name:7}  "
                f"{lower:.1f}-{upper:.1f}             "
                f"{len(selected):5d}   "
                f"{predicted:8.3f}  {actual:6.3f}"
            )


def main():
    with REPORT.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No predictions found in {REPORT}")

    seasons = sorted({row["season_file"] for row in rows})

    for season in seasons:
        season_rows = [
            row for row in rows
            if row["season_file"] == season
        ]

        print("\n" + "=" * 60)
        print(f"SEASON: {season} ({len(season_rows)} matches)")

        for model_name, columns in MODELS.items():
            print_calibration(season_rows, model_name, columns)


if __name__ == "__main__":
    main()
