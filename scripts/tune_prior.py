
import sys
from collections import defaultdict
from math import log
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_prediction.football_data import load_football_data
from football_prediction.predictor import predict_match


DATA_DIR = ROOT / "data" / "raw"
MIN_HISTORY = 60

PRIOR_VALUES = [10.0, 25.0, 50.0, 100.0]

# Use earlier seasons to choose a setting.
TRAIN_SEASONS = {
    "epl_2020_21.csv",
    "epl_2021_22.csv",
    "epl_2022_23.csv",
    "epl_2023_24.csv",
}

# Keep these seasons separate for the final evaluation.
TEST_SEASONS = {
    "epl_2024_25.csv",
    "epl_2025_26.csv",
}


def score_predictions(rows):
    total_brier = 0.0
    total_log_loss = 0.0

    for probabilities, actual in rows:
        for index, probability in enumerate(probabilities):
            observed = 1.0 if index == actual else 0.0
            total_brier += (probability - observed) ** 2

        actual_probability = max(probabilities[actual], 1e-15)
        total_log_loss -= log(actual_probability)

    count = len(rows)

    return total_brier / count, total_log_loss / count


def main():
    source_files = {}
    matches_by_file = defaultdict(list)

    for path in sorted(DATA_DIR.glob("epl_*.csv")):
        matches = load_football_data(path)

        for match in matches:
            key = (
                match.match_date,
                match.home_team,
                match.away_team,
            )

            if key in source_files:
                raise ValueError(
                    f"Duplicate fixture found: {key}"
                )

            source_files[key] = path.name
            matches_by_file[path.name].append(match)

    all_matches = sorted(
        (match for matches in matches_by_file.values()
         for match in matches),
        key=lambda match: match.match_date,
    )

    results = {
        prior: {"train": [], "test": []}
        for prior in PRIOR_VALUES
    }

    skipped = 0
    eligible = 0

    for match in all_matches:
        season_file = source_files[
            (match.match_date, match.home_team, match.away_team)
        ]

        if season_file not in TRAIN_SEASONS | TEST_SEASONS:
            continue

        history = [
            earlier
            for earlier in all_matches
            if earlier.match_date < match.match_date
        ]

        if len(history) < MIN_HISTORY:
            skipped += 1
            continue

        eligible += 1
        split = "train" if season_file in TRAIN_SEASONS else "test"

        actual = (
            0 if match.home_goals > match.away_goals
            else 1 if match.home_goals == match.away_goals
            else 2
        )

        for prior in PRIOR_VALUES:
            prediction = predict_match(
                matches=history,
                home_team=match.home_team,
                away_team=match.away_team,
                match_date=match.match_date,
                prior_matches=prior,
                allow_unknown_teams=True,
            )

            probabilities = (
                prediction.home_win,
                prediction.draw,
                prediction.away_win,
            )

            results[prior][split].append((probabilities, actual))

    print(f"Eligible fixtures: {eligible}")
    print(f"Skipped for insufficient history: {skipped}")
    print("\nLower Brier score and log loss are better.")

    print("\nTRAINING RESULTS (used to choose prior)")
    print(f"{'Prior':>8} {'Matches':>9} {'Brier':>10} {'Log loss':>11}")

    for prior in PRIOR_VALUES:
        rows = results[prior]["train"]
        brier, log_loss = score_predictions(rows)
        print(
            f"{prior:8.1f} {len(rows):9d} "
            f"{brier:10.4f} {log_loss:11.4f}"
        )

    best_prior = min(
        PRIOR_VALUES,
        key=lambda prior: score_predictions(
            results[prior]["train"]
        )[1],
    )

    print(f"\nSelected prior by training log loss: {best_prior:g}")

    print("\nHELD-OUT RESULTS (not used to choose prior)")
    print(f"{'Prior':>8} {'Matches':>9} {'Brier':>10} {'Log loss':>11}")

    for prior in PRIOR_VALUES:
        rows = results[prior]["test"]
        brier, log_loss = score_predictions(rows)
        print(
            f"{prior:8.1f} {len(rows):9d} "
            f"{brier:10.4f} {log_loss:11.4f}"
        )

    chosen_rows = results[best_prior]["test"]
    chosen_brier, chosen_log_loss = score_predictions(chosen_rows)

    print(
        f"\nHeld-out score for selected prior ({best_prior:g}): "
        f"Brier={chosen_brier:.4f}, "
        f"log loss={chosen_log_loss:.4f}"
    )
    print("Historical results do not guarantee future performance.")


if __name__ == "__main__":
    main()
