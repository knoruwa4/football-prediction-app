
import csv
import math
import random
from collections import defaultdict
from datetime import date
from pathlib import Path


MODEL_PATH = Path("reports/epl_multiseason_predictions.csv")
BASELINE_PATH = Path("reports/epl_baseline_predictions.csv")

BOOTSTRAP_REPLICATES = 5000
BLOCK_DATES = 7
RANDOM_SEED = 42


def load_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def brier_loss(probabilities, actual):
    return sum(
        (p - (1.0 if i == actual else 0.0)) ** 2
        for i, p in enumerate(probabilities)
    )


def log_loss(probabilities, actual):
    return -math.log(max(probabilities[actual], 1e-15))


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def mean(values):
    return sum(values) / len(values)


def main():
    for path in (MODEL_PATH, BASELINE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.resolve()}")

    model_rows = load_csv(MODEL_PATH)
    baseline_rows = load_csv(BASELINE_PATH)

    key_fields = ("date", "season_file", "home_team", "away_team")

    baseline_by_key = {
        tuple(row[field] for field in key_fields): row
        for row in baseline_rows
    }

    if len(baseline_by_key) != len(baseline_rows):
        raise ValueError("Duplicate match keys found in baseline report.")

    paired = []

    for model_row in model_rows:
        key = tuple(model_row[field] for field in key_fields)
        baseline_row = baseline_by_key.get(key)

        if baseline_row is None:
            continue

        outcome = model_row["actual_outcome"].strip()
        outcome_index = {"H": 0, "D": 1, "A": 2}.get(outcome)

        if outcome_index is None:
            raise ValueError(f"Unexpected outcome {outcome!r} for {key}")

        venue = [
            float(model_row["venue_home_win"]),
            float(model_row["venue_draw"]),
            float(model_row["venue_away_win"]),
        ]
        baseline = [
            float(baseline_row["baseline_home_win"]),
            float(baseline_row["baseline_draw"]),
            float(baseline_row["baseline_away_win"]),
        ]

        for probabilities in (venue, baseline):
            if any(
                not math.isfinite(p) or p < 0 or p > 1
                for p in probabilities
            ):
                raise ValueError(f"Invalid probabilities for {key}")
            if not math.isclose(sum(probabilities), 1.0, abs_tol=0.001):
                raise ValueError(f"Probabilities do not sum to 1 for {key}")

        paired.append({
            "date": date.fromisoformat(model_row["date"][:10]),
            "season": model_row["season_file"],
            # Positive differences mean Venue Poisson has lower loss.
            "log_diff": (
                log_loss(baseline, outcome_index)
                - log_loss(venue, outcome_index)
            ),
            "brier_diff": (
                brier_loss(baseline, outcome_index)
                - brier_loss(venue, outcome_index)
            ),
        })

    if not paired:
        raise ValueError("No matching fixtures found between the reports.")

    # Organize matches by season and match date.
    seasons = defaultdict(lambda: defaultdict(list))
    for row in paired:
        seasons[row["season"]][row["date"]].append(row)

    log_diffs = [row["log_diff"] for row in paired]
    brier_diffs = [row["brier_diff"] for row in paired]

    observed_log = mean(log_diffs)
    observed_brier = mean(brier_diffs)

    print(f"Matched fixtures: {len(paired)}")
    print(f"Bootstrap replicates: {BOOTSTRAP_REPLICATES}")
    print(f"Block length: {BLOCK_DATES} distinct match dates")
    print("Positive differences favor Venue Poisson.")
    print()
    print("Observed mean loss difference (baseline - Venue Poisson):")
    print(f"  Log loss: {observed_log:+.6f}")
    print(f"  Brier:    {observed_brier:+.6f}")

    # Resample consecutive date blocks within each season, so each
    # season contributes approximately its original number of matches.
    rng = random.Random(RANDOM_SEED)
    bootstrap_log = []
    bootstrap_brier = []

    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_log = []
        sampled_brier = []

        for season in sorted(seasons):
            date_groups = seasons[season]
            dates = sorted(date_groups)
            target_count = sum(len(date_groups[d]) for d in dates)

            season_log = []
            season_brier = []

            while len(season_log) < target_count:
                start = rng.randrange(len(dates))

                for offset in range(min(BLOCK_DATES, len(dates))):
                    chosen_date = dates[(start + offset) % len(dates)]

                    for row in date_groups[chosen_date]:
                        season_log.append(row["log_diff"])
                        season_brier.append(row["brier_diff"])

                    if len(season_log) >= target_count:
                        break

            sampled_log.extend(season_log[:target_count])
            sampled_brier.extend(season_brier[:target_count])

        bootstrap_log.append(mean(sampled_log))
        bootstrap_brier.append(mean(sampled_brier))

    for label, values, observed in (
        ("Log loss", bootstrap_log, observed_log),
        ("Brier score", bootstrap_brier, observed_brier),
    ):
        low = percentile(values, 0.025)
        high = percentile(values, 0.975)
        favorable = sum(value > 0 for value in values) / len(values)

        print(f"\n{label} paired block-bootstrap results:")
        print(f"  Observed difference:      {observed:+.6f}")
        print(f"  Approx. 95% interval:     [{low:+.6f}, {high:+.6f}]")
        print(f"  Replicates favoring Venue Poisson: {favorable:.1%}")

        if low > 0:
            print("  Interpretation: interval favors Venue Poisson.")
        elif high < 0:
            print("  Interpretation: interval favors the baseline.")
        else:
            print("  Interpretation: interval includes no difference.")

    print(
        "\nNote: This is an approximate block-bootstrap diagnostic, "
        "not proof of future performance. Results can depend on "
        "the chosen block length."
    )


if __name__ == "__main__":
    main()
