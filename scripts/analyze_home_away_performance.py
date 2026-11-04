
import csv
import math
from collections import defaultdict
from pathlib import Path

MODEL_PATH = Path("reports/epl_multiseason_predictions.csv")
BASELINE_PATH = Path("reports/epl_baseline_predictions.csv")

OUTCOMES = {"H": 0, "D": 1, "A": 2}


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def loss(probabilities, actual):
    brier = sum(
        (p - (1.0 if i == actual else 0.0)) ** 2
        for i, p in enumerate(probabilities)
    )
    logloss = -math.log(max(probabilities[actual], 1e-15))
    return brier, logloss


def main():
    for path in (MODEL_PATH, BASELINE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.resolve()}")

    model_rows = read_csv(MODEL_PATH)
    baseline_rows = read_csv(BASELINE_PATH)

    key_fields = ("date", "season_file", "home_team", "away_team")
    baseline_by_key = {
        tuple(row[k] for k in key_fields): row
        for row in baseline_rows
    }

    if len(baseline_by_key) != len(baseline_rows):
        raise ValueError("Duplicate match keys in baseline report.")

    team_groups = defaultdict(list)
    season_groups = defaultdict(list)

    for row in model_rows:
        key = tuple(row[k] for k in key_fields)
        base = baseline_by_key.get(key)
        if base is None:
            continue

        actual_label = row["actual_outcome"].strip()
        if actual_label not in OUTCOMES:
            raise ValueError(f"Unexpected outcome: {actual_label}")

        actual = OUTCOMES[actual_label]

        venue = [
            float(row["venue_home_win"]),
            float(row["venue_draw"]),
            float(row["venue_away_win"]),
        ]
        baseline = [
            float(base["baseline_home_win"]),
            float(base["baseline_draw"]),
            float(base["baseline_away_win"]),
        ]

        for probs in (venue, baseline):
            if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs):
                raise ValueError(f"Invalid probability for match {key}")
            if not math.isclose(sum(probs), 1.0, abs_tol=0.001):
                raise ValueError(f"Probabilities do not sum to 1 for {key}")

        venue_brier, venue_log = loss(venue, actual)
        base_brier, base_log = loss(baseline, actual)

        result = {
            "season": row["season_file"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "brier_diff": base_brier - venue_brier,
            "log_diff": base_log - venue_log,
        }

        # Positive differences mean Venue Poisson had lower loss.
        team_groups[("HOME", row["home_team"])].append(result)
        team_groups[("AWAY", row["away_team"])].append(result)
        season_groups[row["season_file"]].append(result)

    print("Positive differences favor Venue Poisson.")
    print("\nTEAM PERFORMANCE: HOME FIXTURES")
    print_team_table(team_groups, "HOME")

    print("\nTEAM PERFORMANCE: AWAY FIXTURES")
    print_team_table(team_groups, "AWAY")

    print("\nSEASON PERFORMANCE")
    print(
        f"{'Season':<20} {'Matches':>8} "
        f"{'Brier Δ':>10} {'Log loss Δ':>12}"
    )
    print("-" * 54)

    for season in sorted(season_groups):
        results = season_groups[season]
        n = len(results)
        brier_diff = sum(r["brier_diff"] for r in results) / n
        log_diff = sum(r["log_diff"] for r in results) / n
        print(
            f"{season:<20} {n:>8} "
            f"{brier_diff:>+10.4f} {log_diff:>+12.4f}"
        )

    print(
        "\nCaution: these are descriptive comparisons, not tests "
        "of statistical significance. Team results can be noisy."
    )


def print_team_table(groups, venue_type):
    results = []

    for (kind, team), matches in groups.items():
        if kind != venue_type or len(matches) < 20:
            continue

        n = len(matches)
        results.append({
            "team": team,
            "matches": n,
            "brier": sum(r["brier_diff"] for r in matches) / n,
            "log": sum(r["log_diff"] for r in matches) / n,
        })

    results.sort(key=lambda r: r["log"], reverse=True)

    print(
        f"{'Team':<24} {'Matches':>8} "
        f"{'Brier Δ':>10} {'Log loss Δ':>12}"
    )
    print("-" * 58)

    for r in results:
        print(
            f"{r['team']:<24} {r['matches']:>8} "
            f"{r['brier']:>+10.4f} {r['log']:>+12.4f}"
        )


if __name__ == "__main__":
    main()
