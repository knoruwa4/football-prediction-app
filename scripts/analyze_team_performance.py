
import csv
import math
from collections import defaultdict
from pathlib import Path

REPORT_PATH = Path("reports/epl_multiseason_predictions.csv")

MODELS = {
    "Venue Poisson": (
        "venue_home_win", "venue_draw", "venue_away_win"
    ),
    "Baseline": (
        "baseline_home_win", "baseline_draw", "baseline_away_win"
    ),
}

BASELINE_PATH = Path("reports/epl_baseline_predictions.csv")
OUTCOMES = {"H": 0, "D": 1, "A": 2}


def loss(probabilities, actual):
    brier = sum(
        (p - (1.0 if i == actual else 0.0)) ** 2
        for i, p in enumerate(probabilities)
    )
    logloss = -math.log(max(probabilities[actual], 1e-15))
    return brier, logloss


def load_rows():
    with REPORT_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        model_rows = list(csv.DictReader(f))

    with BASELINE_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        baseline_rows = list(csv.DictReader(f))

    key_fields = ("date", "season_file", "home_team", "away_team")
    baseline_by_key = {
        tuple(row[k] for k in key_fields): row
        for row in baseline_rows
    }

    combined = []
    for row in model_rows:
        key = tuple(row[k] for k in key_fields)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue

        row = dict(row)
        row.update({
            "baseline_home_win": baseline["baseline_home_win"],
            "baseline_draw": baseline["baseline_draw"],
            "baseline_away_win": baseline["baseline_away_win"],
        })
        combined.append(row)

    if not combined:
        raise ValueError("No matching fixtures between the two reports.")

    return combined


def main():
    rows = load_rows()

    # Group by home team, then evaluate both models on identical matches.
    by_team = defaultdict(list)
    for row in rows:
        by_team[row["home_team"]].append(row)

    results = []
    for team, matches in by_team.items():
        if len(matches) < 30:
            continue

        totals = {
            name: {"brier": 0.0, "logloss": 0.0}
            for name in MODELS
        }

        for row in matches:
            actual = OUTCOMES[row["actual_outcome"].strip()]
            for name, columns in MODELS.items():
                probabilities = [float(row[c]) for c in columns]
                brier, logloss = loss(probabilities, actual)
                totals[name]["brier"] += brier
                totals[name]["logloss"] += logloss

        n = len(matches)
        venue_brier = totals["Venue Poisson"]["brier"] / n
        base_brier = totals["Baseline"]["brier"] / n
        venue_log = totals["Venue Poisson"]["logloss"] / n
        base_log = totals["Baseline"]["logloss"] / n

        results.append({
            "team": team,
            "matches": n,
            "brier_diff": base_brier - venue_brier,
            "logloss_diff": base_log - venue_log,
        })

    results.sort(key=lambda r: r["logloss_diff"], reverse=True)

    print(f"Matches loaded: {len(rows)}")
    print(f"Teams evaluated: {len(results)}")
    print("Positive differences favor Venue Poisson.")
    print("\nBEST 10 TEAMS BY LOG-LOSS DIFFERENCE")
    print(f"{'Home team':<24} {'Matches':>8} {'Brier Δ':>10} {'Log loss Δ':>12}")
    print("-" * 58)

    for r in results[:10]:
        print(
            f"{r['team']:<24} {r['matches']:>8} "
            f"{r['brier_diff']:>+10.4f} {r['logloss_diff']:>+12.4f}"
        )

    print("\nWORST 10 TEAMS BY LOG-LOSS DIFFERENCE")
    print(f"{'Home team':<24} {'Matches':>8} {'Brier Δ':>10} {'Log loss Δ':>12}")
    print("-" * 58)

    for r in results[-10:]:
        print(
            f"{r['team']:<24} {r['matches']:>8} "
            f"{r['brier_diff']:>+10.4f} {r['logloss_diff']:>+12.4f}"
        )

    better = sum(r["logloss_diff"] > 0 for r in results)
    worse = sum(r["logloss_diff"] < 0 for r in results)
    tied = len(results) - better - worse

    print("\nSUMMARY")
    print(f"Teams where Venue Poisson has lower log loss: {better}")
    print(f"Teams where baseline has lower log loss: {worse}")
    print(f"Ties: {tied}")
    print(
        "\nCaution: these are descriptive team-level results. "
        "Team-specific sample sizes are limited, and this is not "
        "a statistical significance test."
    )


if __name__ == "__main__":
    main()
