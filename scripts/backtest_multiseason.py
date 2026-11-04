"""Chronological multi-season EPL backtest."""

import csv
from collections import defaultdict
from math import exp, log
from pathlib import Path

from football_prediction.football_data import load_football_data
from football_prediction.predictor import predict_match

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_PATH = ROOT / "reports" / "epl_multiseason_predictions.csv"

PRIOR = 50.0
MIN_HISTORY = 60
EPSILON = 1e-15


def outcome_of(match):
    if match.home_goals > match.away_goals:
        return 0
    if match.home_goals == match.away_goals:
        return 1
    return 2


def model_probs(prediction):
    return [
        prediction.home_win,
        prediction.draw,
        prediction.away_win,
    ]


def frequency_probs(history):
    counts = [1.0, 1.0, 1.0]
    for match in history:
        counts[outcome_of(match)] += 1
    total = sum(counts)
    return [value / total for value in counts]


def venue_poisson_probs(history):
    """Poisson baseline using league-average home/away scoring."""
    count = len(history)
    prior_games = 5.0

    home_mean = (sum(m.home_goals for m in history) + prior_games * 1.45) / (
        count + prior_games
    )

    away_mean = (sum(m.away_goals for m in history) + prior_games * 1.15) / (
        count + prior_games
    )

    def poisson(mean, max_goals=15):
        probabilities = [exp(-mean)]
        for goals in range(1, max_goals + 1):
            probabilities.append(probabilities[-1] * mean / goals)
        return probabilities

    home = poisson(home_mean)
    away = poisson(away_mean)
    result = [0.0, 0.0, 0.0]

    for hg, hp in enumerate(home):
        for ag, ap in enumerate(away):
            if hg > ag:
                result[0] += hp * ap
            elif hg == ag:
                result[1] += hp * ap
            else:
                result[2] += hp * ap

    total = sum(result)
    return [value / total for value in result]


def score(rows):
    """Calculate multiclass Brier score and log loss."""
    if not rows:
        return None, None

    brier_total = 0.0
    loss_total = 0.0

    for probabilities, actual in rows:
        brier_total += sum(
            (p - (1.0 if i == actual else 0.0)) ** 2
            for i, p in enumerate(probabilities)
        )
        loss_total -= log(max(probabilities[actual], EPSILON))

    return brier_total / len(rows), loss_total / len(rows)


def main():
    files = sorted(DATA_DIR.glob("epl_*.csv"))
    if not files:
        raise FileNotFoundError(f"No EPL CSV files in {DATA_DIR}")

    matches = []
    source_files = {}

    for path in files:
        season_matches = load_football_data(path)

        for match in season_matches:
            key = (
                match.match_date,
                match.home_team,
                match.away_team,
            )
            if key in source_files:
                raise ValueError(f"Duplicate fixture found: {key}")

            source_files[key] = path.name
            matches.append(match)

    matches.sort(
        key=lambda m: (
            m.match_date,
            m.home_team,
            m.away_team,
        )
    )

    rows = []
    skipped = 0
    skip_reasons = defaultdict(int)

    for match in matches:
        # Exclude same-day matches to avoid using results unavailable
        # before that calendar date.
        history = [past for past in matches if past.match_date < match.match_date]

        if len(history) < MIN_HISTORY:
            skipped += 1
            skip_reasons["insufficient_history"] += 1
            continue

       

        try:
            prediction = predict_match(
                history,
                match.home_team,
                match.away_team,
                match.match_date,
                prior_matches=PRIOR,
                allow_unknown_teams=True,
            )
        except (ValueError, KeyError) as error:
            skipped += 1
            skip_reasons[f"prediction_error: {type(error).__name__}"] += 1
            continue

        actual = outcome_of(match)
        model = model_probs(prediction)
        frequency = frequency_probs(history)
        venue = venue_poisson_probs(history)

        key = (
            match.match_date,
            match.home_team,
            match.away_team,
        )

        rows.append(
            {
                "date": match.match_date.isoformat(),
                "season_file": source_files[key],
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_outcome": ("H", "D", "A")[actual],
                "model_home_win": model[0],
                "model_draw": model[1],
                "model_away_win": model[2],
                "frequency_home_win": frequency[0],
                "frequency_draw": frequency[1],
                "frequency_away_win": frequency[2],
                "venue_home_win": venue[0],
                "venue_draw": venue[1],
                "venue_away_win": venue[2],
                "_actual": actual,
                "_model": model,
                "_frequency": frequency,
                "_venue": venue,
            }
        )

    print(f"Input files: {len(files)}")
    print(f"Total matches: {len(matches)}")
    print(f"Evaluated matches: {len(rows)}")
   
    print(f"Skipped matches: {skipped}")
    print("Skip reasons:")
    for reason, count in sorted(skip_reasons.items()):
        print(f"  {reason}: {count}")
    print(f"Model prior: {PRIOR}")


    if not rows:
        print("No eligible matches to score.")
        return

    groups = defaultdict(list)
    for row in rows:
        groups[row["season_file"]].append(row)

    def print_scores(label, selected_rows):
        print(f"\n{label} ({len(selected_rows)} matches)")

        for name, key in (
            ("Team strength", "_model"),
            ("Historical frequency", "_frequency"),
            ("Venue Poisson", "_venue"),
        ):
            brier, loss = score([(row[key], row["_actual"]) for row in selected_rows])
            print(f"  {name}: Brier={brier:.4f}, " f"log loss={loss:.4f}")

    print_scores("Overall", rows)

    for season_file in sorted(groups):
        print_scores(season_file, groups[season_file])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_fields = [key for key in rows[0] if not key.startswith("_")]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row[key] for key in output_fields})

    print(f"\nReport saved to: {REPORT_PATH.relative_to(ROOT)}")
    print("These are historical results, not a guarantee " "of future performance.")


if __name__ == "__main__":
    main()
