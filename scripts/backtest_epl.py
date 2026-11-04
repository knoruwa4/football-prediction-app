
"""Chronological EPL backtest comparing team-strength shrinkage priors."""

import csv
from math import exp, log
from pathlib import Path

from football_prediction.football_data import load_football_data
from football_prediction.predictor import predict_match

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "epl_2025_26.csv"
REPORT_PATH = ROOT / "reports" / "epl_backtest_predictions.csv"

MIN_HISTORY = 60
EPSILON = 1e-15
PRIORS_TO_TEST = (15.0, 20.0, 25.0, 30.0)
REPORT_PRIOR = 25.0


def actual_outcome(match):
    if match.home_goals > match.away_goals:
        return 0
    if match.home_goals == match.away_goals:
        return 1
    return 2


def outcome_label(outcome):
    return ("H", "D", "A")[outcome]


def model_probabilities(prediction):
    return [prediction.home_win, prediction.draw, prediction.away_win]


def benchmark_probabilities(history):
    """Estimate 1X2 probabilities from prior results with smoothing."""
    counts = [1.0, 1.0, 1.0]
    for match in history:
        counts[actual_outcome(match)] += 1
    total = sum(counts)
    return [count / total for count in counts]


def venue_poisson_probabilities(history):
    """League-average Poisson baseline with separate venue goal averages."""
    count = len(history)
    if count == 0:
        raise ValueError("history must not be empty")

    prior_games = 5.0
    prior_home_goals = 1.45
    prior_away_goals = 1.15

    home_xg = (
        sum(m.home_goals for m in history)
        + prior_games * prior_home_goals
    ) / (count + prior_games)
    away_xg = (
        sum(m.away_goals for m in history)
        + prior_games * prior_away_goals
    ) / (count + prior_games)

    def poisson_probs(mean, max_goals=15):
        probs = [exp(-mean)]
        for goals in range(1, max_goals + 1):
            probs.append(probs[-1] * mean / goals)
        return probs

    home = poisson_probs(home_xg)
    away = poisson_probs(away_xg)
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
    return [p / total for p in result]


def score_predictions(rows):
    """Return multiclass Brier score and log loss."""
    if not rows:
        raise ValueError("No predictions available to score")

    brier_total = 0.0
    log_loss_total = 0.0

    for probabilities, outcome in rows:
        brier_total += sum(
            (p - (1.0 if i == outcome else 0.0)) ** 2
            for i, p in enumerate(probabilities)
        )
        log_loss_total -= log(max(probabilities[outcome], EPSILON))

    count = len(rows)
    return brier_total / count, log_loss_total / count


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    matches = sorted(
        load_football_data(DATA_PATH),
        key=lambda m: (m.match_date, m.home_team, m.away_team),
    )

    model_rows = {prior: [] for prior in PRIORS_TO_TEST}
    frequency_rows = []
    venue_rows = []
    report_rows = []
    skipped = 0

    for match in matches:
        # Exclude same-day matches so no result from the current match date
        # can leak into predictions.
        history = [m for m in matches if m.match_date < match.match_date]

        if len(history) < MIN_HISTORY:
            skipped += 1
            continue

        known_teams = {
            team for m in history for team in (m.home_team, m.away_team)
        }
        if match.home_team not in known_teams or match.away_team not in known_teams:
            skipped += 1
            continue

        try:
            predictions = {
                prior: predict_match(
                    history,
                    match.home_team,
                    match.away_team,
                    match.match_date,
                    prior_matches=prior,
                )
                for prior in PRIORS_TO_TEST
            }
            frequency_probs = benchmark_probabilities(history)
            venue_probs = venue_poisson_probabilities(history)
        except ValueError as exc:
            print(
                f"Skipped {match.match_date} {match.home_team} "
                f"vs {match.away_team}: {exc}"
            )
            skipped += 1
            continue

        outcome = actual_outcome(match)

        for prior, prediction in predictions.items():
            model_rows[prior].append(
                (model_probabilities(prediction), outcome)
            )

        frequency_rows.append((frequency_probs, outcome))
        venue_rows.append((venue_probs, outcome))

        selected = predictions[REPORT_PRIOR]
        selected_probs = model_probabilities(selected)

        row = {
            "match_date": match.match_date.isoformat(),
            "home_team": match.home_team,
            "away_team": match.away_team,
            "actual_home_goals": match.home_goals,
            "actual_away_goals": match.away_goals,
            "actual_outcome": outcome_label(outcome),
        }

        for prior, prediction in predictions.items():
            prefix = f"prior_{int(prior)}"
            probs = model_probabilities(prediction)
            row.update({
                f"{prefix}_home_win": probs[0],
                f"{prefix}_draw": probs[1],
                f"{prefix}_away_win": probs[2],
                f"{prefix}_home_xg": prediction.home_xg,
                f"{prefix}_away_xg": prediction.away_xg,
            })

        row.update({
            "frequency_home_win": frequency_probs[0],
            "frequency_draw": frequency_probs[1],
            "frequency_away_win": frequency_probs[2],
            "venue_poisson_home_win": venue_probs[0],
            "venue_poisson_draw": venue_probs[1],
            "venue_poisson_away_win": venue_probs[2],
            "home_xg": selected.home_xg,
            "away_xg": selected.away_xg,
            "model_home_win": selected_probs[0],
            "model_draw": selected_probs[1],
            "model_away_win": selected_probs[2],
        })
        report_rows.append(row)

    print(f"Dataset matches: {len(matches)}")
    print(f"Minimum prior matches: {MIN_HISTORY}")
    print(f"Evaluated matches: {len(report_rows)}")
    print(f"Skipped matches: {skipped}")

    if not report_rows:
        print("No eligible matches were available for evaluation.")
        return

    print("\n1X2 evaluation (lower is better)")
    for prior in PRIORS_TO_TEST:
        brier, loss = score_predictions(model_rows[prior])
        print(
            f"Team strength (prior {int(prior)}): "
            f"Brier={brier:.4f}, log loss={loss:.4f}"
        )

    for name, rows in (
        ("Historical-frequency benchmark", frequency_rows),
        ("Venue Poisson benchmark", venue_rows),
    ):
        brier, loss = score_predictions(rows)
        print(f"{name}: Brier={brier:.4f}, log loss={loss:.4f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nPrediction report saved to: {REPORT_PATH.relative_to(ROOT)}")
    print("Results cover one season and do not guarantee future performance.")


if __name__ == "__main__":
    main()
