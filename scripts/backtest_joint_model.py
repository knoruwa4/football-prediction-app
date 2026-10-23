
"""Compare the current model with a regularized joint attack/defence model.

Each season is predicted using only results available before that season
begins. This gives both models the same historical information.
"""

import csv
from collections import defaultdict
from datetime import date
from math import exp, log
from pathlib import Path

from football_prediction.engine import PoissonModel
from football_prediction.football_data import load_football_data
from football_prediction.predictor import predict_match

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_PATH = ROOT / "reports" / "joint_model_comparison.csv"

PRIOR_MATCHES = 50.0
HALF_LIFE_DAYS = 240.0
REGULARIZATION = 0.5
EPOCHS = 300
LEARNING_RATE = 0.025
EPSILON = 1e-15


class JointAttackDefenceModel:
    """Regularized Poisson model with team attack and defence effects."""

    def __init__(self, matches):
        if not matches:
            raise ValueError("Training history must not be empty")

        self.teams = sorted({
            team
            for match in matches
            for team in (match.home_team, match.away_team)
        })
        self.team_index = {
            team: index for index, team in enumerate(self.teams)
        }
        n = len(self.teams)

        latest_date = max(match.match_date for match in matches)
        weighted = []

        for match in matches:
            age = (latest_date - match.match_date).days
            weight = 0.5 ** (age / HALF_LIFE_DAYS)
            weighted.append((match, weight))

        total_weight = sum(weight for _, weight in weighted)
        home_goals = sum(
            match.home_goals * weight for match, weight in weighted
        )
        away_goals = sum(
            match.away_goals * weight for match, weight in weighted
        )

        # Separate league scoring baselines for home and away teams.
        self.home_intercept = log(max(home_goals / total_weight, 0.05))
        self.away_intercept = log(max(away_goals / total_weight, 0.05))

        # Positive attack means stronger scoring; positive defence means
        # better defending, so it reduces the opponent's expected goals.
        attack = [0.0] * n
        defence = [0.0] * n

        # Adam optimizer state.
        parameters = [
            self.home_intercept,
            self.away_intercept,
            *attack,
            *defence,
        ]
        first_moment = [0.0] * len(parameters)
        second_moment = [0.0] * len(parameters)
        beta1, beta2 = 0.9, 0.999

        for epoch in range(1, EPOCHS + 1):
            gradients = [0.0] * len(parameters)

            for match, weight in weighted:
                hi = self.team_index[match.home_team]
                ai = self.team_index[match.away_team]

                eta_home = (
                    parameters[0] + parameters[2 + hi]
                    - parameters[2 + n + ai]
                )
                eta_away = (
                    parameters[1] + parameters[2 + ai]
                    - parameters[2 + n + hi]
                )

                # Protect the exponential against numerical overflow.
                lambda_home = exp(max(-4.0, min(3.0, eta_home)))
                lambda_away = exp(max(-4.0, min(3.0, eta_away)))

                gh = (lambda_home - match.home_goals) * weight
                ga = (lambda_away - match.away_goals) * weight

                gradients[0] += gh
                gradients[1] += ga
                gradients[2 + hi] += gh
                gradients[2 + n + ai] -= gh
                gradients[2 + ai] += ga
                gradients[2 + n + hi] -= ga

            # Average the likelihood gradient, then shrink team effects.
            for index in range(len(gradients)):
                gradients[index] /= total_weight

            for index in range(2, len(parameters)):
                gradients[index] += REGULARIZATION * parameters[index]

            # Adam update.
            for index, gradient in enumerate(gradients):
                first_moment[index] = (
                    beta1 * first_moment[index]
                    + (1.0 - beta1) * gradient
                )
                second_moment[index] = (
                    beta2 * second_moment[index]
                    + (1.0 - beta2) * gradient * gradient
                )

                m_hat = first_moment[index] / (1.0 - beta1 ** epoch)
                v_hat = second_moment[index] / (1.0 - beta2 ** epoch)

                parameters[index] -= (
                    LEARNING_RATE * m_hat / (v_hat ** 0.5 + 1e-8)
                )

        self.home_intercept = parameters[0]
        self.away_intercept = parameters[1]
        self.attack = {
            team: parameters[2 + index]
            for team, index in self.team_index.items()
        }
        self.defence = {
            team: parameters[2 + n + index]
            for team, index in self.team_index.items()
        }

    def predict(self, home_team, away_team):
        home_attack = self.attack.get(home_team, 0.0)
        away_attack = self.attack.get(away_team, 0.0)
        home_defence = self.defence.get(home_team, 0.0)
        away_defence = self.defence.get(away_team, 0.0)

        home_eta = self.home_intercept + home_attack - away_defence
        away_eta = self.away_intercept + away_attack - home_defence

        # Keep expected goals within the existing engine's accepted range.
        home_xg = min(20.0, max(0.0, exp(max(-4.0, min(3.0, home_eta)))))
        away_xg = min(20.0, max(0.0, exp(max(-4.0, min(3.0, away_eta)))))

        return PoissonModel().predict(home_xg, away_xg)


def outcome_of(match):
    if match.home_goals > match.away_goals:
        return 0
    if match.home_goals == match.away_goals:
        return 1
    return 2


def probabilities(prediction):
    return [
        prediction.home_win,
        prediction.draw,
        prediction.away_win,
    ]


def score(rows):
    if not rows:
        return None, None

    brier = 0.0
    log_loss = 0.0

    for probs, actual in rows:
        brier += sum(
            (p - (1.0 if i == actual else 0.0)) ** 2
            for i, p in enumerate(probs)
        )
        log_loss -= log(max(probs[actual], EPSILON))

    return brier / len(rows), log_loss / len(rows)


def main():
    files = sorted(DATA_DIR.glob("epl_*.csv"))
    if not files:
        raise FileNotFoundError(f"No EPL CSV files in {DATA_DIR}")

    matches = []
    season_matches = defaultdict(list)

    for path in files:
        loaded = load_football_data(path)
        season_matches[path.name].extend(loaded)
        matches.extend(loaded)

    matches.sort(key=lambda m: (
        m.match_date, m.home_team, m.away_team
    ))

    rows = []

    # Each season uses only matches dated before that season's first fixture.
    for season_file in sorted(season_matches):
        fixtures = sorted(
            season_matches[season_file],
            key=lambda m: (m.match_date, m.home_team, m.away_team),
        )
        if not fixtures:
            continue

        season_start = min(match.match_date for match in fixtures)
        history = [
            match for match in matches
            if match.match_date < season_start
        ]

        if len(history) < 60:
            print(
                f"Skipping {season_file}: only {len(history)} "
                "pre-season matches"
            )
            continue

        print(
            f"Training for {season_file}: "
            f"{len(history)} historical matches"
        )

        joint_model = JointAttackDefenceModel(history)

        for match in fixtures:
            # Current model is also frozen at the start of the season.
            current = predict_match(
                history,
                match.home_team,
                match.away_team,
                match.match_date,
                prior_matches=PRIOR_MATCHES,
                half_life_days=120.0,
                allow_unknown_teams=True,
            )

            joint = joint_model.predict(
                match.home_team, match.away_team
            )

            rows.append({
                "date": match.match_date.isoformat(),
                "season_file": season_file,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_outcome": ("H", "D", "A")[outcome_of(match)],
                "current_home": current.home_win,
                "current_draw": current.draw,
                "current_away": current.away_win,
                "joint_home": joint.home_win,
                "joint_draw": joint.draw,
                "joint_away": joint.away_win,
                "_actual": outcome_of(match),
            })

    if not rows:
        print("No fixtures could be evaluated.")
        return

    def report(label, selected):
        print(f"\n{label}: {len(selected)} matches")
        for name, keys in (
            ("Current team-strength model",
             ("current_home", "current_draw", "current_away")),
            ("Joint attack/defence model",
             ("joint_home", "joint_draw", "joint_away")),
        ):
            values = [
                ([row[key] for key in keys], row["_actual"])
                for row in selected
            ]
            brier, loss = score(values)
            print(f"  {name}: Brier={brier:.4f}, log loss={loss:.4f}")

    report("Overall", rows)

    for season in sorted({row["season_file"] for row in rows}):
        report(
            season,
            [row for row in rows if row["season_file"] == season],
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [key for key in rows[0] if not key.startswith("_")]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in fields}
            for row in rows
        )

    print(f"\nReport saved to: {REPORT_PATH.relative_to(ROOT)}")
    print("Lower Brier and log-loss scores are better.")
    print(
        "Note: this experiment freezes each model at season start; "
        "compare its scores only against the other model in this report."
    )


if __name__ == "__main__":
    main()
