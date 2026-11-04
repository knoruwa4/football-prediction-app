
"""Regularized joint attack/defence Poisson model."""

from math import exp, log

from football_prediction.engine import PoissonModel


class JointAttackDefenceModel:
    """Estimate team attack and defence jointly from historical results."""

    name = "joint_attack_defence_poisson"

    def __init__(
        self,
        matches,
        half_life_days: float = 240.0,
        regularization: float = 0.5,
        epochs: int = 300,
        learning_rate: float = 0.025,
        allow_unknown_teams: bool = True,
    ):
        if not matches:
            raise ValueError("Training history must not be empty")
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if regularization < 0:
            raise ValueError("regularization must not be negative")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        competitions = {match.competition for match in matches}
        if len(competitions) != 1:
            raise ValueError("all matches must belong to one competition")

        self.teams = sorted({
            team
            for match in matches
            for team in (match.home_team, match.away_team)
        })
        self.team_index = {
            team: index for index, team in enumerate(self.teams)
        }
        self.allow_unknown_teams = allow_unknown_teams
        n = len(self.teams)

        latest_date = max(match.match_date for match in matches)
        weighted = []

        for match in matches:
            age = (latest_date - match.match_date).days
            weight = 0.5 ** (age / half_life_days)
            weighted.append((match, weight))

        total_weight = sum(weight for _, weight in weighted)
        home_goals = sum(
            match.home_goals * weight for match, weight in weighted
        )
        away_goals = sum(
            match.away_goals * weight for match, weight in weighted
        )

        parameters = [
            log(max(home_goals / total_weight, 0.05)),
            log(max(away_goals / total_weight, 0.05)),
            *([0.0] * n),
            *([0.0] * n),
        ]
        first_moment = [0.0] * len(parameters)
        second_moment = [0.0] * len(parameters)
        beta1, beta2 = 0.9, 0.999

        for epoch in range(1, epochs + 1):
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

            for index in range(len(gradients)):
                gradients[index] /= total_weight

            for index in range(2, len(parameters)):
                gradients[index] += (
                    regularization * parameters[index]
                )

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
                    learning_rate * m_hat / (v_hat ** 0.5 + 1e-8)
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

    def predict(self, home_team: str, away_team: str):
        """Return match probabilities for the requested fixture."""
        if not isinstance(home_team, str) or not home_team.strip():
            raise ValueError("home_team must be a non-empty string")
        if not isinstance(away_team, str) or not away_team.strip():
            raise ValueError("away_team must be a non-empty string")

        home_team = home_team.strip()
        away_team = away_team.strip()

        if home_team == away_team:
            raise ValueError("home_team and away_team must be different")

        unknown = {home_team, away_team} - set(self.teams)
        if unknown and not self.allow_unknown_teams:
            raise ValueError(
                "unknown team(s) in training data: "
                + ", ".join(sorted(unknown))
            )

        home_eta = (
            self.home_intercept
            + self.attack.get(home_team, 0.0)
            - self.defence.get(away_team, 0.0)
        )
        away_eta = (
            self.away_intercept
            + self.attack.get(away_team, 0.0)
            - self.defence.get(home_team, 0.0)
        )

        home_xg = exp(max(-4.0, min(3.0, home_eta)))
        away_xg = exp(max(-4.0, min(3.0, away_eta)))

        result = PoissonModel().predict(home_xg, away_xg)

        # Keep the result's model identifier distinct from plain Poisson.
        from dataclasses import replace
        return replace(result, model=self.name)
