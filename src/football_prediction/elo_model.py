"""Elo-adjusted expected goals as an experimental model candidate."""

from __future__ import annotations

from math import exp, isfinite, log
from collections.abc import Sequence

from football_prediction.data import MatchResult
from football_prediction.engine import MatchProbabilities, PoissonModel
from football_prediction.joint_model import JointAttackDefenceModel


class EloAdjustedPoissonModel:
    """Adjust joint Poisson rates with ratings fitted from prior results.

    Elo ratings use result-only updates. The rating difference modestly shifts
    the two goal rates in opposite directions while preserving their product.
    This is a candidate feature mapping and must be validated chronologically.
    """

    name = "elo_adjusted_joint_poisson"

    def __init__(
        self,
        matches: Sequence[MatchResult],
        *,
        k_factor: float = 20.0,
        initial_rating: float = 1500.0,
        regularization_scale: float = 800.0,
        home_advantage: float = 60.0,
        half_life_days: float = 240.0,
        allow_unknown_teams: bool = True,
    ) -> None:
        if not matches:
            raise ValueError("Training history must not be empty")
        for name, value in (("k_factor", k_factor), ("regularization_scale", regularization_scale)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not isfinite(initial_rating) or not isfinite(home_advantage):
            raise ValueError("initial_rating and home_advantage must be finite")
        self.home_advantage = float(home_advantage)
        self.regularization_scale = float(regularization_scale)
        self.ratings: dict[str, float] = {}

        for match in sorted(matches, key=lambda item: (item.match_date, item.home_team.casefold(), item.away_team.casefold())):
            home_rating = self.ratings.get(match.home_team, initial_rating)
            away_rating = self.ratings.get(match.away_team, initial_rating)
            expected_home = 1.0 / (1.0 + 10.0 ** (-(home_rating + self.home_advantage - away_rating) / 400.0))
            if match.home_goals > match.away_goals:
                actual_home = 1.0
            elif match.home_goals == match.away_goals:
                actual_home = 0.5
            else:
                actual_home = 0.0
            change = float(k_factor) * (actual_home - expected_home)
            self.ratings[match.home_team] = home_rating + change
            self.ratings[match.away_team] = away_rating - change

        self.goal_model = JointAttackDefenceModel(
            list(matches),
            half_life_days=half_life_days,
            allow_unknown_teams=allow_unknown_teams,
        )
        self.allow_unknown_teams = allow_unknown_teams

    def predict(self, home_team: str, away_team: str) -> MatchProbabilities:
        if home_team == away_team:
            raise ValueError("home_team and away_team must be different")
        unknown = {home_team, away_team} - self.ratings.keys()
        if unknown and not self.allow_unknown_teams:
            raise ValueError("unknown team(s) in training data: " + ", ".join(sorted(unknown)))

        baseline = self.goal_model.predict(home_team, away_team)
        difference = self.ratings.get(home_team, 1500.0) - self.ratings.get(away_team, 1500.0)
        multiplier = exp(log(10.0) * difference / self.regularization_scale)
        adjustment = multiplier ** 0.5
        home_xg = min(20.0, max(0.0, baseline.home_xg * adjustment))
        away_xg = min(20.0, max(0.0, baseline.away_xg / adjustment))
        result = PoissonModel().predict(home_xg, away_xg)
        from dataclasses import replace
        return replace(result, model=self.name)
