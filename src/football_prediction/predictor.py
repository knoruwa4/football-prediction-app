
"""Combine historical team strengths with Poisson probability models."""

from datetime import date
from collections.abc import Sequence

from football_prediction.data import MatchResult
from football_prediction.engine import MatchProbabilities, PoissonModel
from football_prediction.joint_model import JointAttackDefenceModel
from football_prediction.team_strength import TeamStrengthModel


def predict_match(
    matches: Sequence[MatchResult],
    home_team: str,
    away_team: str,
    match_date: date,
    prior_matches: float = 50.0,
    half_life_days: float = 120.0,
    allow_unknown_teams: bool = False,
    model_type: str = "team_strength",
    joint_half_life_days: float = 240.0,
    competition: str | None = None,
) -> MatchProbabilities:
    """Predict a fixture using only results before its match date.

    model_type:
        "team_strength" uses the existing model.
        "joint_attack_defence" uses the experimental joint model.
    """
    if not isinstance(match_date, date):
        raise TypeError("match_date must be a datetime.date")

    if model_type not in {"team_strength", "joint_attack_defence"}:
        raise ValueError(
            "model_type must be 'team_strength' or 'joint_attack_defence'"
        )

    historical_matches = [
        match
        for match in matches
        if match.match_date < match_date
        and (competition is None or match.competition == competition)
    ]

    if not historical_matches:
        raise ValueError(
            "no historical matches are available before match_date"
        )

    if model_type == "joint_attack_defence":
        return JointAttackDefenceModel(
            list(historical_matches),
            half_life_days=joint_half_life_days,
            allow_unknown_teams=allow_unknown_teams,
        ).predict(home_team, away_team)

    strength_model = TeamStrengthModel(
        list(historical_matches),
        prior_matches=prior_matches,
        half_life_days=half_life_days,
        allow_unknown_teams=allow_unknown_teams,
    )

    expected_goals = strength_model.predict(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
    )

    return PoissonModel().predict(
        home_xg=expected_goals.home_xg,
        away_xg=expected_goals.away_xg,
    )
