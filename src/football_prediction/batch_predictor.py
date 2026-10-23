
"""Efficient batch predictions for fixtures on the same date."""

from datetime import date
from collections.abc import Sequence

from football_prediction.data import MatchResult
from football_prediction.engine import MatchProbabilities, PoissonModel
from football_prediction.joint_model import JointAttackDefenceModel
from football_prediction.predictor import predict_match


def predict_fixtures(
    matches: Sequence[MatchResult],
    fixtures: Sequence[tuple[str, str]],
    match_date: date,
    model_type: str = "joint_attack_defence",
    prior_matches: float = 50.0,
    half_life_days: float = 120.0,
    joint_half_life_days: float = 240.0,
    allow_unknown_teams: bool = False,
) -> list[tuple[str, str, MatchProbabilities]]:
    """Predict multiple fixtures using history strictly before match_date.

    Each fixture is a (home_team, away_team) tuple.
    The joint model is fitted only once for the entire batch.
    """
    if not isinstance(match_date, date):
        raise TypeError("match_date must be a datetime.date")

    if model_type not in {"team_strength", "joint_attack_defence"}:
        raise ValueError(
            "model_type must be 'team_strength' or 'joint_attack_defence'"
        )

    historical_matches = [
        match for match in matches if match.match_date < match_date
    ]

    if not historical_matches:
        raise ValueError(
            "no historical matches are available before match_date"
        )

    fixture_list = list(fixtures)
    if not fixture_list:
        return []

    results: list[tuple[str, str, MatchProbabilities]] = []

    if model_type == "joint_attack_defence":
        model = JointAttackDefenceModel(
            historical_matches,
            half_life_days=joint_half_life_days,
            allow_unknown_teams=allow_unknown_teams,
        )

        for home_team, away_team in fixture_list:
            probabilities = model.predict(home_team, away_team)
            results.append((home_team, away_team, probabilities))

        return results

    for home_team, away_team in fixture_list:
        probabilities = predict_match(
            matches=historical_matches,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            prior_matches=prior_matches,
            half_life_days=half_life_days,
            allow_unknown_teams=allow_unknown_teams,
            model_type="team_strength",
        )
        results.append((home_team, away_team, probabilities))

    return results
