
from datetime import date

import pytest

from football_prediction.batch_predictor import predict_fixtures
from football_prediction.data import MatchResult


def test_batch_predictor_returns_predictions_for_multiple_fixtures():
    matches = [
        MatchResult(
            home_team="Arsenal",
            away_team="Chelsea",
            home_goals=2,
            away_goals=0,
            match_date=date(2025, 1, 1),
            competition="EPL",
        ),
        MatchResult(
            home_team="Chelsea",
            away_team="Liverpool",
            home_goals=1,
            away_goals=1,
            match_date=date(2025, 1, 8),
            competition="EPL",
        ),
        MatchResult(
            home_team="Liverpool",
            away_team="Arsenal",
            home_goals=0,
            away_goals=2,
            match_date=date(2025, 1, 15),
            competition="EPL",
        ),
    ]

    fixtures = [
        ("Arsenal", "Chelsea"),
        ("Liverpool", "Arsenal"),
    ]

    results = predict_fixtures(
        matches=matches,
        fixtures=fixtures,
        match_date=date(2025, 2, 1),
        model_type="joint_attack_defence",
    )

    assert len(results) == 2

    for home_team, away_team, probabilities in results:
        assert (home_team, away_team) in fixtures
        assert 0 <= probabilities.home_win <= 1
        assert 0 <= probabilities.draw <= 1
        assert 0 <= probabilities.away_win <= 1
        assert probabilities.home_win + probabilities.draw + probabilities.away_win == pytest.approx(1.0)
