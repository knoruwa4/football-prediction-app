from datetime import date

import pytest

from football_prediction.data import MatchResult
from football_prediction.dixon_coles import DixonColesModel
from football_prediction.predictor import predict_match


def test_dixon_coles_returns_consistent_probability_markets():
    result = DixonColesModel(rho=-0.05).predict(1.4, 1.1)

    assert result.model == "dixon_coles_joint_poisson"
    assert result.home_win + result.draw + result.away_win == pytest.approx(1)
    assert result.double_chance_1x == pytest.approx(result.home_win + result.draw)
    assert result.over_2_5 + result.under_2_5 == pytest.approx(1)
    assert result.btts_yes + result.btts_no == pytest.approx(1)


def test_dixon_coles_zero_goal_rates_are_supported():
    result = DixonColesModel().predict(0, 0)

    assert result.draw == pytest.approx(1)
    assert result.home_win == pytest.approx(0)
    assert result.away_win == pytest.approx(0)


@pytest.mark.parametrize("rho", [-1, 1, float("nan"), True])
def test_dixon_coles_rejects_invalid_rho(rho):
    with pytest.raises((TypeError, ValueError)):
        DixonColesModel(rho)


def test_dixon_coles_predictor_excludes_same_day_results():
    history = [
        MatchResult(date(2025, 8, 1), "Test League", "A", "B", 2, 0),
        MatchResult(date(2025, 8, 2), "Test League", "C", "A", 1, 1),
        MatchResult(date(2025, 8, 3), "Test League", "B", "C", 0, 2),
    ]
    baseline = predict_match(history, "A", "B", date(2025, 8, 4), model_type="dixon_coles")
    same_day_result = MatchResult(date(2025, 8, 4), "Test League", "A", "B", 10, 0)
    with_same_day = predict_match(history + [same_day_result], "A", "B", date(2025, 8, 4), model_type="dixon_coles")

    assert baseline == with_same_day
