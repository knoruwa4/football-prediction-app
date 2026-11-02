from datetime import date, timedelta

import pytest

from football_prediction.data import MatchResult
from football_prediction.elo_model import EloAdjustedPoissonModel
from football_prediction.predictor import predict_match


def _history():
    start = date(2025, 1, 1)
    return [
        MatchResult(start + timedelta(days=i), "Test League", "A", "B", 2, 0)
        for i in range(4)
    ] + [
        MatchResult(start + timedelta(days=4), "Test League", "B", "C", 1, 1),
        MatchResult(start + timedelta(days=5), "Test League", "C", "A", 0, 1),
    ]


def test_elo_adjusted_model_returns_valid_poisson_markets():
    result = EloAdjustedPoissonModel(_history()).predict("A", "B")

    assert result.model == "elo_adjusted_joint_poisson"
    assert result.home_win + result.draw + result.away_win == pytest.approx(1)
    assert result.home_xg > 0
    assert result.away_xg > 0


def test_elo_predictor_excludes_same_day_results():
    history = _history()
    match_date = date(2025, 1, 8)
    baseline = predict_match(history, "A", "B", match_date, model_type="elo_poisson")
    same_day = MatchResult(match_date, "Test League", "A", "B", 0, 8)
    with_same_day = predict_match(history + [same_day], "A", "B", match_date, model_type="elo_poisson")

    assert baseline == with_same_day


def test_elo_model_rejects_invalid_training_configuration():
    with pytest.raises(ValueError, match="k_factor"):
        EloAdjustedPoissonModel(_history(), k_factor=0)
