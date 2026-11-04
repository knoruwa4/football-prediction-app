"""Independent Poisson baseline; expected goals are supplied by a validated upstream model."""
from __future__ import annotations
from dataclasses import dataclass
from math import exp
from typing import Protocol

class PredictionModel(Protocol):
    name: str
    def predict(self, home_xg: float, away_xg: float) -> "MatchProbabilities": ...

@dataclass(frozen=True)
class MatchProbabilities:
    model: str
    home_xg: float
    away_xg: float
    home_win: float
    draw: float
    away_win: float
    double_chance_1x: float
    double_chance_x2: float
    double_chance_12: float
    over_1_5: float
    over_2_5: float
    under_2_5: float
    btts_yes: float
    btts_no: float

class PoissonModel:
    """Independent goal model; score tails are truncated below 1e-12."""
    name = "independent_poisson"
    _TAIL_TOLERANCE = 1e-12
    _MAX_EXPECTED_GOALS = 20.0

    def predict(self, home_xg: float, away_xg: float) -> MatchProbabilities:
        self._validate_xg(home_xg, "home_xg")
        self._validate_xg(away_xg, "away_xg")
        home, away = self._distribution(home_xg), self._distribution(away_xg)
        hw = dr = aw = 0.0
        for hg, hp in enumerate(home):
            for ag, ap in enumerate(away):
                p = hp * ap
                if hg > ag: hw += p
                elif hg == ag: dr += p
                else: aw += p
        o15, o25 = self._over_probability(home, away, 1), self._over_probability(home, away, 2)
        btts = (1.0 - home[0]) * (1.0 - away[0])
        return MatchProbabilities(self.name, home_xg, away_xg, hw, dr, aw,
            hw + dr, dr + aw, hw + aw, o15, o25, 1.0 - o25, btts, 1.0 - btts)

    @classmethod
    def _validate_xg(cls, value: float, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be a number")
        if not 0.0 <= value <= cls._MAX_EXPECTED_GOALS:
            raise ValueError(f"{field} must be between 0 and {cls._MAX_EXPECTED_GOALS}")

    @classmethod
    def _distribution(cls, mean: float) -> list[float]:
        p = exp(-mean)
        result, cumulative, goals = [p], p, 0
        while cumulative < 1.0 - cls._TAIL_TOLERANCE:
            goals += 1
            p *= mean / goals
            result.append(p)
            cumulative += p
        return result

    @staticmethod
    def _over_probability(home: list[float], away: list[float], threshold: int) -> float:
        p_under = sum(hp * ap for hg, hp in enumerate(home)
                      for ag, ap in enumerate(away) if hg + ag <= threshold)
        return 1.0 - p_under
