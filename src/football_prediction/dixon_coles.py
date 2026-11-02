"""Dixon–Coles low-score correction over independent Poisson goal rates."""

from __future__ import annotations

from dataclasses import replace
from math import exp

from football_prediction.engine import MatchProbabilities, PoissonModel


class DixonColesModel:
    """Apply the Dixon–Coles correlation correction to two expected goals.

    ``rho`` is an explicit candidate parameter, not fitted from the match
    being predicted. Select it only on a separate chronological validation
    period before assessing a final held-out period.
    """

    name = "dixon_coles_joint_poisson"
    _TAIL_TOLERANCE = 1e-12

    def __init__(self, rho: float = -0.05) -> None:
        if isinstance(rho, bool) or not isinstance(rho, (int, float)):
            raise TypeError("rho must be a number")
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must be strictly between -1 and 1")
        self.rho = float(rho)

    def predict(self, home_xg: float, away_xg: float) -> MatchProbabilities:
        """Return corrected 1X2, goal-total, and BTTS probabilities."""
        baseline = PoissonModel().predict(home_xg, away_xg)
        home = self._distribution(home_xg)
        away = self._distribution(away_xg)
        matrix = [[hp * ap for ap in away] for hp in home]

        corrections = {
            (0, 0): 1.0 - home_xg * away_xg * self.rho,
            (0, 1): 1.0 + home_xg * self.rho,
            (1, 0): 1.0 + away_xg * self.rho,
            (1, 1): 1.0 - self.rho,
        }
        for (home_goals, away_goals), factor in corrections.items():
            if factor < 0:
                raise ValueError(
                    "rho and expected goals produce a negative score probability"
                )
            matrix[home_goals][away_goals] *= factor

        total = sum(sum(row) for row in matrix)
        if total <= 0:
            raise ValueError("corrected score probabilities have no mass")
        matrix = [[probability / total for probability in row] for row in matrix]

        home_win = draw = away_win = 0.0
        over_1_5 = over_2_5 = btts_yes = 0.0
        for home_goals, row in enumerate(matrix):
            for away_goals, probability in enumerate(row):
                if home_goals > away_goals:
                    home_win += probability
                elif home_goals == away_goals:
                    draw += probability
                else:
                    away_win += probability
                if home_goals + away_goals > 1:
                    over_1_5 += probability
                if home_goals + away_goals > 2:
                    over_2_5 += probability
                if home_goals > 0 and away_goals > 0:
                    btts_yes += probability

        return replace(
            baseline,
            model=self.name,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            double_chance_1x=home_win + draw,
            double_chance_x2=draw + away_win,
            double_chance_12=home_win + away_win,
            over_1_5=over_1_5,
            over_2_5=over_2_5,
            under_2_5=1.0 - over_2_5,
            btts_yes=btts_yes,
            btts_no=1.0 - btts_yes,
        )

    @classmethod
    def _distribution(cls, expected_goals: float) -> list[float]:
        probability = exp(-expected_goals)
        probabilities = [probability]
        cumulative = probability
        goals = 0
        while goals < 1 or cumulative < 1.0 - cls._TAIL_TOLERANCE:
            goals += 1
            probability *= expected_goals / goals
            probabilities.append(probability)
            cumulative += probability
        return probabilities
