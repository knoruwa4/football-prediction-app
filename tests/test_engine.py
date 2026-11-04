
import unittest
from math import isfinite

from football_prediction.engine import PoissonModel


class PoissonModelTests(unittest.TestCase):
    def setUp(self):
        self.model = PoissonModel()

    def test_market_probabilities_are_valid(self):
        r = self.model.predict(1.45, 1.05)

        values = [
            getattr(r, name)
            for name in (
                "home_win", "draw", "away_win",
                "double_chance_1x", "double_chance_x2",
                "double_chance_12", "over_1_5",
                "over_2_5", "under_2_5",
                "btts_yes", "btts_no",
            )
        ]

        self.assertTrue(all(isfinite(x) and 0 <= x <= 1 for x in values))
        self.assertAlmostEqual(
            r.home_win + r.draw + r.away_win, 1, places=10
        )
        self.assertAlmostEqual(
            r.over_2_5 + r.under_2_5, 1, places=12
        )
        self.assertAlmostEqual(
            r.btts_yes + r.btts_no, 1, places=12
        )

    def test_double_chance_is_derived_from_1x2(self):
        r = self.model.predict(1.4, 1.1)

        self.assertAlmostEqual(
            r.double_chance_1x, r.home_win + r.draw
        )
        self.assertAlmostEqual(
            r.double_chance_x2, r.draw + r.away_win
        )
        self.assertAlmostEqual(
            r.double_chance_12, r.home_win + r.away_win
        )

    def test_zero_goal_rates(self):
        r = self.model.predict(0, 0)

        self.assertAlmostEqual(r.home_win, 0)
        self.assertAlmostEqual(r.draw, 1)
        self.assertAlmostEqual(r.away_win, 0)
        self.assertAlmostEqual(r.over_1_5, 0)
        self.assertAlmostEqual(r.over_2_5, 0)
        self.assertAlmostEqual(r.btts_yes, 0)

    def test_symmetric_goal_rates(self):
        r = self.model.predict(1.2, 1.2)

        self.assertAlmostEqual(r.home_win, r.away_win)
        self.assertAlmostEqual(r.double_chance_1x, r.double_chance_x2)

    def test_invalid_goal_rates(self):
        invalid_cases = [
            ((-0.1, 1), ValueError),
            ((1, 20.1), ValueError),
            ((True, 1), TypeError),
            ((1, False), TypeError),
            ((float("nan"), 1), ValueError),
            ((1, float("nan")), ValueError),
            ((float("inf"), 1), ValueError),
            ((1, float("-inf")), ValueError),
            (("1.2", 1), TypeError),
        ]

        for args, expected_exception in invalid_cases:
            with self.subTest(args=args):
                with self.assertRaises(expected_exception):
                    self.model.predict(*args)

    def test_maximum_supported_goal_rates(self):
        r = self.model.predict(20.0, 20.0)

        self.assertAlmostEqual(
            r.home_win + r.draw + r.away_win, 1, places=10
        )
        self.assertTrue(0 <= r.home_win <= 1)
        self.assertTrue(0 <= r.draw <= 1)
        self.assertTrue(0 <= r.away_win <= 1)


if __name__ == "__main__":
    unittest.main()
