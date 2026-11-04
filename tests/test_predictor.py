
import unittest
from datetime import date

from football_prediction.data import MatchResult
from football_prediction.predictor import predict_match


class PredictMatchTests(unittest.TestCase):
    def setUp(self):
        self.matches = [
            MatchResult(date(2025, 8, 1), "Test League", "A", "B", 2, 0),
            MatchResult(date(2025, 8, 2), "Test League", "C", "A", 1, 1),
            MatchResult(date(2025, 8, 3), "Test League", "B", "C", 0, 2),
            MatchResult(date(2025, 8, 4), "Test League", "A", "C", 1, 0),
            MatchResult(date(2025, 8, 5), "Test League", "B", "A", 1, 2),
            MatchResult(date(2025, 8, 6), "Test League", "C", "B", 2, 1),
        ]

    def test_returns_valid_market_probabilities(self):
        result = predict_match(
            self.matches, "A", "B", date(2025, 8, 7)
        )

        self.assertAlmostEqual(
            result.home_win + result.draw + result.away_win, 1.0
        )
        self.assertAlmostEqual(result.over_2_5 + result.under_2_5, 1.0)
        self.assertGreater(result.home_xg, 0)
        self.assertGreater(result.away_xg, 0)

    def test_does_not_use_same_day_or_future_results(self):
        baseline = predict_match(
            self.matches, "A", "B", date(2025, 8, 7)
        )

        # This result must not affect a prediction for 7 August.
        extra = MatchResult(
            date(2025, 8, 7), "Test League", "A", "B", 10, 0
        )
        with_extra = predict_match(
            self.matches + [extra], "A", "B", date(2025, 8, 7)
        )

        self.assertEqual(baseline, with_extra)

    def test_rejects_no_prior_history(self):
        with self.assertRaisesRegex(ValueError, "no historical matches"):
            predict_match([], "A", "B", date(2025, 8, 7))

    def test_rejects_unknown_team_by_default(self):
        with self.assertRaisesRegex(ValueError, "unknown team"):
            predict_match(
                self.matches,
                "A",
                "Promoted FC",
                date(2025, 8, 7),
            )

    def test_allows_unknown_team_when_enabled(self):
        result = predict_match(
            self.matches,
            "A",
            "Promoted FC",
            date(2025, 8, 7),
            allow_unknown_teams=True,
        )

        self.assertAlmostEqual(
            result.home_win + result.draw + result.away_win, 1.0
        )
        self.assertGreater(result.home_xg, 0)
        self.assertGreater(result.away_xg, 0)


if __name__ == "__main__":
    unittest.main()
