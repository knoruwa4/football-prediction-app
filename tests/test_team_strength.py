
import unittest
from datetime import date
from math import isfinite

from football_prediction.data import MatchResult
from football_prediction.team_strength import TeamStrengthModel


class TeamStrengthModelTests(unittest.TestCase):
    def setUp(self):
        self.matches = [
            MatchResult(date(2025, 8, 1), "Test League", "A", "B", 2, 0),
            MatchResult(date(2025, 8, 2), "Test League", "C", "A", 1, 1),
            MatchResult(date(2025, 8, 3), "Test League", "B", "C", 0, 2),
            MatchResult(date(2025, 8, 4), "Test League", "A", "C", 1, 0),
            MatchResult(date(2025, 8, 5), "Test League", "B", "A", 1, 2),
            MatchResult(date(2025, 8, 6), "Test League", "C", "B", 2, 1),
        ]
        self.model = TeamStrengthModel(self.matches)

    def test_returns_positive_finite_expected_goals(self):
        result = self.model.predict("A", "B", date(2025, 8, 7))

        self.assertTrue(isfinite(result.home_xg))
        self.assertTrue(isfinite(result.away_xg))
        self.assertGreater(result.home_xg, 0)
        self.assertGreater(result.away_xg, 0)

    def test_rejects_fixture_date_that_could_leak_future_data(self):
        with self.assertRaisesRegex(ValueError, "after the latest"):
            self.model.predict("A", "B", date(2025, 8, 6))

    def test_rejects_unknown_team(self):
        with self.assertRaisesRegex(ValueError, "unknown team"):
            self.model.predict("A", "Unknown", date(2025, 8, 7))

    def test_rejects_same_team(self):
        with self.assertRaisesRegex(ValueError, "must be different"):
            self.model.predict("A", "A", date(2025, 8, 7))

    def test_rejects_empty_training_data(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            TeamStrengthModel([])

    def test_rejects_mixed_competitions(self):
        matches = self.matches + [
            MatchResult(
                date(2025, 8, 7), "Other League", "A", "B", 1, 0
            )
        ]
        with self.assertRaisesRegex(ValueError, "one competition"):
            TeamStrengthModel(matches)

    def test_unknown_team_is_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, "unknown team"):
            self.model.predict(
                "A", "Promoted FC", date(2025, 8, 7)
            )

    def test_unknown_team_can_use_neutral_strengths(self):
        model = TeamStrengthModel(
            self.matches,
            allow_unknown_teams=True,
        )

        result = model.predict(
            "A", "Promoted FC", date(2025, 8, 7)
        )

        self.assertAlmostEqual(
            result.away_xg,
            model.league_away_goals * model.home_defence["A"],
        )
        self.assertGreater(result.home_xg, 0)
        self.assertGreater(result.away_xg, 0)

    def test_neutral_strengths_work_for_unknown_home_team(self):
        model = TeamStrengthModel(
            self.matches,
            allow_unknown_teams=True,
        )

        result = model.predict(
            "Promoted FC", "A", date(2025, 8, 7)
        )

        self.assertAlmostEqual(
            result.home_xg,
            model.league_home_goals * model.away_defence["A"],
        )
        self.assertGreater(result.home_xg, 0)
        self.assertGreater(result.away_xg, 0)


if __name__ == "__main__":
    unittest.main()
