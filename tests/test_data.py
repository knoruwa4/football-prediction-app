
import unittest
from datetime import date

from football_prediction.data import MatchResult


class MatchResultTests(unittest.TestCase):
    def make_result(self, **overrides):
        values = {
            "match_date": date(2025, 8, 15),
            "competition": "Example League",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_goals": 2,
            "away_goals": 1,
        }
        values.update(overrides)
        return MatchResult(**values)

    def test_valid_result_is_created(self):
        result = self.make_result()

        self.assertEqual(result.home_goals, 2)
        self.assertEqual(result.away_goals, 1)
        self.assertEqual(result.match_date, date(2025, 8, 15))

    def test_negative_goals_are_rejected(self):
        for field in ("home_goals", "away_goals"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.make_result(**{field: -1})

    def test_non_integer_goals_are_rejected(self):
        for value in (1.5, "2", True):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.make_result(home_goals=value)

    def test_empty_names_are_rejected(self):
        for field in ("competition", "home_team", "away_team"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.make_result(**{field: "  "})

    def test_identical_teams_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_result(away_team="Team A")

    def test_invalid_date_is_rejected(self):
        with self.assertRaises(TypeError):
            self.make_result(match_date="2025-08-15")


if __name__ == "__main__":
    unittest.main()
