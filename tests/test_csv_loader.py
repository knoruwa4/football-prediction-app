
import tempfile
import unittest
from pathlib import Path

from football_prediction.csv_loader import load_matches


VALID_CSV = """match_date,competition,home_team,away_team,home_goals,away_goals
2025-08-15,Example League,Team A,Team B,2,1
2025-08-16,Example League,Team C,Team D,0,0
"""


class CsvLoaderTests(unittest.TestCase):
    def write_csv(self, folder, content):
        path = Path(folder) / "matches.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_valid_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, VALID_CSV)

            matches = load_matches(path)

            self.assertEqual(len(matches), 2)
            self.assertEqual(matches[0].home_team, "Team A")
            self.assertEqual(matches[0].home_goals, 2)
            self.assertEqual(matches[1].away_goals, 0)

    def test_rejects_missing_required_column(self):
        content = (
            "match_date,competition,home_team,away_team,home_goals\n"
            "2025-08-15,Example League,Team A,Team B,2\n"
        )

        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_matches(path)

    def test_rejects_invalid_date_with_row_number(self):
        content = VALID_CSV.replace("2025-08-15", "not-a-date")

        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)

            with self.assertRaisesRegex(ValueError, "row 2"):
                load_matches(path)

    def test_rejects_negative_goals_with_row_number(self):
        content = VALID_CSV.replace(",2,1", ",-2,1")

        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)

            with self.assertRaisesRegex(ValueError, "row 2"):
                load_matches(path)

    def test_rejects_non_integer_goals(self):
        content = VALID_CSV.replace(",2,1", ",two,1")

        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)

            with self.assertRaisesRegex(ValueError, "row 2"):
                load_matches(path)


if __name__ == "__main__":
    unittest.main()
