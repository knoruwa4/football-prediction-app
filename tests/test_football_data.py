
import tempfile
import unittest
from pathlib import Path

from football_prediction.football_data import load_football_data


VALID_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR
E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H
E0,16/08/2025,12:30,Aston Villa,Newcastle,0,0,D
"""


class FootballDataTests(unittest.TestCase):
    def write_csv(self, folder, content):
        path = Path(folder) / "football_data.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_valid_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, VALID_CSV)

            matches = load_football_data(path)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].competition, "Premier League")
        self.assertEqual(matches[0].match_date.isoformat(), "2025-08-15")
        self.assertEqual(matches[0].home_team, "Liverpool")
        self.assertEqual(matches[0].home_goals, 4)
        self.assertEqual(matches[0].away_goals, 2)

    def test_accepts_zero_zero_score(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, VALID_CSV)

            matches = load_football_data(path)

        self.assertEqual((matches[1].home_goals, matches[1].away_goals), (0, 0))

    def test_rejects_missing_score(self):
        content = VALID_CSV.replace("Bournemouth,4,2,H", "Bournemouth,,,")
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)

            with self.assertRaisesRegex(ValueError, "row 2"):
                load_football_data(path)

    def test_rejects_invalid_date(self):
        content = VALID_CSV.replace("15/08/2025", "not-a-date")
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)

            with self.assertRaisesRegex(ValueError, "row 2"):
                load_football_data(path)

    def test_rejects_negative_goals(self):
        content = VALID_CSV.replace("Liverpool,Bournemouth,4,2,H",
                                    "Liverpool,Bournemouth,-1,2,A")
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)

            with self.assertRaisesRegex(ValueError, "row 2"):
                load_football_data(path)

    def test_rejects_unsupported_competition(self):
        content = VALID_CSV.replace("E0,15/08/2025", "XX,15/08/2025")
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)

            with self.assertRaisesRegex(ValueError, "unsupported competition"):
                load_football_data(path)


if __name__ == "__main__":
    unittest.main()
