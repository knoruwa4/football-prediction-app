
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
        content = VALID_CSV.replace("Bournemouth,4,2,H", "Bournemouth,4,,H")
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

    def test_loads_supported_top_divisions(self):
        for code, competition in (
            ("SP1", "La Liga"),
            ("I1", "Serie A"),
            ("D1", "Bundesliga"),
            ("F1", "Ligue 1"),
        ):
            content = VALID_CSV.replace("E0,", f"{code},")
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                path = self.write_csv(folder, content)
                matches = load_football_data(path, expected_division_code=code)
            self.assertEqual(matches[0].competition, competition)

    def test_rejects_file_with_wrong_division_code(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, VALID_CSV)
            with self.assertRaisesRegex(ValueError, "does not match expected"):
                load_football_data(path, expected_division_code="SP1")

    def test_loads_windows_1252_team_names(self):
        content = (
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\r\n"
            "EC,03/10/20,King’s Lynn,Yeovil,2,2\r\n"
        ).replace("’", "\u2019").encode("cp1252")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "football_data.csv"
            path.write_bytes(content)
            matches = load_football_data(path)
        self.assertEqual(matches[0].home_team, "King’s Lynn")
        self.assertEqual(matches[0].match_date.isoformat(), "2020-10-03")

    def test_skips_unplayed_fixture_rows_and_accepts_unnamed_odds_columns(self):
        content = (
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,\r\n"
            "E0,01/08/25,Played,Opponent,1,0,,\r\n"
            "E0,02/08/25,Future,Opponent,,, ,\r\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_csv(folder, content)
            matches = load_football_data(path)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].home_team, "Played")


if __name__ == "__main__":
    unittest.main()
