
"""Import completed match results from Football-Data.co.uk CSV files."""

import csv
from datetime import datetime
from io import StringIO
from pathlib import Path

from football_prediction.data import MatchResult


REQUIRED_COLUMNS = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}

COMPETITIONS = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Primeira Liga",
    "T1": "Turkish Super Lig",
    "G1": "Greek Super League",
    "ARG": "Argentina Primera Division",
    "AUT": "Austrian Bundesliga",
    "BRA": "Brazilian Serie A",
    "CHN": "Chinese Super League",
    "DNK": "Danish Superliga",
    "FIN": "Finnish Veikkausliiga",
    "IRL": "Irish Premier Division",
    "JPN": "Japanese J1 League",
    "MEX": "Mexican Liga MX",
    "NOR": "Norwegian Eliteserien",
    "POL": "Polish Ekstraklasa",
    "ROU": "Romanian Liga 1",
    "RUS": "Russian Premier League",
    "SWE": "Swedish Allsvenskan",
    "SWZ": "Swiss Super League",
    "USA": "Major League Soccer",
}


def load_football_data(
    path: str | Path, expected_division_code: str | None = None
) -> list[MatchResult]:
    """Load and validate completed matches from a Football-Data CSV."""
    csv_path = Path(path)

    if expected_division_code is not None and expected_division_code not in COMPETITIONS:
        raise ValueError(
            f"unsupported competition code: {expected_division_code!r}"
        )

    content = csv_path.read_bytes()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Football-Data files have historically mixed UTF-8 and Windows-1252.
        text = content.decode("cp1252")

    with StringIO(text, newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file is empty or missing a header")

        headers = [header.strip() if header else "" for header in reader.fieldnames]
        # Some provider files have unnamed, unused trailing odds columns.
        headers = [
            header if header else f"_unnamed_{index}"
            for index, header in enumerate(headers)
        ]

        if len(set(headers)) != len(headers):
            raise ValueError("CSV contains duplicate column names")

        missing = REQUIRED_COLUMNS - set(headers)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        # Normalize header whitespace before reading rows.
        reader.fieldnames = headers
        matches: list[MatchResult] = []

        for row_number, row in enumerate(reader, start=2):
            try:
                division = row["Div"].strip()
                if division not in COMPETITIONS:
                    raise ValueError(
                        f"unsupported competition code: {division!r}"
                    )
                if (
                    expected_division_code is not None
                    and division != expected_division_code
                ):
                    raise ValueError(
                        f"competition code {division!r} does not match "
                        f"expected {expected_division_code!r}"
                    )

                raw_date = row["Date"].strip()
                try:
                    match_date = datetime.strptime(raw_date, "%d/%m/%Y").date()
                except ValueError:
                    match_date = datetime.strptime(raw_date, "%d/%m/%y").date()

                raw_home_goals = row["FTHG"].strip()
                raw_away_goals = row["FTAG"].strip()
                if not raw_home_goals and not raw_away_goals:
                    # Fixtures without a full-time score are not match results.
                    continue
                if not raw_home_goals or not raw_away_goals:
                    raise ValueError("only one full-time score is present")
                home_goals = int(raw_home_goals)
                away_goals = int(raw_away_goals)

                match = MatchResult(
                    match_date=match_date,
                    competition=COMPETITIONS[division],
                    home_team=row["HomeTeam"].strip(),
                    away_team=row["AwayTeam"].strip(),
                    home_goals=home_goals,
                    away_goals=away_goals,
                )
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid Football-Data CSV row {row_number}: {exc}"
                ) from exc

            matches.append(match)

    return matches
