
"""Import completed match results from Football-Data.co.uk CSV files."""

import csv
from datetime import datetime
from pathlib import Path

from football_prediction.data import MatchResult


REQUIRED_COLUMNS = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}

COMPETITIONS = {
    "E0": "Premier League",
}


def load_football_data(path: str | Path) -> list[MatchResult]:
    """Load and validate completed matches from a Football-Data CSV."""
    csv_path = Path(path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file is empty or missing a header")

        headers = [header.strip() if header else "" for header in reader.fieldnames]

        if any(not header for header in headers):
            raise ValueError("CSV contains an empty column name")

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

                raw_date = row["Date"].strip()
                match_date = datetime.strptime(raw_date, "%d/%m/%Y").date()

                home_goals = int(row["FTHG"].strip())
                away_goals = int(row["FTAG"].strip())

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
