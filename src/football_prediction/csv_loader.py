
"""Load and validate historical match results from CSV files."""

import csv
from datetime import date
from pathlib import Path

from football_prediction.data import MatchResult


REQUIRED_COLUMNS = {
    "match_date",
    "competition",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
}


def load_matches(path: str | Path) -> list[MatchResult]:
    """Read match results from a CSV file and validate every row."""
    csv_path = Path(path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or missing a header")

        headers = [header.strip() for header in reader.fieldnames if header]
        if len(headers) != len(reader.fieldnames):
            raise ValueError("CSV contains an empty column name")

        if len(set(headers)) != len(headers):
            raise ValueError("CSV contains duplicate column names")

        missing = REQUIRED_COLUMNS - set(headers)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        matches = []

        for row_number, row in enumerate(reader, start=2):
            try:
                match = MatchResult(
                    match_date=date.fromisoformat(row["match_date"].strip()),
                    competition=row["competition"].strip(),
                    home_team=row["home_team"].strip(),
                    away_team=row["away_team"].strip(),
                    home_goals=int(row["home_goals"]),
                    away_goals=int(row["away_goals"]),
                )
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid CSV data on row {row_number}: {exc}"
                ) from exc

            matches.append(match)

    return matches
