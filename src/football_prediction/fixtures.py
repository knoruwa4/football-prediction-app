"""Load explicitly sourced fixture schedules from CSV."""

import csv
from datetime import date
from pathlib import Path


REQUIRED_FIXTURE_FIELDS = {
    "match_date",
    "kickoff_utc",
    "home_team",
    "away_team",
    "source_url",
}

HISTORICAL_TEAM_NAMES = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
}


def historical_team_name(display_name: str) -> str:
    """Map official fixture names to names used in historical CSV data."""
    return HISTORICAL_TEAM_NAMES.get(display_name, display_name)


def load_verified_fixtures(
    path: str | Path, match_date: date
) -> list[dict[str, str]]:
    """Load fixtures for one date, requiring a source URL for every row."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_FIXTURE_FIELDS - fields
        if missing:
            raise ValueError(
                "fixture CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        fixtures = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            try:
                fixture_date = date.fromisoformat(row["match_date"].strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid fixture date on CSV row {row_number}"
                ) from exc
            if fixture_date != match_date:
                raise ValueError(
                    f"CSV row {row_number} is for {fixture_date}, "
                    f"expected {match_date}"
                )
            home = row["home_team"].strip()
            away = row["away_team"].strip()
            source = row["source_url"].strip()
            kickoff = row["kickoff_utc"].strip()
            if not home or not away or not kickoff or not source.startswith("https://"):
                raise ValueError(f"Incomplete or unsourced fixture on CSV row {row_number}")
            key = (home.casefold(), away.casefold())
            if key in seen:
                raise ValueError(f"Duplicate fixture on CSV row {row_number}")
            seen.add(key)
            fixtures.append({
                "match_date": fixture_date.isoformat(),
                "kickoff_utc": kickoff,
                "home_team": home,
                "away_team": away,
                "source_url": source,
            })
    return fixtures
