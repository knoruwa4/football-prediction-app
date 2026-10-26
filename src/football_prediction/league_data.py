"""Find and load the project's locally downloaded main-league seasons."""

from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from pathlib import Path
from collections.abc import Iterable

from football_prediction.data import MatchResult
from football_prediction.football_data import COMPETITIONS, load_football_data


MAIN_LEAGUE_CODES = (
    "E0", "E1", "E2", "E3", "EC",
    "SC0", "SC1", "SC2", "SC3",
    "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2",
    "N1", "B1", "P1", "T1", "G1",
)
FIRST_SEASON_YEAR = 2020
LAST_SEASON_YEAR = 2026
EXTRA_LEAGUE_CODES = (
    "ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN",
    "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "SWZ", "USA",
)

# Country exports include a League field. Keep cup competitions and lower
# divisions out of the league histories used by the prediction models.
EXTRA_SOURCE_LEAGUES = {
    "ARG": "Liga Profesional",
    "AUT": "Bundesliga",
    "BRA": "Serie A",
    "CHN": "Super League",
    "DNK": "Superliga",
    "FIN": "Veikkausliiga",
    "IRL": "Premier Division",
    "JPN": "J1 League",
    "MEX": "Liga MX",
    "NOR": "Eliteserien",
    "POL": "Ekstraklasa",
    "ROU": "Superliga",
    "RUS": "Premier League",
    "SWE": "Allsvenskan",
    "SWZ": "Super League",
    "USA": "MLS",
}


def season_file(
    data_root: str | Path, season_start_year: int, division_code: str
) -> Path:
    """Return the existing source file path for one local season and league."""
    if division_code not in MAIN_LEAGUE_CODES:
        raise ValueError(f"unsupported season-file division code: {division_code!r}")
    return Path(data_root) / f"data {season_start_year}" / f"{division_code}.csv"


def load_league_history(
    data_root: str | Path,
    division_code: str,
    season_start_years: Iterable[int] = range(
        FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1
    ),
) -> list[MatchResult]:
    """Load seasons for one league without reading or copying other leagues.

    The current season file is a partial snapshot; completed rows present in
    that file are included. No other competition is ever added to the result.
    """
    if division_code not in MAIN_LEAGUE_CODES:
        raise ValueError(f"unsupported season-file division code: {division_code!r}")

    matches: list[MatchResult] = []
    seen_fixtures: set[tuple[object, str, str]] = set()
    years = sorted(set(season_start_years))
    if not years:
        raise ValueError("season_start_years must not be empty")

    for year in years:
        path = season_file(data_root, year, division_code)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {COMPETITIONS[division_code]} season file: {path}"
            )
        season_matches = load_football_data(
            path, expected_division_code=division_code
        )
        for match in season_matches:
            key = (
                match.match_date,
                match.home_team.casefold(),
                match.away_team.casefold(),
            )
            if key in seen_fixtures:
                raise ValueError(
                    f"Duplicate {COMPETITIONS[division_code]} fixture: {key}"
                )
            seen_fixtures.add(key)
            matches.append(match)

    return sorted(
        matches,
        key=lambda match: (
            match.match_date,
            match.home_team.casefold(),
            match.away_team.casefold(),
        ),
    )


def load_main_league_histories(
    data_root: str | Path,
    division_codes: Iterable[str] = MAIN_LEAGUE_CODES,
    season_start_years: Iterable[int] = range(
        FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1
    ),
) -> dict[str, list[MatchResult]]:
    """Load requested leagues separately, keyed by Football-Data code."""
    years = tuple(season_start_years)
    histories = {}
    for code in division_codes:
        histories[code] = load_league_history(
            data_root,
            code,
            season_start_years=years,
        )
    return histories


def extra_league_file(data_root: str | Path, country_code: str) -> Path:
    """Return the official all-seasons country file for an extra league."""
    if country_code not in EXTRA_LEAGUE_CODES:
        raise ValueError(f"unsupported extra-league code: {country_code!r}")
    return Path(data_root) / "extra" / f"{country_code}.csv"


def _season_start_year(value: str) -> int:
    """Read provider season labels like 2020 or 2020/2021."""
    label = value.strip()
    start = label.split("/", 1)[0]
    if len(start) != 4 or not start.isdigit():
        raise ValueError(f"invalid season label: {value!r}")
    return int(start)


def _parse_match_date(value: str):
    raw_date = value.strip()
    for date_format in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw_date, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"invalid match date: {value!r}")


def load_extra_league_history(
    data_root: str | Path,
    country_code: str,
    season_start_years: Iterable[int] = range(
        FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1
    ),
) -> list[MatchResult]:
    """Load one requested extra league from its multi-season country CSV.

    Season labels may be calendar years or European year pairs. The first year
    controls selection, so 2020 and 2020/2021 both belong to start year 2020.
    The provider's League column is filtered to the division in the user's
    country list; e.g. Switzerland's Challenge League and Argentina's Copa de
    la Liga are not mixed into the named top-division history.
    """
    if country_code not in EXTRA_LEAGUE_CODES:
        raise ValueError(f"unsupported extra-league code: {country_code!r}")
    years = set(season_start_years)
    if not years:
        raise ValueError("season_start_years must not be empty")

    path = extra_league_file(data_root, country_code)
    if not path.is_file():
        raise FileNotFoundError(f"Missing extra-league source file: {path}")
    content = path.read_bytes()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1252")

    with StringIO(text, newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{country_code} CSV is empty or missing a header")
        field_map = {name.strip().casefold(): name for name in reader.fieldnames if name}
        required = {"season", "date", "league", "home", "away", "hg", "ag"}
        missing = required - set(field_map)
        if missing:
            raise ValueError(
                f"{country_code} CSV is missing columns: {', '.join(sorted(missing))}"
            )

        target_league = EXTRA_SOURCE_LEAGUES[country_code].casefold()
        matches: list[MatchResult] = []
        seen_fixtures: set[tuple[object, str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            try:
                raw_season = row[field_map["season"]].strip()
                if _season_start_year(raw_season) not in years:
                    continue
                if row[field_map["league"]].strip().casefold() != target_league:
                    continue

                raw_home_goals = row[field_map["hg"]].strip()
                raw_away_goals = row[field_map["ag"]].strip()
                if not raw_home_goals and not raw_away_goals:
                    continue
                if not raw_home_goals or not raw_away_goals:
                    raise ValueError("only one full-time score is present")

                match = MatchResult(
                    match_date=_parse_match_date(row[field_map["date"]]),
                    competition=COMPETITIONS[country_code],
                    home_team=row[field_map["home"]].strip(),
                    away_team=row[field_map["away"]].strip(),
                    home_goals=int(raw_home_goals),
                    away_goals=int(raw_away_goals),
                )
                key = (
                    match.match_date,
                    match.home_team.casefold(),
                    match.away_team.casefold(),
                )
                if key in seen_fixtures:
                    raise ValueError(f"duplicate fixture: {key}")
                seen_fixtures.add(key)
                matches.append(match)
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid {country_code} CSV row {row_number}: {exc}"
                ) from exc

    return sorted(
        matches,
        key=lambda match: (
            match.match_date,
            match.home_team.casefold(),
            match.away_team.casefold(),
        ),
    )


def load_all_competition_histories(
    data_root: str | Path,
    season_start_years: Iterable[int] = range(
        FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1
    ),
) -> dict[str, list[MatchResult]]:
    """Load all configured main and extra leagues into separate histories."""
    years = tuple(season_start_years)
    histories = load_main_league_histories(
        data_root, season_start_years=years
    )
    for code in EXTRA_LEAGUE_CODES:
        histories[code] = load_extra_league_history(
            data_root, code, season_start_years=years
        )
    return histories
