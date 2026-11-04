"""Audit all locally downloaded Football-Data league files and coverage."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_prediction.football_data import COMPETITIONS, load_football_data
from football_prediction.league_data import (
    EXTRA_LEAGUE_CODES,
    EXTRA_SOURCE_LEAGUES,
    FIRST_SEASON_YEAR,
    LAST_SEASON_YEAR,
    MAIN_LEAGUE_CODES,
    extra_league_file,
    load_extra_league_history,
    season_file,
)


def main() -> None:
    print("Football-Data.co.uk locally downloaded main-league files\n")
    total = 0
    for code in MAIN_LEAGUE_CODES:
        league_total = 0
        print(f"{COMPETITIONS[code]} ({code})")
        for year in range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1):
            path = season_file(PROJECT_ROOT / "data", year, code)
            if not path.is_file():
                print(f"  {year}/{str(year + 1)[-2:]}: MISSING {path}")
                continue
            matches = load_football_data(path, expected_division_code=code)
            dates = [match.match_date for match in matches]
            count = len(matches)
            league_total += count
            if dates:
                coverage = f"{min(dates)} to {max(dates)}"
            else:
                coverage = "no completed results"
            print(
                f"  {year}/{str(year + 1)[-2:]}: "
                f"{count} completed matches, {coverage}, {path.stat().st_size:,} bytes"
            )
        total += league_total
        print(f"  Total local completed matches: {league_total}\n")
    print(f"Grand total: {total} completed matches")
    print("Extra leagues (multi-season country files, filtered to requested division)\n")
    extra_total = 0
    for code in EXTRA_LEAGUE_CODES:
        path = extra_league_file(PROJECT_ROOT / "data", code)
        print(f"{EXTRA_SOURCE_LEAGUES[code]} ({code})")
        if not path.is_file():
            print(f"  MISSING {path}\n")
            continue
        league_total = 0
        for year in range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1):
            matches = load_extra_league_history(
                PROJECT_ROOT / "data", code, [year]
            )
            league_total += len(matches)
            dates = [match.match_date for match in matches]
            coverage = f"{min(dates)} to {max(dates)}" if dates else "no rows for this division"
            season_label = f"provider season starting {year}"
            print(f"  {season_label}: {len(matches)} matches, {coverage}")
        print(f"  Total selected matches: {league_total:,}\n")
        extra_total += league_total
    print(f"Extra-league total: {extra_total:,} completed matches")
    print(f"Combined main and extra total: {total + extra_total:,} completed matches")
    print("\nThe current 2026/27 main-league files are in-progress snapshots.")


if __name__ == "__main__":
    main()
